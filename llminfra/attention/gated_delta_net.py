"""Educational Gated DeltaNet implementation.

Gated DeltaNet is the linear-attention family behind Qwen3-Next and Kimi
Delta Attention (KDA). The exact production variants use chunkwise
parallelism and custom kernels; this module keeps the recurrent gated
state update readable for study and small experiments.

Despite the historical name, the update implemented here is gated *linear*
attention, ``S = (1 - g) S + g · k ⊗ v``: the gate interpolates between the
old state and the new key/value outer product. The delta rule proper —
with the error-correction term ``v - Sᵀk`` — is implemented in
``kimi_delta_attention.KimiDeltaAttention``.
"""

from __future__ import annotations

import torch
from torch import nn

from .base_attention import BaseAttention, validate_attention_inputs


class GatedDeltaNet(BaseAttention):
    """Recurrent gated linear attention, ``S = (1 - g) S + g · k ⊗ v``.

    The sigmoid write gate interpolates between the previous state and the
    new key/value outer product. This is *not* the delta rule: there is no
    error-correction term ``v - Sᵀk``. See
    :class:`~llminfra.attention.kimi_delta_attention.KimiDeltaAttention` for
    a delta-rule implementation.

    Args:
        hidden_size: Dimensionality of input and output features.
        num_heads: Number of attention heads.
        feature_dim: Dimension of the key/query feature space.
        beta_init: Initial logit offset for the write gate.
        dropout: Dropout probability applied to the output.
        bias: Whether linear projections use biases.
        normalize: If True, divide the recurrent output by the accumulated
            gate denominator. This is a teaching convenience; exact models
            may not use it.

    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        feature_dim: int | None = None,
        beta_init: float = -2.0,
        dropout: float = 0.0,
        bias: bool = True,
        normalize: bool = False,
    ) -> None:
        super().__init__(hidden_size, num_heads, dropout, bias)
        self.feature_dim = feature_dim or self.head_dim
        self.beta_init = float(beta_init)
        self.normalize = bool(normalize)

        self.q_proj = nn.Linear(hidden_size, num_heads * self.feature_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.feature_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.beta_proj = nn.Linear(hidden_size, num_heads * self.feature_dim, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self._init_projections(
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.beta_proj,
            self.o_proj,
        )
        if self.beta_proj.bias is not None:
            nn.init.constant_(self.beta_proj.bias, self.beta_init)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the recurrent Gated DeltaNet forward pass.

        Gated DeltaNet does not materialize an attention matrix, so
        ``return_attention_weights=True`` is not supported.
        """
        if return_attention_weights:
            raise ValueError("GatedDeltaNet does not materialize attention weights")
        validate_attention_inputs(hidden_state, attention_mask, self.num_heads)
        batch_size, seq_len, _ = hidden_state.size()

        query = self._split(self.q_proj(hidden_state))
        key = self._split(self.k_proj(hidden_state))
        value = self.split_head(self.v_proj(hidden_state))
        gate = torch.sigmoid(self._split(self.beta_proj(hidden_state)))

        key_mask = self._key_padding_mask(attention_mask, batch_size)
        if key_mask is not None:
            key = key * key_mask.unsqueeze(-1)
            value = value * key_mask.unsqueeze(-1)
            gate = gate * key_mask.unsqueeze(-1)

        state = torch.zeros(
            batch_size,
            self.num_heads,
            self.feature_dim,
            self.head_dim,
            device=hidden_state.device,
            dtype=hidden_state.dtype,
        )
        outputs: list[torch.Tensor] = []

        # Hoist step-invariant tensor ops out of the recurrence: the gate
        # complements and (for ``normalize``) the per-step scalar gates are
        # computed once for the whole sequence instead of per step.
        gate_u = gate.unsqueeze(-1)  # (batch, heads, seq, feature_dim, 1)
        keep_gate = 1.0 - gate_u
        gate_scalar = gate.mean(dim=-1) if self.normalize else None
        denominator = torch.zeros(
            batch_size,
            self.num_heads,
            1,
            device=hidden_state.device,
            dtype=hidden_state.dtype,
        )

        for step in range(seq_len):
            gate_t = gate_u[:, :, step]  # (batch, heads, feature_dim, 1)
            key_t = key[:, :, step]
            value_t = value[:, :, step]

            # addcmul fuses ``keep * state + gate * (k ⊗ v)`` into one kernel.
            state = torch.addcmul(
                keep_gate[:, :, step] * state,
                gate_t,
                key_t.unsqueeze(-1) * value_t.unsqueeze(-2),
            )
            # (batch, heads, 1, f) @ (batch, heads, f, d); cheaper per step
            # than einsum, whose equation parsing dominates tiny matmuls.
            output_t = torch.matmul(query[:, :, step].unsqueeze(-2), state).squeeze(-2)
            if self.normalize:
                # ``gate_scalar`` is always computed when normalize is set.
                assert gate_scalar is not None
                scalar_t = gate_scalar[:, :, step].unsqueeze(-1)
                denominator = (1.0 - scalar_t) * denominator + scalar_t
                safe_denominator = denominator.clamp_min(
                    torch.finfo(denominator.dtype).eps
                )
                output_t = output_t / safe_denominator
            outputs.append(output_t)

        if outputs:
            output: torch.Tensor = torch.stack(outputs, dim=2)
        else:
            # Empty sequence: the recurrence never ran; an empty slice of
            # ``value`` keeps the autograd graph alive.
            output = value[:, :, :0]
        output = self.o_proj(self.combine_head(output))
        if self.training and self.dropout_prob > 0:
            output = self.dropout(output)
        return output

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """Split a projection into ``(batch, heads, seq, feature_dim)``."""
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.feature_dim).transpose(
            1, 2
        )

    @staticmethod
    def _key_padding_mask(
        attention_mask: torch.Tensor | None, batch_size: int
    ) -> torch.Tensor | None:
        """Convert a BaseAttention-style mask to a ``(batch, 1, seq)`` mask."""
        if attention_mask is None:
            return None
        if attention_mask.dim() == 4:
            # Causal structure is already enforced by the recurrence. Reduce
            # over query positions to retain only key-padding visibility.
            attention_mask = attention_mask.any(dim=-2)
        elif attention_mask.dim() != 3:
            raise ValueError("attention_mask must be 3D or 4D")
        elif attention_mask.size(1) != 1:
            attention_mask = attention_mask.any(dim=1, keepdim=True)
        if attention_mask.size(0) != batch_size:
            raise ValueError("attention_mask batch size must match hidden_state")
        return attention_mask.bool()

    def extra_repr(self) -> str:
        """Return a string representation of the module's extra information."""
        return (
            f"{super().extra_repr()}, feature_dim={self.feature_dim}, "
            f"beta_init={self.beta_init}, normalize={self.normalize}"
        )
