"""Reference implementation of Kimi Delta Attention (KDA).

KDA is a gated delta-rule linear attention.  The recurrent state stores a
key/value map; each token first decays that map, then writes the prediction
error ``v - S.T @ k`` with a learned update rate.  This implementation keeps
the recurrence explicit for readability and testing.  Production KDA uses
causal convolutions, chunked scans, fused kernels, and persistent decode
state, which are deliberately outside this reference module.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .base_attention import BaseAttention, validate_attention_inputs


class KimiDeltaAttention(BaseAttention):
    """Causal KDA recurrence with per-head, per-key-dimension decay.

    Args:
        hidden_size: Input and output feature dimension.
        num_heads: Number of recurrent heads.
        feature_dim: Query/key state dimension. Defaults to ``head_dim``.
        beta_init: Initial logit for the scalar delta update gate.
        decay_init: Initial raw value for the per-dimension decay gate.
        output_gate: Apply a learned sigmoid gate before the output projection.
        dropout: Dropout probability applied to the projected output.
        bias: Whether projections use bias terms.

    Inputs use the common ``BaseAttention`` shape ``(batch, seq, hidden)``.
    The recurrence is inherently causal, so ``return_attention_weights`` is
    rejected because KDA does not materialize a quadratic attention matrix.
    ``attention_mask`` is interpreted as a key-padding mask; arbitrary dense
    query/key masks are not representable by this linear recurrence.

    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        feature_dim: int | None = None,
        beta_init: float = -2.0,
        decay_init: float = -1.0,
        output_gate: bool = True,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__(hidden_size, num_heads, dropout, bias)
        self.feature_dim = self.head_dim if feature_dim is None else int(feature_dim)
        if self.feature_dim < 1:
            raise ValueError("feature_dim must be >= 1")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be in [0, 1]")

        self.beta_init = float(beta_init)
        self.decay_init = float(decay_init)
        self.use_output_gate = bool(output_gate)
        projection_size = num_heads * self.feature_dim

        self.q_proj = nn.Linear(hidden_size, projection_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, projection_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=bias)
        self.decay_proj = nn.Linear(hidden_size, projection_size, bias=bias)
        self.gate_proj = (
            nn.Linear(hidden_size, hidden_size, bias=bias)
            if self.use_output_gate
            else None
        )
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        projections = [
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.beta_proj,
            self.decay_proj,
            self.o_proj,
        ]
        if self.gate_proj is not None:
            projections.append(self.gate_proj)
        self._init_projections(*projections)
        if self.beta_proj.bias is not None:
            nn.init.constant_(self.beta_proj.bias, self.beta_init)
        if self.decay_proj.bias is not None:
            nn.init.constant_(self.decay_proj.bias, self.decay_init)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the recurrent delta-rule update over a token sequence."""
        if return_attention_weights:
            raise ValueError(
                "KimiDeltaAttention does not materialize attention weights"
            )
        batch_size, seq_len = validate_attention_inputs(
            hidden_state, attention_mask, self.num_heads
        )
        query = self._split(self.q_proj(hidden_state))
        key = self._split(self.k_proj(hidden_state))
        value = self.split_head(self.v_proj(hidden_state))

        # KDA normalizes Q/K before applying the delta rule.  Scaling Q keeps
        # the output magnitude comparable to softmax attention.
        query = F.normalize(query, dim=-1, eps=1e-6) / math.sqrt(self.feature_dim)
        key = F.normalize(key, dim=-1, eps=1e-6)
        beta = torch.sigmoid(self.beta_proj(hidden_state)).transpose(1, 2)
        beta = beta.unsqueeze(-1)
        raw_decay = self._split(self.decay_proj(hidden_state))
        decay = -F.softplus(raw_decay)

        valid = self._key_padding_mask(attention_mask, batch_size, seq_len)
        if valid is not None:
            valid_f = valid.to(dtype=hidden_state.dtype)
            key = key * valid_f.unsqueeze(-1)
            value = value * valid_f.unsqueeze(-1)
            beta = beta * valid_f.unsqueeze(-1)
            # A padding token must not decay the state of a real sequence.
            decay = decay * valid_f.unsqueeze(-1)

        state = hidden_state.new_zeros(
            batch_size,
            self.num_heads,
            self.feature_dim,
            self.head_dim,
        )
        # Hoist step-invariant ops out of the recurrence: one batched exp and
        # broadcast reshape instead of seq_len small ones.
        decay_exp = decay.exp().unsqueeze(-1)  # (batch, heads, seq, f, 1)
        beta_u = beta.unsqueeze(-1)  # (batch, heads, seq, 1, 1)
        outputs: list[torch.Tensor] = []
        for step in range(seq_len):
            key_t = key[:, :, step]
            value_t = value[:, :, step]
            state = state * decay_exp[:, :, step]
            # (b, h, 1, f) @ (b, h, f, d) contracts the feature dim; plain
            # matmul avoids einsum's per-call equation parsing, which
            # dominates these tiny steps.
            prediction = torch.matmul(key_t.unsqueeze(-2), state).squeeze(-2)
            error = value_t - prediction
            # addcmul fuses ``state + beta * (k ⊗ error)`` into one kernel.
            state = torch.addcmul(
                state, beta_u[:, :, step], key_t.unsqueeze(-1) * error.unsqueeze(-2)
            )
            output_t = torch.matmul(query[:, :, step].unsqueeze(-2), state).squeeze(-2)
            outputs.append(output_t)

        # Empty sequence: the recurrence never ran; an empty slice of
        # ``value`` keeps the autograd graph alive.
        core_output = torch.stack(outputs, dim=2) if outputs else value[:, :, :0]
        core_output = self.combine_head(core_output)
        if self.gate_proj is not None:
            core_output = core_output * torch.sigmoid(self.gate_proj(hidden_state))
        output: torch.Tensor = self.o_proj(core_output)
        if valid is not None:
            output = output * valid.transpose(1, 2).to(output.dtype)
        output = self.dropout(output)
        return output

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """Split ``(batch, seq, heads * feature_dim)`` projections."""
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.feature_dim).transpose(
            1, 2
        )

    @staticmethod
    def _key_padding_mask(
        attention_mask: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor | None:
        """Normalize a key-padding mask to ``(batch, 1, seq)`` booleans."""
        if attention_mask is None:
            return None
        if attention_mask.dim() == 4:
            # The recurrence itself is causal. Reducing over query positions
            # extracts the key-padding component from a combined causal mask.
            mask = attention_mask.any(dim=-2)
        elif attention_mask.dim() == 3:
            mask = (
                attention_mask
                if attention_mask.size(1) == 1
                else attention_mask.any(dim=1, keepdim=True)
            )
        else:
            raise ValueError("attention_mask must be 3D or 4D")
        if mask.size(0) != batch_size or mask.size(-1) != seq_len:
            raise ValueError("attention_mask must match batch and sequence lengths")
        return mask.bool()

    def extra_repr(self) -> str:
        """Return a string representation of the module's extra information."""
        return (
            f"{super().extra_repr()}, feature_dim={self.feature_dim}, "
            f"beta_init={self.beta_init}, decay_init={self.decay_init}, "
            f"output_gate={self.use_output_gate}"
        )


KDAAttention = KimiDeltaAttention

__all__ = ["KDAAttention", "KimiDeltaAttention"]
