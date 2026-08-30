"""Educational Gated Linear Attention (GLA) implementation.

GLA (Yang et al., "Gated Linear Attention Transformers with
Hardware-Efficient Training", arXiv:2312.06635) extends linear attention
with a data-dependent, per-feature gate::

    S_t = S_{t-1} ⊙ g_t + k_t^T v_t,   o_t = q_t S_t

where ``g_t = sigmoid(W_g x_t)`` (optionally through a low-rank projection,
as in the paper's parameter-efficient variant). Unlike the interpolating
update ``S = (1 - g) S + g · k^T v`` of
:class:`~llminfra.attention.gated_delta_net.GatedDeltaNet`, the GLA gate
multiplies the old state directly: ``g = 1`` preserves it (ungated linear
attention), ``g = 0`` resets it at every step. The chunked parallel form and
the step-by-step recurrence are algebraically equivalent; both are exposed
for study. The production GLA layer adds short convolutions, an output gate
and fused chunkwise kernels (see the ``fla`` library), which are
deliberately omitted here.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .base_attention import BaseAttention, validate_attention_inputs


class GatedLinearAttention(BaseAttention):
    """Gated linear attention with a data-dependent per-feature state gate.

    Args:
        hidden_size: Dimensionality of input and output features.
        num_heads: Number of attention heads.
        feature_dim: Dimension of the query/key/gate feature space. Defaults
            to ``head_dim``.
        gate_rank: Bottleneck rank of the gate projection. ``None`` uses a
            single ``hidden_size -> num_heads * feature_dim`` projection; an
            integer uses the paper's low-rank variant
            ``hidden_size -> num_heads * gate_rank -> num_heads * feature_dim``.
        chunk_size: Tokens per chunk in the parallel forward pass. It is a
            performance knob and does not change the result within the
            numerical envelope described below.
        dropout: Dropout applied to the output when training.
        bias: Whether linear projections use biases.

    Teaching simplifications: queries are scaled by ``1 / sqrt(feature_dim)``
    to keep magnitudes comparable to softmax attention; the chunked path
    factors the cumulative log-gate into the queries and keys
    (``q ⊙ exp(log_cum)``, ``k ⊙ exp(-log_cum)``). The ``exp(-log_cum)``
    factor grows by ``1 / g_t`` per step, so it overflows once a chunk's
    cumulative ``-log g`` exceeds the dtype's exponential range (~88 in
    float32, ~11 in float16/bfloat16). Saturation is not required: for
    roughly zero-mean gate logits the average ``-log g`` per step is at
    least ``log 2``, so even benign gates overflow float32 when a chunk
    exceeds ~128 tokens, and half precision overflows well within the
    default chunk size. The remedies are a smaller ``chunk_size``, float32
    activations, or the recurrent path, which is the numerically stable
    reference. Masked (padding) positions neither write to nor decay the
    state: their gate is forced to 1 and their key/value are zeroed,
    matching the padding semantics of the other recurrent modules in this
    package.

    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        feature_dim: int | None = None,
        gate_rank: int | None = None,
        chunk_size: int = 64,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__(hidden_size, num_heads, dropout, bias)
        self.feature_dim = self.head_dim if feature_dim is None else int(feature_dim)
        if self.feature_dim < 1:
            raise ValueError("feature_dim must be >= 1")
        if gate_rank is not None and gate_rank < 1:
            raise ValueError(f"gate_rank must be >= 1, got {gate_rank}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self.gate_rank = gate_rank
        self.chunk_size = int(chunk_size)
        self.scale = 1.0 / math.sqrt(self.feature_dim)

        projection_size = num_heads * self.feature_dim
        self.q_proj = nn.Linear(hidden_size, projection_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, projection_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.g_proj: nn.Linear | None = None
        self.g_down_proj: nn.Linear | None = None
        self.g_up_proj: nn.Linear | None = None
        projections = [self.q_proj, self.k_proj, self.v_proj]
        if gate_rank is None:
            self.g_proj = nn.Linear(hidden_size, projection_size, bias=bias)
            projections.append(self.g_proj)
        else:
            self.g_down_proj = nn.Linear(hidden_size, num_heads * gate_rank, bias=bias)
            self.g_up_proj = nn.Linear(
                num_heads * gate_rank, projection_size, bias=bias
            )
            projections.extend([self.g_down_proj, self.g_up_proj])
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        projections.append(self.o_proj)
        self._init_projections(*projections)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the chunked parallel GLA forward pass over ``hidden_state``.

        GLA does not materialize an ``(n, n)`` attention matrix, so
        ``return_attention_weights=True`` is not supported.
        """
        if return_attention_weights:
            raise ValueError(
                "GatedLinearAttention does not materialize attention weights"
            )
        query, key, value, gate = self._prepare(hidden_state, attention_mask)
        output = self._chunked_scan(query, key, value, gate)
        return self._finalize(output)

    def recurrent_forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the step-by-step recurrence ``S = S ⊙ g + k^T v``.

        Numerically equivalent to :meth:`forward`; provided for study and for
        cross-checking the chunked parallel form.
        """
        if return_attention_weights:
            raise ValueError(
                "GatedLinearAttention does not materialize attention weights"
            )
        query, key, value, gate = self._prepare(hidden_state, attention_mask)
        output = self._recurrent_scan(query, key, value, gate)
        return self._finalize(output)

    def _prepare(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project to q/k/v/gate and apply the key-padding mask.

        Returns ``(query, key, value, gate)``, each ``(batch, heads, seq, *)``
        with the gate in ``(0, 1)``. Masked positions get zeroed key/value
        (no write) and a gate of exactly 1 (no decay).
        """
        batch_size, _ = validate_attention_inputs(
            hidden_state, attention_mask, self.num_heads
        )
        query = self._split(self.q_proj(hidden_state)) * self.scale
        key = self._split(self.k_proj(hidden_state))
        value = self.split_head(self.v_proj(hidden_state))

        if self.g_proj is not None:
            gate_logits = self.g_proj(hidden_state)
        else:
            # ``gate_rank`` is set, so the low-rank pair exists.
            assert self.g_down_proj is not None and self.g_up_proj is not None
            gate_logits = self.g_up_proj(self.g_down_proj(hidden_state))
        gate = torch.sigmoid(self._split(gate_logits))

        key_mask = self._key_padding_mask(attention_mask, batch_size)
        if key_mask is not None:
            key = key * key_mask.unsqueeze(-1)
            value = value * key_mask.unsqueeze(-1)
            # A padding token must not decay the state of a real sequence.
            gate = torch.where(key_mask.unsqueeze(-1), gate, torch.ones_like(gate))
        return query, key, value, gate

    def _chunked_scan(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """Chunked parallel form with cumulative log-gates.

        With ``log_cum_i = sum_{t<=i} log g_t`` inside a chunk, key ``j``
        reaches query ``i >= j`` with decay ``exp(log_cum_i - log_cum_j)``,
        which factors into ``q ⊙ exp(log_cum)`` and ``k ⊙ exp(-log_cum)``.
        Keys from previous chunks enter through the running state scaled by
        ``exp(log_cum_i)``; the state is advanced by the chunk's total decay
        ``exp(log_cum_last)`` plus each key/value pair decayed by its
        distance to the chunk end.
        """
        batch_size, _, seq_len, _ = query.shape
        if seq_len == 0:
            # The chunk loop never runs for empty sequences; an empty slice of
            # ``value`` keeps the autograd graph alive.
            return value
        state = query.new_zeros(
            batch_size, self.num_heads, self.feature_dim, self.head_dim
        )
        out_parts: list[torch.Tensor] = []
        # Causal masks are identical for equal-length chunks; build each
        # distinct one only once per forward pass.
        future_masks: dict[int, torch.Tensor] = {}

        for start in range(0, seq_len, self.chunk_size):
            stop = min(start + self.chunk_size, seq_len)
            length = stop - start
            q_chunk = query[:, :, start:stop]
            k_chunk = key[:, :, start:stop]
            v_chunk = value[:, :, start:stop]
            # Gates are strictly positive (sigmoid), so log is finite.
            log_cum = torch.log(gate[:, :, start:stop]).cumsum(dim=2)

            q_tilde = q_chunk * log_cum.exp()
            k_tilde = k_chunk * (-log_cum).exp()
            scores = torch.matmul(q_tilde, k_tilde.transpose(-1, -2))
            future = future_masks.get(length)
            if future is None:
                future = torch.triu(
                    torch.ones(length, length, dtype=torch.bool, device=query.device),
                    diagonal=1,
                )
                future_masks[length] = future
            scores = scores.masked_fill(future, 0.0)
            local = torch.matmul(scores, v_chunk)
            history = torch.matmul(q_tilde, state)
            out_parts.append(local + history)

            chunk_log_decay = log_cum[:, :, -1]  # (batch, heads, feature_dim)
            state = state * chunk_log_decay.exp().unsqueeze(-1) + torch.matmul(
                (k_chunk * (chunk_log_decay.unsqueeze(2) - log_cum).exp()).transpose(
                    -1, -2
                ),
                v_chunk,
            )

        return torch.cat(out_parts, dim=2)

    def _recurrent_scan(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """Step-by-step recurrence ``S_t = S_{t-1} ⊙ g_t + k_t^T v_t``."""
        batch_size, _, seq_len, _ = query.shape
        state = query.new_zeros(
            batch_size, self.num_heads, self.feature_dim, self.head_dim
        )
        gate_u = gate.unsqueeze(-1)  # (batch, heads, seq, feature_dim, 1)
        outputs: list[torch.Tensor] = []
        for step in range(seq_len):
            state = state * gate_u[:, :, step] + (
                key[:, :, step].unsqueeze(-1) * value[:, :, step].unsqueeze(-2)
            )
            # (batch, heads, 1, f) @ (batch, heads, f, d); cheaper per step
            # than einsum, whose equation parsing dominates tiny matmuls.
            output_t = torch.matmul(query[:, :, step].unsqueeze(-2), state).squeeze(-2)
            outputs.append(output_t)

        if outputs:
            return torch.stack(outputs, dim=2)
        # Empty sequence: the recurrence never ran; an empty slice of
        # ``value`` keeps the autograd graph alive.
        return value[:, :, :0]

    def _finalize(self, output: torch.Tensor) -> torch.Tensor:
        """Combine heads, project and apply dropout."""
        projected: torch.Tensor = self.o_proj(self.combine_head(output))
        if self.training and self.dropout_prob > 0:
            projected = self.dropout(projected)
        return projected

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
            f"gate_rank={self.gate_rank}, chunk_size={self.chunk_size}"
        )


__all__ = ["GatedLinearAttention"]
