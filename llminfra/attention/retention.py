"""Educational Retention (RetNet) implementation.

Retention (Sun et al., "Retentive Network: A Successor to Transformer for
Large Language Models", arXiv:2307.08621) replaces softmax attention with a
linear recurrence whose key/value state decays by a fixed per-head constant
``gamma_h = 1 - 2**(-5 - h)`` (paper Eq. 5, head index ``h = 1..num_heads``).
The parallel form ``(Q K^T ⊙ D) V`` with the causal decay mask
``D[n, m] = gamma_h ** (n - m)`` and the recurrent form
``S_n = gamma_h * S_{n-1} + k_n^T v_n``, ``o_n = q_n S_n`` are algebraically
equivalent; this module exposes both for study. The full RetNet block adds a
short convolution, a swish output gate and GroupNorm, which are deliberately
omitted here.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .base_attention import BaseAttention, validate_attention_inputs


class Retention(BaseAttention):
    """Multi-scale retention with fixed per-head exponential decay.

    Args:
        hidden_size: Dimensionality of input and output features.
        num_heads: Number of retention heads.
        feature_dim: Dimension of the query/key feature space. Defaults to
            ``head_dim``.
        chunk_size: Tokens per chunk in the parallel forward pass. It is a
            performance knob and does not change the result; the chunked and
            recurrent forms are algebraically equivalent.
        dropout: Dropout applied to the output when training.
        bias: Whether linear projections use biases.

    Teaching simplifications: queries are scaled by ``1 / sqrt(feature_dim)``
    to keep magnitudes comparable to softmax attention (the paper folds the
    scaling into its weight parametrization), and masked (padding) positions
    neither write to nor decay the state, matching the padding semantics of
    the other recurrent modules in this package.

    """

    # Per-head decay constants, registered as a (non-persistent) buffer.
    head_decay: torch.Tensor

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        feature_dim: int | None = None,
        chunk_size: int = 64,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__(hidden_size, num_heads, dropout, bias)
        self.feature_dim = self.head_dim if feature_dim is None else int(feature_dim)
        if self.feature_dim < 1:
            raise ValueError("feature_dim must be >= 1")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self.chunk_size = int(chunk_size)
        self.scale = 1.0 / math.sqrt(self.feature_dim)

        # Paper Eq. 5: head h (1-indexed) decays by gamma_h = 1 - 2**(-5 - h).
        head_ids = torch.arange(1, num_heads + 1, dtype=torch.float64)
        two = torch.tensor(2.0, dtype=torch.float64)
        head_decay = 1.0 - torch.pow(two, -5.0 - head_ids)
        self.register_buffer(
            "head_decay", head_decay.to(torch.float32), persistent=False
        )

        projection_size = num_heads * self.feature_dim
        self.q_proj = nn.Linear(hidden_size, projection_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, projection_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self._init_projections(self.q_proj, self.k_proj, self.v_proj, self.o_proj)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the chunked parallel retention over ``hidden_state``.

        Retention does not materialize an ``(n, n)`` attention matrix, so
        ``return_attention_weights=True`` is not supported.
        """
        if return_attention_weights:
            raise ValueError("Retention does not materialize attention weights")
        query, key, value, _valid, counts = self._prepare(hidden_state, attention_mask)
        output = self._chunked_retention(query, key, value, counts)
        return self._finalize(output)

    def recurrent_forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the step-by-step recurrent retention, ``S = gamma * S + k^T v``.

        Numerically equivalent to :meth:`forward`; provided for study and for
        cross-checking the chunked parallel form.
        """
        if return_attention_weights:
            raise ValueError("Retention does not materialize attention weights")
        query, key, value, valid, counts = self._prepare(hidden_state, attention_mask)
        del counts  # the recurrence reads the per-step validity directly
        output = self._recurrent_retention(query, key, value, valid)
        return self._finalize(output)

    def _prepare(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project to q/k/v and derive the per-step validity and decay counts.

        ``counts[b, 1, t]`` is the number of valid tokens up to and including
        position ``t``: the effective decay exponent between two positions is
        the difference of their counts, so masked steps neither write nor
        decay the state.
        """
        batch_size, seq_len = validate_attention_inputs(
            hidden_state, attention_mask, self.num_heads
        )
        query = self._split(self.q_proj(hidden_state)) * self.scale
        key = self._split(self.k_proj(hidden_state))
        value = self.split_head(self.v_proj(hidden_state))

        key_mask = self._key_padding_mask(attention_mask, batch_size)
        if key_mask is None:
            valid = torch.ones(
                batch_size, 1, seq_len, dtype=torch.bool, device=hidden_state.device
            )
        else:
            valid = key_mask
            key = key * valid.unsqueeze(-1)
            value = value * valid.unsqueeze(-1)
        counts = valid.to(hidden_state.dtype).cumsum(dim=2)
        return query, key, value, valid, counts

    def _chunked_retention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        """Chunked parallel form ``(Q K^T ⊙ D) V`` plus a carried state.

        Within a chunk, query row ``i`` attends to key column ``j <= i`` with
        decay ``gamma ** (counts_i - counts_j)``. Keys from previous chunks
        enter through the running state, scaled by ``gamma ** counts_i`` for
        the distance to the chunk boundary; the state itself is advanced by
        the chunk's decayed key/value sum. Masked positions have flat counts,
        so they contribute no decay and (already zeroed) no key/value write.
        """
        batch_size, _, seq_len, _ = query.shape
        if seq_len == 0:
            # The chunk loop never runs for empty sequences; an empty slice of
            # ``value`` keeps the autograd graph alive.
            return value
        state = query.new_zeros(
            batch_size, self.num_heads, self.feature_dim, self.head_dim
        )
        decay = self.head_decay.view(1, self.num_heads, 1, 1)
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
            # Chunk-local counts: valid tokens since the chunk boundary. The
            # running state holds sums already decayed up to that boundary.
            c_chunk = counts[:, :, start:stop]  # (batch, 1, length)
            if start > 0:
                c_chunk = c_chunk - counts[:, :, start - 1 : start]

            # D[i, j] = gamma ** (counts_i - counts_j) for j <= i else 0.
            exponent = (c_chunk.unsqueeze(-1) - c_chunk.unsqueeze(-2)).clamp_min(0.0)
            future = future_masks.get(length)
            if future is None:
                future = torch.triu(
                    torch.ones(length, length, dtype=torch.bool, device=query.device),
                    diagonal=1,
                )
                future_masks[length] = future
            decay_mask = (decay**exponent).masked_fill(future, 0.0)

            scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2)) * decay_mask
            local = torch.matmul(scores, v_chunk)
            # Distance from the chunk boundary to query i is counts_i steps.
            row_decay = decay ** c_chunk.unsqueeze(-1)  # (batch, heads, length, 1)
            history = torch.matmul(q_chunk, state) * row_decay
            out_parts.append(local + history)

            # Advance the state by the whole chunk: decay the history by the
            # chunk's valid-token count, then add each key/value pair decayed
            # by its distance to the chunk end.
            chunk_steps = c_chunk[:, :, -1:]  # (batch, 1, 1)
            state_decay = decay ** chunk_steps.unsqueeze(-1)  # (batch, heads, 1, 1)
            key_decay = decay ** (chunk_steps - c_chunk).unsqueeze(-1)
            state = state * state_decay + torch.matmul(
                (k_chunk * key_decay).transpose(-1, -2), v_chunk
            )

        return torch.cat(out_parts, dim=2)

    def _recurrent_retention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Step-by-step recurrence ``S_t = gamma * S_{t-1} + k_t^T v_t``."""
        batch_size, _, seq_len, _ = query.shape
        state = query.new_zeros(
            batch_size, self.num_heads, self.feature_dim, self.head_dim
        )
        # Hoist step-invariant ops out of the recurrence: one batched where
        # instead of seq_len small ones. Masked steps keep the state (decay 1)
        # and write nothing (key/value are zeroed by the caller).
        decay_seq = torch.where(
            valid.unsqueeze(-1),
            self.head_decay.view(1, self.num_heads, 1, 1),
            1.0,
        ).unsqueeze(-1)  # (batch, heads, seq, 1, 1)
        outputs: list[torch.Tensor] = []
        for step in range(seq_len):
            state = state * decay_seq[:, :, step] + (
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
            # Causal structure is already enforced by the decay mask. Reduce
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
            f"chunk_size={self.chunk_size}"
        )


__all__ = ["Retention"]
