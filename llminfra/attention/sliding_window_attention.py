"""Educational Sliding Window Attention (SWA).

SWA restricts each query to a fixed-size window of recent key/value
positions. It is used by Mistral 7B and by the local layers of Gemma 2/4.
This module is a teaching implementation: it still materializes the score
matrix, but the mask makes it easy to reason about the supported attention
pattern and to compare against full attention.
"""

from __future__ import annotations

import torch
from torch import nn

from .base_attention import BaseAttention, validate_attention_inputs


class SlidingWindowAttention(BaseAttention):
    """Grouped-Query Attention restricted to a sliding local window.

    Args:
        hidden_size: Dimensionality of input and output features.
        num_heads: Number of query heads.
        window_size: Number of key/value positions visible to each query,
            the query position itself included. This matches the
            ``sliding_window`` convention of Mistral and HuggingFace
            transformers: a causal query at position ``i`` attends to keys in
            ``[i - window_size + 1, i]``.
        num_kv_groups: Number of shared key/value head groups. Defaults to
            ``num_heads``, which is plain MHA plus the window mask.
        dropout: Dropout probability for attention weights.
        bias: Whether linear projections use biases.
        causal: If True, the window only covers positions ``<= query``.

    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        window_size: int,
        num_kv_groups: int | None = None,
        dropout: float = 0.1,
        bias: bool = True,
        causal: bool = True,
    ) -> None:
        super().__init__(hidden_size, num_heads, dropout, bias)
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")

        num_kv_groups = num_heads if num_kv_groups is None else num_kv_groups
        if num_heads % num_kv_groups != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible "
                f"by num_kv_groups ({num_kv_groups})"
            )

        self.window_size = int(window_size)
        self.num_kv_groups = int(num_kv_groups)
        self.heads_per_group = self.num_heads // self.num_kv_groups
        self.causal = bool(causal)

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(
            hidden_size, self.num_kv_groups * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            hidden_size, self.num_kv_groups * self.head_dim, bias=bias
        )
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        self._init_projections(self.q_proj, self.k_proj, self.v_proj, self.o_proj)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run sliding-window attention over ``hidden_state``."""
        validate_attention_inputs(hidden_state, attention_mask, self.num_heads)
        _batch_size, seq_len, _ = hidden_state.size()

        query = self.split_head(self.q_proj(hidden_state))
        key = self._split_kv(self.k_proj(hidden_state))
        value = self._split_kv(self.v_proj(hidden_state))

        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale_factor
        # Keep the window mask at (1, 1, q, kv): ``masked_fill`` broadcasts
        # it, so there is no need to materialize a per-batch/per-head copy.
        sliding_mask = self._sliding_mask(seq_len, key.size(-2), scores.device)
        combined_mask = self._combine_with_input_mask(sliding_mask, attention_mask)

        attention_weights = self.compute_attention_weights(scores, combined_mask)
        output: torch.Tensor = torch.matmul(attention_weights, value)
        output = self.o_proj(self.combine_head(output))

        if return_attention_weights:
            return output, attention_weights
        return output

    def _split_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Split grouped K/V projections and expand them to query heads."""
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_kv_groups, self.head_dim).transpose(
            1, 2
        )
        x = x.unsqueeze(2).expand(
            batch_size,
            self.num_kv_groups,
            self.heads_per_group,
            seq_len,
            self.head_dim,
        )
        return x.reshape(batch_size, self.num_heads, seq_len, self.head_dim)

    def _sliding_mask(
        self, q_len: int, kv_len: int, device: torch.device
    ) -> torch.Tensor:
        """Return a ``(1, 1, q_len, kv_len)`` boolean window mask."""
        offset = kv_len - q_len
        q_pos = torch.arange(q_len, device=device).view(-1, 1)
        k_pos = torch.arange(kv_len, device=device).view(1, -1)
        distance = q_pos - k_pos + offset
        if self.causal:
            # Mistral/HF convention: the query position is part of the
            # window, so a query sees ``window_size`` positions in total.
            allowed = (distance >= 0) & (distance < self.window_size)
        else:
            allowed = distance.abs() <= self.window_size
        return allowed[None, None]

    @staticmethod
    def _combine_with_input_mask(
        sliding_mask: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Combine the boolean window mask with a user-provided attention mask."""
        if attention_mask is None:
            return sliding_mask
        if attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        return sliding_mask & attention_mask.bool()

    def extra_repr(self) -> str:
        """Return a string representation of the module's extra information."""
        return (
            f"{super().extra_repr()}, window_size={self.window_size}, "
            f"num_kv_groups={self.num_kv_groups}, causal={self.causal}"
        )
