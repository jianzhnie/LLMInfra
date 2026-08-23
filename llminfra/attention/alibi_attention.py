"""ALiBi attention that adds linear position biases to attention scores."""

from __future__ import annotations

import torch
from torch import nn

from ..positional import ALiBiBias
from .base_attention import BaseAttention, validate_attention_inputs


class ALiBiAttention(BaseAttention):
    """Grouped-query attention with ALiBi additive position biases."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_groups: int | None = None,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__(hidden_size, num_heads, dropout, bias)
        num_kv_groups = num_heads if num_kv_groups is None else num_kv_groups
        if num_heads % num_kv_groups != 0:
            raise ValueError("num_heads must be divisible by num_kv_groups")
        self.num_kv_groups = int(num_kv_groups)
        self.heads_per_group = self.num_heads // self.num_kv_groups
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(
            hidden_size, self.num_kv_groups * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            hidden_size, self.num_kv_groups * self.head_dim, bias=bias
        )
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.alibi = ALiBiBias(num_heads, max_seq_len)
        self._init_projections(self.q_proj, self.k_proj, self.v_proj, self.o_proj)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run GQA attention with ALiBi bias."""
        validate_attention_inputs(hidden_state, attention_mask, self.num_heads)
        _batch_size, seq_len, _ = hidden_state.size()
        query = self.split_head(self.q_proj(hidden_state))
        key = self._split_kv(self.k_proj(hidden_state))
        value = self._split_kv(self.v_proj(hidden_state))

        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale_factor
        scores = scores + self.alibi(seq_len).to(scores.dtype)
        weights = self.compute_attention_weights(scores, attention_mask)
        output: torch.Tensor = torch.matmul(weights, value)
        output = self.o_proj(self.combine_head(output))
        if return_attention_weights:
            return output, weights
        return output

    def _split_kv(self, x: torch.Tensor) -> torch.Tensor:
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

    def extra_repr(self) -> str:
        """Return a string representation of the module's extra information."""
        return f"{super().extra_repr()}, num_kv_groups={self.num_kv_groups}"


__all__ = ["ALiBiAttention"]
