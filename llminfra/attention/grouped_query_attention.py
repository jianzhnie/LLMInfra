"""Grouped-query attention that shares key/value heads across query groups."""

from __future__ import annotations

import torch
from torch import nn

from .base_attention import BaseAttention, validate_attention_inputs


class GroupedQueryAttention(BaseAttention):
    """Group Query Attention module (GQA).

    Implements the attention variant described in "GQA: Training Generalized
    Multi-Query Transformer Models from Multi-Head Checkpoints"
    (Chen et al., 2023). This implementation is a hybrid between
    Multi-Head Attention and
    Multi-Query Attention, where queries are grouped and each group shares a
    single key and value head. This reduces memory usage and speeds up
    inference while maintaining performance closer to Multi-Head Attention
    than Multi-Query Attention.

    By configuring `num_kv_groups` (G, the number of groups), this module supports:
        - When num_kv_groups == num_heads: Multi-Head Attention (MHA)
        - When num_kv_groups == 1: Multi-Query Attention (MQA)
        - When 1 < num_kv_groups < num_heads: Generic Grouped Query Attention (GQA)

    Args:
        hidden_size (int): Dimensionality of the input and output features.
        num_heads (int): Number of query heads to use. Must divide hidden_size evenly.
        num_kv_groups (int): Number of groups to divide query heads into.
            Must divide num_heads evenly.
        dropout (float, optional): Dropout probability for attention weights.
            Defaults to 0.1.
        bias (bool, optional): Whether to use bias in linear projections.
            Defaults to True.
        output_gate (bool, optional): Whether to add a sigmoid output gate
            computed from the layer input and applied before ``o_proj``
            (Qwen3-Next / "Gated Attention" style). Defaults to False.

    Attributes:
        num_heads (int): Number of query heads.
        head_dim (int): Dimensionality of each attention head.
        num_kv_groups (int): Number of groups.
        heads_per_group (int): Number of heads per group.
        scale_factor (float): Scaling factor for dot-product attention.
        q_proj (nn.Linear): Linear projection for query vectors.
        k_proj (nn.Linear): Linear projection for key vectors (one per group).
        v_proj (nn.Linear): Linear projection for value vectors (one per group).
        o_proj (nn.Linear): Linear projection for output vectors.
        dropout (nn.Dropout): Dropout layer for attention weights.

    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_groups: int,
        dropout: float = 0.1,
        bias: bool = True,
        qk_norm: bool = False,
        output_gate: bool = False,
    ) -> None:
        super().__init__(hidden_size, num_heads, dropout, bias, qk_norm)
        if num_heads % num_kv_groups != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible "
                f"by num_kv_groups ({num_kv_groups})"
            )

        self.num_kv_groups = num_kv_groups

        # Number of heads per group
        self.heads_per_group = num_heads // num_kv_groups

        # Linear projections for queries, keys, and values
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(
            hidden_size, self.num_kv_groups * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            hidden_size, self.num_kv_groups * self.head_dim, bias=bias
        )

        # Output projection
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        if output_gate:
            # Qwen3-Next derives the gate by doubling the query projection
            # and chunking it; a separate projection is the equivalent
            # parameterization and keeps the two paths independent.
            self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        self._init_projections(self.q_proj, self.k_proj, self.v_proj, self.o_proj)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the Group Query Attention module.

        Args:
            hidden_state (torch.Tensor): Input tensor of shape (batch_size,
                seq_len, hidden_size).
            attention_mask (Optional[torch.Tensor]): Attention mask broadcastable
                against the (batch_size, num_heads, seq_len, seq_len) scores,
                e.g. a (batch_size, 1, 1, seq_len) padding mask or a full
                per-head mask. 1 indicates positions to attend to, 0 indicates
                positions to mask out.
            return_attention_weights (bool, optional): Whether to return
                attention weights. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                Output tensor of shape (batch_size, seq_len, hidden_size).
                If return_attention_weights is True, returns a tuple
                (output, attention_weights).

        """
        validate_attention_inputs(hidden_state, attention_mask, self.num_heads)

        # Linear projections
        # query has shape (batch_size, seq_len, hidden_size)
        # key and value have shape (batch_size, seq_len, num_kv_groups * head_dim)
        query = self.split_head(self.q_proj(hidden_state))
        key = self.split_head_grouped(self.k_proj(hidden_state))
        value = self.split_head_grouped(self.v_proj(hidden_state))
        query, key = self._apply_qk_norm(query, key)

        # Scaled dot-product attention:
        # (batch_size, num_heads, seq_len, head_dim)
        # * (batch_size, num_heads, head_dim, seq_len)
        # -> (batch_size, num_heads, seq_len, seq_len)
        attention_scores = (
            torch.matmul(query, key.transpose(-1, -2)) * self.scale_factor
        )
        attention_weights = self.compute_attention_weights(
            attention_scores, attention_mask
        )

        # Weighted sum of values, merge heads, optional output gate, projection
        output: torch.Tensor = torch.matmul(attention_weights, value)
        output = self._apply_output_gate(hidden_state, self.combine_head(output))
        output = self.o_proj(output)

        if return_attention_weights:
            return output, attention_weights
        return output

    def split_head_grouped(self, x: torch.Tensor) -> torch.Tensor:
        """Split the input tensor into grouped attention heads for keys and values.

        This method splits keys/values into groups, then expands each to serve
        multiple query heads in the same group.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len,
                num_kv_groups * head_dim).

        Returns:
            torch.Tensor: Tensor of shape (batch_size, num_heads, seq_len, head_dim).

        """
        batch_size, seq_len, _ = x.size()

        # Split into groups: (batch_size, seq_len, num_kv_groups * head_dim)
        # -> (batch_size, num_kv_groups, seq_len, head_dim)
        x = x.view(batch_size, seq_len, self.num_kv_groups, self.head_dim).transpose(
            1, 2
        )

        # Expand each group's key/value to serve multiple query heads
        # (batch_size, num_kv_groups, seq_len, head_dim)
        # -> (batch_size, num_kv_groups, heads_per_group, seq_len, head_dim)
        x = x.unsqueeze(2).expand(
            batch_size, self.num_kv_groups, self.heads_per_group, seq_len, self.head_dim
        )

        # Reshape to match query heads: (batch_size, num_heads, seq_len, head_dim)
        return x.reshape(batch_size, self.num_heads, seq_len, self.head_dim)

    def extra_repr(self) -> str:
        """Return a string representation of the module's extra information."""
        return (
            f"{super().extra_repr()}, num_kv_groups={self.num_kv_groups}, "
            f"heads_per_group={self.heads_per_group}"
        )


__all__ = ["GroupedQueryAttention"]
