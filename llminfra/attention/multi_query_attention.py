"""Multi-query attention where all query heads share one key/value head."""

from __future__ import annotations

import torch
from torch import nn

from .base_attention import BaseAttention, validate_attention_inputs


class MultiQueryAttention(BaseAttention):
    """Multi-Query Attention module (MQA).

    Implements the attention variant described in "Fast Transformer Decoding:
    One Write-Head is All You Need" (Shazeer, 2019). This implementation
    uses a single key and value head for all query heads,
    which reduces memory usage and speeds up inference compared to standard
    Multi-Head Attention.

    Args:
        hidden_size (int): Dimensionality of the input and output features.
        num_heads (int): Number of query heads to use. Must divide hidden_size evenly.
        dropout (float, optional): Dropout probability for attention weights.
            Defaults to 0.1.
        bias (bool, optional): Whether to use bias in linear projections.
            Defaults to True.

    Attributes:
        num_heads (int): Number of query heads.
        head_dim (int): Dimensionality of each attention head.
        scale_factor (float): Scaling factor for dot-product attention.
        q_proj (nn.Linear): Linear projection for query vectors.
        k_proj (nn.Linear): Linear projection for key vectors (single head).
        v_proj (nn.Linear): Linear projection for value vectors (single head).
        o_proj (nn.Linear): Linear projection for output vectors.
        dropout (nn.Dropout): Dropout layer for attention weights.

    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        qk_norm: bool = False,
    ) -> None:
        super().__init__(hidden_size, num_heads, dropout, bias, qk_norm)

        # Projection matrices: multiple heads for queries, single head for
        # keys and values
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.head_dim, bias=bias)

        # Output projection
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        self._init_projections(self.q_proj, self.k_proj, self.v_proj, self.o_proj)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the Multi-Query Attention module.

        Args:
            hidden_state (torch.Tensor): Input tensor of shape (batch_size,
                seq_len, hidden_size).
            attention_mask (Optional[torch.Tensor]): Attention mask broadcastable
                against the (batch_size, num_heads, seq_len, seq_len) scores,
                e.g. a (batch_size, 1, 1, seq_len) padding mask or a full
                per-head mask. 1 indicates positions to attend to, 0 indicates
                positions to mask out.
            return_attention_weights (bool): Whether to return attention
                weights along with the output. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                Output tensor of shape (batch_size, seq_len, hidden_size).
                If return_attention_weights is True, returns a tuple
                (output, attention_weights).

        """
        validate_attention_inputs(hidden_state, attention_mask, self.num_heads)

        # Linear projections; queries get all heads, keys and values share one
        query = self.split_head(self.q_proj(hidden_state))
        key = self.split_head(self.k_proj(hidden_state), num_heads=1)
        value = self.split_head(self.v_proj(hidden_state), num_heads=1)
        query, key = self._apply_qk_norm(query, key)

        # Scaled dot-product attention; the single key/value head broadcasts
        # over all query heads:
        # (batch_size, num_heads, seq_len, head_dim)
        # * (batch_size, 1, head_dim, seq_len)
        # -> (batch_size, num_heads, seq_len, seq_len)
        attention_scores = (
            torch.matmul(query, key.transpose(-1, -2)) * self.scale_factor
        )
        attention_weights = self.compute_attention_weights(
            attention_scores, attention_mask
        )

        # Weighted sum of values (broadcast again), merge heads, projection
        output: torch.Tensor = torch.matmul(attention_weights, value)
        output = self.o_proj(self.combine_head(output))

        if return_attention_weights:
            return output, attention_weights
        return output

    def forward_with_cache(
        self,
        hidden_state: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with a KV cache for incremental decoding.

        Computes queries/keys/values for the ``q_len`` new positions in
        ``hidden_state`` only, appends the new keys/values to
        ``past_key_value`` (along the sequence dimension) when given, and
        attends over the full ``past_len + q_len`` key/value set. The cache
        keeps the single shared key/value head, so this is where MQA's
        memory saving over MHA comes from.

        Args:
            hidden_state (torch.Tensor): New-token input of shape
                (batch_size, q_len, hidden_size).
            past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]]):
                Cached ``(key, value)`` from previous steps, each of shape
                (batch_size, 1, past_len, head_dim).
            attention_mask (Optional[torch.Tensor]): Keep-mask broadcastable
                against the (batch_size, num_heads, q_len, past_len + q_len)
                scores; same 1/0 semantics as :meth:`forward`, with the key
                dimension spanning past and new positions.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: The
                output of shape (batch_size, q_len, hidden_size) and the
                updated ``(key, value)`` cache including the new positions.

        """
        if hidden_state.dim() != 3:
            raise ValueError(f"hidden_state must be 3D, got {hidden_state.dim()}D")
        query = self.split_head(self.q_proj(hidden_state))
        key = self.split_head(self.k_proj(hidden_state), num_heads=1)
        value = self.split_head(self.v_proj(hidden_state), num_heads=1)
        query, key = self._apply_qk_norm(query, key)
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)
        if attention_mask is not None and attention_mask.size(-1) != key.size(2):
            raise ValueError(
                f"attention_mask key length {attention_mask.size(-1)} must "
                f"match the cached key length {key.size(2)}"
            )

        attention_scores = (
            torch.matmul(query, key.transpose(-1, -2)) * self.scale_factor
        )
        attention_weights = self.compute_attention_weights(
            attention_scores, attention_mask
        )

        output: torch.Tensor = torch.matmul(attention_weights, value)
        output = self.o_proj(self.combine_head(output))
        return output, (key, value)
