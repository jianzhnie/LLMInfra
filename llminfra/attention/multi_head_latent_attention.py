"""Multi-head latent attention that projects Q/K/V through latent spaces."""

from __future__ import annotations

import torch
from torch import nn

from .base_attention import BaseAttention, validate_attention_inputs


class MultiHeadLatentAttention(BaseAttention):
    """Multi-head Latent Attention module that operates on latent space representations.

    This implementation extends the standard multi-head attention mechanism to
    work with latent space representations: queries, keys and values are
    produced by projecting the input down to a latent space and back up,
    allowing for more flexible and powerful attention patterns.

    Simplifications relative to DeepSeek-V2/V3 MLA: there is no decoupled
    RoPE key/query segment, no RMSNorm on the latent states, no YaRN mscale
    adjustment, and the K/V head dims equal the Q head dim. The structure
    (shared KV down-projection plus separate K/V up-projections) matches the
    DeepSeek ``kv_a_proj_with_mqa``/``kv_b_proj`` decomposition.

    Args:
        hidden_size (int): Dimensionality of the input and output features.
        num_heads (int): Number of attention heads to use. Must divide
            hidden_size evenly.
        q_latent_size (int): Latent dimension of the query branch.
        kv_latent_size (int): Latent dimension shared by the key and value
            branches.
        dropout (float, optional): Dropout probability for attention weights.
            Defaults to 0.0.
        bias (bool, optional): Whether to use bias in linear projections.
            Defaults to True.

    Attributes:
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        q_latent_size (int): Latent dimension of the query branch.
        kv_latent_size (int): Latent dimension of the key/value branch.
        scale_factor (float): Scaling factor for dot-product attention.
        q_down_proj (nn.Linear): Query projection into latent space.
        q_up_proj (nn.Linear): Query projection back to hidden size.
        kv_down_proj (nn.Linear): Shared key/value projection into latent space.
        k_up_proj (nn.Linear): Key projection back to hidden size.
        v_up_proj (nn.Linear): Value projection back to hidden size.
        output_proj (nn.Linear): Linear projection for output vectors.
        dropout (nn.Dropout): Dropout layer for attention weights.

    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        q_latent_size: int,
        kv_latent_size: int,
        dropout: float = 0.0,
        bias: bool = True,
        qk_norm: bool = False,
    ) -> None:
        super().__init__(hidden_size, num_heads, dropout, bias, qk_norm)

        self.q_latent_size = q_latent_size
        self.kv_latent_size = kv_latent_size

        # Projections for Q, K, V (operating through latent spaces)
        self.q_down_proj = nn.Linear(hidden_size, q_latent_size, bias=bias)
        self.q_up_proj = nn.Linear(q_latent_size, hidden_size, bias=bias)
        self.kv_down_proj = nn.Linear(hidden_size, kv_latent_size, bias=bias)
        self.k_up_proj = nn.Linear(kv_latent_size, hidden_size, bias=bias)
        self.v_up_proj = nn.Linear(kv_latent_size, hidden_size, bias=bias)

        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        self._init_projections(
            self.q_down_proj,
            self.q_up_proj,
            self.kv_down_proj,
            self.k_up_proj,
            self.v_up_proj,
            self.output_proj,
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the Multi-head Latent Attention module.

        Args:
            hidden_state (torch.Tensor): Input tensor of shape (batch_size,
                seq_len, hidden_size).
            attention_mask (Optional[torch.Tensor]): Attention mask broadcastable
                against the (batch_size, num_heads, seq_len, seq_len) scores,
                e.g. a (batch_size, 1, 1, seq_len) padding mask or a full
                per-head mask. 1 indicates positions to attend to, 0 indicates
                positions to mask out.
            return_attention_weights (bool, optional): If True, returns
                attention weights. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                Output tensor of shape (batch_size, seq_len, hidden_size).
                If return_attention_weights is True, returns a tuple
                (output, attention_weights).

        """
        validate_attention_inputs(hidden_state, attention_mask, self.num_heads)

        # Query branch: project down to the latent space and back up
        query = self.q_up_proj(self.q_down_proj(hidden_state))

        # Key/value branches: share the down-projection into latent space
        latent_state = self.kv_down_proj(hidden_state)
        key = self.k_up_proj(latent_state)
        value = self.v_up_proj(latent_state)

        # Split into multiple heads
        query = self.split_head(query)
        key = self.split_head(key)
        value = self.split_head(value)
        query, key = self._apply_qk_norm(query, key)

        # Scaled dot-product attention
        attention_scores = (
            torch.matmul(query, key.transpose(-1, -2)) * self.scale_factor
        )
        attention_weights = self.compute_attention_weights(
            attention_scores, attention_mask
        )

        # Weighted sum of values, merge heads, output projection
        output: torch.Tensor = torch.matmul(attention_weights, value)
        output = self.output_proj(self.combine_head(output))

        if return_attention_weights:
            return output, attention_weights
        return output

    def extra_repr(self) -> str:
        """Return a string representation of the module's extra information."""
        return (
            f"{super().extra_repr()}, q_latent_size={self.q_latent_size}, "
            f"kv_latent_size={self.kv_latent_size}"
        )
