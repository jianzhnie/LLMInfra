"""Educational block-sparse attention.

Modern long-context models such as MiniMax M3 (MSA), DeepSeek-V4 (CSA/DSA)
and GLM-5 use sparse selection at block granularity. This module provides a
small, PyTorch-only approximation of that pattern: key/value blocks are
selected per query block and attention is then computed over the selected
blocks only. A real production implementation would additionally fuse the
selection with a CUDA kernel and page table; this file intentionally keeps
the mathematical structure readable.
"""

from __future__ import annotations

import torch
from torch import nn

from .base_attention import BaseAttention, validate_attention_inputs


class BlockSparseAttention(BaseAttention):
    """Grouped-Query attention over a subset of key/value blocks.

    Args:
        hidden_size: Dimensionality of input and output features.
        num_heads: Number of query heads.
        block_size: Number of key/value tokens per block.
        num_kv_groups: Number of shared key/value head groups.
        top_k: Default number of blocks selected per query block when
            ``block_indices`` is not supplied. The fallback pattern is a
            local causal window over blocks.
        dropout: Dropout probability for attention weights.
        bias: Whether linear projections use biases.
        causal: Whether to enforce causal masking in addition to block sparsity.

    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        block_size: int,
        num_kv_groups: int | None = None,
        top_k: int = 1,
        dropout: float = 0.1,
        bias: bool = True,
        causal: bool = True,
    ) -> None:
        super().__init__(hidden_size, num_heads, dropout, bias)
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        num_kv_groups = num_heads if num_kv_groups is None else num_kv_groups
        if num_heads % num_kv_groups != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible "
                f"by num_kv_groups ({num_kv_groups})"
            )

        self.block_size = int(block_size)
        self.num_kv_groups = int(num_kv_groups)
        self.heads_per_group = self.num_heads // self.num_kv_groups
        self.top_k = int(top_k)
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
        block_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run block-sparse attention.

        ``block_indices`` may have shape ``(num_heads, num_q_blocks, top_k)``
        or ``(batch, num_heads, num_q_blocks, top_k)``. If omitted, a simple
        local-block fallback is used.
        """
        validate_attention_inputs(hidden_state, attention_mask, self.num_heads)
        batch_size, seq_len, _ = hidden_state.size()

        query = self.split_head(self.q_proj(hidden_state))
        key = self._split_kv(self.k_proj(hidden_state))
        value = self._split_kv(self.v_proj(hidden_state))

        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale_factor
        sparse_mask = self._build_mask(
            batch_size=batch_size,
            q_len=seq_len,
            kv_len=key.size(-2),
            block_indices=block_indices,
            device=scores.device,
        )
        combined_mask = self._combine_with_input_mask(sparse_mask, attention_mask)

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

    def _build_mask(
        self,
        *,
        batch_size: int,
        q_len: int,
        kv_len: int,
        block_indices: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a ``(batch, heads, q_len, kv_len)`` allowed-block mask."""
        if q_len == 0:
            # ``q_block.max()`` below is undefined on empty sequences.
            return torch.zeros(
                batch_size, self.num_heads, 0, kv_len, dtype=torch.bool, device=device
            )
        q_block = torch.arange(q_len, device=device) // self.block_size
        k_block = torch.arange(kv_len, device=device) // self.block_size
        num_q_blocks = int(q_block.max()) + 1
        num_kv_blocks = int(k_block.max()) + 1

        if block_indices is None:
            block_indices = self._fallback_block_indices(
                batch_size, num_q_blocks, device
            )
        elif block_indices.dim() == 3:
            block_indices = block_indices.unsqueeze(0).expand(batch_size, -1, -1, -1)

        if block_indices.dim() != 4:
            raise ValueError(
                "block_indices must have shape (heads, q_blocks, top_k) or "
                "(batch, heads, q_blocks, top_k)"
            )
        if block_indices.size(0) != batch_size:
            raise ValueError("block_indices batch size must match hidden_state")
        if block_indices.size(1) != self.num_heads:
            raise ValueError("block_indices head dimension must match num_heads")
        if block_indices.size(2) != num_q_blocks:
            raise ValueError(
                f"block_indices has {block_indices.size(2)} query blocks, "
                f"expected {num_q_blocks}"
            )
        if block_indices.numel() == 0 or (
            block_indices.min() < 0 or block_indices.max() >= num_kv_blocks
        ):
            raise ValueError("block_indices contains an out-of-range block id")

        selected = torch.zeros(
            batch_size,
            self.num_heads,
            num_q_blocks,
            num_kv_blocks,
            dtype=torch.bool,
            device=device,
        )
        selected.scatter_(
            dim=-1,
            index=block_indices.to(device=device),
            value=True,
        )
        # Two consecutive small gathers measurably beat one fused 4D
        # fancy-index lookup when expanding blocks to token granularity.
        allowed = selected[:, :, q_block][:, :, :, k_block]
        if self.causal:
            # Right-align the query window against the KV window: query i is
            # the global position i + (kv_len - q_len), as in cached decoding.
            offset = kv_len - q_len
            q_pos = torch.arange(q_len, device=device).view(-1, 1)
            k_pos = torch.arange(kv_len, device=device).view(1, -1)
            allowed = allowed & ((q_pos + offset) >= k_pos)[None, None]
        return allowed

    def _fallback_block_indices(
        self, batch_size: int, num_q_blocks: int, device: torch.device
    ) -> torch.Tensor:
        """Select a local causal window of blocks when no indexer is supplied."""
        rows = []
        for qb in range(num_q_blocks):
            if self.causal:
                start = max(0, qb - self.top_k + 1)
            else:
                start = max(0, qb - self.top_k // 2)
            stop = min(num_q_blocks, start + self.top_k)
            start = max(0, stop - self.top_k)
            indices = torch.full(
                (batch_size, self.num_heads, self.top_k),
                num_q_blocks - 1,
                device=device,
                dtype=torch.long,
            )
            count = stop - start
            if count:
                indices[..., :count] = (
                    torch.arange(start, stop, device=device).unsqueeze(0).unsqueeze(0)
                )
            rows.append(indices)
        return torch.stack(rows, dim=2)

    @staticmethod
    def _combine_with_input_mask(
        sparse_mask: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Combine the boolean block mask with a user-provided attention mask."""
        if attention_mask is None:
            return sparse_mask
        if attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        return sparse_mask & attention_mask.bool()

    def extra_repr(self) -> str:
        """Return a string representation of the module's extra information."""
        return (
            f"{super().extra_repr()}, block_size={self.block_size}, "
            f"num_kv_groups={self.num_kv_groups}, top_k={self.top_k}, "
            f"causal={self.causal}"
        )
