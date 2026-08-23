"""Two-dimensional (block, within-block) position embedding."""

from __future__ import annotations

import torch
from torch import nn

from .base import BasePositionalEncoding


class TwoDimensionalPositionEmbedding(BasePositionalEncoding):
    """Simple 2D position embedding for long-document layouts."""

    def __init__(
        self,
        embedding_dim: int,
        max_blocks: int,
        max_positions_per_block: int,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.max_blocks = int(max_blocks)
        self.max_positions_per_block = int(max_positions_per_block)
        self.block_embeddings = nn.Embedding(max_blocks, embedding_dim)
        self.position_embeddings = nn.Embedding(max_positions_per_block, embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        block_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add block and within-block embeddings to ``x``."""
        seq_len = x.size(-2)
        if block_ids is None:
            block_ids = (
                torch.arange(seq_len, device=x.device) // self.max_positions_per_block
            )
        if positions is None:
            positions = (
                torch.arange(seq_len, device=x.device) % self.max_positions_per_block
            )
        if block_ids.numel() and (
            block_ids.min() < 0 or block_ids.max() >= self.max_blocks
        ):
            raise ValueError("block_ids must be in [0, max_blocks)")
        if positions.numel() and (
            positions.min() < 0 or positions.max() >= self.max_positions_per_block
        ):
            raise ValueError("positions must be in [0, max_positions_per_block)")
        block: torch.Tensor = self.block_embeddings(block_ids)
        position: torch.Tensor = self.position_embeddings(positions)
        if block.dim() == x.dim() - 1:
            block = block.unsqueeze(0)
            position = position.unsqueeze(0)
        return x + block + position
