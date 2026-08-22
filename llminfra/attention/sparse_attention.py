"""Learned sparse-attention compositions used by recent long-context models.

The implementations in this module are PyTorch reference paths. They expose
the architecture and masking semantics of DeepSeek Sparse Attention (DSA),
MiniMax Sparse Attention (MSA), and Hierarchical Compressed Attention (HCA),
but do not replace the fused block-gather kernels used by production systems.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from .base_attention import BaseAttention, validate_attention_inputs
from .block_sparse_attention import BlockSparseAttention
from .compressed_sparse_attention import CompressedSparseAttention


class QueryKeyBlockIndexer(nn.Module):
    """Select key blocks from learned query/key block representations.

    Args:
        hidden_size: Input feature dimension.
        num_index_heads: Number of independently learned index heads.
        block_size: Tokens summarized into one index block.
        top_k: Key blocks selected for every query block.
        max_seq_len: Maximum supported sequence length.
        index_dim: Per-index-head projection dimension. Defaults to the input
            dimension divided across index heads.
        causal: Restrict every query block to itself and earlier key blocks.

    Notes:
        Hard ``topk`` indices are discrete. Use :meth:`routing_scores` when an
        auxiliary distillation or ranking loss should train the indexer.

    """

    def __init__(
        self,
        hidden_size: int,
        num_index_heads: int,
        block_size: int,
        top_k: int,
        max_seq_len: int,
        index_dim: int | None = None,
        causal: bool = True,
    ) -> None:
        super().__init__()
        if min(hidden_size, num_index_heads, block_size, top_k, max_seq_len) < 1:
            raise ValueError("indexer dimensions and top_k must be >= 1")
        index_dim = index_dim or max(8, hidden_size // num_index_heads)
        if index_dim < 1:
            raise ValueError("index_dim must be >= 1")

        self.hidden_size = int(hidden_size)
        self.num_index_heads = int(num_index_heads)
        self.block_size = int(block_size)
        self.top_k = int(top_k)
        self.max_seq_len = int(max_seq_len)
        self.index_dim = int(index_dim)
        self.causal = bool(causal)

        projection_size = self.num_index_heads * self.index_dim
        self.query_proj = nn.Linear(hidden_size, projection_size, bias=False)
        self.key_proj = nn.Linear(hidden_size, projection_size, bias=False)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Return indices shaped ``(batch, index_heads, q_blocks, top_k)``."""
        scores = self.routing_scores(hidden_state)
        block_count = scores.size(-1)
        rows: list[torch.Tensor] = []
        for query_block in range(block_count):
            candidates = (
                scores[:, :, query_block, : query_block + 1]
                if self.causal
                else scores[:, :, query_block]
            )
            count = min(self.top_k, candidates.size(-1))
            selected = torch.topk(candidates, count, dim=-1).indices
            if count < self.top_k:
                fallback = query_block if self.causal else block_count - 1
                selected = F.pad(
                    selected,
                    (0, self.top_k - count),
                    value=fallback,
                )
            rows.append(selected)
        if not rows:
            # Empty sequence: no query blocks to select for.
            return scores.new_zeros(
                scores.size(0), scores.size(1), 0, self.top_k, dtype=torch.long
            )
        return torch.stack(rows, dim=2)

    def routing_scores(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Return differentiable query-to-key block scores.

        The result has shape ``(batch, index_heads, q_blocks, kv_blocks)``.
        Causally invalid entries are set to the minimum finite value of the
        score dtype so a downstream ranking loss can still remain finite.
        """
        if hidden_state.dim() != 3 or hidden_state.size(-1) != self.hidden_size:
            raise ValueError(
                "hidden_state must have shape (batch, seq_len, hidden_size)"
            )
        _, seq_len, _ = hidden_state.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        blocks = self._pool_blocks(hidden_state)
        batch_size, block_count, _ = blocks.shape
        query = self.query_proj(blocks).view(
            batch_size, block_count, self.num_index_heads, self.index_dim
        )
        key = self.key_proj(blocks).view(
            batch_size, block_count, self.num_index_heads, self.index_dim
        )
        scores = torch.einsum("bqhd,bkhd->bhqk", query, key)
        scores = scores / math.sqrt(self.index_dim)
        if self.causal:
            allowed = torch.tril(
                torch.ones(
                    block_count,
                    block_count,
                    dtype=torch.bool,
                    device=hidden_state.device,
                )
            )
            scores = scores.masked_fill(
                ~allowed[None, None], torch.finfo(scores.dtype).min
            )
        return scores

    def _pool_blocks(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Mean-pool blocks without biasing a final partial block."""
        batch_size, seq_len, hidden_size = hidden_state.shape
        block_count = math.ceil(seq_len / self.block_size)
        padded_len = block_count * self.block_size
        padded = F.pad(hidden_state, (0, 0, 0, padded_len - seq_len))
        block_sum = padded.view(
            batch_size, block_count, self.block_size, hidden_size
        ).sum(dim=2)
        valid = torch.arange(padded_len, device=hidden_state.device) < seq_len
        counts = valid.view(block_count, self.block_size).sum(dim=1)
        return block_sum / counts.to(hidden_state.dtype)[None, :, None]

    def extra_repr(self) -> str:
        """Return a string representation of the module's extra information."""
        return (
            f"hidden_size={self.hidden_size}, num_index_heads={self.num_index_heads}, "
            f"block_size={self.block_size}, top_k={self.top_k}, "
            f"index_dim={self.index_dim}, causal={self.causal}"
        )


class DynamicSparseAttention(BaseAttention):
    """DSA-style learned block selection followed by exact sparse GQA.

    The indexer uses one learned selector per query head. Production DSA also
    relies on specialized index training objectives and fused gather kernels;
    those concerns remain explicit extension points here.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        block_size: int = 64,
        top_k: int = 8,
        max_seq_len: int = 4096,
        num_kv_groups: int | None = None,
        index_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = True,
    ) -> None:
        super().__init__(hidden_size, num_heads, dropout, bias)
        self.indexer = QueryKeyBlockIndexer(
            hidden_size,
            num_heads,
            block_size,
            top_k,
            max_seq_len,
            index_dim=index_dim,
            causal=causal,
        )
        self.sparse_attention = BlockSparseAttention(
            hidden_size,
            num_heads,
            block_size,
            num_kv_groups=num_kv_groups,
            top_k=top_k,
            dropout=dropout,
            bias=bias,
            causal=causal,
        )

    def select_blocks(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Return the hard block choices used by the sparse attention path."""
        # Module calls return Any in torch's stubs; the indexer yields a Tensor.
        return cast(torch.Tensor, self.indexer(hidden_state))

    def routing_scores(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Return differentiable index scores for an auxiliary routing loss."""
        return self.indexer.routing_scores(hidden_state)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Select blocks and execute exact attention on selected tokens."""
        validate_attention_inputs(hidden_state, attention_mask, self.num_heads)
        return cast(
            "torch.Tensor | tuple[torch.Tensor, torch.Tensor]",
            self.sparse_attention(
                hidden_state,
                attention_mask=attention_mask,
                return_attention_weights=return_attention_weights,
                block_indices=self.select_blocks(hidden_state),
            ),
        )


class MiniMaxSparseAttention(DynamicSparseAttention):
    """MSA-style group-specific block selection on top of GQA.

    Unlike :class:`DynamicSparseAttention`, index heads are shared by all
    query heads in the same KV group. This mirrors MSA's per-GQA-group index
    branch while retaining the readable :class:`BlockSparseAttention` path.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        block_size: int = 64,
        top_k: int = 8,
        max_seq_len: int = 4096,
        num_kv_groups: int | None = None,
        index_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = True,
    ) -> None:
        num_kv_groups = num_heads if num_kv_groups is None else num_kv_groups
        if num_heads % num_kv_groups != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_groups ({num_kv_groups})"
            )
        BaseAttention.__init__(self, hidden_size, num_heads, dropout, bias)
        self.num_kv_groups = int(num_kv_groups)
        self.heads_per_group = num_heads // num_kv_groups
        self.indexer = QueryKeyBlockIndexer(
            hidden_size,
            num_kv_groups,
            block_size,
            top_k,
            max_seq_len,
            index_dim=index_dim,
            causal=causal,
        )
        self.sparse_attention = BlockSparseAttention(
            hidden_size,
            num_heads,
            block_size,
            num_kv_groups=num_kv_groups,
            top_k=top_k,
            dropout=dropout,
            bias=bias,
            causal=causal,
        )

    def select_blocks(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Return group selections expanded over their query heads."""
        group_indices: torch.Tensor = self.indexer(hidden_state)
        return group_indices.repeat_interleave(self.heads_per_group, dim=1)

    def routing_scores(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Return differentiable per-KV-group routing scores."""
        return self.indexer.routing_scores(hidden_state)


class HierarchicalCompressedAttention(BaseAttention):
    """HCA-style mixture of fine sparse and coarse global compressed paths.

    The fine branch selects a small number of lightly compressed entries. The
    coarse branch uses a larger compression ratio and exposes every available
    compressed entry. A learnable sigmoid gate mixes both outputs.

    ``return_attention_weights=True`` returns weights from the fine branch;
    the coarse branch has a different key length and cannot share one tensor.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        fine_compress_ratio: int = 4,
        coarse_compress_ratio: int = 16,
        fine_top_k: int = 8,
        max_seq_len: int = 4096,
        num_kv_groups: int | None = None,
        gate_init: float = 0.0,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = True,
    ) -> None:
        super().__init__(hidden_size, num_heads, dropout, bias)
        if fine_compress_ratio < 1 or coarse_compress_ratio <= fine_compress_ratio:
            raise ValueError("compression ratios must satisfy 1 <= fine < coarse")
        if max_seq_len < coarse_compress_ratio:
            raise ValueError("max_seq_len must cover at least one coarse block")
        self.fine_compress_ratio = int(fine_compress_ratio)
        self.coarse_compress_ratio = int(coarse_compress_ratio)
        self.fine_attention = CompressedSparseAttention(
            hidden_size,
            num_heads,
            compress_ratio=fine_compress_ratio,
            num_kv_groups=num_kv_groups,
            top_k=fine_top_k,
            causal=causal,
            dropout=dropout,
            bias=bias,
        )
        self.coarse_attention = CompressedSparseAttention(
            hidden_size,
            num_heads,
            compress_ratio=coarse_compress_ratio,
            num_kv_groups=num_kv_groups,
            top_k=math.ceil(max_seq_len / coarse_compress_ratio),
            causal=causal,
            dropout=dropout,
            bias=bias,
        )
        self.mix_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run fine and coarse compressed paths and mix their outputs."""
        validate_attention_inputs(hidden_state, attention_mask, self.num_heads)
        seq_len = hidden_state.size(1)
        if seq_len % self.coarse_compress_ratio != 0:
            raise ValueError(
                "seq_len must be divisible by coarse_compress_ratio "
                f"({seq_len} % {self.coarse_compress_ratio} != 0)"
            )
        fine_result = self.fine_attention(
            hidden_state,
            attention_mask=attention_mask,
            return_attention_weights=return_attention_weights,
        )
        if return_attention_weights:
            fine_output, fine_weights = fine_result
        else:
            fine_output = fine_result
        coarse_output = self.coarse_attention(
            hidden_state,
            attention_mask=attention_mask,
        )
        gate = torch.sigmoid(self.mix_logit).to(hidden_state.dtype)
        output: torch.Tensor = gate * fine_output + (1.0 - gate) * coarse_output
        if return_attention_weights:
            return output, fine_weights
        return output

    def extra_repr(self) -> str:
        """Return a string representation of the module's extra information."""
        return (
            f"{super().extra_repr()}, fine_compress_ratio="
            f"{self.fine_compress_ratio}, coarse_compress_ratio="
            f"{self.coarse_compress_ratio}"
        )


DeepSeekSparseAttention = DynamicSparseAttention


__all__ = [
    "DeepSeekSparseAttention",
    "DynamicSparseAttention",
    "HierarchicalCompressedAttention",
    "MiniMaxSparseAttention",
    "QueryKeyBlockIndexer",
]
