"""Educational Compressed Sparse Attention (CSA) implementation.

CSA compresses a group of key/value tokens into one entry and then applies
block-sparse selection. This is a simplified version of the compressed
attention used by DeepSeek-V4 and GLM-5.
"""

from __future__ import annotations

import torch
from torch import nn

from .base_attention import BaseAttention, validate_attention_inputs


class CompressedSparseAttention(BaseAttention):
    """Grouped-query attention over compressed key/value blocks.

    Args:
        hidden_size: Dimensionality of input and output features.
        num_heads: Number of query heads.
        compress_ratio: Number of original key/value tokens per entry. The
            sequence length must be divisible by it.
        num_kv_groups: Number of shared key/value head groups.
        top_k: Number of compressed blocks selected per query block.
        causal: Apply causal masking. A compressed entry becomes visible to a
            query only once its *last* source token is in the past.
        dropout: Dropout probability for attention weights.
        bias: Whether linear projections use biases.

    Note:
        ``attention_mask`` is compressed to block granularity: a block is
        visible if any of its tokens is valid. The compressed K/V mean still
        averages in masked tokens, which is a deliberate teaching
        simplification.

    Note:
        In causal mode the first ``compress_ratio - 1`` query positions
        attend to nothing: entry 0 only becomes visible once its last source
        token (position ``compress_ratio - 1``) is in the past, and the
        fallback selection for the first query block also only points at
        entry 0. Their attention weights are all-masked, so those positions
        produce an all-zero output (via ``nan_to_num``).

    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        compress_ratio: int,
        num_kv_groups: int | None = None,
        top_k: int = 4,
        causal: bool = True,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__(hidden_size, num_heads, dropout, bias)
        if compress_ratio < 1 or top_k < 1:
            raise ValueError("compress_ratio and top_k must be >= 1")
        num_kv_groups = num_heads if num_kv_groups is None else num_kv_groups
        if num_heads % num_kv_groups != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible "
                f"by num_kv_groups ({num_kv_groups})"
            )
        self.compress_ratio = int(compress_ratio)
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
        """Run compressed sparse attention over ``hidden_state``.

        Args:
            hidden_state: Input of shape ``(batch, seq_len, hidden_size)``;
                ``seq_len`` must be divisible by ``compress_ratio``.
            attention_mask: Optional key padding mask, ``(batch, 1, 1, seq)``
                or broadcastable; compressed to block granularity (see the
                class docstring for the simplification).
            return_attention_weights: Also return the sparse weights.
            block_indices: Optional explicit selected blocks, shape
                ``(heads, num_q_blocks, top_k)`` or
                ``(batch, heads, num_q_blocks, top_k)``.

        """
        validate_attention_inputs(hidden_state, attention_mask, self.num_heads)
        batch_size, seq_len, _ = hidden_state.size()
        if seq_len % self.compress_ratio != 0:
            raise ValueError(
                "seq_len must be divisible by compress_ratio "
                f"({seq_len} % {self.compress_ratio} != 0)"
            )

        query = self.split_head(self.q_proj(hidden_state))
        key = self._compress(self.k_proj(hidden_state))
        value = self._compress(self.v_proj(hidden_state))
        compressed_len = key.size(-2)

        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale_factor
        sparse_mask = self._build_mask(
            batch_size,
            seq_len,
            compressed_len,
            block_indices,
            scores.device,
        )
        if attention_mask is not None:
            # Compress the key mask to block granularity: a block stays
            # visible if any of its source tokens is valid.
            key_mask = attention_mask
            if key_mask.dim() == 4:
                # Reduce over query positions: a key stays visible if any
                # query can attend to it (a pure causal mask has no padding).
                key_mask = key_mask.any(dim=-2)
            key_mask = key_mask.reshape(key_mask.size(0), -1).bool()
            key_blocks = key_mask.view(
                batch_size, compressed_len, self.compress_ratio
            ).any(dim=-1)
            sparse_mask = sparse_mask & key_blocks[:, None, None, :]

        weights = self.compute_attention_weights(scores, sparse_mask)
        output: torch.Tensor = torch.matmul(weights, value)
        output = self.o_proj(self.combine_head(output))
        if return_attention_weights:
            return output, weights
        return output

    def _compress(self, x: torch.Tensor) -> torch.Tensor:
        """Compress ``x`` along the sequence dimension."""
        batch_size, seq_len, features = x.size()
        ratio = self.compress_ratio
        x = x.view(batch_size, seq_len // ratio, ratio, features).mean(dim=2)
        return self._split_kv(x)

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

    def _build_mask(
        self,
        batch_size: int,
        q_len: int,
        compressed_len: int,
        block_indices: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        """Build a ``(batch, heads, q_len, compressed_len)`` selection mask."""
        if q_len == 0:
            # ``q_block.max()`` below is undefined on empty sequences.
            return torch.zeros(
                batch_size,
                self.num_heads,
                0,
                compressed_len,
                dtype=torch.bool,
                device=device,
            )
        q_block = torch.arange(q_len, device=device) // self.compress_ratio
        num_q_blocks = int(q_block.max()) + 1
        if block_indices is None:
            indices = self._fallback_indices(batch_size, num_q_blocks, device)
        elif block_indices.dim() == 3:
            indices = block_indices.unsqueeze(0).expand(batch_size, -1, -1, -1)
        elif block_indices.dim() == 4:
            indices = block_indices
        else:
            raise ValueError(
                "block_indices must be 3D (heads, q_blocks, top_k) or "
                f"4D (batch, heads, q_blocks, top_k), got {block_indices.dim()}D"
            )
        indices = indices.to(device)
        if indices.numel() and (indices.min() < 0 or indices.max() >= compressed_len):
            raise ValueError("block_indices out of range")

        selected = torch.zeros(
            batch_size,
            self.num_heads,
            num_q_blocks,
            compressed_len,
            dtype=torch.bool,
            device=device,
        )
        selected.scatter_(-1, indices, True)
        allowed = selected[:, :, q_block, :]
        if self.causal:
            # Entry k aggregates tokens [k*ratio, (k+1)*ratio - 1]; a query may
            # only see it once the entry's LAST source token is in the past.
            entry_end = (
                torch.arange(compressed_len, device=device) + 1
            ) * self.compress_ratio - 1
            allowed = allowed & (
                entry_end[None, None, None, :]
                <= torch.arange(q_len, device=device)[None, None, :, None]
            )
        return allowed

    def _fallback_indices(
        self, batch_size: int, num_q_blocks: int, device: torch.device
    ) -> torch.Tensor:
        """Select a local window of compressed blocks."""
        rows = []
        for qb in range(num_q_blocks):
            start = (
                max(0, qb - self.top_k + 1)
                if self.causal
                else max(0, qb - self.top_k // 2)
            )
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

    def extra_repr(self) -> str:
        """Return a string representation of the module's extra information."""
        return (
            f"{super().extra_repr()}, compress_ratio={self.compress_ratio}, "
            f"num_kv_groups={self.num_kv_groups}, top_k={self.top_k}, "
            f"causal={self.causal}"
        )
