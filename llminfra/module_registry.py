"""Registry helpers for constructing attention and positional modules."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from .attention.alibi_attention import ALiBiAttention
from .attention.base_attention import BaseAttention
from .attention.block_sparse_attention import BlockSparseAttention
from .attention.compressed_sparse_attention import CompressedSparseAttention
from .attention.gated_delta_net import GatedDeltaNet
from .attention.grouped_query_attention import GroupedQueryAttention
from .attention.hybrid_attention import HybridAttention
from .attention.kimi_delta_attention import KimiDeltaAttention
from .attention.lightning_attention import LightningAttention
from .attention.linear_attention import LinearAttention
from .attention.multi_head_attention import MultiHeadAttention
from .attention.multi_head_latent_attention import MultiHeadLatentAttention
from .attention.multi_query_attention import MultiQueryAttention
from .attention.ring_attention import RingAttention
from .attention.sliding_window_attention import SlidingWindowAttention
from .attention.sparse_attention import (
    DynamicSparseAttention,
    HierarchicalCompressedAttention,
    MiniMaxSparseAttention,
)
from .positional import (
    BasePositionalEncoding,
)
from .positional import (
    build_positional_encoding as _build_positional_encoding,
)

ATTENTION_REGISTRY: dict[str, type[BaseAttention]] = {
    "alibi": ALiBiAttention,
    "mha": MultiHeadAttention,
    "mqa": MultiQueryAttention,
    "gqa": GroupedQueryAttention,
    "mla": MultiHeadLatentAttention,
    "swa": SlidingWindowAttention,
    "block_sparse": BlockSparseAttention,
    "compressed_sparse": CompressedSparseAttention,
    "dsa": DynamicSparseAttention,
    "dynamic_sparse": DynamicSparseAttention,
    "msa": MiniMaxSparseAttention,
    "minimax_sparse": MiniMaxSparseAttention,
    "hca": HierarchicalCompressedAttention,
    "hierarchical_compressed": HierarchicalCompressedAttention,
    "ring": RingAttention,
    "linear": LinearAttention,
    "lightning": LightningAttention,
    "gated_delta": GatedDeltaNet,
    "hybrid": HybridAttention,
    "kda": KimiDeltaAttention,
    "kimi_delta": KimiDeltaAttention,
}


def build_attention(
    name: str,
    hidden_size: int,
    num_heads: int,
    **kwargs: object,
) -> BaseAttention:
    """Build an attention module by registry name.

    Required extra keyword arguments depend on the architecture, for example
    ``window_size`` for ``swa``, ``block_size`` for ``block_sparse``, or
    ``q_latent_size`` / ``kv_latent_size`` for ``mla``.
    """
    try:
        factory = ATTENTION_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown attention: {name}; available: {list_attentions()}"
        ) from exc
    # Registry entries share only the base-class signature; the remaining
    # keyword arguments are specific to each concrete architecture.
    constructor = cast("Callable[..., BaseAttention]", factory)
    return constructor(hidden_size=hidden_size, num_heads=num_heads, **kwargs)


def list_attentions() -> list[str]:
    """Return the available attention registry names."""
    return list(ATTENTION_REGISTRY)


def build_positional_encoding(
    name: str,
    *,
    dim: int,
    num_heads: int | None = None,
    max_seq_len: int = 4096,
    **kwargs: object,
) -> BasePositionalEncoding:
    """Build a positional encoding module by name."""
    return _build_positional_encoding(
        name,
        dim=dim,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        **kwargs,
    )
