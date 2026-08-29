"""Attention mechanisms: classic variants, sparse/linear forms and hybrids.

All full-attention modules share the `BaseAttention` interface; linear and
state-space-flavored variants (``linear``, ``lightning``, ``gated_delta_net``)
live here too since they are drop-in attention layers. File names carry the
full mechanism name with an ``_attention`` suffix where applicable.
"""

from .alibi_attention import ALiBiAttention
from .attention_residual import AttentionResidual
from .base_attention import BaseAttention, validate_attention_inputs
from .block_sparse_attention import BlockSparseAttention
from .compressed_sparse_attention import CompressedSparseAttention
from .flash_mla_attention import FlashMLA
from .gated_delta_net import GatedDeltaNet
from .gated_linear_attention import GatedLinearAttention
from .grouped_query_attention import GroupedQueryAttention
from .hybrid_attention import HybridAttention
from .kimi_delta_attention import KimiDeltaAttention
from .lightning_attention import LightningAttention
from .linear_attention import LinearAttention
from .multi_head_attention import MultiHeadAttention
from .multi_head_latent_attention import MultiHeadLatentAttention
from .multi_query_attention import MultiQueryAttention
from .retention import Retention
from .ring_attention import RingAttention, distributed_ring_attention, ring_attention
from .rwkv import RWKVLayer
from .sliding_window_attention import SlidingWindowAttention
from .sparse_attention import (
    DynamicSparseAttention,
    HierarchicalCompressedAttention,
    MiniMaxSparseAttention,
    QueryKeyBlockIndexer,
)
from .ttt import TTTLayer

__all__ = [
    "ALiBiAttention",
    "AttentionResidual",
    "BaseAttention",
    "BlockSparseAttention",
    "CompressedSparseAttention",
    "DynamicSparseAttention",
    "FlashMLA",
    "GatedDeltaNet",
    "GatedLinearAttention",
    "GroupedQueryAttention",
    "HierarchicalCompressedAttention",
    "HybridAttention",
    "KimiDeltaAttention",
    "LightningAttention",
    "LinearAttention",
    "MiniMaxSparseAttention",
    "MultiHeadAttention",
    "MultiHeadLatentAttention",
    "MultiQueryAttention",
    "QueryKeyBlockIndexer",
    "RWKVLayer",
    "Retention",
    "RingAttention",
    "SlidingWindowAttention",
    "TTTLayer",
    "distributed_ring_attention",
    "ring_attention",
    "validate_attention_inputs",
]
