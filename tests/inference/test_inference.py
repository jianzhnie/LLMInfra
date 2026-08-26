"""Tests for KV-cache, paged-attention, and sparse-indexing infrastructure."""

import pytest
import torch
import torch.nn.functional as F

import llminfra.inference as inference
from llminfra import (
    BlockSparseAttention,
    BlockSparseIndexer,
    OnDiskKVStore,
    PagedAttentionCache,
    TieredKVCache,
    paged_attention,
)


def test_inference_all_exports_resolve() -> None:
    for name in inference.__all__:
        assert hasattr(inference, name), name


def test_paged_cache_append_and_gather_matches_dense() -> None:
    cache = PagedAttentionCache(num_blocks=4, block_size=2, num_heads=2, head_dim=4)
    key = torch.randn(5, 2, 4)
    value = torch.randn(5, 2, 4)
    cache.append(0, key, value)
    actual_key, actual_value = cache.get(0)
    torch.testing.assert_close(actual_key, key)
    torch.testing.assert_close(actual_value, value)


def test_paged_attention_matches_dense_attention() -> None:
    cache = PagedAttentionCache(num_blocks=4, block_size=2, num_heads=2, head_dim=4)
    key = torch.randn(5, 2, 4)
    value = torch.randn(5, 2, 4)
    query = torch.randn(3, 2, 4)
    cache.append(0, key, value)

    actual = paged_attention(
        query,
        cache.key_cache,
        cache.value_cache,
        cache.block_tables[0],
        num_tokens=5,
        block_size=2,
        causal=True,
    )
    scores = torch.einsum("qhd,khd->hqk", query, key) / 2.0
    mask = torch.arange(5) <= (torch.arange(3)[:, None] + 2)
    weights = torch.softmax(scores.masked_fill(~mask[None], float("-inf")), dim=-1)
    expected = torch.einsum("hqk,khd->qhd", weights, value)
    torch.testing.assert_close(actual, expected)


def test_paged_attention_computes_scores_per_head() -> None:
    torch.manual_seed(0)
    cache = PagedAttentionCache(num_blocks=4, block_size=2, num_heads=2, head_dim=4)
    key = torch.randn(5, 2, 4)
    value = torch.randn(5, 2, 4)
    query = torch.randn(3, 2, 4)
    cache.append(0, key, value)

    actual = paged_attention(
        query,
        cache.key_cache,
        cache.value_cache,
        cache.block_tables[0],
        num_tokens=5,
        block_size=2,
        causal=True,
    )
    mask = torch.arange(5) <= (torch.arange(3)[:, None] + 2)
    for head in range(2):
        scores = query[:, head] @ key[:, head].T / 4**0.5
        weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
        torch.testing.assert_close(actual[:, head], weights @ value[:, head])


def test_paged_cache_get_empty_sequence_returns_empty_tensors() -> None:
    cache = PagedAttentionCache(num_blocks=4, block_size=2, num_heads=2, head_dim=4)
    cache.append(0, torch.randn(0, 2, 4), torch.randn(0, 2, 4))
    keys, values = cache.get(0)
    assert keys.shape == (0, 2, 4)
    assert values.shape == (0, 2, 4)

    output = paged_attention(
        torch.randn(3, 2, 4),
        cache.key_cache,
        cache.value_cache,
        cache.block_tables[0],
        num_tokens=0,
        block_size=2,
    )
    assert output.shape == (3, 2, 4)


def test_paged_cache_clone_uses_copy_on_write() -> None:
    cache = PagedAttentionCache(num_blocks=8, block_size=4, num_heads=1, head_dim=2)
    prefix_key = torch.randn(3, 1, 2)
    prefix_value = torch.randn(3, 1, 2)
    cache.append(0, prefix_key, prefix_value)
    cache.clone_sequence(0, 1)
    shared_block = cache.block_tables[0][-1]

    new_key = torch.randn(1, 1, 2)
    new_value = torch.randn(1, 1, 2)
    cache.append(1, new_key, new_value)

    assert cache.block_tables[0][-1] == shared_block
    assert cache.block_tables[1][-1] != shared_block
    torch.testing.assert_close(cache.get(0)[0], prefix_key)
    torch.testing.assert_close(cache.get(1)[0], torch.cat((prefix_key, new_key)))


def test_paged_cache_reset_preserves_cloned_blocks() -> None:
    cache = PagedAttentionCache(num_blocks=4, block_size=4, num_heads=1, head_dim=2)
    key = torch.randn(2, 1, 2)
    value = torch.randn(2, 1, 2)
    cache.append(0, key, value)
    cache.clone_sequence(0, 1)
    shared_block = cache.block_tables[0][0]
    cache.reset(0)
    assert shared_block in cache.allocator.allocated
    torch.testing.assert_close(cache.get(1)[0], key)


def test_paged_cache_reset_releases_unshared_blocks() -> None:
    cache = PagedAttentionCache(num_blocks=4, block_size=2, num_heads=1, head_dim=2)
    cache.append(0, torch.randn(3, 1, 2), torch.randn(3, 1, 2))
    cache.reset(0)
    assert 0 not in cache.block_tables
    assert len(cache.allocator.free_blocks) == 4


def test_on_disk_kv_store_round_trip(tmp_path) -> None:
    store = OnDiskKVStore(tmp_path / "kv")
    key = torch.randn(4, 2, 8)
    value = torch.randn(4, 2, 8)
    store.save(1, key, value)
    actual_key, actual_value = store.load(1)
    torch.testing.assert_close(actual_key, key)
    torch.testing.assert_close(actual_value, value)
    store.delete(1)
    with pytest.raises(FileNotFoundError):
        store.load(1)


def test_tiered_kv_cache_copies_input_tensors(tmp_path) -> None:
    cache = TieredKVCache(
        tmp_path / "kv", max_hbm_entries=2, max_cpu_entries=2, hbm_device="cpu"
    )
    key = torch.randn(4, 2, 8)
    value = torch.randn(4, 2, 8)
    original_key = key.clone()
    original_value = value.clone()
    cache.put(0, key, value)
    key.fill_(999.0)
    value.fill_(999.0)
    # Force an HBM->CPU eviction; the evicted record must not alias the inputs.
    cache.put(1, torch.randn(4, 2, 8), torch.randn(4, 2, 8))
    cache.put(2, torch.randn(4, 2, 8), torch.randn(4, 2, 8))
    actual_key, actual_value = cache.get(0)
    torch.testing.assert_close(actual_key, original_key)
    torch.testing.assert_close(actual_value, original_value)


def test_sparse_indexer_is_causal_and_integrates_with_attention() -> None:
    hidden_size = 32
    num_heads = 4
    indexer = BlockSparseIndexer(
        hidden_size,
        num_heads,
        block_size=2,
        top_k=4,
        max_seq_len=16,
        causal=True,
    )
    attention = BlockSparseAttention(
        hidden_size,
        num_heads,
        block_size=2,
        num_kv_groups=2,
        top_k=4,
    )
    hidden_state = torch.randn(2, 7, hidden_size)
    block_indices = indexer(hidden_state)
    for query_block in range(block_indices.size(2)):
        assert (block_indices[:, :, query_block] <= query_block).all()
    output = attention(hidden_state, block_indices=block_indices)
    assert output.shape == hidden_state.shape


def test_sparse_indexer_handles_empty_sequence() -> None:
    indexer = BlockSparseIndexer(8, 2, block_size=2, top_k=2, max_seq_len=16)
    block_indices = indexer(torch.randn(3, 0, 8))
    assert block_indices.shape == (3, 2, 0, 2)
    assert block_indices.dtype == torch.long


def _loop_indexer_reference(
    indexer: BlockSparseIndexer, hidden_state: torch.Tensor
) -> torch.Tensor:
    """Slow per-query-block selection, mirroring the original loop."""
    batch_size, seq_len, _ = hidden_state.size()
    padded_len = (
        (seq_len + indexer.block_size - 1) // indexer.block_size
    ) * indexer.block_size
    if padded_len != seq_len:
        hidden_state = F.pad(hidden_state, (0, 0, 0, padded_len - seq_len))
    block_count = padded_len // indexer.block_size
    block_vectors = hidden_state.view(
        batch_size, block_count, indexer.block_size, indexer.hidden_size
    ).mean(dim=2)
    scores = indexer.score_proj(block_vectors).transpose(1, 2)
    rows: list[torch.Tensor] = []
    for query_block in range(block_count):
        candidate_scores = scores[:, :, : query_block + 1] if indexer.causal else scores
        count = min(indexer.top_k, candidate_scores.size(-1))
        selected = torch.topk(candidate_scores, count, dim=-1).indices
        if count < indexer.top_k:
            last = candidate_scores.size(-1) - 1
            selected = F.pad(selected, (0, indexer.top_k - count), value=last)
        rows.append(selected)
    return torch.stack(rows, dim=2)


def test_sparse_indexer_vectorized_matches_loop_reference() -> None:
    torch.manual_seed(0)
    configs = [
        # (seq_len, block_size, top_k, num_heads, batch_size)
        (8, 2, 4, 4, 2),  # divisible, top_k < block_count
        (7, 2, 4, 4, 2),  # ragged tail block
        (5, 2, 8, 1, 3),  # top_k > block_count
        (3, 4, 2, 2, 1),  # single padded block
        (16, 4, 1, 8, 2),  # top_k = 1
        (9, 3, 3, 2, 4),  # divisible
    ]
    for causal in (True, False):
        for seq_len, block_size, top_k, num_heads, batch_size in configs:
            indexer = BlockSparseIndexer(
                16,
                num_heads,
                block_size=block_size,
                top_k=top_k,
                max_seq_len=32,
                causal=causal,
            )
            hidden_state = torch.randn(batch_size, seq_len, 16)
            actual = indexer(hidden_state)
            expected = _loop_indexer_reference(indexer, hidden_state)
            assert actual.dtype == torch.long
            assert torch.equal(actual, expected), (
                causal,
                seq_len,
                block_size,
                top_k,
                num_heads,
                batch_size,
            )


def test_paged_cache_append_casts_mismatched_dtype_across_blocks() -> None:
    """Regression: the multi-block scatter path must cast inputs to cache dtype.

    Basic-indexing assignment casts implicitly, but the vectorized scatter
    uses ``index_put_``, which requires matching dtypes; fp32 inputs into an
    fp16 cache used to work and must keep working.
    """
    cache = PagedAttentionCache(
        num_blocks=4, block_size=2, num_heads=2, head_dim=4, dtype=torch.float16
    )
    key = torch.randn(3, 2, 4)  # spans two physical blocks -> scatter path
    value = torch.randn(3, 2, 4)
    cache.append(0, key, value)

    # A second chunk starting mid-block and crossing into the next block.
    extra_key = torch.randn(2, 2, 4)
    extra_value = torch.randn(2, 2, 4)
    cache.append(0, extra_key, extra_value)

    actual_key, actual_value = cache.get(0)
    assert actual_key.dtype == torch.float16
    expected_key = torch.cat((key, extra_key)).to(torch.float16)
    expected_value = torch.cat((value, extra_value)).to(torch.float16)
    torch.testing.assert_close(actual_key, expected_key)
    torch.testing.assert_close(actual_value, expected_value)


def test_tiered_kv_cache_validates_and_hits_hbm(tmp_path) -> None:
    with pytest.raises(ValueError, match=">= 1"):
        TieredKVCache(tmp_path / "kv", max_hbm_entries=0)
    # Default HBM device falls back to CPU when CUDA is unavailable.
    cache = TieredKVCache(tmp_path / "kv2", max_hbm_entries=2, max_cpu_entries=2)
    key = torch.randn(2, 3)
    value = torch.randn(2, 3)
    cache.put(0, key, value)
    got_key, got_value = cache.get(0)  # HBM hit path
    torch.testing.assert_close(got_key, key.to(cache.hbm_device))
    torch.testing.assert_close(got_value, value.to(cache.hbm_device))
    with pytest.raises(ValueError, match="shapes must match"):
        cache.put(1, key, torch.randn(3, 3))
    with pytest.raises(KeyError, match="unknown KV sequence"):
        cache.get(99)


def test_tiered_kv_cache_reput_after_nvme_eviction_cleans_disk(tmp_path) -> None:
    """Re-inserting an NVMe-evicted sequence must delete its on-disk copy."""
    cache = TieredKVCache(tmp_path / "kv", max_hbm_entries=1, max_cpu_entries=1)
    cache.put(0, torch.zeros(1), torch.zeros(1))
    cache.put(1, torch.ones(1), torch.ones(1))  # evicts 0: HBM -> CPU
    cache.put(2, torch.full((1,), 2.0), torch.full((1,), 2.0))  # 0: CPU -> NVMe
    assert cache.tier_counts() == {"hbm": 1, "cpu": 1, "nvme": 1}
    assert 0 in cache.nvme

    cache.put(0, torch.full((1,), 3.0), torch.full((1,), 3.0))
    assert 0 in cache.hbm and 0 not in cache.nvme
    got_key, _ = cache.get(0)
    torch.testing.assert_close(got_key, torch.full((1,), 3.0))

    cache.delete(1)  # sequence 1 was evicted to NVMe by the put above
    assert cache.tier_counts() == {"hbm": 1, "cpu": 1, "nvme": 0}
    with pytest.raises(KeyError):
        cache.get(1)


def test_paged_block_allocator_validates_and_exhausts() -> None:
    from llminfra.inference.paged_attention import PagedKVBlockAllocator

    with pytest.raises(ValueError, match="num_blocks"):
        PagedKVBlockAllocator(0)
    allocator = PagedKVBlockAllocator(1)
    block = allocator.allocate()
    with pytest.raises(RuntimeError, match="full"):
        allocator.allocate()
    with pytest.raises(ValueError, match="not allocated"):
        allocator.retain(7)
    with pytest.raises(ValueError, match="not allocated"):
        allocator.free(7)
    with pytest.raises(ValueError, match="out of range"):
        allocator.reference_count(5)
    allocator.free(block)
    assert allocator.reference_count(block) == 0


def test_paged_cache_validates_arguments() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        PagedAttentionCache(num_blocks=1, block_size=0, num_heads=1, head_dim=1)
    cache = PagedAttentionCache(num_blocks=4, block_size=2, num_heads=2, head_dim=4)
    key = torch.randn(3, 2, 4)
    with pytest.raises(ValueError, match=r"seq, heads, head_dim"):
        cache.append(0, torch.randn(2, 4), torch.randn(2, 4))
    with pytest.raises(ValueError, match="head configuration"):
        cache.append(0, torch.randn(3, 4, 4), torch.randn(3, 4, 4))
    with pytest.raises(ValueError, match="same shape"):
        cache.append(0, key, torch.randn(4, 2, 4))
    with pytest.raises(KeyError, match="Unknown sequence id"):
        cache.get(7)
    cache.append(0, key, key)
    with pytest.raises(ValueError, match="already exists"):
        cache.clone_sequence(0, 0)
    with pytest.raises(KeyError, match="Unknown sequence id"):
        cache.clone_sequence(9, 1)


def test_paged_attention_validates_query_shape() -> None:
    with pytest.raises(ValueError, match=r"q_len, heads, head_dim"):
        paged_attention(
            torch.randn(2, 4),
            torch.randn(1, 2, 2, 4),
            torch.randn(1, 2, 2, 4),
            block_table=[0],
            num_tokens=2,
            block_size=2,
        )


def test_sparse_indexer_validates_arguments() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        BlockSparseIndexer(8, 2, block_size=0, top_k=1, max_seq_len=16)
    indexer = BlockSparseIndexer(8, 2, block_size=2, top_k=1, max_seq_len=8)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        indexer(torch.randn(1, 9, 8))
