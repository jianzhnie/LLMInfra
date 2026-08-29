"""Tests for the quantized KV cache (per-channel K, per-token V)."""

import pytest
import torch

from llminfra import PagedAttentionCache, paged_attention
from llminfra.inference.kv_cache_quantization import (
    QuantizedKVCache,
    quantized_paged_attention,
)


def _append_and_get(bits, key, value, residual_length=0):
    cache = QuantizedKVCache(
        num_heads=key.size(1),
        head_dim=key.size(2),
        bits=bits,
        residual_length=residual_length,
    )
    cache.append(0, key, value)
    return cache, *cache.get(0)


def test_round_trip_error_is_bounded_by_half_scale() -> None:
    torch.manual_seed(0)
    key = torch.randn(9, 2, 8)
    value = torch.randn(9, 2, 8)
    for bits in (8, 4):
        cache, got_key, got_value = _append_and_get(bits, key, value)
        chunk = cache.chunks[0][0]
        key_error = (got_key - key).abs()
        value_error = (got_value - value).abs()
        assert (key_error <= chunk.key_scale / 2 + 1e-6).all()
        assert (value_error <= chunk.value_scale / 2 + 1e-6).all()


def test_int8_is_more_accurate_than_int4() -> None:
    torch.manual_seed(0)
    key = torch.randn(16, 2, 8)
    value = torch.randn(16, 2, 8)
    _, key8, value8 = _append_and_get(8, key, value)
    _, key4, value4 = _append_and_get(4, key, value)
    error8 = (key8 - key).abs().mean() + (value8 - value).abs().mean()
    error4 = (key4 - key).abs().mean() + (value4 - value).abs().mean()
    assert error8 < error4


def test_scale_shapes_per_channel_key_per_token_value() -> None:
    cache, _, _ = _append_and_get(8, torch.randn(5, 3, 4), torch.randn(5, 3, 4))
    chunk = cache.chunks[0][0]
    assert chunk.key_q.dtype == torch.int8
    assert chunk.value_q.dtype == torch.int8
    # One scale/zero-point per (head, channel) for K.
    assert chunk.key_scale.shape == (1, 3, 4)
    assert chunk.key_zero.shape == (1, 3, 4)
    # One scale/zero-point per (token, head) for V.
    assert chunk.value_scale.shape == (5, 3, 1)
    assert chunk.value_zero.shape == (5, 3, 1)


def test_attention_matches_full_precision_cache() -> None:
    torch.manual_seed(0)
    num_heads, head_dim, seq_len, q_len = 2, 16, 12, 3
    key = torch.randn(seq_len, num_heads, head_dim)
    value = torch.randn(seq_len, num_heads, head_dim)
    query = torch.randn(q_len, num_heads, head_dim)

    dense = PagedAttentionCache(
        num_blocks=4, block_size=4, num_heads=num_heads, head_dim=head_dim
    )
    dense.append(0, key, value)
    expected = paged_attention(
        query,
        dense.key_cache,
        dense.value_cache,
        dense.block_tables[0],
        num_tokens=seq_len,
        block_size=4,
        causal=True,
    )

    errors = {}
    for bits in (8, 4):
        cache = QuantizedKVCache(
            num_heads=num_heads, head_dim=head_dim, bits=bits, residual_length=0
        )
        cache.append(0, key, value)
        actual = quantized_paged_attention(query, cache, 0, causal=True)
        assert actual.shape == expected.shape
        errors[bits] = (actual - expected).abs().max().item()
    assert errors[8] < errors[4]
    assert errors[8] < 0.1


def test_residual_buffer_keeps_recent_tokens_exact() -> None:
    torch.manual_seed(0)
    key = torch.randn(6, 2, 4)
    value = torch.randn(6, 2, 4)
    cache, got_key, got_value = _append_and_get(4, key, value, residual_length=4)

    # Only the 2 oldest tokens overflowed the residual buffer and were
    # quantized as one batch; the 4 most recent stay full precision.
    assert len(cache.chunks[0]) == 1
    assert cache.chunks[0][0].key_q.size(0) == 2
    torch.testing.assert_close(got_key[2:], key[2:])
    torch.testing.assert_close(got_value[2:], value[2:])
    # The int4-quantized prefix actually lost precision.
    assert not torch.equal(got_key[:2], key[:2])


def test_incremental_appends_flush_in_batches() -> None:
    torch.manual_seed(0)
    cache = QuantizedKVCache(num_heads=2, head_dim=4, bits=8, residual_length=2)
    key = torch.randn(5, 2, 4)
    value = torch.randn(5, 2, 4)
    for index in range(5):
        cache.append(0, key[index : index + 1], value[index : index + 1])

    assert cache.num_tokens[0] == 5
    # 5 tokens with a residual of 2 flush 1 + 1 + 1 rows in three batches.
    assert [chunk.key_q.size(0) for chunk in cache.chunks[0]] == [1, 1, 1]
    got_key, got_value = cache.get(0)
    assert got_key.shape == key.shape
    # The last 2 tokens are still full precision.
    torch.testing.assert_close(got_key[3:], key[3:])
    torch.testing.assert_close(got_value[3:], value[3:])


def test_single_token_and_constant_rows() -> None:
    cache = QuantizedKVCache(num_heads=2, head_dim=4, bits=8, residual_length=0)
    key = torch.full((1, 2, 4), 3.0)  # degenerate min == max range
    value = torch.full((1, 2, 4), -2.0)
    cache.append(0, key, value)
    got_key, got_value = cache.get(0)
    assert got_key.shape == (1, 2, 4)
    torch.testing.assert_close(got_key, key, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(got_value, value, atol=1e-4, rtol=1e-4)


def test_empty_sequence_and_empty_attention() -> None:
    cache = QuantizedKVCache(num_heads=2, head_dim=4)
    cache.append(0, torch.randn(0, 2, 4), torch.randn(0, 2, 4))
    keys, values = cache.get(0)
    assert keys.shape == (0, 2, 4)
    assert values.shape == (0, 2, 4)
    output = quantized_paged_attention(torch.randn(3, 2, 4), cache, 0)
    assert output.shape == (3, 2, 4)
    assert (output == 0).all()


def test_interface_matches_paged_cache_shapes() -> None:
    torch.manual_seed(0)
    key = torch.randn(7, 2, 4)
    value = torch.randn(7, 2, 4)
    dense = PagedAttentionCache(num_blocks=4, block_size=2, num_heads=2, head_dim=4)
    quantized = QuantizedKVCache(num_heads=2, head_dim=4, bits=8, residual_length=0)
    # Same append pattern spanning multiple blocks/chunks.
    dense.append(0, key[:3], value[:3])
    dense.append(0, key[3:], value[3:])
    quantized.append(0, key[:3], value[:3])
    quantized.append(0, key[3:], value[3:])

    dense_key, dense_value = dense.get(0)
    got_key, got_value = quantized.get(0)
    assert got_key.shape == dense_key.shape
    assert got_value.shape == dense_value.shape
    assert quantized.num_tokens == dense.num_tokens
    assert len(quantized.chunks[0]) == 2  # one chunk per append


def test_quantized_storage_is_smaller_than_dense() -> None:
    torch.manual_seed(0)
    seq_len, num_heads, head_dim = 16, 2, 8
    shape = (seq_len, num_heads, head_dim)
    cache, _, _ = _append_and_get(8, torch.randn(shape), torch.randn(shape))
    dense_bytes = 2 * seq_len * num_heads * head_dim * 4  # fp32 K + V
    stored = cache.storage_nbytes(0)
    assert stored < dense_bytes
    chunk = cache.chunks[0][0]
    assert chunk.key_q.element_size() == 1
    assert chunk.value_q.element_size() == 1


def test_cache_validates_arguments() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        QuantizedKVCache(num_heads=0, head_dim=4)
    with pytest.raises(ValueError, match="bits"):
        QuantizedKVCache(num_heads=1, head_dim=4, bits=7)
    with pytest.raises(ValueError, match=">= 0"):
        QuantizedKVCache(num_heads=1, head_dim=4, residual_length=-1)
    cache = QuantizedKVCache(num_heads=2, head_dim=4)
    with pytest.raises(ValueError, match=r"seq, heads, head_dim"):
        cache.append(0, torch.randn(2, 4), torch.randn(2, 4))
    with pytest.raises(ValueError, match="head configuration"):
        cache.append(0, torch.randn(3, 4, 4), torch.randn(3, 4, 4))
    with pytest.raises(ValueError, match="same shape"):
        cache.append(0, torch.randn(3, 2, 4), torch.randn(4, 2, 4))
    with pytest.raises(KeyError, match="Unknown sequence id"):
        cache.get(7)
    with pytest.raises(KeyError, match="Unknown sequence id"):
        cache.storage_nbytes(7)


def test_reset_drops_sequence_state() -> None:
    cache, _, _ = _append_and_get(8, torch.randn(4, 2, 4), torch.randn(4, 2, 4))
    cache.reset(0)
    assert 0 not in cache.chunks
    assert 0 not in cache.num_tokens
    with pytest.raises(KeyError):
        cache.get(0)
