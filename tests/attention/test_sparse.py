"""Tests for the sparse attention variants.

Covers sliding window attention, block sparse attention and compressed sparse
attention: output shapes, gradient flow, sparsity patterns, constructor
validation and equivalence with dense GQA when the sparsity is disabled.
"""

import pytest
import torch
from helpers import make_causal_mask, make_hidden_state

from llminfra import (
    BlockSparseAttention,
    CompressedSparseAttention,
    GroupedQueryAttention,
    SlidingWindowAttention,
)

HIDDEN = 64
HEADS = 4
BATCH = 2
SEQ = 7


@pytest.fixture()
def swa():
    return SlidingWindowAttention(
        HIDDEN,
        HEADS,
        window_size=2,
        num_kv_groups=2,
        dropout=0.0,
    )


def test_sliding_window_output_shape_and_gradient(swa):
    x = make_hidden_state(BATCH, SEQ, HIDDEN).requires_grad_(True)
    out = swa(x)
    assert out.shape == (BATCH, SEQ, HIDDEN)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_sliding_window_weights_respect_window(swa):
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    _, weights = swa(x, return_attention_weights=True)
    assert weights.shape == (BATCH, HEADS, SEQ, SEQ)
    for i in range(SEQ):
        for j in range(SEQ):
            allowed = 0 <= i - j <= 2
            if not allowed:
                assert weights[0, 0, i, j].item() == 0.0


def test_sliding_window_large_window_equals_gqa():
    swa_full = SlidingWindowAttention(
        HIDDEN,
        HEADS,
        window_size=SEQ,
        num_kv_groups=2,
        dropout=0.0,
        causal=False,
    )
    gqa = GroupedQueryAttention(HIDDEN, HEADS, num_kv_groups=2, dropout=0.0)
    swa_full.load_state_dict(gqa.state_dict())
    swa_full.eval()
    gqa.eval()

    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    torch.testing.assert_close(swa_full(x), gqa(x))


def test_sliding_window_rejects_bad_constructor_args():
    with pytest.raises(ValueError, match="window_size"):
        SlidingWindowAttention(HIDDEN, HEADS, window_size=0)
    with pytest.raises(ValueError, match="divisible"):
        SlidingWindowAttention(HIDDEN, HEADS, window_size=4, num_kv_groups=3)


@pytest.fixture()
def bsa():
    return BlockSparseAttention(
        HIDDEN,
        HEADS,
        block_size=2,
        num_kv_groups=2,
        top_k=1,
        dropout=0.0,
    )


def test_block_sparse_shape_and_gradient(bsa):
    x = make_hidden_state(BATCH, SEQ, HIDDEN).requires_grad_(True)
    out = bsa(x)
    assert out.shape == (BATCH, SEQ, HIDDEN)
    out.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_block_sparse_selects_only_local_block(bsa):
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    _, weights = bsa(x, return_attention_weights=True)
    # block_size=2 and top_k=1 selects the query's own block.
    for i in range(SEQ):
        for j in range(SEQ):
            if i // 2 != j // 2:
                assert weights[0, 0, i, j].item() == 0.0


def test_block_sparse_with_all_blocks_equals_causal_gqa():
    num_blocks = (SEQ + 1) // 2
    indices = torch.arange(num_blocks).expand(HEADS, num_blocks, num_blocks)
    sparse = BlockSparseAttention(
        HIDDEN,
        HEADS,
        block_size=2,
        num_kv_groups=2,
        top_k=num_blocks,
        dropout=0.0,
    )
    gqa = GroupedQueryAttention(HIDDEN, HEADS, num_kv_groups=2, dropout=0.0)
    sparse.load_state_dict(gqa.state_dict())
    sparse.eval()
    gqa.eval()

    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    torch.testing.assert_close(
        sparse(x, block_indices=indices),
        gqa(x, attention_mask=make_causal_mask(BATCH, SEQ)),
    )


def test_block_sparse_rejects_out_of_range_block(bsa):
    indices = torch.full((HEADS, (SEQ + 1) // 2, 1), 99, dtype=torch.long)
    with pytest.raises(ValueError, match="out-of-range"):
        bsa(make_hidden_state(BATCH, SEQ, HIDDEN), block_indices=indices)


def test_block_sparse_rejects_negative_block(bsa):
    """Negative block ids must raise ValueError, not scatter's RuntimeError."""
    indices = torch.full((HEADS, (SEQ + 1) // 2, 1), -1, dtype=torch.long)
    with pytest.raises(ValueError, match="out-of-range"):
        bsa(make_hidden_state(BATCH, SEQ, HIDDEN), block_indices=indices)


def test_block_sparse_causal_mask_right_aligns_longer_kv(bsa):
    """With kv_len > q_len, query i is the global position i + kv_len - q_len."""
    module = BlockSparseAttention(
        HIDDEN, HEADS, block_size=1, top_k=8, dropout=0.0, causal=True
    )
    # Select every KV block explicitly so only the causal term can mask.
    indices = torch.arange(6).expand(1, HEADS, 2, 6)
    mask = module._build_mask(
        batch_size=1,
        q_len=2,
        kv_len=6,
        block_indices=indices,
        device=torch.device("cpu"),
    )
    # Query 0 sits at global position 4 and sees keys 0..4; query 1 sees all.
    expected = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    assert (mask == expected[None, None]).all()


def test_compressed_sparse_attention_shape_and_gradient():
    layer = CompressedSparseAttention(
        HIDDEN,
        HEADS,
        compress_ratio=2,
        num_kv_groups=2,
        top_k=4,
    )
    x = make_hidden_state(BATCH, 8, HIDDEN).requires_grad_(True)
    out = layer(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_csa_padding_mask_with_compression():
    """Padding masks must be compressed to block granularity, not crash."""
    module = CompressedSparseAttention(HIDDEN, HEADS, compress_ratio=2, top_k=2).eval()
    x = make_hidden_state(BATCH, 8, HIDDEN)
    mask = torch.ones(BATCH, 1, 1, 8, dtype=torch.bool)
    mask[0, 0, 0, -2:] = False

    out = module(x, attention_mask=mask)

    assert out.shape == (BATCH, 8, HIDDEN)
    assert torch.isfinite(out).all()


def test_csa_dense_causal_mask_matches_implicit_causal():
    """A dense causal mask carries no padding: it must be a no-op for CSA."""
    module = CompressedSparseAttention(
        HIDDEN, HEADS, compress_ratio=2, top_k=2, causal=True
    ).eval()
    x = make_hidden_state(BATCH, 8, HIDDEN)
    causal_mask = torch.tril(torch.ones(BATCH, 1, 8, 8, dtype=torch.bool))

    torch.testing.assert_close(module(x, attention_mask=causal_mask), module(x))


def test_csa_accepts_3d_query_key_mask():
    """A (batch, q_len, kv_len) mask is contract-valid and must not crash."""
    module = CompressedSparseAttention(
        HIDDEN, HEADS, compress_ratio=2, top_k=2, causal=True
    ).eval()
    x = make_hidden_state(BATCH, 8, HIDDEN)
    causal_mask = torch.tril(torch.ones(BATCH, 8, 8, dtype=torch.bool))

    torch.testing.assert_close(module(x, attention_mask=causal_mask), module(x))


def test_csa_accepts_per_head_4d_mask():
    """A (batch, heads, q_len, kv_len) causal mask must not crash either."""
    module = CompressedSparseAttention(
        HIDDEN, HEADS, compress_ratio=2, top_k=2, causal=True
    ).eval()
    x = make_hidden_state(BATCH, 8, HIDDEN)
    causal_mask = torch.tril(torch.ones(BATCH, HEADS, 8, 8, dtype=torch.bool))

    torch.testing.assert_close(module(x, attention_mask=causal_mask), module(x))


def test_csa_causal_boundary_excludes_unfinished_entry():
    """A compressed entry is hidden until its last source token is in the past."""
    module = CompressedSparseAttention(
        HIDDEN, HEADS, compress_ratio=2, top_k=4, causal=True
    ).eval()
    x = make_hidden_state(1, 4, HIDDEN)
    perturbed = x.clone()
    perturbed[:, 3] += 10.0  # belongs to entry 1 (tokens 2-3), future for queries 0-2

    out = module(x)
    out_perturbed = module(perturbed)
    torch.testing.assert_close(out[:, :3], out_perturbed[:, :3])
    assert not torch.allclose(out[:, 3], out_perturbed[:, 3])
