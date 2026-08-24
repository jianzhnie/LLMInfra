"""Numerical correctness tests for the educational FlashAttention versions.

Every version is checked against the dense `reference_attention` for forward
outputs and against autograd through that reference for gradients, across
dtypes, causal masking, padding masks and cross-attention shapes. Also covers
the FA4 conditional-rescale rule and FA3's simulated FP8 path.
"""

import math

import pytest
import torch
from helpers import make_key_padding_mask, make_qkv, reference_with_grads

from llminfra.flash_attention import fa1, fa2, fa3, fa4, flash_attention
from llminfra.flash_attention.common import (
    FlashAttentionConfig,
    reference_attention,
)
from llminfra.flash_attention.flash_attention_v4 import (
    _correction_merge,
    _fa4_rescale_threshold,
)

MODULES = {"fa1": fa1, "fa2": fa2, "fa3": fa3, "fa4": fa4}

# Small blocks so every test exercises multi-block online-softmax merges.
TILED_CONFIG = FlashAttentionConfig(block_size_q=16, block_size_kv=8)

TOLERANCES = {
    torch.float32: 2e-5,
    torch.float16: 1e-2,
    torch.bfloat16: 5e-2,
}

# (batch, heads, q_len, kv_len, head_dim, value_dim, causal, use_mask, dtype)
CASES = [
    (2, 3, 37, 37, 16, 24, False, False, torch.float32),  # self-attn, d_v != d_qk
    (2, 3, 40, 40, 16, 16, True, False, torch.float32),  # causal
    (2, 3, 37, 51, 16, 8, False, True, torch.float32),  # cross-attn + padding
    (2, 2, 33, 47, 32, 16, True, True, torch.float32),  # causal + padding
    (2, 2, 64, 64, 32, 32, True, False, torch.float16),  # fp16
    (2, 2, 20, 28, 16, 16, False, True, torch.bfloat16),  # bf16 + padding
]


@pytest.mark.parametrize("version", MODULES.values(), ids=MODULES.keys())
@pytest.mark.parametrize(
    (
        "batch",
        "heads",
        "q_len",
        "kv_len",
        "head_dim",
        "value_dim",
        "causal",
        "use_mask",
        "dtype",
    ),
    CASES,
)
def test_forward_backward_match_reference(
    version, batch, heads, q_len, kv_len, head_dim, value_dim, causal, use_mask, dtype
):
    q, k, v = make_qkv(batch, heads, q_len, kv_len, head_dim, value_dim, dtype=dtype)
    mask = make_key_padding_mask(batch, kv_len) if use_mask else None
    generator = torch.Generator().manual_seed(2)
    grad_out = torch.randn(
        batch, heads, q_len, value_dim, dtype=dtype, generator=generator
    )
    tol = TOLERANCES[dtype]

    ref_out, ref_grad_q, ref_grad_k, ref_grad_v = reference_with_grads(
        q, k, v, causal=causal, key_padding_mask=mask, grad_out=grad_out
    )

    fwd = version.forward(
        q, k, v, causal=causal, key_padding_mask=mask, config=TILED_CONFIG
    )
    assert (fwd.out.float() - ref_out.float()).abs().max().item() <= tol

    bwd = version.backward(
        q,
        k,
        v,
        grad_out,
        fwd,
        causal=causal,
        key_padding_mask=mask,
        config=TILED_CONFIG,
    )
    assert (bwd.grad_q.float() - ref_grad_q.float()).abs().max().item() <= tol
    assert (bwd.grad_k.float() - ref_grad_k.float()).abs().max().item() <= tol
    assert (bwd.grad_v.float() - ref_grad_v.float()).abs().max().item() <= tol


@pytest.mark.parametrize("version", MODULES.values(), ids=MODULES.keys())
def test_lse_matches_logsumexp(version):
    q, k, v = make_qkv(1, 1, 10, 12, 8, 8)
    scores = torch.einsum("bhid,bhjd->bhij", q / math.sqrt(8), k)
    ref_lse = torch.logsumexp(scores, dim=-1, keepdim=True)

    fwd = version.forward(q, k, v, config=TILED_CONFIG)
    assert (fwd.lse - ref_lse).abs().max().item() <= 1e-4


@pytest.mark.parametrize("version_name", MODULES.keys())
def test_single_tile_matches_reference(version_name):
    """Block sizes larger than the sequence must degrade to plain attention."""
    q, k, v = make_qkv(1, 1, 5, 5, 8, 8)
    big_config = FlashAttentionConfig(block_size_q=1024, block_size_kv=1024)

    ref = reference_attention(q, k, v, causal=True)
    out = flash_attention(q, k, v, version=version_name, causal=True, config=big_config)
    assert (out - ref).abs().max().item() <= 2e-5


@pytest.mark.parametrize("version", MODULES.values(), ids=MODULES.keys())
def test_invalid_shapes_raise(version):
    q, k, v = make_qkv(1, 2, 8, 8, 16, 16)
    bad_k = torch.randn(1, 4, 8, 16)  # head mismatch

    with pytest.raises(ValueError, match="head"):
        version.forward(q, bad_k, v)

    bad_mask = torch.ones(3, 8, dtype=torch.bool)  # batch mismatch
    with pytest.raises(ValueError, match="key_padding_mask"):
        version.forward(q, k, v, key_padding_mask=bad_mask)


@pytest.mark.parametrize("version", MODULES.values(), ids=MODULES.keys())
def test_fully_masked_rows_produce_zero_output(version):
    q, k, v = make_qkv(1, 1, 6, 6, 8, 8)
    mask = torch.zeros(1, 6, dtype=torch.bool)

    fwd = version.forward(q, k, v, key_padding_mask=mask, config=TILED_CONFIG)
    assert torch.isfinite(fwd.out).all()
    assert (fwd.out == 0).all()


@pytest.mark.parametrize("version", MODULES.values(), ids=MODULES.keys())
def test_mixed_input_dtypes_raise(version):
    """Mixed q/k/v dtypes must fail fast in forward instead of crashing in backward."""
    q, k, v = make_qkv(1, 2, 8, 8, 16, 16)

    with pytest.raises(ValueError, match="same dtype"):
        version.forward(q, k, v.to(torch.float16))


@pytest.mark.parametrize("version", MODULES.values(), ids=MODULES.keys())
@pytest.mark.parametrize("case", ["empty_q", "empty_kv", "zero_head_dim"])
def test_degenerate_shapes_raise(version, case):
    """Empty sequences / zero head dims must raise a descriptive ValueError."""
    q, k, v = make_qkv(1, 1, 4, 4, 8, 8)
    generator = torch.Generator().manual_seed(0)
    if case == "empty_q":
        q = torch.randn(1, 1, 0, 8, generator=generator)
        match = "non-zero sequence length"
    elif case == "empty_kv":
        k = torch.randn(1, 1, 0, 8, generator=generator)
        v = torch.randn(1, 1, 0, 8, generator=generator)
        match = "non-zero sequence length"
    else:
        q = torch.randn(1, 1, 4, 0, generator=generator)
        k = torch.randn(1, 1, 4, 0, generator=generator)
        match = "hidden dimension must be non-zero"

    with pytest.raises(ValueError, match=match):
        version.forward(q, k, v)


# --- FA3: simulated FP8 path -------------------------------------------------


def test_fa3_fp8_forward_matches_reference_and_tracks_metadata():
    q, k, v = make_qkv(1, 2, 32, 32, 16, 16)
    fp8_config = FlashAttentionConfig(block_size_q=16, block_size_kv=8, fp8=True)

    fwd = fa3.forward(q, k, v, config=fp8_config)
    ref = reference_attention(q, k, v)

    # Simulated E4M3 quantization is lossy but should stay in the ballpark.
    assert (fwd.out - ref).abs().max().item() <= 0.2

    # Every pipeline stage records its per-tile quantization scales.
    assert fwd.saved_state["fp8_enabled"]
    first_stage = fwd.saved_state["pipeline_trace"][0]
    assert first_stage["fp8"]
    assert first_stage["q_scale"] is not None
    assert first_stage["k_scale"] is not None
    assert first_stage["v_scale"] is not None


def test_fa3_fp8_backward_raises():
    q, k, v = make_qkv(1, 1, 16, 16, 8, 8)
    fp8_config = FlashAttentionConfig(fp8=True)
    fwd = fa3.forward(q, k, v, config=fp8_config)

    with pytest.raises(ValueError, match="FP8 backward"):
        fa3.backward(q, k, v, torch.ones_like(fwd.out), fwd, config=fp8_config)


# --- FA4: thresholded selective rescaling ------------------------------------

SCALE_LOG2 = 1.0 / math.log(2.0)


def _correction_merge_for(block_max: float, **overrides):
    """Run `_correction_merge` on a minimal 1x1x1x2 state, FA4 threshold 8.0."""
    state = {
        "out_acc_block": torch.tensor([[[[2.0, 4.0]]]]),
        "normalizer_block": torch.tensor([[[[3.0]]]]),
        "row_max_block": torch.tensor([[[[10.0]]]]),
        "block_sum": torch.tensor([[[[5.0]]]]),
        "weighted_values": torch.tensor([[[[7.0, 11.0]]]]),
    }
    state.update(overrides)
    return _correction_merge(
        **state,
        block_max=torch.tensor([[[[block_max]]]]),
        scale_log2=SCALE_LOG2,
        rescale_threshold=8.0,
    )


def test_fa4_rescale_threshold_matches_official_policy():
    assert _fa4_rescale_threshold(torch.float16) == 8.0
    assert _fa4_rescale_threshold(torch.bfloat16) == 8.0
    assert _fa4_rescale_threshold(torch.float32) == 0.0


def test_fa4_correction_skips_rescale_for_small_max_increase():
    # block_max 11 vs running max 10: the log2-domain delta stays within the
    # threshold, so the old row max is kept and no full rescale happens.
    _, _, row_max, rescaled = _correction_merge_for(11.0)
    assert not rescaled.any().item()
    assert torch.equal(row_max, torch.tensor([[[[10.0]]]]))


def test_fa4_correction_rescales_for_large_max_increase():
    # block_max 20 vs running max 10: beyond the threshold, full rescale path.
    _, _, row_max, rescaled = _correction_merge_for(20.0)
    assert rescaled.all().item()
    # The merged row max must take the new block max exactly.
    torch.testing.assert_close(row_max, torch.tensor([[[[20.0]]]]))


def test_fa4_correction_initializes_empty_state():
    # Uninitialized rows (row max -inf) must take the full merge path so the
    # running row max becomes finite even with thresholding enabled.
    _, _, row_max, rescaled = _correction_merge_for(
        5.0,
        out_acc_block=torch.zeros(1, 1, 1, 2),
        normalizer_block=torch.zeros(1, 1, 1, 1),
        row_max_block=torch.full((1, 1, 1, 1), float("-inf")),
    )
    assert rescaled.all().item()
    assert torch.isfinite(row_max).all()
