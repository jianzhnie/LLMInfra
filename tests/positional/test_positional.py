"""Tests for the positional encoding modules.

Covers RoPE (and ``apply_rotary_pos_emb``), YaRN, dynamic NTK, partial RoPE,
position interpolation, LongRoPE, 2D position embedding, the ALiBi bias and
the ``get_positional_encoding`` factory.
"""

import math

import pytest
import torch

from llminfra import (
    ALiBiBias,
    DynamicNTKRotaryEmbedding,
    LongRoPEScaledRotaryEmbedding,
    MultiModalRotaryPositionEmbedding,
    PartialRotaryPositionEmbedding,
    PositionInterpolation,
    RotaryPositionEmbedding,
    T5RelativePositionBias,
    TwoDimensionalPositionEmbedding,
    YaRNParameters,
    YaRNScaledRotaryEmbedding,
    apply_rotary_pos_emb,
    get_positional_encoding,
)


def test_rotary_embedding_preserves_norm_and_shape():
    rope = RotaryPositionEmbedding(dim=8, max_seq_len=16)
    x = torch.randn(2, 3, 7, 8)
    y = rope(x)
    assert y.shape == x.shape
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-4)


def test_apply_rotary_pos_emb_rejects_odd_dim():
    x = torch.randn(1, 1, 3, 7)
    cos = torch.ones(1, 1, 3, 7)
    sin = torch.zeros(1, 1, 3, 7)
    with pytest.raises(ValueError, match="even"):
        apply_rotary_pos_emb(x, cos, sin)


def test_yarn_and_dynamic_ntk_are_finite():
    params = YaRNParameters(
        factor=4.0,
        original_max_position_embeddings=2048,
    )
    yarn = YaRNScaledRotaryEmbedding(8, max_seq_len=4096, params=params)
    ntk = DynamicNTKRotaryEmbedding(
        8,
        original_max_position_embeddings=2048,
        max_seq_len=8192,
    )
    x = torch.randn(1, 2, 64, 8)
    assert torch.isfinite(yarn(x)).all()
    assert torch.isfinite(ntk(x)).all()


def test_yarn_interpolates_high_and_keeps_low_frequencies():
    params = YaRNParameters(
        factor=4.0,
        original_max_position_embeddings=2048,
    )
    yarn = YaRNScaledRotaryEmbedding(8, max_seq_len=4096, params=params)
    plain = RotaryPositionEmbedding(8, max_seq_len=4096)
    x = torch.randn(1, 1, 2, 8)
    yarn_out = yarn(x)
    plain_out = plain(x)
    # The lowest-frequency pair keeps the original (unscaled) frequency.
    torch.testing.assert_close(yarn_out[..., -2:], plain_out[..., -2:])
    # The highest-frequency pair is interpolated: angle = position / factor.
    a, b = x[0, 0, 1, 0], x[0, 0, 1, 1]
    theta = 1.0 / params.factor  # inv_freq[0] == 1
    cos, sin = math.cos(theta), math.sin(theta)
    expected = torch.stack([a * cos - b * sin, a * sin + b * cos])
    torch.testing.assert_close(yarn_out[0, 0, 1, :2], expected)


def test_dynamic_ntk_requires_dim_greater_than_two():
    with pytest.raises(ValueError, match="dim > 2"):
        DynamicNTKRotaryEmbedding(
            2,
            original_max_position_embeddings=128,
            max_seq_len=256,
        )


def test_alibi_shape_and_causal_mask():
    alibi = ALiBiBias(num_heads=4, max_seq_len=8)
    x = torch.randn(1, 4, 6, 8)
    bias = alibi(x)
    assert bias.shape == (1, 4, 6, 6)
    assert torch.isinf(bias[0, 0, 0, 1])
    assert bias[0, 0, 5, 4] < 0
    assert bias[0, 0, 5, 4] < bias[0, 1, 5, 4]


def test_partial_rope_shape_and_norm_preserved():
    rope = PartialRotaryPositionEmbedding(
        dim=8, partial_rotary_factor=0.5, max_seq_len=16
    )
    x = torch.randn(2, 3, 7, 8)
    y = rope(x)
    assert y.shape == x.shape
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-4)


def test_position_interpolation_is_finite():
    interpolate = PositionInterpolation(
        dim=8,
        original_max_position_embeddings=2048,
        max_seq_len=4096,
    )
    x = torch.randn(1, 1, 128, 8)
    assert torch.isfinite(interpolate(x)).all()


def test_longrope_and_2d_position_are_finite():
    factors = [1.0, 2.0, 3.0, 4.0]
    longrope = LongRoPEScaledRotaryEmbedding(
        dim=8,
        original_max_position_embeddings=4,
        max_seq_len=16,
        long_factor=factors,
        short_factor=factors,
    )
    x = torch.randn(1, 1, 8, 8)
    assert torch.isfinite(longrope(x)).all()
    two_d = TwoDimensionalPositionEmbedding(8, max_blocks=4, max_positions_per_block=4)
    y = torch.randn(1, 8, 8)
    assert torch.isfinite(two_d(y)).all()


def test_two_dimensional_position_embedding_preserves_shape():
    two_d = TwoDimensionalPositionEmbedding(8, max_blocks=4, max_positions_per_block=4)
    # 2-D input (seq, dim) with default ids must not gain a batch dimension.
    x2d = torch.randn(8, 8)
    assert two_d(x2d).shape == x2d.shape
    # Batched ids must broadcast per batch instead of adding a leading dim.
    x = torch.randn(2, 8, 8)
    block_ids = (torch.arange(8) // 4).expand(2, 8)
    positions = (torch.arange(8) % 4).expand(2, 8)
    out = two_d(x, block_ids, positions)
    assert out.shape == x.shape
    expected = torch.stack(
        [
            x[i]
            + two_d.block_embeddings(block_ids[i])
            + two_d.position_embeddings(positions[i])
            for i in range(x.size(0))
        ]
    )
    torch.testing.assert_close(out, expected)


def test_two_dimensional_position_embedding_allows_empty_input():
    two_d = TwoDimensionalPositionEmbedding(8, max_blocks=4, max_positions_per_block=4)
    x = torch.randn(1, 0, 8)
    assert two_d(x).shape == x.shape


def test_two_dimensional_position_embedding_rejects_negative_ids():
    two_d = TwoDimensionalPositionEmbedding(8, max_blocks=4, max_positions_per_block=4)
    x = torch.randn(1, 4, 8)
    with pytest.raises(ValueError, match="block_ids"):
        two_d(x, block_ids=torch.tensor([0, 1, -1, 2]))
    with pytest.raises(ValueError, match="positions"):
        two_d(x, positions=torch.tensor([0, 1, -2, 3]))


def test_positional_encoding_factory():
    assert isinstance(get_positional_encoding("rope", dim=8), RotaryPositionEmbedding)
    assert isinstance(get_positional_encoding("alibi", dim=8, num_heads=4), ALiBiBias)
    with pytest.raises(ValueError, match="Unknown"):
        get_positional_encoding("unknown", dim=8)


def test_positional_factory_new_modes():
    assert isinstance(
        get_positional_encoding("partial_rope", dim=8, partial_rotary_factor=0.5),
        PartialRotaryPositionEmbedding,
    )
    assert isinstance(
        get_positional_encoding(
            "interpolation",
            dim=8,
            original_max_position_embeddings=2048,
        ),
        PositionInterpolation,
    )
    assert isinstance(
        get_positional_encoding("mrope", dim=8, mrope_section=(1, 1, 2)),
        MultiModalRotaryPositionEmbedding,
    )


def test_positional_factory_2d_rejects_extra_kwargs():
    assert isinstance(
        get_positional_encoding("2d", dim=8, max_blocks=4, max_positions_per_block=4),
        TwoDimensionalPositionEmbedding,
    )
    with pytest.raises(ValueError, match="unsupported 2d"):
        get_positional_encoding(
            "2d", dim=8, max_blocks=4, max_positions_per_block=4, dropout=0.1
        )


def test_mrope_preserves_norm_and_uses_independent_axes():
    mrope = MultiModalRotaryPositionEmbedding(8, mrope_section=(1, 1, 2))
    x = torch.randn(2, 3, 5, 8)
    position_ids = torch.stack(
        [
            torch.arange(5).expand(2, -1),
            torch.zeros(2, 5, dtype=torch.long),
            torch.arange(5).flip(0).expand(2, -1),
        ]
    )
    output = mrope(x, position_ids)
    assert output.shape == x.shape
    torch.testing.assert_close(
        output.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-4
    )


def test_mrope_validates_sections_and_position_shape():
    with pytest.raises(ValueError, match="sum"):
        MultiModalRotaryPositionEmbedding(8, mrope_section=(1, 1))
    mrope = MultiModalRotaryPositionEmbedding(8, mrope_section=(1, 1, 2))
    with pytest.raises(ValueError, match="position_ids"):
        mrope(torch.randn(2, 5, 8), torch.zeros(3, 1, 5))


def test_alibi_bias_cache_follows_module_dtype():
    # The memoized bias must rebuild when .half()/.to(dtype) re-casts the
    # slopes buffer; otherwise stale float32 values are served forever.
    alibi = ALiBiBias(num_heads=4, max_seq_len=16)
    assert alibi(8).dtype == torch.float32
    alibi.half()
    bias = alibi(8)
    assert bias.dtype == torch.float16
    fresh = ALiBiBias(num_heads=4, max_seq_len=16).half()
    torch.testing.assert_close(bias, fresh(8), rtol=0, atol=0)
    alibi.to(torch.float64)
    assert alibi(8).dtype == torch.float64
    fresh64 = ALiBiBias(num_heads=4, max_seq_len=16).to(torch.float64)
    torch.testing.assert_close(alibi(8), fresh64(8), rtol=0, atol=0)


def test_t5_bucket_cache_built_in_inference_mode_allows_backward():
    # Buckets memoized under inference_mode must stay usable as embedding
    # indices in a later autograd-tracked call.
    t5 = T5RelativePositionBias(num_heads=2, max_seq_len=16)
    with torch.inference_mode():
        t5(8)
    bias = t5(8)
    bias.sum().backward()
    assert t5.relative_attention_bias.weight.grad is not None


def test_mrope_text_only_matches_explicit_positions_low_precision():
    # The text-only fast path must agree with the explicit-position_ids
    # path even where bf16 cannot represent token indices exactly.
    mrope = MultiModalRotaryPositionEmbedding(8, mrope_section=(1, 1, 2))
    seq_len = 2100  # beyond bf16's exact-integer range
    x = torch.randn(2, seq_len, 8, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len).expand(3, seq_len)
    assert torch.equal(mrope(x), mrope(x, position_ids))


def test_apply_rotary_pos_emb_allows_empty_sequence():
    # The complex fast path must not choke on the ambiguous -1 view of an
    # empty (seq_len == 0) input.
    x = torch.randn(1, 0, 8)
    cos = torch.randn(0, 4)
    sin = torch.randn(0, 4)
    out = apply_rotary_pos_emb(x, cos, sin)
    assert out.shape == x.shape
    rope = RotaryPositionEmbedding(dim=8, max_seq_len=16)
    assert rope(x).shape == x.shape


def test_rotary_cache_growth_matches_uncached_reference():
    # Serving seq 32 -> 96 -> 32 from the growing table must match a fresh
    # per-call computation of the scalar rotation formula.
    rope = RotaryPositionEmbedding(dim=8, max_seq_len=128)
    inv_freq = rope.inv_freq
    assert isinstance(inv_freq, torch.Tensor)

    def reference(x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(-2), dtype=torch.float32)
        freqs = torch.einsum("s,f->sf", positions, inv_freq.to(torch.float32))
        cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
        sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return x * cos + rotated * sin

    x32 = torch.randn(2, 32, 8)
    x96 = torch.randn(2, 96, 8)
    rope(x32)
    torch.testing.assert_close(rope(x96), reference(x96))
    torch.testing.assert_close(rope(x32), reference(x32))
