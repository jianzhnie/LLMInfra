"""Tests for the non-attention layer modules.

Covers the FFN variants (FeedForward, SwiGLU), RMSNorm, the Mamba2 SSM layer
and TransformerBlock, including blocks wired with custom attention or MoE FFNs.
"""

import pytest
import torch
from helpers import make_hidden_state

from llminfra import (
    FeedForward,
    HybridAttention,
    Mamba2Layer,
    MixtureOfExperts,
    RMSNorm,
    SwiGLUFFN,
    TransformerBlock,
)

# Batch 1 modules are imported directly from their submodules; the package
# level exports are wired separately.
from llminfra.attention.multi_head_attention import MultiHeadAttention
from llminfra.layers.activations import ACTIVATIONS, get_activation
from llminfra.layers.gated_feed_forward import (
    ClampedSwiGLUFFN,
    GeGLUFFN,
    ReGLUFFN,
    build_feed_forward,
)
from llminfra.layers.hybrid_layers import HybridLayerStack, HybridSSMBlock
from llminfra.layers.normalization import DeepNorm, LayerNorm, LayerScale

HIDDEN = 32
HEADS = 4
SEQ = 7
BATCH = 2


def test_rms_norm_normalizes_last_dimension():
    norm = RMSNorm(HIDDEN)
    x = torch.randn(BATCH, SEQ, HIDDEN)
    y = norm(x)
    mean_square = y.pow(2).mean(dim=-1)
    torch.testing.assert_close(
        mean_square, torch.ones_like(mean_square), atol=1e-5, rtol=1e-4
    )


def test_swiglu_ffn_shape_and_gradient():
    ffn = SwiGLUFFN(HIDDEN, intermediate_size=64)
    x = make_hidden_state(BATCH, SEQ, HIDDEN).requires_grad_(True)
    out = ffn(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_feed_forward_shape():
    ffn = FeedForward(HIDDEN, intermediate_size=64, activation="relu")
    out = ffn(make_hidden_state(BATCH, SEQ, HIDDEN))
    assert out.shape == (BATCH, SEQ, HIDDEN)


def test_mamba2_layer_shape_and_gradient():
    layer = Mamba2Layer(HIDDEN, d_state=8)
    x = make_hidden_state(BATCH, SEQ, HIDDEN).requires_grad_(True)
    out, state = layer(x)
    assert out.shape == x.shape
    assert state.ssm.shape == (BATCH, HIDDEN, 8)
    assert state.convolution.shape == (BATCH, HIDDEN, 3)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_mamba2_state_threading_matches_full_forward():
    """Chunked decoding with a threaded state must equal one full pass."""
    layer = Mamba2Layer(HIDDEN, d_state=8).eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)

    full_out, _ = layer(x)
    first_out, state = layer(x[:, :3])
    second_out, state = layer(x[:, 3:], state=state)
    torch.testing.assert_close(torch.cat([first_out, second_out], dim=1), full_out)


def test_mamba2_empty_sequence():
    layer = Mamba2Layer(HIDDEN, d_state=8)
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    out, state = layer(x[:, :0])
    assert out.shape == (BATCH, 0, HIDDEN)
    assert state.ssm.shape == (BATCH, HIDDEN, 8)


def test_mamba2_per_step_discretize_matches_full_materialization():
    """Per-step ZOH discretization must match slicing the full tensors."""
    layer = Mamba2Layer(HIDDEN, d_state=8).eval()
    u = torch.randn(BATCH, SEQ, layer.d_inner)
    b = torch.randn(BATCH, SEQ, layer.d_state)
    dt = torch.rand(BATCH, SEQ, layer.d_inner) * 0.1
    a = -torch.exp(layer.A_log).to(dtype=u.dtype)

    full_a_bar = torch.exp(dt[..., None] * a[None, None])
    full_b_bar = dt[..., None] * b[:, :, None, :] * u[..., None]
    for step in range(SEQ):
        a_bar, b_bar = layer._discretize(u[:, step], b[:, step], dt[:, step], a)
        torch.testing.assert_close(a_bar, full_a_bar[:, step], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(b_bar, full_b_bar[:, step], atol=1e-6, rtol=1e-6)


def test_transformer_block_shape_and_gradient():
    block = TransformerBlock(
        HIDDEN,
        HEADS,
        intermediate_size=64,
    )
    x = make_hidden_state(BATCH, SEQ, HIDDEN).requires_grad_(True)
    out = block(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_transformer_block_with_hybrid_attention():
    hybrid = HybridAttention(
        HIDDEN,
        HEADS,
        linear_interval=3,
        full_interval=1,
        linear_feature_dim=8,
        num_kv_groups=2,
    )
    block = TransformerBlock(
        HIDDEN,
        HEADS,
        intermediate_size=64,
        attention=hybrid,
    )
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    assert block(x, layer_index=0).shape == x.shape
    assert block(x, layer_index=3).shape == x.shape


def test_transformer_block_with_moe_ffn():
    moe = MixtureOfExperts(
        hidden_size=HIDDEN,
        num_experts=4,
        intermediate_size=64,
        top_k=2,
    )
    block = TransformerBlock(
        HIDDEN,
        HEADS,
        intermediate_size=64,
        ffn=moe,
    )
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    assert block(x).shape == x.shape


def test_transformer_block_post_norm_matches_manual_computation():
    """post-norm: attention sees the raw input; norms follow the residuals."""
    block = TransformerBlock(HIDDEN, HEADS, intermediate_size=64, pre_norm=False).eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)

    expected = block.norm1(x + block.attention(x))
    expected = block.norm2(expected + block.ffn(expected))

    torch.testing.assert_close(block(x), expected)


@pytest.mark.parametrize("norm_style", ["pre", "post", "sandwich"])
def test_transformer_block_norm_styles_shape_and_gradient(norm_style):
    block = TransformerBlock(HIDDEN, HEADS, intermediate_size=64, norm_style=norm_style)
    x = make_hidden_state(BATCH, SEQ, HIDDEN).requires_grad_(True)
    out = block(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_transformer_block_invalid_norm_style_raises():
    with pytest.raises(ValueError, match="norm_style"):
        TransformerBlock(HIDDEN, HEADS, intermediate_size=64, norm_style="sideways")


def test_transformer_block_pre_norm_bool_maps_to_norm_style():
    block = TransformerBlock(HIDDEN, HEADS, intermediate_size=64, pre_norm=False)
    assert block.norm_style == "post"
    assert not block.pre_norm
    block = TransformerBlock(HIDDEN, HEADS, intermediate_size=64, pre_norm=True)
    assert block.norm_style == "pre"
    assert block.pre_norm


def test_transformer_block_parallel_shape_and_gradient():
    block = TransformerBlock(HIDDEN, HEADS, intermediate_size=64, parallel=True)
    x = make_hidden_state(BATCH, SEQ, HIDDEN).requires_grad_(True)
    out = block(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_transformer_block_attention_residual_takes_effect():
    torch.manual_seed(0)
    plain = TransformerBlock(HIDDEN, HEADS, intermediate_size=64).eval()
    torch.manual_seed(0)
    gated = TransformerBlock(
        HIDDEN, HEADS, intermediate_size=64, attention_residual=True
    ).eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)

    # The gate initializes to 1.0, so it reproduces the plain residual exactly.
    torch.testing.assert_close(gated(x), plain(x))

    # Rescaling the learned gate must change the block output.
    with torch.no_grad():
        gated.attn_res.weight.mul_(0.5)
    assert not torch.allclose(gated(x), plain(x))

    out = gated(x.requires_grad_(True))
    out.sum().backward()
    assert gated.attn_res.weight.grad is not None
    assert torch.isfinite(gated.attn_res.weight.grad).all()


def test_transformer_block_post_norm_differs_from_pre_norm():
    torch.manual_seed(0)
    pre = TransformerBlock(HIDDEN, HEADS, intermediate_size=64, norm_style="pre").eval()
    torch.manual_seed(0)
    post = TransformerBlock(
        HIDDEN, HEADS, intermediate_size=64, norm_style="post"
    ).eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    assert not torch.allclose(pre(x), post(x))


def test_transformer_block_supports_layernorm_and_layerscale():
    block = TransformerBlock(
        HIDDEN,
        HEADS,
        intermediate_size=64,
        norm_type="layernorm",
        layer_scale_init=1e-2,
    )
    x = make_hidden_state(BATCH, SEQ, HIDDEN).requires_grad_(True)
    output = block(x)
    assert output.shape == x.shape
    assert isinstance(block.norm1, LayerNorm)
    assert isinstance(block.attention_scale, LayerScale)
    output.sum().backward()
    assert block.attention_scale.weight.grad is not None


def test_transformer_block_deepnorm_shape_and_gradient():
    block = TransformerBlock(
        HIDDEN,
        HEADS,
        intermediate_size=64,
        norm_style="deepnorm",
        deepnorm_alpha=2.0,
    )
    x = make_hidden_state(BATCH, SEQ, HIDDEN).requires_grad_(True)
    output = block(x)
    assert output.shape == x.shape
    output.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_transformer_block_only_builds_norms_used_by_the_style():
    deepnorm = TransformerBlock(
        HIDDEN, HEADS, intermediate_size=64, norm_style="deepnorm"
    )
    assert deepnorm.norm1 is None and deepnorm.norm2 is None
    post_parallel = TransformerBlock(
        HIDDEN, HEADS, intermediate_size=64, norm_style="post", parallel=True
    )
    assert post_parallel.norm1 is not None
    assert post_parallel.norm2 is None
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    assert post_parallel(x).shape == x.shape


_VALID_BLOCK_COMBOS = [
    (style, parallel, attention_residual)
    for style in ("pre", "post", "sandwich", "deepnorm")
    for parallel in (False, True)
    for attention_residual in (False, True)
    if not (style == "deepnorm" and attention_residual)
]


@pytest.mark.parametrize(
    ("norm_style", "parallel", "attention_residual"), _VALID_BLOCK_COMBOS
)
def test_transformer_block_every_layout_combination(
    norm_style, parallel, attention_residual
):
    """Every documented norm_style x parallel x residual combo must forward."""
    block = TransformerBlock(
        HIDDEN,
        HEADS,
        intermediate_size=64,
        norm_style=norm_style,
        parallel=parallel,
        attention_residual=attention_residual,
    )
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    output = block(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("norm_style", ["pre", "post", "sandwich"])
def test_transformer_block_manifold_hyper_connection_combinations(norm_style):
    block = TransformerBlock(
        HIDDEN,
        HEADS,
        intermediate_size=64,
        norm_style=norm_style,
        manifold_hyper_connection=True,
    )
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    output = block(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"norm_style": "deepnorm", "attention_residual": True},
        {"manifold_hyper_connection": True, "attention_residual": True},
        {"norm_style": "deepnorm", "manifold_hyper_connection": True},
        {"parallel": True, "manifold_hyper_connection": True},
    ],
)
def test_transformer_block_impossible_combinations_raise(kwargs):
    with pytest.raises(ValueError, match=r"combin|deepnorm|sequential"):
        TransformerBlock(HIDDEN, HEADS, intermediate_size=64, **kwargs)


def test_mamba2_chunked_scan_matches_recurrent():
    layer = Mamba2Layer(HIDDEN, d_state=8).eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)

    out_recurrent, state_recurrent = layer(x)
    out_chunked, state_chunked = layer(x, scan="chunked", chunk_size=3)

    torch.testing.assert_close(out_chunked, out_recurrent)
    torch.testing.assert_close(state_chunked.ssm, state_recurrent.ssm)
    torch.testing.assert_close(
        state_chunked.convolution,
        state_recurrent.convolution,
    )


def test_mamba2_chunked_scan_threads_state():
    layer = Mamba2Layer(HIDDEN, d_state=8).eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)

    full_out, _ = layer(x, scan="chunked", chunk_size=2)
    first_out, state = layer(x[:, :3], scan="chunked", chunk_size=2)
    second_out, _ = layer(x[:, 3:], state=state, scan="chunked", chunk_size=2)
    torch.testing.assert_close(torch.cat([first_out, second_out], dim=1), full_out)


def test_mamba2_invalid_scan_raises():
    layer = Mamba2Layer(HIDDEN, d_state=8)
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    try:
        layer(x, scan="parallel")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown scan mode")


def test_hybrid_ssm_block_follows_pattern():
    torch.manual_seed(0)
    block = HybridSSMBlock(HIDDEN, num_heads=HEADS, pattern="ssm:attn").eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)

    ssm_out, _ = block.layers[0](x)
    # Attention sublayers default to a causal mask when none is given.
    causal = torch.ones(SEQ, SEQ, dtype=torch.bool).tril_()
    expected = block.layers[1](
        ssm_out, attention_mask=causal.view(1, 1, SEQ, SEQ).expand(BATCH, 1, SEQ, SEQ)
    )
    torch.testing.assert_close(block(x), expected)


def test_hybrid_ssm_block_attention_is_causal_by_default():
    block = HybridSSMBlock(HIDDEN, num_heads=HEADS, pattern="ssm:attn").eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    perturbed = x.clone()
    perturbed[:, -1] += 1.0
    torch.testing.assert_close(block(x)[:, :-1], block(perturbed)[:, :-1])


def test_hybrid_layer_stack_full_attention_is_causal_by_default():
    stack = HybridLayerStack(
        hidden_size=HIDDEN,
        num_heads=HEADS,
        intermediate_size=64,
        layer_map="linear:full",
    ).eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    perturbed = x.clone()
    perturbed[:, -1] += 1.0
    torch.testing.assert_close(stack(x)[:, :-1], stack(perturbed)[:, :-1])


@pytest.mark.parametrize(
    "layer_map",
    ["linear", "ssm", "full", "attn", "linear:ssm:full"],
)
def test_hybrid_layer_stack_every_layer_map_token(layer_map):
    stack = HybridLayerStack(
        hidden_size=HIDDEN,
        num_heads=HEADS,
        intermediate_size=64,
        layer_map=layer_map,
    )
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    output, states = stack(x, return_state=True)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    # Only ssm positions produce a state; "attn" is an alias for "full".
    expected = [token == "ssm" for token in stack.layer_map]
    assert [state is not None for state in states] == expected
    if layer_map == "attn":
        assert stack.layer_map == ("full",)


def test_hybrid_layer_stack_rejects_unknown_tokens():
    with pytest.raises(ValueError, match="unknown layer types"):
        HybridLayerStack(
            hidden_size=HIDDEN,
            num_heads=HEADS,
            intermediate_size=64,
            layer_map="linear:bogus",
        )


def test_hybrid_ssm_block_string_and_list_patterns_match():
    torch.manual_seed(0)
    from_string = HybridSSMBlock(HIDDEN, num_heads=HEADS, pattern="ssm:attn").eval()
    torch.manual_seed(0)
    from_list = HybridSSMBlock(HIDDEN, num_heads=HEADS, pattern=["ssm", "attn"]).eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    torch.testing.assert_close(from_string(x), from_list(x))


def test_hybrid_ssm_block_pattern_changes_output():
    torch.manual_seed(0)
    ssm_first = HybridSSMBlock(HIDDEN, num_heads=HEADS, pattern="ssm:attn").eval()
    torch.manual_seed(0)
    attn_first = HybridSSMBlock(HIDDEN, num_heads=HEADS, pattern="attn:ssm").eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    assert not torch.allclose(ssm_first(x), attn_first(x))


def test_hybrid_ssm_block_shape_and_gradient():
    block = HybridSSMBlock(HIDDEN, num_heads=HEADS, pattern="ssm:ssm:attn", d_state=8)
    x = make_hidden_state(BATCH, SEQ, HIDDEN).requires_grad_(True)
    out = block(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# Batch 1: activations registry, gated FFN variants, norm additions, qk_norm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(ACTIVATIONS))
def test_activation_registry_functions_shape_and_finite(name):
    activation = get_activation(name)
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    out = activation(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_activation_registry_gelu_variants_differ():
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    exact = get_activation("gelu")(x)
    tanh = get_activation("gelu_tanh")(x)
    assert not torch.allclose(exact, tanh)


def test_activation_registry_squared_relu():
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    torch.testing.assert_close(
        get_activation("squared_relu")(x),
        torch.tensor([0.0, 0.0, 0.0, 0.25, 4.0]),
    )


def test_get_activation_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown activation"):
        get_activation("mish")


@pytest.mark.parametrize("ffn_cls", [GeGLUFFN, ReGLUFFN, ClampedSwiGLUFFN])
def test_gated_ffn_variants_shape_and_gradient(ffn_cls):
    ffn = ffn_cls(HIDDEN, intermediate_size=64)
    x = make_hidden_state(BATCH, SEQ, HIDDEN).requires_grad_(True)
    out = ffn(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_geglu_ffn_matches_manual_computation():
    ffn = GeGLUFFN(HIDDEN, intermediate_size=64).eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)

    gated = torch.nn.functional.gelu(ffn.gate_proj(x)) * ffn.up_proj(x)
    expected = ffn.down_proj(gated)

    torch.testing.assert_close(ffn(x), expected)


def test_reglu_ffn_matches_manual_computation():
    ffn = ReGLUFFN(HIDDEN, intermediate_size=64).eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)

    gated = torch.relu(ffn.gate_proj(x)) * ffn.up_proj(x)
    expected = ffn.down_proj(gated)

    torch.testing.assert_close(ffn(x), expected)


def test_clamped_swiglu_ffn_output_bounded_for_large_inputs():
    """With huge inputs the clamps keep the down-projection input bounded."""
    limit = 0.5
    ffn = ClampedSwiGLUFFN(HIDDEN, intermediate_size=64, swiglu_limit=limit).eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN) * 1e4

    out = ffn(x)
    assert torch.isfinite(out).all()
    # |hidden| <= limit elementwise, so each output channel is bounded by
    # limit * ||W_down row||_1 + |bias|.
    bound = limit * ffn.down_proj.weight.abs().sum(dim=1)
    if ffn.down_proj.bias is not None:
        bound = bound + ffn.down_proj.bias.abs()
    assert (out.abs() <= bound).all()


def test_clamped_swiglu_ffn_clamp_takes_effect():
    """A tiny limit must saturate the output; a huge one must not."""
    x = make_hidden_state(BATCH, SEQ, HIDDEN) * 1e3
    torch.manual_seed(0)
    tight = ClampedSwiGLUFFN(HIDDEN, intermediate_size=64, swiglu_limit=0.1).eval()
    loose = ClampedSwiGLUFFN(HIDDEN, intermediate_size=64, swiglu_limit=1e6).eval()
    loose.load_state_dict(tight.state_dict())

    assert not torch.allclose(tight(x), loose(x))


def test_build_feed_forward_sizing_rules():
    ffn_4x = build_feed_forward("4x", HIDDEN)
    assert isinstance(ffn_4x, FeedForward)
    assert ffn_4x.intermediate_size == 4 * HIDDEN

    ffn_llama = build_feed_forward("8/3x", HIDDEN)
    assert isinstance(ffn_llama, SwiGLUFFN)
    assert ffn_llama.intermediate_size == round(8 * HIDDEN / 3)

    ffn_custom = build_feed_forward("custom", HIDDEN, ratio=2.5)
    assert isinstance(ffn_custom, SwiGLUFFN)
    assert ffn_custom.intermediate_size == round(2.5 * HIDDEN)

    ffn_override = build_feed_forward("4x", HIDDEN, intermediate_size=17)
    assert ffn_override.intermediate_size == 17

    out = ffn_llama(make_hidden_state(BATCH, SEQ, HIDDEN))
    assert out.shape == (BATCH, SEQ, HIDDEN)


def test_build_feed_forward_invalid_args_raise():
    with pytest.raises(ValueError, match="Unknown ffn kind"):
        build_feed_forward("5x", HIDDEN)
    with pytest.raises(ValueError, match="ratio"):
        build_feed_forward("custom", HIDDEN)


def test_layer_norm_normalizes_last_dimension():
    norm = LayerNorm(HIDDEN).eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    y = norm(x)
    torch.testing.assert_close(
        y.mean(dim=-1), torch.zeros(BATCH, SEQ), atol=1e-5, rtol=0
    )
    torch.testing.assert_close(
        y.var(dim=-1, unbiased=False), torch.ones(BATCH, SEQ), atol=1e-4, rtol=1e-3
    )


def test_layer_norm_shape_gradient_and_no_bias_option():
    norm = LayerNorm(HIDDEN, bias=False)
    assert norm.bias is None
    x = make_hidden_state(BATCH, SEQ, HIDDEN).requires_grad_(True)
    out = norm(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert norm.weight.grad is not None


def test_layer_norm_matches_manual_computation():
    norm = LayerNorm(HIDDEN, eps=1e-6).eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    mean = x.mean(dim=-1, keepdim=True)
    var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
    expected = (x - mean) / torch.sqrt(var + 1e-6) * norm.weight + norm.bias
    torch.testing.assert_close(norm(x), expected)


def test_deepnorm_residual_scaling_formula():
    """DeepNorm must compute norm(alpha * residual + sublayer_output)."""
    alpha = 2.5
    deepnorm = DeepNorm(HIDDEN, alpha=alpha, eps=1e-6).eval()
    residual = make_hidden_state(BATCH, SEQ, HIDDEN, seed=0)
    sublayer_output = make_hidden_state(BATCH, SEQ, HIDDEN, seed=1)

    combined = alpha * residual + sublayer_output
    mean = combined.mean(dim=-1, keepdim=True)
    var = (combined - mean).pow(2).mean(dim=-1, keepdim=True)
    normalized = (combined - mean) / torch.sqrt(var + 1e-6)
    expected = normalized * deepnorm.norm.weight + deepnorm.norm.bias

    torch.testing.assert_close(deepnorm(residual, sublayer_output), expected)


def test_deepnorm_shape_and_gradient():
    deepnorm = DeepNorm(HIDDEN, alpha=2.0)
    residual = make_hidden_state(BATCH, SEQ, HIDDEN, seed=0)
    sublayer_output = make_hidden_state(BATCH, SEQ, HIDDEN, seed=1).requires_grad_(True)
    out = deepnorm(residual, sublayer_output)
    assert out.shape == residual.shape
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert sublayer_output.grad is not None
    assert torch.isfinite(sublayer_output.grad).all()


def test_layer_scale_initializes_to_identity():
    layer_scale = LayerScale(HIDDEN)
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    torch.testing.assert_close(layer_scale(x), x)


def test_layer_scale_shape_gradient_and_init_value():
    layer_scale = LayerScale(HIDDEN, init_value=0.5)
    x = make_hidden_state(BATCH, SEQ, HIDDEN).requires_grad_(True)
    out = layer_scale(x)
    torch.testing.assert_close(out, x * 0.5)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer_scale.weight.grad is not None


def test_mha_qk_norm_output_finite_and_differs():
    torch.manual_seed(0)
    plain = MultiHeadAttention(HIDDEN, HEADS, dropout=0.0).eval()
    normed = MultiHeadAttention(HIDDEN, HEADS, dropout=0.0, qk_norm=True).eval()
    normed.load_state_dict(plain.state_dict())

    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    out_plain = plain(x)
    out_normed = normed(x)

    assert out_normed.shape == x.shape
    assert torch.isfinite(out_normed).all()
    assert not torch.allclose(out_plain, out_normed)


def test_mha_qk_norm_gives_unit_rms_heads():
    mha = MultiHeadAttention(HIDDEN, HEADS, dropout=0.0, qk_norm=True)
    q = torch.randn(BATCH, HEADS, SEQ, HIDDEN // HEADS)
    k = q.clone()

    q_normed, k_normed = mha._apply_qk_norm(q, k)
    rms = q_normed.pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-5, rtol=1e-4)
    rms_k = k_normed.pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms_k, torch.ones_like(rms_k), atol=1e-5, rtol=1e-4)

    # Disabled qk_norm must be an exact no-op.
    plain = MultiHeadAttention(HIDDEN, HEADS, dropout=0.0)
    q_out, k_out = plain._apply_qk_norm(q, k)
    assert q_out is q and k_out is k
