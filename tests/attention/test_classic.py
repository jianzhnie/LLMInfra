"""Behavioral and numerical tests for the classic full-attention modules.

Covers MHA, GQA, MQA and MLA (output shapes, masking semantics, attention
weight properties, gradient flow, constructor validation, and numerical
equivalence between related variants and against PyTorch's SDPA) plus ring
attention, which computes exact full attention in chunks.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from helpers import make_causal_mask, make_hidden_state, make_padding_mask, make_qkv

from llminfra import (
    GroupedQueryAttention,
    MultiHeadAttention,
    MultiHeadLatentAttention,
    MultiQueryAttention,
    RingAttention,
    ring_attention,
)

HIDDEN = 64
HEADS = 4
BATCH = 2
SEQ = 7

MODULE_FACTORIES = {
    "mha": lambda: MultiHeadAttention(HIDDEN, HEADS, dropout=0.0),
    "gqa": lambda: GroupedQueryAttention(HIDDEN, HEADS, num_kv_groups=2, dropout=0.0),
    "mqa": lambda: MultiQueryAttention(HIDDEN, HEADS, dropout=0.0),
    "mla": lambda: MultiHeadLatentAttention(
        HIDDEN, HEADS, q_latent_size=16, kv_latent_size=24, dropout=0.0
    ),
}


@pytest.fixture(params=list(MODULE_FACTORIES), ids=list(MODULE_FACTORIES))
def module(request):
    mod = MODULE_FACTORIES[request.param]()
    mod.eval()
    return mod


def test_output_shape_and_finiteness(module):
    out = module(make_hidden_state(BATCH, SEQ, HIDDEN))
    assert out.shape == (BATCH, SEQ, HIDDEN)
    assert torch.isfinite(out).all()


def test_eval_mode_is_deterministic(module):
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    torch.testing.assert_close(module(x), module(x))


def test_causal_mask_zeroes_future_attention(module):
    out, weights = module(
        make_hidden_state(BATCH, SEQ, HIDDEN),
        attention_mask=make_causal_mask(BATCH, SEQ),
        return_attention_weights=True,
    )

    assert out.shape == (BATCH, SEQ, HIDDEN)
    assert weights.shape == (BATCH, HEADS, SEQ, SEQ)
    # Rows are valid probability distributions...
    torch.testing.assert_close(
        weights.sum(dim=-1), torch.ones(BATCH, HEADS, SEQ), atol=1e-6, rtol=1e-5
    )
    # ...and no attention leaks to future positions.
    future = torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), diagonal=1)
    assert weights[:, :, future].abs().max().item() == 0.0


def test_padding_mask_shape_is_supported(module):
    """A broadcastable (batch, 1, 1, seq) padding mask must work, as documented."""
    out, weights = module(
        make_hidden_state(BATCH, SEQ, HIDDEN),
        attention_mask=make_padding_mask(BATCH, SEQ),
        return_attention_weights=True,
    )
    assert out.shape == (BATCH, SEQ, HIDDEN)
    # Masked key positions receive zero weight in the masked batch row.
    assert weights[0, :, :, -2:].abs().max().item() == 0.0
    # The unmasked batch row is free to attend everywhere.
    assert weights[1].sum().item() == pytest.approx(HEADS * SEQ, rel=1e-4)


def test_return_attention_weights_flag(module):
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    assert isinstance(module(x), torch.Tensor)
    out, weights = module(x, return_attention_weights=True)
    assert isinstance(out, torch.Tensor)
    assert isinstance(weights, torch.Tensor)


def test_gradient_flows_to_inputs_and_parameters(module):
    x = make_hidden_state(BATCH, SEQ, HIDDEN).requires_grad_(True)
    module(x).sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, param in module.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"


def test_train_mode_dropout_is_stochastic():
    mod = MultiHeadAttention(HIDDEN, HEADS, dropout=0.5)
    mod.train()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    assert not torch.allclose(mod(x), mod(x))


def test_constructor_rejects_indivisible_hidden_size():
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(30, HEADS)
    with pytest.raises(ValueError, match="divisible"):
        MultiQueryAttention(30, HEADS)
    with pytest.raises(ValueError, match="divisible"):
        GroupedQueryAttention(30, HEADS, num_kv_groups=1)
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadLatentAttention(30, HEADS, q_latent_size=8, kv_latent_size=8)


def test_gqa_rejects_indivisible_groups():
    with pytest.raises(ValueError, match="divisible"):
        GroupedQueryAttention(HIDDEN, HEADS, num_kv_groups=3)


def test_gqa_with_full_groups_equals_mha():
    """GQA with num_kv_groups == num_heads must reduce to exact MHA."""
    mha = MultiHeadAttention(HIDDEN, HEADS, dropout=0.0)
    gqa = GroupedQueryAttention(HIDDEN, HEADS, num_kv_groups=HEADS, dropout=0.0)
    gqa.load_state_dict(mha.state_dict())
    mha.eval()
    gqa.eval()

    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    torch.testing.assert_close(gqa(x), mha(x))


def test_gqa_with_single_group_equals_mqa():
    """GQA with num_kv_groups == 1 must reduce to exact MQA."""
    mqa = MultiQueryAttention(HIDDEN, HEADS, dropout=0.0)
    gqa = GroupedQueryAttention(HIDDEN, HEADS, num_kv_groups=1, dropout=0.0)
    gqa.load_state_dict(mqa.state_dict())
    mqa.eval()
    gqa.eval()

    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    torch.testing.assert_close(gqa(x), mqa(x))


@pytest.mark.parametrize("use_mask", [False, True], ids=["no-mask", "causal"])
def test_mha_matches_pytorch_sdpa(use_mask):
    """MHA output must match F.scaled_dot_product_attention on the same projections."""
    mha = MultiHeadAttention(HIDDEN, HEADS, dropout=0.0)
    mha.eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)

    query = mha.split_head(mha.q_proj(x))
    key = mha.split_head(mha.k_proj(x))
    value = mha.split_head(mha.v_proj(x))
    mask = make_causal_mask(BATCH, SEQ) if use_mask else None

    ref = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
    ref = mha.o_proj(mha.combine_head(ref))

    torch.testing.assert_close(mha(x, attention_mask=mask), ref)


def test_attention_mask_actually_changes_output(module):
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    plain = module(x)
    masked = module(x, attention_mask=make_causal_mask(BATCH, SEQ))
    assert not torch.allclose(plain, masked)


def test_fully_masked_rows_produce_zero_output(module):
    """A row with every key masked must yield zeros, not NaN."""
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    mask = torch.zeros(BATCH, 1, 1, SEQ, dtype=torch.bool)

    out = module(x, attention_mask=mask)

    assert torch.isfinite(out).all()
    # Freshly initialized modules have zero output-projection biases, so the
    # output of an all-masked row is exactly zero.
    assert (out == 0).all()


def test_rejects_non_3d_input(module):
    with pytest.raises(ValueError, match="3D"):
        module(torch.randn(SEQ, HIDDEN))


def test_rejects_mask_with_wrong_batch_size(module):
    bad_mask = torch.ones(BATCH + 1, 1, 1, SEQ, dtype=torch.bool)
    with pytest.raises(ValueError, match="batch size"):
        module(make_hidden_state(BATCH, SEQ, HIDDEN), attention_mask=bad_mask)


def test_mla_reports_latent_configuration():
    mla = MultiHeadLatentAttention(
        HIDDEN, HEADS, q_latent_size=16, kv_latent_size=24, dropout=0.0
    )
    assert "q_latent_size=16" in mla.extra_repr()
    assert "kv_latent_size=24" in mla.extra_repr()


def test_mla_uses_shared_projection_init():
    """MLA must apply the same Xavier + zero-bias init as the other modules."""
    mla = MultiHeadLatentAttention(
        HIDDEN, HEADS, q_latent_size=16, kv_latent_size=24, bias=True
    )
    for name, param in mla.named_parameters():
        if name.endswith("bias"):
            assert (param == 0).all(), f"{name} was not zero-initialized"


# --- Ring attention: exact full attention computed in chunks -------------------


def dense_reference(q, k, v, causal=True):
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.einsum("bhid,bhjd->bhij", q, k) * scale
    if causal:
        mask = torch.tril(torch.ones(q.size(2), k.size(2), dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v


def test_ring_attention_matches_dense():
    q, k, v = make_qkv(BATCH, HEADS, 8, 8, 8, 8)
    actual = ring_attention(q, k, v, causal=True, num_chunks=3)
    torch.testing.assert_close(actual, dense_reference(q, k, v), atol=1e-5, rtol=1e-4)


def test_ring_attention_module_shape_and_gradient():
    layer = RingAttention(HIDDEN, HEADS, num_chunks=3)
    x = make_hidden_state(BATCH, 8, HIDDEN).requires_grad_(True)
    out = layer(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_ring_attention_supports_different_value_dim():
    """The output accumulator must use V's feature dim, not Q/K's head dim."""
    q, k, _ = make_qkv(1, 2, 8, 8, 16, 16)
    v = torch.randn(1, 2, 8, 24, generator=torch.Generator().manual_seed(3))

    out = ring_attention(q, k, v, causal=True, num_chunks=3)

    assert out.shape == (1, 2, 8, 24)
    assert torch.isfinite(out).all()


def test_ring_attention_rejects_attention_mask():
    module = RingAttention(HIDDEN, HEADS, num_chunks=2)
    with pytest.raises(ValueError, match="attention_mask"):
        module(
            make_hidden_state(BATCH, SEQ, HIDDEN),
            attention_mask=make_causal_mask(BATCH, SEQ),
        )
