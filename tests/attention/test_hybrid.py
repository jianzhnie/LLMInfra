"""Tests for hybrid and auxiliary attention modules.

Covers HybridAttention (linear/full routing by layer index), ALiBi attention,
the attention-residual connector and the FlashMLA prefill/decode interface.
"""

import pytest
import torch
from helpers import make_hidden_state

from llminfra import (
    ALiBiAttention,
    AttentionResidual,
    FlashMLA,
    HybridAttention,
)

HIDDEN = 32
HEADS = 4
SEQ = 7
BATCH = 2


def test_hybrid_attention_routes_by_layer_index():
    hybrid = HybridAttention(
        HIDDEN,
        HEADS,
        linear_interval=3,
        full_interval=1,
        linear_feature_dim=8,
        num_kv_groups=2,
    )
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    assert hybrid.is_linear_layer(0)
    assert hybrid.is_linear_layer(2)
    assert not hybrid.is_linear_layer(3)
    assert hybrid(x, layer_index=0).shape == x.shape
    assert hybrid(x, layer_index=3).shape == x.shape


def test_hybrid_attention_gradient_flows():
    hybrid = HybridAttention(
        HIDDEN,
        HEADS,
        linear_interval=1,
        full_interval=1,
        linear_feature_dim=8,
    )
    x = make_hidden_state(BATCH, SEQ, HIDDEN).requires_grad_(True)
    hybrid(x, layer_index=1).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_hybrid_attention_returns_full_attention_weights():
    hybrid = HybridAttention(
        HIDDEN,
        HEADS,
        linear_interval=1,
        full_interval=1,
        num_kv_groups=2,
    )
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    out, weights = hybrid(x, return_attention_weights=True, layer_index=1)
    assert out.shape == (BATCH, SEQ, HIDDEN)
    assert weights.shape == (BATCH, HEADS, SEQ, SEQ)


def test_hybrid_attention_rejects_bad_intervals():
    with pytest.raises(ValueError, match="must be >= 1"):
        HybridAttention(HIDDEN, HEADS, linear_interval=0, full_interval=1)


def test_alibi_attention_shape_and_causal_weights():
    layer = ALiBiAttention(HIDDEN, HEADS, num_kv_groups=2)
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    out, weights = layer(x, return_attention_weights=True)
    assert out.shape == x.shape
    future = torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), diagonal=1)
    assert weights[:, :, future].abs().max().item() == 0.0


def test_attention_residual_shape_and_gradient():
    residual = AttentionResidual(HIDDEN)
    x = make_hidden_state(BATCH, SEQ, HIDDEN).requires_grad_(True)
    attention_output = make_hidden_state(BATCH, SEQ, HIDDEN, seed=1)
    out = residual(x, attention_output)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_flash_mla_interface():
    layer = FlashMLA(HIDDEN, HEADS, q_latent_size=8, kv_latent_size=12)
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    out = layer.prefill(x)
    assert out.shape == x.shape
    assert len(layer.latent_cache) == 1
    assert layer.decode(x).shape == x.shape
    layer.reset_cache()
    assert len(layer.latent_cache) == 0
