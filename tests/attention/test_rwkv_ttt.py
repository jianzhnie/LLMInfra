"""Tests for the RWKV and TTT linear-time sequence modules.

Covers output shapes, gradient flow, constructor validation, recurrent vs
naive-parallel WKV equivalence, agreement with the WKV formulas of
transformers' ``modeling_rwkv.py`` (hand-written reference, since the local
transformers checkout is not importable), TTT's mini-batch gradient descent
against a per-token reference, causality and padding semantics.
"""

import pytest
import torch
from helpers import make_hidden_state
from torch import nn

from llminfra.attention.rwkv import RWKVChannelMix, RWKVLayer, RWKVTimeMix
from llminfra.attention.ttt import TTTLayer

HIDDEN = 32
BATCH = 2
SEQ = 7


# ---------------------------------------------------------------------------
# RWKV
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scan", ["recurrent", "parallel"])
def test_rwkv_shape_and_gradient(scan):
    layer = RWKVLayer(HIDDEN)
    x = make_hidden_state(BATCH, SEQ, HIDDEN).requires_grad_(True)
    out = layer(x, scan=scan)
    assert out.shape == (BATCH, SEQ, HIDDEN)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.time_mix.time_decay.grad is not None
    assert torch.isfinite(layer.time_mix.time_decay.grad).all()


def test_rwkv_constructor_validation():
    with pytest.raises(ValueError, match="hidden_size must be >= 1"):
        RWKVLayer(0)
    with pytest.raises(ValueError, match="intermediate_size must be >= 1"):
        RWKVLayer(HIDDEN, intermediate_size=0)
    layer = RWKVLayer(HIDDEN)
    with pytest.raises(ValueError, match="scan must be"):
        layer(make_hidden_state(BATCH, SEQ, HIDDEN), scan="bogus")
    with pytest.raises(ValueError, match="3D or 4D"):
        layer(
            make_hidden_state(BATCH, SEQ, HIDDEN),
            attention_mask=torch.ones(BATCH, SEQ, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="must have shape"):
        layer(make_hidden_state(BATCH, SEQ, HIDDEN + 1))


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("seq_len", [1, 5, 16, 17])
def test_rwkv_recurrent_matches_parallel(seq_len, masked):
    """The step-by-step WKV scan must equal the naive quadratic form."""
    layer = RWKVLayer(HIDDEN).double().eval()
    x = make_hidden_state(BATCH, seq_len, HIDDEN).double()
    mask = None
    if masked:
        mask = torch.ones(BATCH, 1, 1, seq_len, dtype=torch.bool)
        mask[0, 0, 0, -2:] = False
    torch.testing.assert_close(
        layer(x, attention_mask=mask, scan="recurrent"),
        layer(x, attention_mask=mask, scan="parallel"),
        rtol=1e-8,
        atol=1e-10,
    )


def _transformers_wkv_reference(time_decay, time_first, key, value):
    """Hand-written from ``rwkv_linear_attention_cpu`` (modeling_rwkv.py).

    Same (num, den, max) state, same order of operations: output with the
    time-first bonus first, then decay the state and absorb the token.
    """
    num_state = torch.zeros_like(key[:, 0])
    den_state = torch.zeros_like(key[:, 0])
    max_state = torch.full_like(key[:, 0], -1e38)
    decay = -torch.exp(time_decay)
    outputs = []
    for t in range(key.size(1)):
        key_t = key[:, t]
        value_t = value[:, t]
        max_for_output = torch.maximum(max_state, key_t + time_first)
        e1 = torch.exp(max_state - max_for_output)
        e2 = torch.exp(key_t + time_first - max_for_output)
        outputs.append((e1 * num_state + e2 * value_t) / (e1 * den_state + e2))
        max_for_state = torch.maximum(max_state + decay, key_t)
        e1 = torch.exp(max_state + decay - max_for_state)
        e2 = torch.exp(key_t - max_for_state)
        num_state = e1 * num_state + e2 * value_t
        den_state = e1 * den_state + e2
        max_state = max_for_state
    return torch.stack(outputs, dim=1)


def test_rwkv_wkv_matches_transformers_formula():
    """The WKV scan must agree term by term with transformers' RWKV-4 code."""
    time_mix = RWKVTimeMix(HIDDEN).double()
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        time_mix.time_decay.copy_(
            torch.randn(HIDDEN, generator=generator, dtype=torch.float64)
        )
        time_mix.time_first.copy_(
            torch.randn(HIDDEN, generator=generator, dtype=torch.float64)
        )
    key = torch.randn(BATCH, SEQ, HIDDEN, generator=generator, dtype=torch.float64)
    value = torch.randn(BATCH, SEQ, HIDDEN, generator=generator, dtype=torch.float64)

    reference = _transformers_wkv_reference(
        time_mix.time_decay, time_mix.time_first, key, value
    )
    torch.testing.assert_close(
        time_mix._wkv_recurrent(key, value), reference, rtol=1e-9, atol=1e-12
    )
    torch.testing.assert_close(
        time_mix._wkv_parallel(key, value), reference, rtol=1e-8, atol=1e-10
    )


@pytest.mark.parametrize("scan", ["recurrent", "parallel"])
def test_rwkv_causal_does_not_see_future(scan):
    """Perturbing the last token must not change earlier outputs."""
    layer = RWKVLayer(HIDDEN).eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    perturbed = x.clone()
    perturbed[:, -1] += 10.0
    torch.testing.assert_close(
        layer(x, scan=scan)[:, :-1], layer(perturbed, scan=scan)[:, :-1]
    )


@pytest.mark.parametrize("scan", ["recurrent", "parallel"])
def test_rwkv_padding_tokens_do_not_leak(scan):
    """Padded tokens must not influence any output, not even via token shift."""
    layer = RWKVLayer(HIDDEN).eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    mask = torch.ones(BATCH, 1, 1, SEQ, dtype=torch.bool)
    mask[0, 0, 0, -2:] = False
    perturbed = x.clone()
    perturbed[0, -2:] += 100.0
    torch.testing.assert_close(
        layer(x, attention_mask=mask, scan=scan),
        layer(perturbed, attention_mask=mask, scan=scan),
    )


def test_rwkv_fully_masked_row_is_finite():
    layer = RWKVLayer(HIDDEN).eval()
    mask = torch.zeros(BATCH, 1, SEQ, dtype=torch.bool)
    out = layer(make_hidden_state(BATCH, SEQ, HIDDEN), attention_mask=mask)
    assert torch.isfinite(out).all()


def test_rwkv_empty_sequence():
    layer = RWKVLayer(HIDDEN).eval()
    out = layer(make_hidden_state(BATCH, 0, HIDDEN))
    assert out.shape == (BATCH, 0, HIDDEN)


def test_rwkv_submodules_empty_sequence():
    """Time-mix/channel-mix must not crash on empty inputs when used directly."""
    x = make_hidden_state(BATCH, 0, HIDDEN)
    time_mix = RWKVTimeMix(HIDDEN).eval()
    channel_mix = RWKVChannelMix(HIDDEN).eval()
    assert time_mix(x).shape == (BATCH, 0, HIDDEN)
    assert time_mix(x, scan="parallel").shape == (BATCH, 0, HIDDEN)
    assert channel_mix(x).shape == (BATCH, 0, HIDDEN)


@pytest.mark.parametrize("scan", ["recurrent", "parallel"])
def test_rwkv_half_precision_is_finite(scan):
    """The scan must run in half precision: the state floor and the decay
    rate must stay inside the dtype range instead of overflowing to NaN."""
    layer = RWKVLayer(HIDDEN).half().eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN).half()
    assert torch.isfinite(layer(x, scan=scan)).all()
    # Padded prefix and fully masked rows exercise the state-floor path.
    mask = torch.ones(BATCH, 1, 1, SEQ, dtype=torch.bool)
    mask[0, 0, 0, :2] = False
    assert torch.isfinite(layer(x, attention_mask=mask, scan=scan)).all()
    mask = torch.zeros(BATCH, 1, 1, SEQ, dtype=torch.bool)
    assert torch.isfinite(layer(x, attention_mask=mask, scan=scan)).all()


@pytest.mark.parametrize(
    "dtype,time_decay", [(torch.float32, 90.0), (torch.float16, 12.0)]
)
def test_rwkv_parallel_matches_recurrent_with_extreme_decay(dtype, time_decay):
    """A learned decay whose exp() overflows the dtype must saturate instead
    of producing NaN, and both scans must agree on the saturated result."""
    time_mix = RWKVTimeMix(HIDDEN).to(dtype).eval()
    with torch.no_grad():
        time_mix.time_decay.copy_(torch.full((HIDDEN,), time_decay, dtype=dtype))
    key = torch.randn(BATCH, SEQ, HIDDEN, dtype=dtype)
    value = torch.randn(BATCH, SEQ, HIDDEN, dtype=dtype)
    recurrent = time_mix._wkv_recurrent(key, value)
    parallel = time_mix._wkv_parallel(key, value)
    assert torch.isfinite(recurrent).all()
    assert torch.isfinite(parallel).all()
    torch.testing.assert_close(recurrent, parallel, rtol=1e-2, atol=1e-3)


# ---------------------------------------------------------------------------
# TTT
# ---------------------------------------------------------------------------


def test_ttt_shape_and_gradient():
    layer = TTTLayer(HIDDEN, chunk_size=3, learnable_lr=True)
    x = make_hidden_state(BATCH, SEQ, HIDDEN).requires_grad_(True)
    out = layer(x)
    assert out.shape == (BATCH, SEQ, HIDDEN)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.init_weight.grad).all()
    inner_lr_grad = layer.inner_lr.grad
    assert inner_lr_grad is not None and torch.isfinite(inner_lr_grad).all()


def test_ttt_constructor_validation():
    with pytest.raises(ValueError, match="hidden_size must be >= 1"):
        TTTLayer(0)
    with pytest.raises(ValueError, match="chunk_size must be >= 1"):
        TTTLayer(HIDDEN, chunk_size=0)
    with pytest.raises(ValueError, match="inner_lr must be > 0"):
        TTTLayer(HIDDEN, inner_lr=0.0)
    layer = TTTLayer(HIDDEN)
    with pytest.raises(ValueError, match="3D or 4D"):
        layer(
            make_hidden_state(BATCH, SEQ, HIDDEN),
            attention_mask=torch.ones(BATCH, SEQ, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="must have shape"):
        layer(make_hidden_state(BATCH, SEQ, HIDDEN + 1))


def test_ttt_fixed_lr_is_a_buffer():
    layer = TTTLayer(HIDDEN, learnable_lr=False)
    assert not isinstance(layer.inner_lr, nn.Parameter)
    assert "inner_lr" in layer.state_dict()


def _ttt_minibatch_reference(layer, x, key_padding_mask=None):
    """Per-token reference written directly from the paper's update rule.

    Within each chunk all gradients are evaluated at the chunk-initial
    weight; token t's output uses W after the gradient steps of tokens up to
    and including t (inclusive). Unlike the layer itself, this reference
    loops over tokens one by one and never batches the gradient
    accumulation.
    """
    key = layer.k_proj(x)
    value = layer.v_proj(x)
    query = layer.q_proj(x)
    batch_size, seq_len, _ = x.shape
    weight = layer.init_weight.unsqueeze(0).repeat(batch_size, 1, 1)
    lr = layer.inner_lr.view(1, -1, 1)
    if key_padding_mask is None:
        key_padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    outputs = []
    for start in range(0, seq_len, layer.chunk_size):
        stop = min(start + layer.chunk_size, seq_len)
        chunk_grad = torch.zeros_like(weight)
        for t in range(start, stop):
            valid = key_padding_mask[:, t].unsqueeze(-1).to(x.dtype)
            residual = torch.einsum("bij,bj->bi", weight, key[:, t]) - value[:, t]
            residual = residual * valid
            chunk_grad = chunk_grad + torch.einsum("bi,bj->bij", residual, key[:, t])
            # Inclusive update: token t's gradient is applied before its output.
            weight_t = weight - lr * chunk_grad
            outputs.append(torch.einsum("bij,bj->bi", weight_t, query[:, t]))
        weight = weight - lr * chunk_grad
    return layer.o_proj(torch.stack(outputs, dim=1))


@pytest.mark.parametrize("chunk_size", [1, 3, 16])
@pytest.mark.parametrize("masked", [False, True])
def test_ttt_matches_minibatch_reference(chunk_size, masked):
    """The batched chunk computation must equal the per-token update rule."""
    layer = TTTLayer(HIDDEN, chunk_size=chunk_size, inner_lr=0.02).double().eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN).double()
    mask = None
    if masked:
        mask = torch.ones(BATCH, 1, SEQ, dtype=torch.bool)
        mask[0, 0, -2:] = False
    key_padding_mask = layer._key_padding_mask(mask, BATCH)
    torch.testing.assert_close(
        layer(x, attention_mask=mask),
        _ttt_minibatch_reference(layer, x, key_padding_mask),
        rtol=1e-8,
        atol=1e-10,
    )


def test_ttt_single_step_matches_manual_gradient_descent():
    """One token: z = W_1 q with W_1 = W_0 - lr (W_0 k - v) k^T by hand."""
    layer = (
        TTTLayer(HIDDEN, chunk_size=16, inner_lr=0.05, learnable_lr=False)
        .double()
        .eval()
    )
    generator = torch.Generator().manual_seed(4)
    with torch.no_grad():
        layer.init_weight.copy_(
            torch.randn(HIDDEN, HIDDEN, generator=generator, dtype=torch.float64) * 0.1
        )
    x = make_hidden_state(1, 1, HIDDEN, seed=5).double()

    key = layer.k_proj(x)[0, 0]
    value = layer.v_proj(x)[0, 0]
    query = layer.q_proj(x)[0, 0]
    residual = layer.init_weight @ key - value
    weight_1 = layer.init_weight - layer.inner_lr.view(-1, 1) * torch.outer(
        residual, key
    )
    expected = layer.o_proj((weight_1 @ query).reshape(1, 1, HIDDEN))
    torch.testing.assert_close(layer(x), expected, rtol=1e-9, atol=1e-12)


def test_ttt_update_reduces_reconstruction_loss():
    """With q tied to k, one step must move the inner model toward v."""
    layer = (
        TTTLayer(HIDDEN, chunk_size=16, inner_lr=0.01, learnable_lr=False)
        .double()
        .eval()
    )
    with torch.no_grad():
        layer.q_proj.weight.copy_(layer.k_proj.weight)
        layer.o_proj.weight.copy_(torch.eye(HIDDEN, dtype=torch.float64))
    x = make_hidden_state(1, 1, HIDDEN, seed=6).double()
    key = layer.k_proj(x)[0, 0]
    value = layer.v_proj(x)[0, 0]

    loss_before = 0.5 * ((layer.init_weight @ key - value) ** 2).sum()
    # With q == k and an identity output projection, the layer output is the
    # inner model's reconstruction of k after one gradient step.
    reconstruction = layer(x)[0, 0]
    loss_after = 0.5 * ((reconstruction - value) ** 2).sum()
    assert loss_after < loss_before


def test_ttt_causal_later_tokens_do_not_affect_earlier_outputs():
    """Perturbing a token in a later chunk must not change earlier outputs."""
    layer = TTTLayer(HIDDEN, chunk_size=3).eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    perturbed = x.clone()
    perturbed[:, 5] += 10.0  # second chunk; positions 0-4 must be unchanged
    torch.testing.assert_close(layer(x)[:, :5], layer(perturbed)[:, :5])


def test_ttt_padding_tokens_do_not_update_inner_model():
    """Padded tokens must not influence the outputs at valid positions."""
    layer = TTTLayer(HIDDEN, chunk_size=2).eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    mask = torch.ones(BATCH, 1, SEQ, dtype=torch.bool)
    mask[0, 0, -2:] = False
    perturbed = x.clone()
    perturbed[0, -2:] += 100.0
    out = layer(x, attention_mask=mask)
    out_perturbed = layer(perturbed, attention_mask=mask)
    torch.testing.assert_close(out[0, :-2], out_perturbed[0, :-2])
    torch.testing.assert_close(out[1], out_perturbed[1])


def test_ttt_numerically_stable_no_nan():
    layer = TTTLayer(HIDDEN, chunk_size=8).eval()
    out = layer(make_hidden_state(BATCH, 64, HIDDEN, seed=7))
    assert torch.isfinite(out).all()
    mask = torch.zeros(BATCH, 1, 64, dtype=torch.bool)
    x = make_hidden_state(BATCH, 64, HIDDEN, seed=8)
    out_masked = layer(x, attention_mask=mask)
    assert torch.isfinite(out_masked).all()


def test_ttt_empty_sequence():
    layer = TTTLayer(HIDDEN).eval()
    out = layer(make_hidden_state(BATCH, 0, HIDDEN))
    assert out.shape == (BATCH, 0, HIDDEN)
