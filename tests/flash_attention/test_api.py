"""Tests for the PyTorch-style FlashAttention API."""

import pytest
import torch
from helpers import make_key_padding_mask, make_qkv, with_grad

from llminfra import FlashAttention, flash_attention
from llminfra.flash_attention import (
    ATTENTION_FN_REGISTRY,
    get_version_module,
    list_versions,
)
from llminfra.flash_attention.common import FlashAttentionConfig
from llminfra.flash_attention.fp8_support import validate_fp8_support

CONFIG = FlashAttentionConfig(block_size_q=8, block_size_kv=8)
VERSIONS = list_versions()


@pytest.mark.parametrize("version", VERSIONS)
def test_functional_matches_version_modules(version):
    q, k, v = make_qkv(2, 3, 20, 20, 16, 24)
    direct = ATTENTION_FN_REGISTRY[version](q, k, v, causal=True, config=CONFIG)
    wrapped = flash_attention(q, k, v, version=version, causal=True, config=CONFIG)
    torch.testing.assert_close(wrapped, direct)


@pytest.mark.parametrize("version", VERSIONS)
def test_attention_is_differentiable(version):
    """The flash_attention_v* functions must route into the tiled backward."""
    module = get_version_module(version)
    attention_fn = ATTENTION_FN_REGISTRY[version]
    q, k, v = with_grad(*make_qkv(2, 3, 20, 20, 16, 24))
    grad_out = torch.randn(2, 3, 20, 24, generator=torch.Generator().manual_seed(1))

    out = attention_fn(q, k, v, causal=True, config=CONFIG)
    assert out.grad_fn is not None
    out.backward(grad_out)

    # The autograd path must reproduce the manual forward/backward pair exactly.
    fwd = module.forward(q.detach(), k.detach(), v.detach(), causal=True, config=CONFIG)
    manual = module.backward(
        q.detach(), k.detach(), v.detach(), grad_out, fwd, causal=True, config=CONFIG
    )
    torch.testing.assert_close(out.detach(), fwd.out)
    torch.testing.assert_close(q.grad, manual.grad_q)
    torch.testing.assert_close(k.grad, manual.grad_k)
    torch.testing.assert_close(v.grad, manual.grad_v)


@pytest.mark.parametrize("version", VERSIONS)
def test_attention_supports_padding_mask_with_autograd(version):
    q, k, v = with_grad(*make_qkv(2, 3, 20, 20, 16, 24))
    mask = make_key_padding_mask(2, 20, fully_masked_row=False)

    out = flash_attention(
        q, k, v, version=version, key_padding_mask=mask, config=CONFIG
    )
    out.sum().backward()
    for t in (q, k, v):
        assert t.grad is not None and torch.isfinite(t.grad).all()


def test_module_wrapper_matches_functional():
    q, k, v = make_qkv(2, 3, 20, 20, 16, 24)
    attn = FlashAttention(version="fa3", causal=True, config=CONFIG)

    torch.testing.assert_close(
        attn(q, k, v),
        flash_attention(q, k, v, version="fa3", causal=True, config=CONFIG),
    )
    assert "fa3" in attn.extra_repr()
    assert "causal=True" in attn.extra_repr()


def test_module_wrapper_is_differentiable():
    q, k, v = with_grad(*make_qkv(2, 3, 20, 20, 16, 24))
    attn = FlashAttention()  # defaults to fa2

    out = attn(q, k, v)
    assert out.shape == (2, 3, 20, 24)
    out.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


def test_unknown_version_raises():
    q, k, v = make_qkv(2, 3, 20, 20, 16, 24)
    with pytest.raises(ValueError, match="Unknown FlashAttention version"):
        flash_attention(q, k, v, version="fa9")
    with pytest.raises(ValueError, match="Unknown FlashAttention version"):
        FlashAttention(version="fa9")


def test_fp8_guardrails():
    """`validate_fp8_support` enforces FA3's forward-only FP8 boundary."""
    with pytest.raises(ValueError, match="only implemented for --version fa3"):
        validate_fp8_support(version="fa4", fp8=True, script_name="flash_attention")
    with pytest.raises(ValueError, match="backward is unsupported"):
        validate_fp8_support(version="fa3", fp8=True, script_name="check_backward")
    with pytest.raises(ValueError, match="only applies to the FA3 flash path"):
        validate_fp8_support(
            version="fa3", fp8=True, script_name="bench", benchmark_type="normal"
        )
