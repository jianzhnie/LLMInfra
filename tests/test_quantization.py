"""Tests for portable quantization-aware training utilities."""

import pytest
import torch
from torch import nn

from llminfra import FakeQuantizer, QuantizationConfig, build_quantized


@pytest.mark.parametrize("mode", ["int4", "int8", "fp8_e4m3"])
def test_fake_quantizer_is_finite_and_preserves_gradient(mode: str):
    quantizer = FakeQuantizer(QuantizationConfig(mode=mode))
    x = torch.linspace(-10, 10, 33, requires_grad=True)
    output = quantizer(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_int4_has_fewer_levels_than_int8():
    x = torch.linspace(-1, 1, 257)
    int4 = FakeQuantizer(QuantizationConfig(mode="int4"))(x)
    int8 = FakeQuantizer(QuantizationConfig(mode="int8"))(x)
    assert int4.unique().numel() < int8.unique().numel()


@pytest.mark.parametrize(("mode", "qmax"), [("int4", 7), ("int8", 127)])
def test_symmetric_integer_quantization_respects_error_bound(mode: str, qmax: int):
    """Round-to-nearest with scale = max_abs / qmax must err by <= scale / 2."""
    x = torch.randn(64, generator=torch.Generator().manual_seed(0)) * 3
    quantized = FakeQuantizer(QuantizationConfig(mode=mode))(x)
    scale = x.abs().max() / qmax
    assert (quantized - x).abs().max() <= scale / 2 + 1e-6


def test_qat_wrapper_quantizes_module_without_mutating_parameters():
    module = nn.Linear(8, 4)
    original = module.weight.detach().clone()
    wrapped = build_quantized(module, mode="int8")
    x = torch.randn(2, 3, 8, requires_grad=True)

    output = wrapped(x)
    assert output.shape == (2, 3, 4)
    output.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert module.weight.grad is not None
    torch.testing.assert_close(module.weight, original)


def test_quantization_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported"):
        QuantizationConfig(mode="nf3")  # type: ignore[arg-type]
