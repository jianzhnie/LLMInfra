"""Tests for portable quantization-aware training utilities."""

import pytest
import torch
from torch import nn

from llminfra import FakeQuantizer, QuantizationConfig, build_quantized


@pytest.mark.parametrize("mode", ["int4", "int8", "fp8_e4m3", "mxfp4"])
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


def test_quantization_config_rejects_non_positive_eps():
    with pytest.raises(ValueError, match="eps"):
        QuantizationConfig(eps=0.0)


def test_fake_quantizer_passes_through_non_float_and_empty_tensors():
    quantizer = FakeQuantizer(QuantizationConfig(mode="int8"))
    ints = torch.arange(5)
    assert quantizer(ints) is ints
    empty = torch.empty(0)
    assert quantizer(empty) is empty


def test_qat_wrapper_maps_nested_containers_and_leaves():
    """List inputs are quantized elementwise; non-tensor leaves pass through."""

    class ListModule(nn.Module):
        def forward(self, tensors, scale):
            return [tensors[0] + tensors[1], scale]

    wrapped = build_quantized(ListModule(), config=QuantizationConfig(mode="int8"))
    generator = torch.Generator().manual_seed(0)
    a = torch.randn(4, generator=generator) * 50  # not exactly representable
    b = torch.randn(4, generator=generator) * 50
    out = wrapped([a, b], 3)
    assert isinstance(out, list)
    assert out[1] == 3
    assert out[0].shape == (4,)
    assert not torch.equal(out[0], a + b)


def test_qat_wrapper_can_skip_input_and_weight_quantization():
    module = nn.Linear(8, 4)
    config = QuantizationConfig(
        mode="int8", quantize_inputs=False, quantize_weights=False
    )
    wrapped = build_quantized(module, config=config)
    x = torch.full((2, 8), 100.0)
    # With inputs and weights untouched, only the output is fake-quantized.
    reference = FakeQuantizer(QuantizationConfig(mode="int8"))(module(x))
    torch.testing.assert_close(wrapped(x), reference)


def test_mxfp4_quantizes_to_e2m1_grid():
    """With block max_abs = 6 the shared scale is 1 and values snap to E2M1."""
    values = torch.zeros(32)
    values[0] = 6.0  # block max -> scale = 2 ** floor(log2(6/6)) = 1
    values[1] = 0.4  # -> 0.5
    values[2] = 1.2  # -> 1.0
    values[3] = 1.8  # -> 2.0
    values[4] = -2.6  # -> -3.0
    values[5] = 4.9  # -> 4.0
    quantized = FakeQuantizer(QuantizationConfig(mode="mxfp4"))(values)
    expected = torch.zeros(32)
    expected[0:6] = torch.tensor([6.0, 0.5, 1.0, 2.0, -3.0, 4.0])
    torch.testing.assert_close(quantized, expected)


def test_mxfp4_block_scales_are_independent_powers_of_two():
    """Each 32-element block gets its own power-of-two E8M0 scale."""
    values = torch.zeros(64)
    values[0] = 48.0  # block 0: scale = 2 ** floor(log2(48/6)) = 8
    values[1] = 24.0  # 24 / 8 = 3 is exactly representable
    values[32] = 0.4  # block 1 max -> scale = 2 ** floor(log2(0.4/6)) = 1/16
    # 0.4 / (1/16) = 6.4 -> nearest E2M1 magnitude 6 -> 6/16 = 0.375
    quantized = FakeQuantizer(QuantizationConfig(mode="mxfp4"))(values)
    assert quantized[0].item() == 48.0
    assert quantized[1].item() == 24.0
    assert quantized[32].item() == 0.375


def test_mxfp4_preserves_shape_for_ragged_last_dim():
    x = torch.randn(2, 33, generator=torch.Generator().manual_seed(0))
    quantized = FakeQuantizer(QuantizationConfig(mode="mxfp4"))(x)
    assert quantized.shape == x.shape
    assert torch.isfinite(quantized).all()
