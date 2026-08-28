"""Quantization-aware training utilities for Transformer components.

The implementations in this module are portable PyTorch references. They
simulate low-precision numerics with straight-through estimators (STE), so
they are useful for architecture experiments and QAT unit tests. They do not
replace vendor FP8/INT8 kernels used by production inference engines.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from torch import nn
from torch.func import functional_call

QuantizationMode = Literal["int4", "int8", "fp8_e4m3", "mxfp4", "nvfp4"]

# Magnitudes representable in the FP4 E2M1 element format (sign applied
# separately), shared by the MXFP4 and NVFP4 block formats.
_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@dataclass(frozen=True)
class QuantizationConfig:
    """Configuration for fake quantization.

    Args:
        mode: Target numerical format. ``"mxfp4"`` is the OCP Microscaling
            FP4 format (E2M1 elements with a shared power-of-two E8M0 scale
            per 32-element block, as used by GPT-OSS); ``"nvfp4"`` is the
            NVIDIA FP4 variant (E2M1 elements with a finer FP8-E4M3 scale per
            16-element block, as used by Blackwell-era models). Both block
            formats ignore ``per_channel``/``channel_axis``; the two-level
            global scale of real NVFP4 recipes is treated as 1.
        per_channel: Compute one scale per channel instead of one scale for
            the entire tensor. This is most useful for weight tensors.
        channel_axis: Axis retained when ``per_channel=True``.
        quantize_weights: Fake-quantize floating-point parameters.
        quantize_inputs: Fake-quantize tensor inputs.
        quantize_outputs: Fake-quantize tensor outputs.
        eps: Lower bound for scale values.

    """

    mode: QuantizationMode = "int8"
    per_channel: bool = False
    channel_axis: int = 0
    quantize_weights: bool = True
    quantize_inputs: bool = True
    quantize_outputs: bool = True
    eps: float = 1e-8

    def __post_init__(self) -> None:
        """Reject unsupported modes and non-positive ``eps`` values."""
        if self.mode not in {"int4", "int8", "fp8_e4m3", "mxfp4", "nvfp4"}:
            raise ValueError(f"Unsupported quantization mode: {self.mode!r}")
        if self.eps <= 0:
            raise ValueError("eps must be > 0")


class FakeQuantizer(nn.Module):
    """Apply differentiable fake quantization with an STE backward pass."""

    def __init__(self, config: QuantizationConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a fake-quantized tensor while preserving identity gradients."""
        if not tensor.is_floating_point() or tensor.numel() == 0:
            return tensor
        with torch.no_grad():
            if self.config.mode == "fp8_e4m3":
                quantized = self._fake_fp8_e4m3(tensor)
            elif self.config.mode == "mxfp4":
                quantized = self._fake_mxfp4(tensor)
            elif self.config.mode == "nvfp4":
                quantized = self._fake_nvfp4(tensor)
            else:
                bits = 4 if self.config.mode == "int4" else 8
                quantized = self._fake_symmetric_integer(tensor, bits)
            # The STE residual is a constant for autograd; computing it under
            # ``no_grad`` keeps a useless ``sub`` node out of the graph.
            delta = quantized - tensor
        return tensor + delta

    def _reduction_dims(self, tensor: torch.Tensor) -> tuple[int, ...]:
        if not self.config.per_channel:
            return tuple(range(tensor.dim()))
        axis = self.config.channel_axis % tensor.dim()
        return tuple(dim for dim in range(tensor.dim()) if dim != axis)

    def _fake_symmetric_integer(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        # Annotated explicitly because ``int.__pow__`` is typed as returning
        # ``Any`` in typeshed (negative exponents yield floats).
        qmax: int = 2 ** (bits - 1) - 1
        reduce_dims = self._reduction_dims(tensor)
        if reduce_dims:
            # The inf-norm is exactly ``abs().amax()`` but reduces in a
            # single pass without materializing the absolute values.
            max_abs = torch.linalg.vector_norm(
                tensor, ord=torch.inf, dim=reduce_dims, keepdim=True
            )
        else:
            max_abs = tensor.detach().abs()
        # Both operands are fresh temporaries, so the in-place chain avoids
        # extra allocations without touching the caller's tensor.
        scale = max_abs.div_(qmax).clamp_min_(self.config.eps)
        quantized = tensor / scale
        # ``Tensor.round_`` is typed as returning ``Any`` in the stubs.
        result: torch.Tensor = quantized.round_().clamp_(-qmax, qmax).mul_(scale)
        return result

    @staticmethod
    def _fake_fp8_e4m3(tensor: torch.Tensor) -> torch.Tensor:
        """Approximate finite E4M3 values without requiring FP8 hardware.

        E4M3 has three explicit mantissa bits and a maximum finite magnitude
        of 448. This reference rounds normal values to a power-of-two step
        selected by their exponent. Subnormal edge behavior is approximated.
        """
        max_finite = 448.0
        clamped = tensor.clamp(-max_finite, max_finite)
        magnitude = clamped.abs()
        safe = magnitude.clamp_min_(torch.finfo(clamped.dtype).tiny)
        exponent = torch.floor(torch.log2(safe)).clamp_(-6, 8)
        # ``exp2`` is the same power-of-two step as ``pow(2, e - 3)`` but is
        # a single elementwise kernel and much cheaper than generic ``pow``.
        step = torch.exp2(exponent.sub_(3))
        rounded = (clamped / step).round_().mul_(step)
        # Zero inputs round to exactly zero on their own (0 / step == 0), so
        # no explicit ``where`` masking is needed.
        return rounded

    @staticmethod
    def _fake_fp4_blocks(
        tensor: torch.Tensor,
        block_size: int,
        scale_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Shared E2M1 block-quantization core for the FP4 formats.

        ``scale_fn`` maps each block's ``max_abs`` (already clamped away from
        zero) to its scale, which is what distinguishes MXFP4 (power-of-two
        E8M0 scale) from NVFP4 (FP8 E4M3 scale).
        """
        levels = torch.tensor(_E2M1_LEVELS, dtype=tensor.dtype, device=tensor.device)
        last_dim = tensor.size(-1)
        pad = (-last_dim) % block_size
        padded = torch.nn.functional.pad(tensor, (0, pad)) if pad else tensor
        blocks = padded.reshape(*padded.shape[:-1], -1, block_size)

        max_abs = blocks.abs().amax(dim=-1, keepdim=True)
        scale = scale_fn(max_abs.clamp_min(torch.finfo(tensor.dtype).tiny))
        normalized = (blocks / scale).abs()
        # Round to the nearest E2M1 magnitude.
        nearest = (normalized.unsqueeze(-1) - levels).abs().argmin(dim=-1)
        quantized = levels[nearest] * blocks.sign() * scale

        quantized = quantized.reshape(*padded.shape)
        return quantized[..., :last_dim] if pad else quantized

    @staticmethod
    def _fake_mxfp4(tensor: torch.Tensor) -> torch.Tensor:
        """Simulate OCP Microscaling FP4 (MXFP4) without FP4 hardware.

        Elements use the E2M1 format and share one power-of-two E8M0 scale
        per 32-element block along the last axis. The scale is chosen as
        ``2 ** floor(log2(max_abs / 6))`` so the block maximum maps inside
        the representable range without saturation.
        """

        def power_of_two_scale(max_abs: torch.Tensor) -> torch.Tensor:
            return torch.exp2(torch.floor(torch.log2(max_abs / 6.0)))

        return FakeQuantizer._fake_fp4_blocks(tensor, 32, power_of_two_scale)

    @staticmethod
    def _fake_nvfp4(tensor: torch.Tensor) -> torch.Tensor:
        """Simulate NVIDIA FP4 (NVFP4) without FP4 hardware.

        Compared to MXFP4, NVFP4 uses smaller 16-element blocks and stores
        the per-block scale itself in FP8 E4M3 instead of a power of two.
        The two-level global FP32 scale of real NVFP4 recipes is treated as
        1 here (a calibration concern, not a format property).
        """

        def e4m3_scale(max_abs: torch.Tensor) -> torch.Tensor:
            # Clamp to the smallest E4M3 subnormal so the element division
            # never divides by zero.
            return FakeQuantizer._fake_fp8_e4m3(max_abs / 6.0).clamp_min(2.0**-9)

        return FakeQuantizer._fake_fp4_blocks(tensor, 16, e4m3_scale)

    def extra_repr(self) -> str:
        """Show the quantizer mode and channel configuration in ``repr``."""
        return (
            f"mode={self.config.mode!r}, "
            f"per_channel={self.config.per_channel}, "
            f"channel_axis={self.config.channel_axis}"
        )


class QATWrapper(nn.Module):
    """Wrap an arbitrary module with input, weight and output fake quantization.

    Parameters are supplied through :func:`torch.func.functional_call`, so the
    wrapped module is never mutated in-place and gradients still reach its
    original parameters. Nested tensor tuples/lists/dicts are supported for
    modules such as attention layers that optionally return weights.
    """

    def __init__(self, module: nn.Module, config: QuantizationConfig) -> None:
        super().__init__()
        self.module = module
        self.config = config
        self.activation_quantizer = FakeQuantizer(config)
        weight_config = QuantizationConfig(
            mode=config.mode,
            per_channel=True,
            channel_axis=config.channel_axis,
            quantize_weights=config.quantize_weights,
            quantize_inputs=config.quantize_inputs,
            quantize_outputs=config.quantize_outputs,
            eps=config.eps,
        )
        self.weight_quantizer = FakeQuantizer(weight_config)

    def _map_tensors(self, value: object) -> object:
        """Recursively fake-quantize every tensor in a nested structure.

        Tuples, lists and dicts are rebuilt with each tensor leaf passed
        through the activation quantizer; non-tensor leaves are returned
        unchanged.
        """
        if isinstance(value, torch.Tensor):
            return self.activation_quantizer(value)
        if isinstance(value, tuple):
            return tuple(self._map_tensors(item) for item in value)
        if isinstance(value, list):
            return [self._map_tensors(item) for item in value]
        if isinstance(value, dict):
            return {key: self._map_tensors(item) for key, item in value.items()}
        return value

    def forward(self, *args: object, **kwargs: object) -> object:
        """Run the wrapped module with the configured fake quantization."""
        if self.config.quantize_inputs:
            # ``_map_tensors`` preserves the outer container type; the casts
            # only recover what the ``object`` annotation cannot express.
            call_args = cast("tuple[Any, ...]", self._map_tensors(args))
            call_kwargs = cast("dict[str, Any]", self._map_tensors(kwargs))
        else:
            call_args = args
            call_kwargs = kwargs

        if self.config.quantize_weights:
            state: dict[str, torch.Tensor] = {}
            for name, parameter in self.module.named_parameters():
                state[name] = self.weight_quantizer(parameter)
            state.update(dict(self.module.named_buffers()))
            output: object = functional_call(self.module, state, call_args, call_kwargs)
        else:
            output = self.module(*call_args, **call_kwargs)

        if self.config.quantize_outputs:
            output = self._map_tensors(output)
        return output

    def extra_repr(self) -> str:
        """Show the quantization mode and wrapped module type in ``repr``."""
        return f"mode={self.config.mode!r}, module={type(self.module).__name__}"


def build_quantized(
    module: nn.Module,
    config: QuantizationConfig | None = None,
    *,
    mode: QuantizationMode = "int8",
    per_channel: bool = False,
    channel_axis: int = 0,
) -> QATWrapper:
    """Build a QAT wrapper around ``module``.

    Passing an explicit ``config`` takes precedence over the convenience
    keyword arguments.
    """
    resolved = config or QuantizationConfig(
        mode=mode,
        per_channel=per_channel,
        channel_axis=channel_axis,
    )
    return QATWrapper(module, resolved)
