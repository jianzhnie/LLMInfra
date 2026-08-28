"""Quantization-aware training utilities for Transformer components.

The implementations in this module are portable PyTorch references. They
simulate low-precision numerics with straight-through estimators (STE), so
they are useful for architecture experiments and QAT unit tests. They do not
replace vendor FP8/INT8 kernels used by production inference engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from torch import nn
from torch.func import functional_call

QuantizationMode = Literal["int4", "int8", "fp8_e4m3", "mxfp4"]


@dataclass(frozen=True)
class QuantizationConfig:
    """Configuration for fake quantization.

    Args:
        mode: Target numerical format. ``"mxfp4"`` is the OCP Microscaling
            FP4 format (E2M1 elements with a shared power-of-two E8M0 scale
            per 32-element block, as used by GPT-OSS); it always operates on
            32-element blocks along the last axis and ignores
            ``per_channel``/``channel_axis``.
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
        if self.mode not in {"int4", "int8", "fp8_e4m3", "mxfp4"}:
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
    def _fake_mxfp4(tensor: torch.Tensor) -> torch.Tensor:
        """Simulate OCP Microscaling FP4 (MXFP4) without FP4 hardware.

        Elements use the E2M1 format (magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6)
        and share one power-of-two E8M0 scale per 32-element block along the
        last axis. The scale is chosen as ``2 ** floor(log2(max_abs / 6))``
        so the block maximum maps inside the representable range without
        saturation.
        """
        block_size = 32
        # Magnitudes representable in E2M1 (sign is applied separately).
        levels = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        last_dim = tensor.size(-1)
        pad = (-last_dim) % block_size
        padded = torch.nn.functional.pad(tensor, (0, pad)) if pad else tensor
        blocks = padded.reshape(*padded.shape[:-1], -1, block_size)

        max_abs = blocks.abs().amax(dim=-1, keepdim=True)
        safe_max = max_abs.clamp_min(torch.finfo(tensor.dtype).tiny)
        scale = torch.exp2(torch.floor(torch.log2(safe_max / 6.0)))
        normalized = (blocks / scale).abs()
        # Round to the nearest E2M1 magnitude.
        nearest = (normalized.unsqueeze(-1) - levels).abs().argmin(dim=-1)
        quantized = levels[nearest] * blocks.sign() * scale

        quantized = quantized.reshape(*padded.shape)
        return quantized[..., :last_dim] if pad else quantized

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
