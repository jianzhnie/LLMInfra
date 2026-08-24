"""Gated feed-forward network variants and a sizing factory.

GeGLU/ReGLU share the gate/up/down structure of
:class:`~llminfra.layers.feed_forward.SwiGLUFFN` but swap the gating activation, as
studied in "GLU Variants Improve Transformer" (Shazeer, 2020).
:class:`ClampedSwiGLUFFN` is a teaching-grade simplification of the GPT-OSS
FFN, and :func:`build_feed_forward` builds an FFN from a sizing rule.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

import torch
from torch import nn

from .activations import get_activation
from .feed_forward import FeedForward, SwiGLUFFN


class GatedFFN(nn.Module):
    """Base class for gated FFNs computing ``down(act(gate(x)) * up(x))``."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.activation = activation
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        return cast(torch.Tensor, self.down_proj(gate * up))

    def _init_weights(self) -> None:
        for module in (self.gate_proj, self.up_proj, self.down_proj):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}"
        )


class GeGLUFFN(GatedFFN):
    """GeGLU feed-forward network: ``down(gelu(gate(x)) * up(x))``.

    Same gate/up/down structure as :class:`SwiGLUFFN`, but the gate uses the
    exact GELU instead of SiLU.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__(hidden_size, intermediate_size, get_activation("gelu"), bias)


class ReGLUFFN(GatedFFN):
    """ReGLU feed-forward network: ``down(relu(gate(x)) * up(x))``.

    Same gate/up/down structure as :class:`SwiGLUFFN`, but the gate uses ReLU
    instead of SiLU.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__(hidden_size, intermediate_size, get_activation("relu"), bias)


class ClampedSwiGLUFFN(GatedFFN):
    """GPT-OSS-style SwiGLU with clamped activations.

    Teaching-grade simplification of the GPT-OSS FFN. The forward pass is::

        gate = silu(gate_proj(x))
        up = (up_proj(x) * alpha).clamp(max=swiglu_limit)
        hidden = (gate * up).clamp(-swiglu_limit, swiglu_limit)
        out = down_proj(hidden)

    The clamps bound the activations entering the down projection, which
    improves numerical stability at scale. Unlike GPT-OSS this module keeps
    separate gate/up projections (no fused interleaved layout) and applies
    the activation with plain SiLU.

    Args:
        hidden_size: Feature dimension of input and output.
        intermediate_size: Hidden dimension of the gate/up branches.
        swiglu_limit: Clamp bound applied after the up projection and again
            to the gated product before the down projection.
        alpha: Scale applied to the up projection before clamping.
        bias: Whether the linear projections use biases.

    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        swiglu_limit: float = 7.0,
        alpha: float = 1.702,
        bias: bool = True,
    ) -> None:
        super().__init__(hidden_size, intermediate_size, get_activation("silu"), bias)
        self.swiglu_limit = float(swiglu_limit)
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SiLU gating with the clamped up-projection and product."""
        gate = self.activation(self.gate_proj(x))
        up = (self.up_proj(x) * self.alpha).clamp(max=self.swiglu_limit)
        hidden = (gate * up).clamp(-self.swiglu_limit, self.swiglu_limit)
        return cast(torch.Tensor, self.down_proj(hidden))

    def extra_repr(self) -> str:
        """Show the layer sizes and clamp settings in ``repr(self)``."""
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}, "
            f"swiglu_limit={self.swiglu_limit}, alpha={self.alpha}"
        )


def build_feed_forward(
    kind: str,
    hidden_size: int,
    intermediate_size: int | None = None,
    ratio: float | None = None,
    bias: bool = True,
    multiple_of: int = 1,
) -> nn.Module:
    """Build a feed-forward network from a sizing rule.

    Args:
        kind: Sizing rule. ``"4x"`` builds the classic transformer
            :class:`FeedForward` with ``intermediate = 4 * hidden_size``;
            ``"8/3x"`` builds a :class:`SwiGLUFFN` with
            ``intermediate = round(8/3 * hidden_size)`` (the Llama sizing,
            without Llama's round-to-multiple-of-256 simplification);
            ``"custom"`` builds a :class:`SwiGLUFFN` with
            ``intermediate = round(ratio * hidden_size)`` and requires
            ``ratio``.
        hidden_size: Feature dimension of input and output.
        intermediate_size: Explicit intermediate dimension; overrides the
            value derived from ``kind``/``ratio`` when given.
        ratio: Intermediate-to-hidden ratio for ``kind="custom"``.
        bias: Whether the linear projections use biases.
        multiple_of: Round a derived intermediate size up to this hardware-
            friendly multiple. Explicit ``intermediate_size`` values are kept
            unchanged.

    Returns:
        The constructed feed-forward module.

    Raises:
        ValueError: If ``kind`` is unknown, or ``kind="custom"`` is used
            without a ``ratio``.

    """
    if hidden_size < 1 or multiple_of < 1:
        raise ValueError("hidden_size and multiple_of must be >= 1")
    if kind == "4x":
        derived = 4 * hidden_size
        module_cls: type[nn.Module] = FeedForward
    elif kind == "8/3x":
        derived = round(8 * hidden_size / 3)
        module_cls = SwiGLUFFN
    elif kind == "custom":
        if ratio is None or ratio <= 0:
            raise ValueError(
                "build_feed_forward(kind='custom') requires a positive ratio"
            )
        derived = round(ratio * hidden_size)
        module_cls = SwiGLUFFN
    else:
        raise ValueError(
            f"Unknown ffn kind: {kind!r}. Available: ['4x', '8/3x', 'custom']"
        )
    if intermediate_size is None:
        intermediate_size = math.ceil(derived / multiple_of) * multiple_of
    elif intermediate_size < 1:
        raise ValueError("intermediate_size must be >= 1")
    return module_cls(hidden_size, intermediate_size, bias=bias)
