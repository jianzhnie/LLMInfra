"""Activation function registry used by feed-forward network variants.

All activations are plain functions on tensors so they can be reused both by
``nn.Module`` implementations and by hand-written reference computations in
tests. Use :func:`get_activation` to resolve a name to its function.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

#: Upper clamp used by ``clamped_silu``, mirroring the GPT-OSS swiglu limit.
CLAMPED_SILU_LIMIT = 7.0


def gelu_exact(x: torch.Tensor) -> torch.Tensor:
    """Exact GELU using the error-function formulation (no approximation)."""
    return F.gelu(x)


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """GELU with the tanh approximation used by GPT-2/BERT-style models."""
    return F.gelu(x, approximate="tanh")


def relu(x: torch.Tensor) -> torch.Tensor:
    """Rectified linear unit, ``max(x, 0)``."""
    return F.relu(x)


def squared_relu(x: torch.Tensor) -> torch.Tensor:
    """Squared ReLU, ``relu(x) ** 2``, used by PaLM and Qwen-style models."""
    return F.relu(x).pow(2)


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU (a.k.a. swish), ``x * sigmoid(x)``."""
    return F.silu(x)


def clamped_silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU whose output is clamped above at ``CLAMPED_SILU_LIMIT``.

    A teaching simplification of the GPT-OSS activation clamping: the bounded
    output keeps activations (and therefore logits) from exploding.
    """
    return F.silu(x).clamp(max=CLAMPED_SILU_LIMIT)


#: Registry mapping activation names to their implementations.
ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "gelu": gelu_exact,
    "gelu_exact": gelu_exact,
    "gelu_tanh": gelu_tanh,
    "relu": relu,
    "squared_relu": squared_relu,
    "silu": silu,
    "swish": silu,
    "clamped_silu": clamped_silu,
}


def get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Resolve an activation name to its function.

    Args:
        name: A key in :data:`ACTIVATIONS`, including exact/tanh GELU,
            ReLU, squared ReLU, SiLU/Swish, and clamped SiLU.

    Returns:
        The activation function.

    Raises:
        ValueError: If ``name`` is not a registered activation.

    """
    if name not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation: {name!r}. Available: {sorted(ACTIVATIONS)}"
        )
    return ACTIVATIONS[name]
