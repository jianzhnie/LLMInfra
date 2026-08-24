"""Educational PyTorch reimplementations of FlashAttention v1-v4.

Each ``faX`` module exposes the same three building blocks — ``forward``,
``backward`` and a version-named differentiable entry point
(``flash_attention_v1`` .. ``flash_attention_v4``) — built on the shared
primitives in `common`. The implementations mirror the *algorithmic*
structure of each paper (loop order, work partitioning, pipelining,
scheduling) while staying plain, readable PyTorch; no CUDA details are
simulated.

For everyday use, prefer the PyTorch-style entry points defined here:

- `flash_attention`: a functional interface in the spirit of
  ``torch.nn.functional.scaled_dot_product_attention``, with a ``version``
  knob selecting the FA generation.
- `FlashAttention`: an ``nn.Module`` wrapper holding the version/causal/
  config choices, so call sites read like any other PyTorch layer.
- `flash_attention_v1` .. `flash_attention_v4`: the per-version functions,
  re-exported here for direct import.

All of them are fully differentiable: gradients flow through the tiled
backward pass of the selected version.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType

import torch
from torch import nn

from . import flash_attention_v1 as fa1
from . import flash_attention_v2 as fa2
from . import flash_attention_v3 as fa3
from . import flash_attention_v4 as fa4
from .common import FlashAttentionConfig, reference_attention
from .flash_attention_v1 import flash_attention_v1
from .flash_attention_v2 import flash_attention_v2
from .flash_attention_v3 import flash_attention_v3
from .flash_attention_v4 import flash_attention_v4

__all__ = [
    "ATTENTION_FN_REGISTRY",
    "VERSION_REGISTRY",
    "FlashAttention",
    "FlashAttentionConfig",
    "flash_attention",
    "flash_attention_v1",
    "flash_attention_v2",
    "flash_attention_v3",
    "flash_attention_v4",
    "get_version_module",
    "list_versions",
    "reference_attention",
]

VERSION_REGISTRY: dict[str, ModuleType] = {
    "fa1": fa1,
    "fa2": fa2,
    "fa3": fa3,
    "fa4": fa4,
}

#: Version key -> differentiable attention entry point.
ATTENTION_FN_REGISTRY: dict[str, Callable[..., torch.Tensor]] = {
    "fa1": flash_attention_v1,
    "fa2": flash_attention_v2,
    "fa3": flash_attention_v3,
    "fa4": flash_attention_v4,
}


def get_version_module(version: str) -> ModuleType:
    """Return the module implementing ``version`` (one of ``fa1``..``fa4``).

    Raises:
        ValueError: If ``version`` is not a known FlashAttention version.

    """
    try:
        return VERSION_REGISTRY[version]
    except KeyError as exc:
        raise ValueError(f"Unknown FlashAttention version: {version}") from exc


def list_versions() -> list[str]:
    """Return the available FlashAttention version keys."""
    return list(VERSION_REGISTRY)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    version: str = "fa2",
    causal: bool = False,
    key_padding_mask: torch.Tensor | None = None,
    config: FlashAttentionConfig | None = None,
) -> torch.Tensor:
    """PyTorch-style functional interface to the educational FlashAttention versions.

    Mirrors the calling convention of
    ``torch.nn.functional.scaled_dot_product_attention``: tensors use the
    ``(batch, heads, seq, dim)`` layout and the result is a plain,
    differentiable output tensor.

    Args:
        q: Queries, shape ``(batch, heads, q_len, head_dim)``.
        k: Keys, shape ``(batch, heads, kv_len, head_dim)``.
        v: Values, shape ``(batch, heads, kv_len, value_dim)``.
        version: Which educational implementation to run: ``"fa1"``,
            ``"fa2"``, ``"fa3"`` or ``"fa4"``. Each maps to the corresponding
            `flash_attention_v1` .. `flash_attention_v4` function.
        causal: Apply a causal mask. When ``kv_len > q_len`` the diagonal is
            aligned with the end of the key sequence.
        key_padding_mask: Optional mask of shape ``(batch, kv_len)``; ``True``
            marks valid key positions.
        config: Tiling and debug knobs; ``FlashAttentionConfig()`` defaults
            are used when omitted.

    Returns:
        The attention output tensor of shape
        ``(batch, heads, q_len, value_dim)``, with gradients wired through the
        selected version's tiled backward pass.

    Raises:
        ValueError: If ``version`` is not a known FlashAttention version.

    """
    try:
        attention_fn = ATTENTION_FN_REGISTRY[version]
    except KeyError as exc:
        raise ValueError(f"Unknown FlashAttention version: {version}") from exc
    return attention_fn(
        q,
        k,
        v,
        causal=causal,
        key_padding_mask=key_padding_mask,
        config=config,
    )


class FlashAttention(nn.Module):
    """``nn.Module`` wrapper around `flash_attention`.

    Holds the version, causality and tiling configuration so call sites read
    like any other PyTorch layer::

        >>> attn = FlashAttention(version="fa2", causal=True)
        >>> out = attn(q, k, v)          # differentiable
        >>> out.sum().backward()         # uses fa2's tiled backward

    Args:
        version: Which educational implementation to run: ``"fa1"``,
            ``"fa2"``, ``"fa3"`` or ``"fa4"``.
        causal: Apply a causal mask.
        config: Tiling and debug knobs; ``FlashAttentionConfig()`` defaults
            are used when omitted.

    Raises:
        ValueError: If ``version`` is not a known FlashAttention version.

    """

    def __init__(
        self,
        version: str = "fa2",
        *,
        causal: bool = False,
        config: FlashAttentionConfig | None = None,
    ) -> None:
        super().__init__()
        get_version_module(version)  # validate the version eagerly
        self.version = version
        self.causal = causal
        self.config = config

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run attention over ``(batch, heads, seq, dim)`` q/k/v tensors."""
        return flash_attention(
            q,
            k,
            v,
            version=self.version,
            causal=self.causal,
            key_padding_mask=key_padding_mask,
            config=self.config,
        )

    def extra_repr(self) -> str:
        """Return a string representation of the module's configuration."""
        return f"version={self.version!r}, causal={self.causal}"
