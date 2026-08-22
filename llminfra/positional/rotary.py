"""Rotary Position Embedding (RoPE) core: rotation kernel and base module."""

from __future__ import annotations

from collections.abc import Callable

import torch

from .base_position import BasePositionalEncoding

# Cap on cached cos/sin table elements (~16 MB per float32 table) so presets
# with very large ``max_seq_len`` (e.g. LongRoPE 1M) do not explode memory.
_MAX_CACHED_TABLE_ELEMENTS = 1 << 22


def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary position embedding to the last dimension of ``x``.

    Args:
        x: Tensor with an even final dimension.
        cos: Cosine frequencies broadcastable to ``x``.
        sin: Sine frequencies broadcastable to ``x``.

    """
    if x.size(-1) % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")
    if cos.size(-1) == x.size(-1) // 2:
        # Half-sized frequencies: each adjacent pair (x1, x2) rotates as
        # (x1*cos - x2*sin, x2*cos + x1*sin), which is complex multiplication
        # by (cos + i*sin). Every path below computes exactly this formula.
        if (
            x.dtype in (torch.float32, torch.float64)
            and x.is_contiguous()
            and x.storage_offset() % 2 == 0
        ):
            # Fast path: viewing interleaved pairs as complex numbers turns
            # the rotation into a single complex multiply, avoiding the
            # repeat_interleave/stack temporaries of the scalar formula. The
            # pair count is written explicitly because a -1 view dimension
            # is ambiguous for empty (seq_len == 0) inputs.
            pairs = torch.view_as_complex(x.view(*x.shape[:-1], x.size(-1) // 2, 2))
            freqs = torch.complex(cos.to(x.dtype), sin.to(x.dtype))
            return torch.view_as_real(pairs * freqs).flatten(-2)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin
        return torch.stack((out1, out2), dim=-1).flatten(-2)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return x * cos + rotated * sin


def _default_inv_freq(
    dim: int, base: float, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Compute the standard inverse frequency for RoPE."""
    if dim % 2 != 0:
        raise ValueError("RoPE dimension must be even")
    indices = torch.arange(0, dim, 2, dtype=dtype)
    return 1.0 / (base ** (indices / dim))


def _cos_sin_with_cache(
    module: BasePositionalEncoding,
    seq_len: int,
    dtype: torch.dtype,
    max_seq_len: int,
    half_dim: int,
    build: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
    slot: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cos/sin for ``seq_len`` positions, caching the float32 table.

    The table is built lazily by ``build`` on first use, amortized up to
    ``max_seq_len`` and capped by ``_MAX_CACHED_TABLE_ELEMENTS``. Tables are
    registered as non-persistent buffers so ``module.to(...)`` moves them;
    ``slot`` distinguishes regimes (e.g. LongRoPE short vs. long factors).
    """
    cos_cached = getattr(module, f"cos_cached{slot}", None)
    if cos_cached is not None and cos_cached.size(0) >= seq_len:
        sin_cached = getattr(module, f"sin_cached{slot}")
        assert isinstance(cos_cached, torch.Tensor)
        assert isinstance(sin_cached, torch.Tensor)
        # Slices of the precomputed table; .to() is a no-op when the caller
        # is already float32, costing just two views per call.
        return cos_cached[:seq_len].to(dtype), sin_cached[:seq_len].to(dtype)
    if seq_len * half_dim <= _MAX_CACHED_TABLE_ELEMENTS:
        build_len = max(
            seq_len, min(max_seq_len, _MAX_CACHED_TABLE_ELEMENTS // half_dim)
        )
        cos, sin = build(build_len)
        module.register_buffer(f"cos_cached{slot}", cos, persistent=False)
        module.register_buffer(f"sin_cached{slot}", sin, persistent=False)
        return cos[:seq_len].to(dtype), sin[:seq_len].to(dtype)
    # Sequences beyond the cap fall back to on-the-fly computation instead
    # of growing the cache unboundedly.
    cos, sin = build(seq_len)
    return cos.to(dtype), sin.to(dtype)


class RotaryPositionEmbedding(BasePositionalEncoding):
    """Rotary Position Embedding.

    Args:
        dim: Rotated feature dimension, normally ``head_dim``.
        base: RoPE base frequency.
        max_seq_len: Maximum sequence length used for precomputed frequencies.

    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        max_seq_len: int = 4096,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.base = float(base)
        self.max_seq_len = int(max_seq_len)
        inv_freq = _default_inv_freq(self.dim, self.base, dtype)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _positions(self, seq_len: int) -> torch.Tensor:
        """Return the float32 position indices for ``[0, seq_len)``."""
        inv_freq = self.inv_freq
        assert isinstance(inv_freq, torch.Tensor)
        return torch.arange(seq_len, device=inv_freq.device, dtype=torch.float32)

    def _inv_freq_float32(self) -> torch.Tensor:
        """Return the inverse frequencies used for the cos/sin table."""
        inv_freq = self.inv_freq
        assert isinstance(inv_freq, torch.Tensor)
        return inv_freq.to(torch.float32)

    def _build_cos_sin(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the float32 cos/sin table for positions ``[0, seq_len)``."""
        freqs = torch.outer(self._positions(seq_len), self._inv_freq_float32())
        return torch.cos(freqs), torch.sin(freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate ``x`` by position indices derived from its sequence length."""
        # The table is kept in float32 so half-precision inputs do not lose
        # positional accuracy on long sequences.
        cos, sin = _cos_sin_with_cache(
            self,
            x.size(-2),
            x.dtype,
            self.max_seq_len,
            self.dim // 2,
            self._build_cos_sin,
        )
        return apply_rotary_pos_emb(x, cos, sin)

    def extra_repr(self) -> str:
        """Show the rotated dimension, base frequency, and cached length."""
        return f"dim={self.dim}, base={self.base}, max_seq_len={self.max_seq_len}"
