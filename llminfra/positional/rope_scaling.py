"""Long-context RoPE scaling variants.

Teaching implementations of the interpolation/extrapolation formulas used by
mainstream long-context models: YaRN, dynamic NTK-aware scaling, partial
RoPE, position interpolation and LongRoPE. Exact numerical behavior should
be checked against the official implementations, especially for the YaRN
coefficients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .base import BasePositionalEncoding
from .rotary import (
    _MAX_CACHED_TABLE_ELEMENTS,
    RotaryPositionEmbedding,
    _cos_sin_with_cache,
    _default_inv_freq,
    apply_rotary_pos_emb,
)


@dataclass(frozen=True)
class YaRNParameters:
    """Parameters used by the YaRN frequency scaling approximation."""

    factor: float = 4.0
    original_max_position_embeddings: int = 4096
    beta_fast: float = 32.0
    beta_slow: float = 1.0


@dataclass(frozen=True)
class LongRoPEPreset:
    """Named LongRoPE coefficient set with provenance metadata.

    Scalar factors are expanded over all rotary frequencies. Tuples preserve
    frequency-specific coefficients copied from a verified model config.
    Factors are *multiplicative* coefficients applied to ``inv_freq``;
    official configs that divide ``inv_freq`` by their ``long_factor`` must
    be imported as reciprocals.
    """

    original_max_position_embeddings: int
    max_position_embeddings: int
    long_factor: float | tuple[float, ...]
    short_factor: float | tuple[float, ...]
    source: str


LONGROPE_PRESETS: dict[str, LongRoPEPreset] = {
    "reference_uniform_256k": LongRoPEPreset(
        4096,
        262144,
        4096 / 262144,
        1.0,
        "Uniform position interpolation reference; not official coefficients",
    ),
    "reference_uniform_512k": LongRoPEPreset(
        4096,
        524288,
        4096 / 524288,
        1.0,
        "Uniform position interpolation reference; not official coefficients",
    ),
    "reference_uniform_1m": LongRoPEPreset(
        4096,
        1048576,
        4096 / 1048576,
        1.0,
        "Uniform position interpolation reference; not official coefficients",
    ),
}


def register_longrope_preset(
    name: str,
    preset: LongRoPEPreset,
    *,
    overwrite: bool = False,
) -> None:
    """Register a verified LongRoPE config for reuse by name.

    Factors must be multiplicative coefficients for ``inv_freq``; use the
    reciprocal when importing configs that divide by their factors.
    """
    if not name:
        raise ValueError("preset name must not be empty")
    if name in LONGROPE_PRESETS and not overwrite:
        raise ValueError(f"LongRoPE preset already exists: {name}")
    LONGROPE_PRESETS[name] = preset


def get_longrope_preset(name: str) -> LongRoPEPreset:
    """Return a named LongRoPE preset or raise a descriptive error."""
    try:
        return LONGROPE_PRESETS[name]
    except KeyError as error:
        available = ", ".join(sorted(LONGROPE_PRESETS))
        raise ValueError(
            f"Unknown LongRoPE preset {name!r}; available: {available}"
        ) from error


def _yarn_find_correction_dim(
    num_rotations: float,
    dim: int,
    base: float,
    original_max_position_embeddings: int,
) -> float:
    return (
        dim
        * math.log(original_max_position_embeddings / (num_rotations * 2 * math.pi))
        / (2 * math.log(base))
    )


def _yarn_linear_ramp_mask(min_val: float, max_val: float, dim: int) -> torch.Tensor:
    if min_val == max_val:
        return torch.ones(dim)
    indices = torch.arange(dim, dtype=torch.float32)
    return 1.0 - ((max_val - indices) / (max_val - min_val)).clamp(0.0, 1.0)


class YaRNScaledRotaryEmbedding(BasePositionalEncoding):
    """YaRN-scaled RoPE for long-context extension.

    High-frequency dimensions keep their original frequencies (extrapolation)
    while low-frequency dimensions are interpolated (frequencies divided by
    ``params.factor``); a linear ramp blends the two regimes, following the
    YaRN paper (Peng et al., 2023) and the Transformers reference. This is a
    teaching implementation of the interpolation/extrapolation blending
    formula. Exact numerical behavior should be checked against Transformers
    and the official YaRN repository.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        params: YaRNParameters,
        base: float = 10000.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.max_seq_len = int(max_seq_len)
        self.params = params
        self.base = float(base)

        inv_freq = _default_inv_freq(self.dim, self.base, dtype)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._register_ramp_mask()

    def _register_ramp_mask(self) -> None:
        low = _yarn_find_correction_dim(
            self.params.beta_fast,
            self.dim,
            self.base,
            self.params.original_max_position_embeddings,
        )
        high = _yarn_find_correction_dim(
            self.params.beta_slow,
            self.dim,
            self.base,
            self.params.original_max_position_embeddings,
        )
        low = max(0, min(low, self.dim // 2 - 1))
        high = max(0, min(high, self.dim // 2 - 1))
        ramp = _yarn_linear_ramp_mask(low, high, self.dim // 2)
        self.register_buffer("ramp", ramp, persistent=False)

    def _build_cos_sin(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the float32 cos/sin table with the blended YaRN frequencies."""
        base_inv_freq = self.inv_freq
        ramp = self.ramp
        assert isinstance(base_inv_freq, torch.Tensor)
        assert isinstance(ramp, torch.Tensor)
        extrapolation = base_inv_freq
        interpolation = base_inv_freq / self.params.factor
        # ramp is 0 on high-frequency dims (below beta_fast's correction dim,
        # which extrapolate) and 1 on low-frequency dims (above beta_slow's,
        # which interpolate); swapping the two weights inverts the blend and
        # contradicts the YaRN paper and the Transformers reference.
        inv_freq = interpolation * ramp + extrapolation * (1.0 - ramp)
        # The table is built in float32 (like the reference kernels) so
        # half-precision inputs do not lose positional accuracy.
        positions = torch.arange(
            seq_len, device=base_inv_freq.device, dtype=torch.float32
        )
        freqs = torch.outer(positions, inv_freq.to(torch.float32))
        return torch.cos(freqs), torch.sin(freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply YaRN-scaled RoPE to ``x``."""
        # The frequency blend is invariant, so cos/sin come from the shared
        # lazily-built cache instead of being recomputed on every call.
        cos, sin = _cos_sin_with_cache(
            self,
            x.size(-2),
            x.dtype,
            self.max_seq_len,
            self.dim // 2,
            self._build_cos_sin,
        )
        return apply_rotary_pos_emb(x, cos, sin)


class DynamicNTKRotaryEmbedding(BasePositionalEncoding):
    """Dynamic NTK-aware RoPE scaling.

    The base frequency is increased when the sequence length exceeds the
    original training context, following the NTK-aware scaling idea used by
    several long-context models.
    """

    def __init__(
        self,
        dim: int,
        original_max_position_embeddings: int,
        max_seq_len: int,
        base: float = 10000.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        if self.dim <= 2:
            raise ValueError("dynamic NTK scaling requires dim > 2")
        self.original_max_position_embeddings = int(original_max_position_embeddings)
        self.max_seq_len = int(max_seq_len)
        self.base = float(base)
        self.dtype = dtype
        self.register_buffer(
            "base_inv_freq",
            _default_inv_freq(self.dim, self.base, dtype),
            persistent=False,
        )
        # Memo for cos/sin tables of scaled (beyond-original-context) lengths,
        # keyed by (seq_len, device); bounded and rebuilt after device moves.
        self._scaled_cache: dict[
            tuple[int, str], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def _build_cos_sin(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the unscaled float32 cos/sin table for short sequences."""
        base_inv_freq = self.base_inv_freq
        assert isinstance(base_inv_freq, torch.Tensor)
        positions = torch.arange(
            seq_len, device=base_inv_freq.device, dtype=torch.float32
        )
        freqs = torch.outer(positions, base_inv_freq.to(torch.float32))
        return torch.cos(freqs), torch.sin(freqs)

    def _scaled_cos_sin(
        self, seq_len: int, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin for scaled lengths, memoized per (length, device)."""
        key = (seq_len, str(x.device))
        cached = self._scaled_cache.get(key)
        if cached is not None:
            return cached[0].to(x.dtype), cached[1].to(x.dtype)
        inv_freq = self._scaled_inv_freq(seq_len).to(
            device=x.device, dtype=torch.float32
        )
        positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        if seq_len * (self.dim // 2) <= _MAX_CACHED_TABLE_ELEMENTS:
            # Layers in one forward pass share a length, so a small memo
            # turns per-layer trig into one computation per length.
            if len(self._scaled_cache) >= 32:
                self._scaled_cache.clear()
            self._scaled_cache[key] = (cos, sin)
        return cos.to(x.dtype), sin.to(x.dtype)

    def _scaled_inv_freq(self, seq_len: int) -> torch.Tensor:
        if seq_len <= self.original_max_position_embeddings:
            base_inv_freq = self.base_inv_freq
            assert isinstance(base_inv_freq, torch.Tensor)
            return base_inv_freq
        scale = seq_len / self.original_max_position_embeddings
        adjusted_base = self.base * (scale ** (self.dim / (self.dim - 2)))
        return _default_inv_freq(self.dim, adjusted_base, self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dynamic NTK-scaled RoPE to ``x``."""
        seq_len = x.size(-2)
        if seq_len <= self.original_max_position_embeddings:
            # Unscaled regime: identical to the base table, amortized only
            # up to the original context so longer rows are never served.
            cos, sin = _cos_sin_with_cache(
                self,
                seq_len,
                x.dtype,
                min(self.max_seq_len, self.original_max_position_embeddings),
                self.dim // 2,
                self._build_cos_sin,
            )
        else:
            cos, sin = self._scaled_cos_sin(seq_len, x)
        return apply_rotary_pos_emb(x, cos, sin)


class PartialRotaryPositionEmbedding(RotaryPositionEmbedding):
    """Partial RoPE used by Gemma 4 and DeepSeek-V4-style p-RoPE designs.

    Only the first ``rotated_dim`` channels are rotated; the remaining
    channels pass through unchanged.
    """

    def __init__(
        self,
        dim: int,
        partial_rotary_factor: float = 0.25,
        base: float = 10000.0,
        max_seq_len: int = 4096,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not 0.0 < partial_rotary_factor <= 1.0:
            raise ValueError("partial_rotary_factor must be in (0, 1]")
        rotated_dim = int(dim * partial_rotary_factor)
        rotated_dim = max(2, rotated_dim - rotated_dim % 2)
        if rotated_dim > dim:
            rotated_dim = dim - dim % 2
        self.full_dim = int(dim)
        self.partial_rotary_factor = float(partial_rotary_factor)
        self.rotated_dim = int(rotated_dim)
        super().__init__(
            self.rotated_dim,
            base=base,
            max_seq_len=max_seq_len,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate only the first ``rotated_dim`` channels of ``x``."""
        if x.size(-1) != self.full_dim:
            raise ValueError(
                f"x last dim {x.size(-1)} must equal full_dim {self.full_dim}"
            )
        # The channel slice is non-contiguous; one small copy lets the
        # rotary kernel use its contiguous fast path.
        rotated = super().forward(x[..., : self.rotated_dim].contiguous())
        return torch.cat([rotated, x[..., self.rotated_dim :]], dim=-1)

    def extra_repr(self) -> str:
        """Show the full dimension, rotated dimension, and rotary factor."""
        return (
            f"full_dim={self.full_dim}, rotated_dim={self.rotated_dim}, "
            f"partial_rotary_factor={self.partial_rotary_factor}"
        )


class PositionInterpolation(RotaryPositionEmbedding):
    """Simple position interpolation for RoPE long-context extension."""

    def __init__(
        self,
        dim: int,
        original_max_position_embeddings: int,
        max_seq_len: int,
        base: float = 10000.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(dim, base=base, max_seq_len=max_seq_len, dtype=dtype)
        self.original_max_position_embeddings = int(original_max_position_embeddings)

    def _positions(self, seq_len: int) -> torch.Tensor:
        """Return positions rescaled into the original training context."""
        scale = self.original_max_position_embeddings / self.max_seq_len
        return super()._positions(seq_len) * scale

    def extra_repr(self) -> str:
        """Append the original training context length to the base repr."""
        return (
            f"{super().extra_repr()}, "
            f"original_max_position_embeddings="
            f"{self.original_max_position_embeddings}"
        )


class LongRoPEScaledRotaryEmbedding(RotaryPositionEmbedding):
    """Simplified LongRoPE with separate short/long frequency factors."""

    def __init__(
        self,
        dim: int,
        original_max_position_embeddings: int,
        max_seq_len: int,
        long_factor: list[float] | tuple[float, ...],
        short_factor: list[float] | tuple[float, ...],
        base: float = 10000.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(dim, base=base, max_seq_len=max_seq_len, dtype=dtype)
        if len(long_factor) != dim // 2 or len(short_factor) != dim // 2:
            raise ValueError("long_factor and short_factor must have dim//2 entries")
        self.original_max_position_embeddings = int(original_max_position_embeddings)
        self.register_buffer(
            "long_factor",
            torch.tensor(long_factor, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "short_factor",
            torch.tensor(short_factor, dtype=dtype),
            persistent=False,
        )

    @classmethod
    def from_preset(
        cls,
        name: str,
        *,
        dim: int,
        base: float = 10000.0,
        dtype: torch.dtype = torch.float32,
    ) -> LongRoPEScaledRotaryEmbedding:
        """Construct an embedding from a registered coefficient set."""
        preset = get_longrope_preset(name)

        def expand_factor(factor: float | tuple[float, ...]) -> tuple[float, ...]:
            if isinstance(factor, float):
                return (factor,) * (dim // 2)
            return factor

        return cls(
            dim,
            preset.original_max_position_embeddings,
            preset.max_position_embeddings,
            expand_factor(preset.long_factor),
            expand_factor(preset.short_factor),
            base=base,
            dtype=dtype,
        )

    def _build_factored_cos_sin(
        self, seq_len: int, factors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the float32 cos/sin table for one factor regime."""
        base_inv_freq = self.inv_freq
        assert isinstance(base_inv_freq, torch.Tensor)
        inv_freq = base_inv_freq * factors.to(base_inv_freq.device)
        positions = torch.arange(
            seq_len, device=base_inv_freq.device, dtype=torch.float32
        )
        freqs = torch.outer(positions, inv_freq.to(torch.float32))
        return torch.cos(freqs), torch.sin(freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LongRoPE with the appropriate frequency factor."""
        seq_len = x.size(-2)
        is_long = seq_len > self.original_max_position_embeddings
        factors = self.long_factor if is_long else self.short_factor
        assert isinstance(factors, torch.Tensor)
        # Each factor regime gets its own cached table, distinguished by slot.
        cos, sin = _cos_sin_with_cache(
            self,
            seq_len,
            x.dtype,
            self.max_seq_len,
            self.dim // 2,
            lambda length: self._build_factored_cos_sin(length, factors),
            slot="_long" if is_long else "_short",
        )
        return apply_rotary_pos_emb(x, cos, sin)
