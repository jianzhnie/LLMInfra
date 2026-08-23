"""Multi-axis rotary position embedding for multimodal token sequences."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .base import BasePositionalEncoding
from .rotary import _default_inv_freq, apply_rotary_pos_emb


class MultiModalRotaryPositionEmbedding(BasePositionalEncoding):
    """Apply RoPE sections driven by separate temporal/height/width positions.

    ``mrope_section`` is expressed in rotary *pairs*, not scalar dimensions.
    For example, ``dim=64`` and ``mrope_section=(11, 11, 10)`` allocate all
    32 rotary pairs across three axes. Text-only callers can omit
    ``position_ids``; every axis then receives the standard 1-D token index,
    which reduces to ordinary RoPE.

    Args:
        dim: Even feature dimension to rotate.
        mrope_section: Number of rotary pairs assigned to each axis.
        base: RoPE base frequency.
        max_seq_len: Documented maximum sequence length.
        dtype: Buffer dtype for inverse frequencies.

    """

    def __init__(
        self,
        dim: int,
        mrope_section: Sequence[int],
        base: float = 10000.0,
        max_seq_len: int = 4096,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("mRoPE requires an even feature dimension")
        sections = tuple(int(section) for section in mrope_section)
        if not sections or any(section <= 0 for section in sections):
            raise ValueError("mrope_section must contain positive integers")
        if sum(sections) != dim // 2:
            raise ValueError(
                "sum(mrope_section) must equal dim // 2; "
                f"got {sum(sections)} and {dim // 2}"
            )
        self.dim = int(dim)
        self.mrope_section = sections
        self.base = float(base)
        self.max_seq_len = int(max_seq_len)
        self.register_buffer(
            "inv_freq",
            _default_inv_freq(dim, self.base, dtype),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rotate ``x`` using per-axis position ids.

        Args:
            x: Tensor shaped ``(batch, seq, dim)`` or
                ``(batch, heads, seq, dim)``.
            position_ids: Optional integer/float tensor shaped
                ``(axes, batch, seq)`` or ``(axes, seq)``.

        """
        if x.dim() not in {3, 4}:
            raise ValueError(
                "x must have shape (batch, seq, dim) or (batch, heads, seq, dim)"
            )
        if x.size(-1) != self.dim:
            raise ValueError(f"x last dimension must be {self.dim}")
        batch_size = x.size(0)
        seq_len = x.size(-2)
        num_axes = len(self.mrope_section)

        inv_freq = self.inv_freq
        assert isinstance(inv_freq, torch.Tensor)
        if position_ids is None:
            # Text-only fast path: every axis receives the token index, so
            # the per-axis loop collapses into one outer product whose
            # (seq, freq) cos/sin broadcasts over batch and heads. The index
            # is built as an integer and then cast: torch.arange in bf16
            # rounds positions >= 2057 differently than an integer->bf16
            # cast, which would diverge from the explicit-position_ids path.
            positions = torch.arange(seq_len, device=x.device).to(dtype=x.dtype)
            frequency = torch.outer(positions, inv_freq.to(dtype=x.dtype))
            cos = frequency.cos()
            sin = frequency.sin()
            return apply_rotary_pos_emb(x, cos, sin)

        positions = position_ids.to(device=x.device)
        if positions.dim() == 2:
            positions = positions[:, None, :].expand(-1, batch_size, -1)
        expected = (num_axes, batch_size, seq_len)
        if tuple(positions.shape) != expected:
            raise ValueError(
                f"position_ids must have shape {expected}, got {tuple(positions.shape)}"
            )

        frequencies: list[torch.Tensor] = []
        start = 0
        for axis, section in enumerate(self.mrope_section):
            stop = start + section
            axis_frequency = torch.einsum(
                "bs,f->bsf",
                positions[axis].to(dtype=x.dtype),
                inv_freq[start:stop].to(dtype=x.dtype),
            )
            frequencies.append(axis_frequency)
            start = stop
        frequency = torch.cat(frequencies, dim=-1)
        cos = frequency.cos()
        sin = frequency.sin()
        if x.dim() == 4:
            cos = cos[:, None]
            sin = sin[:, None]
        return apply_rotary_pos_emb(x, cos, sin)

    def extra_repr(self) -> str:
        """Show the dimension, per-axis rotary-pair split, base, and length."""
        return (
            f"dim={self.dim}, mrope_section={self.mrope_section}, "
            f"base={self.base}, max_seq_len={self.max_seq_len}"
        )
