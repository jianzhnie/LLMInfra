"""Attention with Linear Biases (ALiBi) positional bias."""

from __future__ import annotations

import torch

from .base import BasePositionalEncoding


def _alibi_slopes(num_heads: int, slope_base: float) -> list[float]:
    """Return per-head ALiBi slopes following the BLOOM convention."""

    def power_of_two_slopes(count: int) -> list[float]:
        return [slope_base ** (-8 * (head + 1) / count) for head in range(count)]

    if num_heads & (num_heads - 1) == 0:
        return power_of_two_slopes(num_heads)
    # Non-power-of-two head counts: take the nearest smaller power of two
    # and pad with every second slope of the doubled sequence, exactly like
    # ``modeling_bloom.build_alibi_tensor``.
    nearest = 1 << (num_heads.bit_length() - 1)
    extra = power_of_two_slopes(2 * nearest)[0::2][: num_heads - nearest]
    return power_of_two_slopes(nearest) + extra


class ALiBiBias(BasePositionalEncoding):
    """Attention with Linear Biases (ALiBi).

    ``forward`` returns a bias tensor of shape
    ``(1, num_heads, seq_len, seq_len)`` that can be added to attention
    scores. When ``causal`` is True, future positions receive ``-inf``.

    Head slopes follow the paper/BLOOM convention
    (``modeling_bloom.build_alibi_tensor``): for power-of-two head counts the
    geometric sequence ``slope_base ** (-8 * (h + 1) / num_heads)``; for other
    head counts, the sequence of the nearest smaller power of two is
    concatenated with every second slope of the sequence for twice that
    count, so e.g. 12 heads reuse the 8-slope sequence plus 4 interleaved
    slopes from the 16-slope sequence.
    """

    def __init__(
        self,
        num_heads: int,
        max_seq_len: int,
        causal: bool = True,
        slope_base: float = 2.0,
    ) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads must be >= 1")
        self.num_heads = int(num_heads)
        self.max_seq_len = int(max_seq_len)
        self.causal = bool(causal)
        self.slope_base = float(slope_base)
        slopes = _alibi_slopes(self.num_heads, self.slope_base)
        self.register_buffer("slopes", torch.tensor(slopes), persistent=False)
        # Bias depends only on (seq_len, device, dtype), so it is memoized
        # within an element budget; larger grids fall back to on-the-fly
        # computation.
        self._bias_cache: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}

    def _build_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build the ``(heads, seq_len, seq_len)`` ALiBi bias tensor."""
        slopes = self.slopes
        assert isinstance(slopes, torch.Tensor)
        q_pos = torch.arange(seq_len, device=device).view(-1, 1)
        k_pos = torch.arange(seq_len, device=device).view(1, -1)
        if self.causal:
            # For k <= q, slope * (k - q) equals -slope * |q - k| exactly,
            # and future entries are overwritten with -inf below, so the
            # abs() pass of the symmetric formula can be skipped here.
            bias = slopes[:, None, None] * (k_pos - q_pos)
            future = q_pos < k_pos
            bias = bias.masked_fill(future[None], float("-inf"))
        else:
            distance = q_pos - k_pos
            bias = -slopes[:, None, None] * distance.abs()
        return bias

    def forward(self, x: torch.Tensor | int | None = None) -> torch.Tensor:
        """Return ALiBi bias for the sequence length of ``x`` if provided."""
        if isinstance(x, int):
            seq_len = x
        else:
            seq_len = self.max_seq_len if x is None else x.size(-2)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )
        # Derive the device from the registered buffer so the module works
        # after .to("cuda") / .to("mps") moves.
        slopes = self.slopes
        assert isinstance(slopes, torch.Tensor)
        device = slopes.device
        # dtype is part of the key: the plain-dict cache is invisible to
        # .half()/.float(), which re-cast the slopes buffer in place.
        key = (seq_len, str(device), slopes.dtype)
        bias = self._bias_cache.get(key)
        if bias is None:
            bias = self._build_bias(seq_len, device)
            # Cap cached elements (~32 MB of float32); a few entries cover
            # per-layer reuse at one sequence length.
            if bias.numel() <= 1 << 23:
                if len(self._bias_cache) >= 8:
                    self._bias_cache.clear()
                self._bias_cache[key] = bias
        return bias.unsqueeze(0)
