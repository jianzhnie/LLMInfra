"""Classic absolute, sinusoidal, relative-bias, and NoPE position modules."""

from __future__ import annotations

import math

import torch
from torch import nn

from .base_position import BasePositionalEncoding


class NoPositionEncoding(BasePositionalEncoding):
    """Identity position module for architectures that intentionally use NoPE."""

    def __init__(self, dim: int, max_seq_len: int = 4096) -> None:
        super().__init__()
        if dim < 1 or max_seq_len < 1:
            raise ValueError("dim and max_seq_len must be >= 1")
        self.dim = int(dim)
        self.max_seq_len = int(max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` unchanged after validating its feature dimension."""
        if x.size(-1) != self.dim:
            raise ValueError(f"expected feature dimension {self.dim}, got {x.size(-1)}")
        if x.size(-2) > self.max_seq_len:
            raise ValueError(
                f"seq_len {x.size(-2)} exceeds max_seq_len {self.max_seq_len}"
            )
        return x


class SinusoidalPositionEmbedding(BasePositionalEncoding):
    """Add the fixed sinusoidal encoding from the original Transformer.

    Inputs may have any leading dimensions but must end in ``(seq_len, dim)``.
    Optional ``position_ids`` must be shaped like the input without its final
    feature dimension and allow packed or offset positions.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: float = 10000.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim < 1 or max_seq_len < 1 or base <= 0:
            raise ValueError("dim/max_seq_len must be >= 1 and base must be > 0")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be in [0, 1]")
        self.dim = int(dim)
        self.max_seq_len = int(max_seq_len)
        self.base = float(base)
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_seq_len, dtype=torch.float32)[:, None]
        frequencies = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(base) / dim)
        )
        angles = position * frequencies[None]
        encoding = torch.zeros(max_seq_len, dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(angles)
        if dim > 1:
            encoding[:, 1::2] = torch.cos(angles[:, : dim // 2])
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add fixed positional vectors to ``x``."""
        if position_ids is None:
            # Fast path: sequential positions are a row slice of the table,
            # which broadcasts over batch and skips the gather entirely.
            if x.dim() < 2 or x.size(-1) != self.dim:
                raise ValueError(f"x must end in (seq_len, {self.dim})")
            if x.size(-2) > self.max_seq_len:
                raise ValueError(
                    f"seq_len {x.size(-2)} exceeds max_seq_len {self.max_seq_len}"
                )
            encoding = self.encoding
            assert isinstance(encoding, torch.Tensor)
            positional = encoding[: x.size(-2)].to(dtype=x.dtype)
        else:
            positions = _resolve_position_ids(
                x,
                self.dim,
                self.max_seq_len,
                position_ids,
            )
            encoding = self.encoding
            assert isinstance(encoding, torch.Tensor)
            positional = encoding[positions].to(dtype=x.dtype)
        output: torch.Tensor = self.dropout(x + positional)
        return output


class LearnedAbsolutePositionEmbedding(BasePositionalEncoding):
    """Add a trainable absolute position embedding to hidden states."""

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim < 1 or max_seq_len < 1:
            raise ValueError("dim and max_seq_len must be >= 1")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be in [0, 1]")
        self.dim = int(dim)
        self.max_seq_len = int(max_seq_len)
        self.embedding = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add learned positional vectors to ``x``."""
        if position_ids is None:
            # Fast path: embedding(arange(n)) is exactly weight[:n], which
            # broadcasts over batch and skips the gather entirely.
            if x.dim() < 2 or x.size(-1) != self.dim:
                raise ValueError(f"x must end in (seq_len, {self.dim})")
            if x.size(-2) > self.max_seq_len:
                raise ValueError(
                    f"seq_len {x.size(-2)} exceeds max_seq_len {self.max_seq_len}"
                )
            output: torch.Tensor = self.dropout(x + self.embedding.weight[: x.size(-2)])
            return output
        positions = _resolve_position_ids(
            x,
            self.dim,
            self.max_seq_len,
            position_ids,
        )
        output = self.dropout(x + self.embedding(positions))
        return output


class T5RelativePositionBias(BasePositionalEncoding):
    """Bucketed relative position bias used by T5-style attention.

    This module returns an additive score bias rather than transformed hidden
    states. Causal visibility remains the attention module's responsibility.

    Args:
        num_heads: Number of attention heads.
        num_buckets: Number of relative-distance buckets.
        max_distance: Distances at or beyond this value share the final bucket.
        bidirectional: Use separate positive/negative buckets for encoders.
        max_seq_len: Default length when :meth:`forward` receives no input.

    """

    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
        bidirectional: bool = True,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        if min(num_heads, num_buckets, max_distance, max_seq_len) < 1:
            raise ValueError("T5 bias dimensions must be >= 1")
        if num_buckets < 4:
            raise ValueError("num_buckets must be >= 4")
        if bidirectional and num_buckets % 2:
            raise ValueError("bidirectional T5 bias requires an even num_buckets")
        directional_buckets = num_buckets // 2 if bidirectional else num_buckets
        if max_distance <= directional_buckets // 2:
            raise ValueError("max_distance must exceed the exact bucket range")
        self.num_heads = int(num_heads)
        self.num_buckets = int(num_buckets)
        self.max_distance = int(max_distance)
        self.bidirectional = bool(bidirectional)
        self.max_seq_len = int(max_seq_len)
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
        nn.init.normal_(self.relative_attention_bias.weight, mean=0.0, std=0.02)
        # Buckets depend only on (query, key) lengths, so they are memoized
        # per (lengths, device); embedding values are not cached because the
        # weights may update during training.
        self._bucket_cache: dict[tuple[int, int, str], torch.Tensor] = {}

    def forward(
        self,
        x: torch.Tensor | int | tuple[int, int] | None = None,
        key_length: int | None = None,
    ) -> torch.Tensor:
        """Return bias shaped ``(1, heads, query_length, key_length)``."""
        query_length, resolved_key_length = self._resolve_lengths(x, key_length)
        device = self.relative_attention_bias.weight.device
        key = (query_length, resolved_key_length, str(device))
        buckets = self._bucket_cache.get(key)
        if buckets is None:
            # inference_mode(False) keeps the memoized indices usable in
            # later autograd-tracked calls even when the first call happens
            # under torch.inference_mode(); embedding saves its indices for
            # backward, and inference tensors cannot be saved.
            with torch.inference_mode(False):
                context = torch.arange(query_length, device=device)[:, None]
                memory = torch.arange(resolved_key_length, device=device)[None, :]
                relative_position = memory - context
                buckets = self._relative_position_bucket(relative_position)
            # Cap cached elements (~32 MB of int64) so huge grids still work
            # without unbounded memory; a few entries cover layer reuse.
            if buckets.numel() <= 1 << 22:
                if len(self._bucket_cache) >= 16:
                    self._bucket_cache.clear()
                self._bucket_cache[key] = buckets
        values: torch.Tensor = self.relative_attention_bias(buckets)
        return values.permute(2, 0, 1).unsqueeze(0)

    def _resolve_lengths(
        self,
        x: torch.Tensor | int | tuple[int, int] | None,
        key_length: int | None,
    ) -> tuple[int, int]:
        if isinstance(x, tuple):
            if len(x) != 2:
                raise ValueError("length tuple must contain (query, key)")
            query_length, resolved_key_length = x
        elif isinstance(x, int):
            query_length = x
            resolved_key_length = x if key_length is None else key_length
        elif isinstance(x, torch.Tensor):
            query_length = x.size(-2)
            resolved_key_length = query_length if key_length is None else key_length
        elif x is None:
            query_length = self.max_seq_len
            resolved_key_length = self.max_seq_len if key_length is None else key_length
        else:
            raise TypeError("x must be a tensor, int, length tuple, or None")
        if min(query_length, resolved_key_length) < 1:
            raise ValueError("query and key lengths must be >= 1")
        if max(query_length, resolved_key_length) > self.max_seq_len:
            raise ValueError("query/key length exceeds max_seq_len")
        return int(query_length), int(resolved_key_length)

    def _relative_position_bucket(
        self, relative_position: torch.Tensor
    ) -> torch.Tensor:
        buckets = torch.zeros_like(relative_position, dtype=torch.long)
        num_buckets = self.num_buckets
        if self.bidirectional:
            num_buckets //= 2
            buckets += (relative_position > 0).to(torch.long) * num_buckets
            distance = relative_position.abs()
        else:
            distance = -torch.minimum(
                relative_position,
                torch.zeros_like(relative_position),
            )

        max_exact = num_buckets // 2
        is_small = distance < max_exact
        safe_distance = torch.clamp(distance, min=max_exact).to(torch.float32)
        logarithmic = max_exact + (
            torch.log(safe_distance / max_exact)
            / math.log(self.max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        logarithmic = torch.minimum(
            logarithmic,
            torch.full_like(logarithmic, num_buckets - 1),
        )
        return buckets + torch.where(is_small, distance, logarithmic)


def _resolve_position_ids(
    x: torch.Tensor,
    dim: int,
    max_seq_len: int,
    position_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Validate or create position ids matching ``x.shape[:-1]``."""
    if x.dim() < 2 or x.size(-1) != dim:
        raise ValueError(f"x must end in (seq_len, {dim})")
    seq_len = x.size(-2)
    if position_ids is None:
        positions = torch.arange(seq_len, device=x.device)
        positions = positions.view(*([1] * (x.dim() - 2)), seq_len)
        positions = positions.expand(x.shape[:-1])
    else:
        if position_ids.shape != x.shape[:-1]:
            raise ValueError("position_ids must have shape x.shape[:-1]")
        positions = position_ids.to(device=x.device, dtype=torch.long)
    if positions.numel() and (positions.min() < 0 or positions.max() >= max_seq_len):
        raise ValueError("position_ids contain an out-of-range position")
    return positions


__all__ = [
    "LearnedAbsolutePositionEmbedding",
    "NoPositionEncoding",
    "SinusoidalPositionEmbedding",
    "T5RelativePositionBias",
]
