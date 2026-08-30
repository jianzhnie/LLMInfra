"""Quantized KV cache with per-channel keys and per-token values.

Teaching reference for low-bit KV cache quantization, following KIVI
(arXiv:2402.02750) and KVQuant (arXiv:2401.18079). Both papers observe that
the outlier structure of K and V is asymmetric across the token and channel
dimensions — outliers in K are concentrated in a few fixed channels while V
shows no such pattern — so the two tensors should not share one quantization
granularity. This module therefore quantizes keys per channel (one scale per
(head, channel)) and values per token (one scale per (token, head)), and
keeps the most recent tokens in a full-precision residual buffer that is
quantized in batches once it fills up (the KIVI "residual length" idea).

Teaching simplifications (declared, not hidden):

- int4 codes are stored in int8 containers; no nibble bit-packing is done.
- Zero-points are stored as floats instead of integers on the quantized grid,
  which keeps degenerate (constant) tensors exactly representable.
- Key/value statistics are per flush batch rather than global running
  min/max.
- No CUDA kernels, no fused dequantize-attention: reads dequantize to dense
  tensors and reuse the existing :func:`paged_attention` code path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from llminfra.inference.paged_attention import paged_attention

SUPPORTED_BITS = (4, 8)


@dataclass
class QuantizedChunk:
    """One batch of KV rows quantized together (a KIVI-style flush).

    Attributes:
        key_q: Quantized key codes in an int8 container, shape
            ``(tokens, num_heads, head_dim)``.
        key_scale: Per-channel key scales in float32, shape
            ``(1, num_heads, head_dim)``: one scale per (head, channel),
            shared by all tokens of this chunk.
        key_zero: Per-channel key zero-points in float32, same shape as
            ``key_scale``.
        value_q: Quantized value codes in an int8 container, shape
            ``(tokens, num_heads, head_dim)``.
        value_scale: Per-token value scales in float32, shape
            ``(tokens, num_heads, 1)``: one scale per (token, head).
        value_zero: Per-token value zero-points in float32, same shape as
            ``value_scale``.

    """

    key_q: torch.Tensor
    key_scale: torch.Tensor
    key_zero: torch.Tensor
    value_q: torch.Tensor
    value_scale: torch.Tensor
    value_zero: torch.Tensor

    def nbytes(self) -> int:
        """Return the bytes held by this chunk (codes plus scales/zeros)."""
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.key_q,
                self.key_scale,
                self.key_zero,
                self.value_q,
                self.value_scale,
                self.value_zero,
            )
        )


def _quantize_affine(
    tensor: torch.Tensor,
    bits: int,
    reduce_dims: tuple[int, ...],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Min-max asymmetric affine quantization over ``reduce_dims``.

    Returns ``(codes, scale, zero_point)`` such that
    ``tensor ≈ (codes - zero_point) * scale`` with a per-element error of at
    most ``scale / 2``. The zero-point is kept as a float (see the module
    docstring), so constant tensors quantize losslessly up to ``eps``.
    """
    # Annotated explicitly because ``int.__pow__`` is typed as returning
    # ``Any`` in typeshed (negative exponents yield floats).
    qmin: int = -(2 ** (bits - 1))
    qmax: int = 2 ** (bits - 1) - 1
    # Statistics and codes are computed in float32 even when the cache
    # stores a lower-precision dtype: in float16 an eps-clamped scale is
    # subnormal and ``lo / scale`` overflows, turning constant groups into
    # NaN. This also matches the fp32 scales/zero-points that
    # ``storage_nbytes`` documents.
    lo = tensor.amin(dim=reduce_dims, keepdim=True).float()
    hi = tensor.amax(dim=reduce_dims, keepdim=True).float()
    scale = (hi - lo) / (qmax - qmin)
    # A near-constant group clamps the scale to eps, where ``lo / scale``
    # overflows for large |lo| (e.g. a constant 1e38 in fp32). Anchoring
    # such a group on the extreme code keeps the zero-point finite and the
    # constant exactly representable.
    degenerate = scale < eps
    scale = torch.where(degenerate, lo.abs() / qmax, scale)
    # Finite endpoints can still overflow the span itself (e.g. +-3e38 in
    # fp32); dividing before subtracting keeps the scale finite.
    scale = torch.where(
        torch.isinf(scale), hi / (qmax - qmin) - lo / (qmax - qmin), scale
    )
    scale = scale.clamp_min(eps)
    zero = qmin - lo / scale
    codes = (tensor.float() / scale + zero).round().clamp(qmin, qmax).to(torch.int8)
    return codes, scale, zero


def _dequantize_affine(
    codes: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Invert :func:`_quantize_affine` and cast to ``dtype``."""
    return ((codes.to(scale.dtype) - zero) * scale).to(dtype)


class QuantizedKVCache:
    """KV cache that stores low-bit codes and dequantizes on read.

    Append/get semantics mirror :class:`PagedAttentionCache`: ``append``
    takes ``(seq_len, num_heads, head_dim)`` rows for a sequence id and
    ``get`` returns dense tensors of the same shape. Keys are quantized per
    channel (one scale per (head, channel) per flush batch), values per token
    (one scale per (token, head)), following the asymmetric K/V outlier
    observation of KIVI (arXiv:2402.02750).

    The most recent ``residual_length`` tokens are kept at full precision and
    are only quantized once newer tokens push them out of the buffer
    (teaching simplification of KIVI's residual-length scheme, where the
    buffer hides quantization latency for the tokens that matter most to the
    next-step distribution).
    """

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        bits: int = 8,
        residual_length: int = 128,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-8,
    ) -> None:
        if num_heads < 1 or head_dim < 1:
            raise ValueError("num_heads and head_dim must be >= 1")
        if bits not in SUPPORTED_BITS:
            raise ValueError(f"bits must be one of {SUPPORTED_BITS}, got {bits}")
        if residual_length < 0:
            raise ValueError("residual_length must be >= 0")
        if eps <= 0:
            raise ValueError("eps must be > 0")
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.bits = int(bits)
        self.residual_length = int(residual_length)
        self.device = torch.device(device)
        self.dtype = dtype
        self.eps = float(eps)
        self.chunks: dict[int, list[QuantizedChunk]] = {}
        self.residual_key: dict[int, torch.Tensor] = {}
        self.residual_value: dict[int, torch.Tensor] = {}
        self.num_tokens: dict[int, int] = {}

    def _empty_residual(self) -> torch.Tensor:
        return torch.zeros(
            0, self.num_heads, self.head_dim, device=self.device, dtype=self.dtype
        )

    def append(
        self,
        seq_id: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Append ``key``/``value`` rows to ``seq_id``.

        Rows first land in the full-precision residual buffer; once the
        buffer exceeds ``residual_length`` the overflow is quantized as one
        batch (a :class:`QuantizedChunk`).

        Args:
            seq_id: Logical sequence id.
            key: Shape ``(seq_len, num_heads, head_dim)``.
            value: Same shape as ``key``.

        """
        if key.dim() != 3 or value.dim() != 3:
            raise ValueError("key and value must have shape (seq, heads, head_dim)")
        if key.size(1) != self.num_heads or key.size(2) != self.head_dim:
            raise ValueError("key/value head configuration does not match cache")
        if key.shape != value.shape:
            raise ValueError("key and value must have the same shape")

        self.chunks.setdefault(seq_id, [])
        self.residual_key.setdefault(seq_id, self._empty_residual())
        self.residual_value.setdefault(seq_id, self._empty_residual())
        self.num_tokens.setdefault(seq_id, 0)
        if key.size(0) == 0:
            return

        # Copy inputs: later in-place mutation of the caller's tensors must
        # not corrupt the cache.
        key = key.detach().to(device=self.device, dtype=self.dtype).clone()
        value = value.detach().to(device=self.device, dtype=self.dtype).clone()
        self.residual_key[seq_id] = torch.cat((self.residual_key[seq_id], key))
        self.residual_value[seq_id] = torch.cat((self.residual_value[seq_id], value))
        self.num_tokens[seq_id] += key.size(0)

        # Flush in batches, not per token, so quantization statistics are
        # computed over a batch as in KIVI; the tail stays full precision.
        excess = self.residual_key[seq_id].size(0) - self.residual_length
        if excess > 0:
            self._flush(seq_id, excess)

    def _flush(self, seq_id: int, num_flush: int) -> None:
        """Quantize the oldest ``num_flush`` residual rows into a chunk."""
        key = self.residual_key[seq_id][:num_flush]
        value = self.residual_value[seq_id][:num_flush]
        # Keys: one scale/zero-point per (head, channel) -> reduce tokens.
        key_q, key_scale, key_zero = _quantize_affine(key, self.bits, (0,), self.eps)
        # Values: one scale/zero-point per (token, head) -> reduce head_dim.
        value_q, value_scale, value_zero = _quantize_affine(
            value, self.bits, (2,), self.eps
        )
        self.chunks[seq_id].append(
            QuantizedChunk(key_q, key_scale, key_zero, value_q, value_scale, value_zero)
        )
        self.residual_key[seq_id] = self.residual_key[seq_id][num_flush:]
        self.residual_value[seq_id] = self.residual_value[seq_id][num_flush:]

    def get(self, seq_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return dequantized K/V for ``seq_id`` as dense tensors."""
        if seq_id not in self.num_tokens:
            raise KeyError(f"Unknown sequence id: {seq_id}")
        keys = [
            _dequantize_affine(chunk.key_q, chunk.key_scale, chunk.key_zero, self.dtype)
            for chunk in self.chunks[seq_id]
        ]
        keys.append(self.residual_key[seq_id])
        values = [
            _dequantize_affine(
                chunk.value_q, chunk.value_scale, chunk.value_zero, self.dtype
            )
            for chunk in self.chunks[seq_id]
        ]
        values.append(self.residual_value[seq_id])
        return torch.cat(keys), torch.cat(values)

    def reset(self, seq_id: int) -> None:
        """Drop all cached state for ``seq_id``."""
        self.chunks.pop(seq_id, None)
        self.residual_key.pop(seq_id, None)
        self.residual_value.pop(seq_id, None)
        self.num_tokens.pop(seq_id, None)

    def storage_nbytes(self, seq_id: int) -> int:
        """Return the bytes of the quantized representation of ``seq_id``.

        Counts int8 codes, fp32 scales/zero-points and the full-precision
        residual, i.e. everything needed to reconstruct the cache.
        """
        if seq_id not in self.num_tokens:
            raise KeyError(f"Unknown sequence id: {seq_id}")
        total = sum(chunk.nbytes() for chunk in self.chunks[seq_id])
        for residual in (self.residual_key[seq_id], self.residual_value[seq_id]):
            total += residual.numel() * residual.element_size()
        return total


def quantized_paged_attention(
    query: torch.Tensor,
    cache: QuantizedKVCache,
    seq_id: int,
    *,
    causal: bool = True,
) -> torch.Tensor:
    """Run attention over a :class:`QuantizedKVCache` via ``paged_attention``.

    The dequantized dense K/V are wrapped as a single physical block
    (``block_table=[0]``), so the gather, causal masking and softmax logic of
    :func:`paged_attention` is reused verbatim; this demonstrates that the
    quantized cache stays numerically compatible with the paged code path.

    Args:
        query: Shape ``(q_len, num_heads, head_dim)``.
        cache: The quantized cache to attend over.
        seq_id: Logical sequence id inside ``cache``.
        causal: If True, each query may attend only to preceding cached tokens.

    Returns:
        Output tensor of shape ``(q_len, num_heads, head_dim)``.

    """
    key, value = cache.get(seq_id)
    num_tokens = key.size(0)
    return paged_attention(
        query,
        key.unsqueeze(0),
        value.unsqueeze(0),
        block_table=[0] if num_tokens else [],
        num_tokens=num_tokens,
        block_size=max(num_tokens, 1),
        causal=causal,
    )
