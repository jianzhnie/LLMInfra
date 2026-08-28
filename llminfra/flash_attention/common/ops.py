"""Numerical primitives shared by all educational FlashAttention versions.

Everything here operates either on a single (query block, key/value block)
pair or on the per-row online-softmax state carried between blocks. Scores
and softmax statistics are computed in float32 regardless of the input
dtype; fully masked rows are defined to contribute zeros and never produce
NaNs.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .masking import build_block_mask
from .types import ForwardResult


def scaled_scores(q_block: torch.Tensor, k_block: torch.Tensor) -> torch.Tensor:
    """Scaled dot-product scores ``Q K^T / sqrt(d)``, computed in float32.

    Inputs are upcast to float32 *before* the matmul so the score block stays
    accurate for 16-bit inputs, mirroring the fp32-accumulation behavior of
    real GPU kernels.
    """
    scale = 1.0 / math.sqrt(q_block.shape[-1])
    q_fp32 = q_block.to(torch.float32) * scale
    # ``matmul`` on the trailing two dims is equivalent to
    # ``einsum("bhid,bhjd->bhij")`` but skips einsum's equation parsing.
    return torch.matmul(q_fp32, k_block.to(torch.float32).transpose(-1, -2))


def block_scores_and_mask(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    q_slice: slice,
    k_slice: slice,
    causal: bool,
    key_padding_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Score block for one (query block, key block) pair plus its validity mask."""
    scores = scaled_scores(q[:, :, q_slice, :], k[:, :, k_slice, :])
    mask = build_block_mask(
        batch_size=q.shape[0],
        q_slice=q_slice,
        k_slice=k_slice,
        q_len=q.shape[2],
        kv_len=k.shape[2],
        causal=causal,
        key_padding_mask=key_padding_mask,
        device=q.device,
    )
    return scores, mask


def compute_block_softmax(
    scores: torch.Tensor,
    valid_mask: torch.Tensor | None,
    v_block: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unnormalized softmax statistics of a single score block.

    Returns ``(block_max, block_sum, weighted_values)``: the row-wise maximum,
    the row-wise sum of ``exp(scores - block_max)``, and the unnormalized
    ``exp(scores - block_max) @ V`` contribution. Fully masked rows yield a
    ``block_max`` of 0 and zero contributions, so later merges stay finite.
    """
    if valid_mask is not None:
        scores = scores.masked_fill(~valid_mask, float("-inf"))

    block_max = scores.max(dim=-1, keepdim=True).values
    if valid_mask is not None:
        # A fully masked row has max -inf; substitute 0 so exp() yields 0 for
        # that row instead of NaN. Masked positions are exp(-inf - m) = 0
        # either way. Without a mask every row is finite, so this guard can
        # be skipped entirely.
        block_max = torch.where(
            torch.isfinite(block_max), block_max, torch.zeros_like(block_max)
        )
    exp_scores = torch.exp(scores - block_max)

    block_sum = exp_scores.sum(dim=-1, keepdim=True)
    weighted_values = torch.matmul(exp_scores.to(v_block.dtype), v_block)
    return block_max, block_sum, weighted_values


def merge_normalized_block(
    out: torch.Tensor,
    normalizer: torch.Tensor,
    row_max: torch.Tensor,
    block_max: torch.Tensor,
    block_sum: torch.Tensor,
    weighted_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fold one block's contribution into a running *normalized* output block.

    FA1-style merge: the output block is renormalized at every step, so it
    always holds the exact attention output over the keys seen so far.
    ``out`` is a float32 accumulator; ``weighted_values`` may be lower
    precision and is upcast on accumulation.
    """
    has_contribution = block_sum > 0
    new_row_max = torch.where(
        has_contribution, torch.maximum(row_max, block_max), row_max
    )
    # Fully masked blocks (``block_sum == 0``) carry a substitute block max of
    # 0 that must not raise the running row max: for a row whose true scores
    # are all very negative, ``exp(row_max - 0)`` would underflow to zero (or
    # produce inf ratios in the subnormal range) and wipe out the accumulated
    # state. The where() overrides force the identity update for such rows, so
    # the discarded exp() results (inf/nan) never leak into the state.
    old_scale = torch.where(
        has_contribution,
        torch.exp(row_max - new_row_max),
        torch.ones_like(row_max),
    )
    new_scale = torch.where(
        has_contribution,
        torch.exp(block_max - new_row_max),
        torch.zeros_like(block_max),
    )
    new_normalizer = old_scale * normalizer + new_scale * block_sum

    safe_normalizer = torch.where(
        new_normalizer > 0, new_normalizer, torch.ones_like(new_normalizer)
    )
    out = (old_scale * normalizer / safe_normalizer) * out + (
        new_scale / safe_normalizer
    ) * weighted_values
    out = torch.where(new_normalizer > 0, out, torch.zeros_like(out))
    return out, new_normalizer, new_row_max


def merge_unnormalized_block(
    out_acc: torch.Tensor,
    normalizer: torch.Tensor,
    row_max: torch.Tensor,
    block_max: torch.Tensor,
    block_sum: torch.Tensor,
    weighted_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fold one block's contribution into the running *unnormalized* state.

    FA2-style merge: the accumulator stays scaled by ``exp(-row_max)`` and is
    only divided by the normalizer in `finalize_output`, once all blocks have
    been merged. ``out_acc`` is a float32 accumulator; ``weighted_values``
    may be lower precision and is upcast on accumulation.
    """
    has_contribution = block_sum > 0
    new_row_max = torch.where(
        has_contribution, torch.maximum(row_max, block_max), row_max
    )
    # Fully masked blocks (``block_sum == 0``) carry a substitute block max of
    # 0 that must not raise the running row max: for a row whose true scores
    # are all very negative, ``exp(row_max - 0)`` would underflow to zero (or
    # produce inf ratios in the subnormal range) and wipe out the accumulated
    # state. The where() overrides force the identity update for such rows, so
    # the discarded exp() results (inf/nan) never leak into the state.
    old_scale = torch.where(
        has_contribution,
        torch.exp(row_max - new_row_max),
        torch.ones_like(row_max),
    )
    new_scale = torch.where(
        has_contribution,
        torch.exp(block_max - new_row_max),
        torch.zeros_like(block_max),
    )
    out_acc = old_scale * out_acc + new_scale * weighted_values
    normalizer = old_scale * normalizer + new_scale * block_sum
    return out_acc, normalizer, new_row_max


def finalize_output(
    out_acc: torch.Tensor,
    normalizers: torch.Tensor,
) -> torch.Tensor:
    """Apply the deferred softmax division to an unnormalized accumulator."""
    safe_normalizers = torch.where(
        normalizers > 0, normalizers, torch.ones_like(normalizers)
    )
    out = out_acc / safe_normalizers.to(out_acc.dtype)
    return torch.where(normalizers > 0, out, torch.zeros_like(out))


def lse_from_state(normalizers: torch.Tensor, row_max: torch.Tensor) -> torch.Tensor:
    """Log-sum-exp of the masked scores from the final online-softmax state.

    Fully masked rows (normalizer 0) get an LSE of 0 by convention.
    """
    safe_normalizers = torch.where(
        normalizers > 0, normalizers, torch.ones_like(normalizers)
    )
    lse = row_max + torch.log(safe_normalizers)
    return torch.where(normalizers > 0, lse, torch.zeros_like(lse))


def assemble_forward_result(
    out_blocks: list[torch.Tensor],
    normalizer_blocks: list[torch.Tensor],
    row_max_blocks: list[torch.Tensor],
    *,
    normalized: bool,
    out_dtype: torch.dtype,
    saved_state: dict[str, Any],
) -> ForwardResult:
    """Concatenate per-block state and build the `ForwardResult`.

    With ``normalized=False`` the deferred softmax division (FA2-style) is
    applied here. Accumulators are float32; the output is cast back to
    ``out_dtype`` so it matches the input precision.
    """
    out_acc = torch.cat(out_blocks, dim=2)
    normalizers = torch.cat(normalizer_blocks, dim=2)
    row_max = torch.cat(row_max_blocks, dim=2)
    out = out_acc if normalized else finalize_output(out_acc, normalizers)
    return ForwardResult(
        out=out.to(out_dtype),
        lse=lse_from_state(normalizers, row_max),
        normalizers=normalizers,
        row_max=row_max,
        saved_state=saved_state,
    )


def probabilities_from_lse(
    scores: torch.Tensor,
    valid_mask: torch.Tensor | None,
    lse_block: torch.Tensor,
) -> torch.Tensor:
    """Rebuild a block's softmax probabilities from saved log-sum-exp values."""
    if valid_mask is not None:
        scores = scores.masked_fill(~valid_mask, float("-inf"))
    return torch.exp(scores - lse_block)


def compute_block_gradients(
    *,
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    v_block: torch.Tensor,
    out_block: torch.Tensor,
    grad_out_block: torch.Tensor,
    scores: torch.Tensor,
    valid_mask: torch.Tensor | None,
    lse_block: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact attention gradients for one (query block, key/value block) pair.

    Probabilities are recomputed from the saved log-sum-exp instead of a
    stored attention matrix, then the standard attention derivatives apply:

    .. code-block:: text

        P          = exp(S - LSE)           recomputed probabilities
        grad_v     = P^T @ grad_out
        grad_probs = grad_out @ V^T
        row_dot    = rowsum(grad_out * O)   saved-output trick
        grad_s     = P * (grad_probs - row_dot)
        grad_q     = grad_s @ K / sqrt(d)
        grad_k     = grad_s^T @ Q / sqrt(d)

    Returns ``(grad_q, grad_k, grad_v)`` for the block.
    """
    probabilities = probabilities_from_lse(scores, valid_mask, lse_block).to(
        grad_out_block.dtype
    )
    # Each ``matmul`` below is the einsum in the formula above with the
    # transpose expressed as a strided view, which avoids einsum overhead.
    grad_v = torch.matmul(probabilities.transpose(-1, -2), grad_out_block)
    grad_probs = torch.matmul(grad_out_block, v_block.transpose(-1, -2))
    row_dot = torch.sum(grad_out_block * out_block, dim=-1, keepdim=True)
    grad_s = probabilities * (grad_probs - row_dot)
    scale = 1.0 / math.sqrt(q_block.shape[-1])
    grad_q = scale * torch.matmul(grad_s, k_block)
    grad_k = scale * torch.matmul(grad_s.transpose(-1, -2), q_block)
    return grad_q, grad_k, grad_v
