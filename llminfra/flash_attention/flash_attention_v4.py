"""FA4: scheduled tiles with explicit main/softmax/correction phases.

Simplified algorithm in this module:
1. Build an explicit tile schedule that groups work into waves, so execution is
   driven by scheduler metadata rather than only by nested loops.
2. Split each wave into three conceptual roles:
   a main score-production phase, a softmax-statistics phase, and a correction
   phase that merges the new contribution into the running output state.
3. Track whether a merge requires rescaling the accumulated state, so the code
   makes the late / conditional rescaling decision visible even though the
   actual tensor path remains mathematically exact and sequential.
4. Reuse the same scheduled-wave view in backward so the version still reads
   like a scheduler-driven algorithm rather than a generic tiled kernel.

Compared with FA3, the educational improvement is the explicit scheduler and
role split. Real FA4 further co-designs the algorithm around Blackwell-era
features such as TMEM, async MMA, multi-role warpgroups, and deeper overlap.
This module does not simulate those hardware details, but it does expose the
main/softmax/correction decomposition and the conditional-rescaling idea that
differentiate FA4 from the earlier versions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import torch

from .common import (
    BackwardResult,
    FlashAttentionConfig,
    ForwardResult,
    TiledAttentionFunction,
    assemble_forward_result,
    block_scores_and_mask,
    block_slices,
    compute_block_gradients,
    compute_block_softmax,
    init_block_state,
    init_gradients,
    merge_unnormalized_block,
    prepare_inputs,
)

__all__ = ["backward", "flash_attention_v4", "forward"]


@dataclass(frozen=True)
class ScheduledTile:
    """One unit of scheduled work: a (query tile, key tile) pair in a wave."""

    wave_id: int
    query_tile: int
    key_tile: int
    q_slice: slice
    k_slice: slice


def _fa4_rescale_threshold(dtype: torch.dtype) -> float:
    """Rescale-skip threshold matching official FA4's dtype-based policy.

    Official FA4 sets ``rescale_threshold=8.0`` for 16-bit query types and
    0.0 otherwise. We mirror that policy here.
    """
    return 8.0 if dtype in (torch.float16, torch.bfloat16) else 0.0


def _build_schedule(
    q_slices: list[slice], k_slices: list[slice]
) -> list[list[ScheduledTile]]:
    """Group all (query tile, key tile) pairs into one wave per query tile.

    The schedule is intentionally explicit so readers can see that FA4 is
    driven by scheduler metadata, not just by plain nested loops. Real FA4
    uses a richer scheduler to map tiles to warpgroups / CTA roles.
    """
    waves = []
    for wave_id, q_slice in enumerate(q_slices):
        waves.append(
            [
                ScheduledTile(
                    wave_id=wave_id,
                    query_tile=wave_id,
                    key_tile=key_tile,
                    q_slice=q_slice,
                    k_slice=k_slice,
                )
                for key_tile, k_slice in enumerate(k_slices)
            ]
        )
    return waves


def _correction_merge(
    *,
    out_acc_block: torch.Tensor,
    normalizer_block: torch.Tensor,
    row_max_block: torch.Tensor,
    block_max: torch.Tensor,
    block_sum: torch.Tensor,
    weighted_values: torch.Tensor,
    scale_log2: float,
    rescale_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """FA4's "correction" role: merge one tile, with conditional rescaling.

    This helper represents the correction role in FA4. In the real kernels
    that role is separated so output rescaling and correction do not block the
    main compute path. Here we keep the separation conceptually, but execute
    it inline in a plain tensor function.

    Official FA4 uses a thresholded selective-rescaling rule: when the row max
    grows by less than ``rescale_threshold`` (in log2 units), the old row max
    is kept and the new contribution is folded in with a relative scale
    instead of rescaling the whole accumulator. Both paths are mathematically
    exact; only the bookkeeping differs.

    Returns the merged ``(out_acc, normalizer, row_max)`` triple plus a
    boolean tensor recording which rows took the full-rescale path.
    """
    has_prior_state = torch.isfinite(row_max_block)
    # Fully masked tiles (block_sum == 0) contribute nothing: the guarded
    # merge below is an identity update for them, so they must not count as
    # "requiring a rescale" in the debug trace either.
    is_empty_block = block_sum <= 0
    requires_rescale = (~has_prior_state | (block_max > row_max_block)) & (
        ~is_empty_block
    )

    merged_out_acc, merged_normalizer, merged_row_max = merge_unnormalized_block(
        out_acc_block,
        normalizer_block,
        row_max_block,
        block_max,
        block_sum,
        weighted_values,
    )

    if rescale_threshold <= 0.0:
        # Without selective rescaling, every row that would keep the old max
        # gets a bitwise-identical result from the full merge (the new row max
        # equals the old one, so old_scale is exactly 1). Return early and
        # skip the relative-scale exp and the three torch.where selects.
        return merged_out_acc, merged_normalizer, merged_row_max, requires_rescale

    safe_row_max_block = torch.where(has_prior_state, row_max_block, block_max)
    acc_scale_log2 = (safe_row_max_block - block_max) * scale_log2
    selective_skip = (
        has_prior_state
        & (block_max > row_max_block)
        & (acc_scale_log2 >= -rescale_threshold)
    )
    requires_rescale = requires_rescale & ~selective_skip

    # Selective-skip path: keep the old row max and fold the new tile in with
    # a relative scale, leaving the accumulated state untouched. Empty tiles
    # are excluded: their ``block_max`` placeholder of 0 could make the
    # relative scale overflow (``exp(0 - row_max)``), and the guarded merged
    # state is an exact identity for them anyway.
    same_scale = ~requires_rescale & ~is_empty_block
    relative_scale = torch.exp(block_max - safe_row_max_block)
    same_scale_out_acc = (
        out_acc_block + relative_scale.to(weighted_values.dtype) * weighted_values
    )
    same_scale_normalizer = normalizer_block + relative_scale * block_sum
    same_scale_row_max = row_max_block

    merged_out_acc = torch.where(same_scale, same_scale_out_acc, merged_out_acc)
    merged_normalizer = torch.where(
        same_scale, same_scale_normalizer, merged_normalizer
    )
    merged_row_max = torch.where(same_scale, same_scale_row_max, merged_row_max)
    return merged_out_acc, merged_normalizer, merged_row_max, requires_rescale


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    key_padding_mask: torch.Tensor | None = None,
    config: FlashAttentionConfig | None = None,
) -> ForwardResult:
    """Run the FA4 scheduler-driven forward pass (main/softmax/correction roles).

    Args:
        q: Queries, shape ``(batch, heads, q_len, head_dim)``.
        k: Keys, shape ``(batch, heads, kv_len, head_dim)``.
        v: Values, shape ``(batch, heads, kv_len, value_dim)``; ``value_dim``
            may differ from ``head_dim``.
        causal: Apply a causal mask. When ``kv_len > q_len`` the diagonal is
            aligned with the end of the key sequence.
        key_padding_mask: Optional mask of shape ``(batch, kv_len)``; ``True``
            marks valid key positions.
        config: Tiling and debug knobs; ``FlashAttentionConfig()`` defaults
            are used when omitted.

    Returns:
        A `ForwardResult` with the attention output, the per-row log-sum-exp,
        and the final online-softmax statistics.

    """
    config = config or FlashAttentionConfig()
    q, k, key_padding_mask = prepare_inputs(
        q,
        k,
        v,
        key_padding_mask=key_padding_mask,
    )

    q_slices = block_slices(q.shape[2], config.block_size_q)
    k_slices = block_slices(k.shape[2], config.block_size_kv)
    scale_log2 = 1.0 / math.log(2.0)
    rescale_threshold = _fa4_rescale_threshold(q.dtype)
    # Each scheduled wave owns one query tile and iterates across the K/V tiles
    # for that query tile. That matches the FA4 forward mental model from the
    # paper/blog more closely: load a Q tile, then loop over K/V blocks.
    #
    # Real FA4 couples the three roles to Blackwell-era hardware features such
    # as TMEM, async MMA, and multi-role warpgroups. We keep the role split and
    # the scheduling metadata, but intentionally do not simulate those
    # primitives.
    out_acc_blocks, normalizer_blocks, row_max_blocks = init_block_state(q, v, q_slices)
    schedule = _build_schedule(q_slices, k_slices)
    scheduler_trace: list[dict[str, Any]] = []

    for wave in schedule:
        # Main role: with one Q tile resident for this wave, step through the
        # K/V tiles and produce the corresponding score tiles. The three role
        # stages are written as generators, so each score block is handed
        # lazily from one role to the next and released right after the
        # correction merge: peak memory stays one (block_q, block_kv) score
        # block instead of the whole (block_q, kv_len) row of a wave. That
        # lazy hand-off also matches real FA4, where the roles run
        # concurrently in different warpgroups rather than phase by phase.
        main_outputs = (
            (
                task,
                *block_scores_and_mask(
                    q=q,
                    k=k,
                    q_slice=task.q_slice,
                    k_slice=task.k_slice,
                    causal=causal,
                    key_padding_mask=key_padding_mask,
                ),
            )
            for task in wave
        )

        # Softmax role: update the per-row statistics and build the local
        # weighted-value contribution from the score tile.
        softmax_outputs = (
            (
                task,
                *compute_block_softmax(
                    scores,
                    valid_mask,
                    v[:, :, task.k_slice, :],
                ),
            )
            for task, scores, valid_mask in main_outputs
        )

        for task, block_max, block_sum, weighted_values in softmax_outputs:
            # Correction role: merge the new tile's contribution into the running
            # output state and record whether a rescale was conceptually needed.
            (
                out_acc_blocks[task.query_tile],
                normalizer_blocks[task.query_tile],
                row_max_blocks[task.query_tile],
                rescaled,
            ) = _correction_merge(
                out_acc_block=out_acc_blocks[task.query_tile],
                normalizer_block=normalizer_blocks[task.query_tile],
                row_max_block=row_max_blocks[task.query_tile],
                block_max=block_max,
                block_sum=block_sum,
                weighted_values=weighted_values,
                scale_log2=scale_log2,
                rescale_threshold=rescale_threshold,
            )
            if config.keep_debug_state:
                scheduler_trace.append(
                    {
                        "wave_id": task.wave_id,
                        "query_tile": task.query_tile,
                        "key_tile": task.key_tile,
                        "rescaled": bool(rescaled.any().item()),
                        "rescale_threshold": rescale_threshold,
                    }
                )

    # The deferred softmax division is applied inside `assemble_forward_result`.
    return assemble_forward_result(
        out_acc_blocks,
        normalizer_blocks,
        row_max_blocks,
        normalized=False,
        out_dtype=q.dtype,
        saved_state={"scheduler_trace": scheduler_trace}
        if config.keep_debug_state
        else {},
    )


def backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
    forward_result: ForwardResult,
    *,
    causal: bool = False,
    key_padding_mask: torch.Tensor | None = None,
    config: FlashAttentionConfig | None = None,
) -> BackwardResult:
    """Run the FA4 scheduler-driven backward pass.

    Follows the same Q-major wave ordering as the forward pass so the
    educational implementation stays aligned with the forward scheduling
    story. The derivative formulas are the exact attention derivatives shared
    by all versions.

    Args:
        q: Queries, as passed to `forward`, shape
            ``(batch, heads, q_len, head_dim)``.
        k: Keys, as passed to `forward`, shape
            ``(batch, heads, kv_len, head_dim)``.
        v: Values, as passed to `forward`, shape
            ``(batch, heads, kv_len, value_dim)``.
        grad_out: Gradient w.r.t. the forward output, same shape as
            ``forward_result.out``.
        forward_result: The result returned by `forward`.
        causal: Must match the forward call.
        key_padding_mask: Must match the forward call.
        config: Must match the forward call.

    Returns:
        A `BackwardResult` with the gradients grad_q, grad_k and grad_v.

    """
    config = config or FlashAttentionConfig()
    q, k, key_padding_mask = prepare_inputs(
        q,
        k,
        v,
        key_padding_mask=key_padding_mask,
    )

    grad_q, grad_k, grad_v = init_gradients(q, k, v)
    q_slices = block_slices(q.shape[2], config.block_size_q)
    k_slices = block_slices(k.shape[2], config.block_size_kv)
    schedule = _build_schedule(q_slices, k_slices)
    scheduler_trace: list[dict[str, Any]] = []

    for wave in schedule:
        # Backward follows the same Q-major wave ordering so the educational
        # implementation stays aligned with the forward scheduling story. As in
        # the forward pass, the generator pipeline keeps only one score block
        # alive at a time instead of materializing the wave's whole row.
        main_outputs = (
            (
                task,
                *block_scores_and_mask(
                    q=q,
                    k=k,
                    q_slice=task.q_slice,
                    k_slice=task.k_slice,
                    causal=causal,
                    key_padding_mask=key_padding_mask,
                ),
            )
            for task in wave
        )

        # The derivative formulas remain exact attention derivatives. The FA4
        # distinction here is the role/schedule decomposition, not new math.
        softmax_outputs = (
            (
                task,
                *compute_block_gradients(
                    q_block=q[:, :, task.q_slice, :],
                    k_block=k[:, :, task.k_slice, :],
                    v_block=v[:, :, task.k_slice, :],
                    out_block=forward_result.out[:, :, task.q_slice, :],
                    grad_out_block=grad_out[:, :, task.q_slice, :],
                    scores=scores,
                    valid_mask=valid_mask,
                    lse_block=forward_result.lse[:, :, task.q_slice, :],
                ),
            )
            for task, scores, valid_mask in main_outputs
        )

        for task, local_grad_q, local_grad_k, local_grad_v in softmax_outputs:
            grad_q[:, :, task.q_slice, :] += local_grad_q
            grad_k[:, :, task.k_slice, :] += local_grad_k
            grad_v[:, :, task.k_slice, :] += local_grad_v
            if config.keep_debug_state:
                scheduler_trace.append(
                    {
                        "wave_id": task.wave_id,
                        "query_tile": task.query_tile,
                        "key_tile": task.key_tile,
                    }
                )

    return BackwardResult(
        grad_q=grad_q.to(q.dtype),
        grad_k=grad_k.to(k.dtype),
        grad_v=grad_v.to(v.dtype),
        debug_state={"scheduler_trace": scheduler_trace}
        if config.keep_debug_state
        else {},
    )


def flash_attention_v4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    key_padding_mask: torch.Tensor | None = None,
    config: FlashAttentionConfig | None = None,
) -> torch.Tensor:
    """Differentiable FA4 attention, callable like any PyTorch operation.

    Unlike calling `forward` directly, the returned tensor records the
    autograd graph: ``loss.backward()`` routes through this module's tiled
    `backward` implementation. Arguments match `forward`.
    """
    # `Function.apply` is untyped in the torch stubs; it returns the forward
    # output tensor, so cast the Any result back.
    return cast(
        torch.Tensor,
        TiledAttentionFunction.apply(  # type: ignore[no-untyped-call]
            q, k, v, causal, key_padding_mask, config, forward, backward
        ),
    )
