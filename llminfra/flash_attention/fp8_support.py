"""Guardrails for scripts that exercise the educational FlashAttention versions.

Centralizes the FP8 support boundary so command-line entry points (forward
runners, backward checkers, benchmarks) reject unsupported flag combinations
with a clear message instead of failing deep inside a kernel.
"""

from __future__ import annotations


def validate_fp8_support(
    *,
    version: str,
    fp8: bool,
    script_name: str,
    benchmark_type: str = "flash",
) -> None:
    """Validate that an FP8 request stays within FA3's support boundary.

    The official FA3 release supports FP8 on the forward path only, and this
    educational repo mirrors that boundary.

    Args:
        version: Selected FlashAttention version (``fa1``..``fa4``).
        fp8: Whether FP8 mode was requested.
        script_name: Entry point making the request (e.g. ``"flash_attention"``,
            ``"check_backward"``, ``"bench"``).
        benchmark_type: Benchmark mode; FP8 only applies to the ``"flash"`` path.

    Raises:
        ValueError: If the FP8 request is not supported for the given
            version / script / benchmark combination.

    """
    if not fp8:
        return
    if version != "fa3":
        raise ValueError(
            f"--fp8 is only implemented for --version fa3 (got --version {version})"
        )
    if script_name == "check_backward":
        raise ValueError(
            "FA3 FP8 backward is unsupported; run check_backward in fp16/bf16 instead"
        )
    if script_name == "bench" and benchmark_type != "flash":
        raise ValueError(
            "--fp8 only applies to the FA3 flash path; use --benchmark-type flash"
        )
