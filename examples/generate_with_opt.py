"""Generate text with a real model through LLMInfra's two decode loops.

Loads the local OPT-125m checkpoint (Hugging Face format) and drives it with:

1. ``llminfra.generate`` — the naive loop, re-running the full forward pass
   over the whole sequence at every step;
2. ``llminfra.generate_with_cache`` — the KV-cache loop, which prefills the
   prompt once and then feeds only the new token at every step
   (``past_key_values`` is threaded through as the opaque cache state).

Both loops share the same sampling/processing semantics, so their greedy
outputs must be identical; the KV-cache path is dramatically cheaper per
step. Run from the repository root:

    python examples/generate_with_opt.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

# The package is not pip-installed in this repo; make the script runnable
# from a source checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llminfra import generate, generate_with_cache

MODEL_PATH = Path("/Users/robin/hfhub/models/facebook/opt-125m")
PROMPT = "The future of artificial intelligence is"
MAX_NEW_TOKENS = 32


def load_opt():
    """Load the local OPT-125m checkpoint and tokenizer (offline)."""
    from transformers import GPT2Tokenizer, OPTForCausalLM

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    model = OPTForCausalLM.from_pretrained(MODEL_PATH)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def run_naive(model, prompt_ids: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Decode greedily through ``llminfra.generate`` (full recompute)."""

    def logits_fn(ids: torch.Tensor) -> torch.Tensor:
        return model(ids).logits[:, -1]

    start = time.perf_counter()
    out = generate(logits_fn, prompt_ids, max_new_tokens=MAX_NEW_TOKENS)
    return out, time.perf_counter() - start


@torch.no_grad()
def run_cached(model, prompt_ids: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Decode greedily through ``llminfra.generate_with_cache``."""

    def step_fn(tokens: torch.Tensor, past: tuple | None) -> tuple[torch.Tensor, tuple]:
        outputs = model(tokens, past_key_values=past, use_cache=True)
        return outputs.logits[:, -1], outputs.past_key_values

    start = time.perf_counter()
    out = generate_with_cache(step_fn, prompt_ids, max_new_tokens=MAX_NEW_TOKENS)
    return out, time.perf_counter() - start


def main() -> None:
    """Run both decode loops, compare outputs and report timings."""
    model, tokenizer = load_opt()
    prompt_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    print(f"Prompt: {PROMPT!r} ({prompt_ids.size(1)} tokens)\n")

    naive_out, naive_time = run_naive(model, prompt_ids)
    cached_out, cached_time = run_cached(model, prompt_ids)

    naive_text = tokenizer.decode(naive_out[0], skip_special_tokens=True)
    cached_text = tokenizer.decode(cached_out[0], skip_special_tokens=True)

    # Greedy decoding is deterministic, so the two loops must agree exactly.
    assert torch.equal(naive_out, cached_out), (naive_text, cached_text)
    print(f"[naive]    {naive_time:.3f}s  {naive_text!r}")
    print(f"[kv-cache] {cached_time:.3f}s  {cached_text!r}")
    print(
        f"\nOutputs identical: True; speedup: {naive_time / cached_time:.1f}x "
        f"(CPU, {MAX_NEW_TOKENS} new tokens)"
    )


if __name__ == "__main__":
    main()
