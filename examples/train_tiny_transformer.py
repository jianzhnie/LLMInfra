"""Train a tiny ``CausalLMModel`` on a synthetic pattern, then generate with it.

Teaching-grade end-to-end example built only from llminfra's public APIs:

1. Build a small ``CausalLMModel`` (embedding + transformer blocks + LM head).
2. Train it for a few hundred steps on a trivially learnable sequence rule:
   each token is the previous token plus a fixed step, modulo the vocabulary.
3. Print the loss as it drops and assert it decreased meaningfully.
4. Generate a few tokens greedily and check the model absorbed the rule.

Runs on CPU in a few seconds. Run from the repository root:

    python examples/train_tiny_transformer.py
"""

from __future__ import annotations

import sys
from itertools import pairwise
from pathlib import Path

import torch

# The package is not pip-installed in this repo; make the script runnable
# directly from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llminfra.models import CausalLMModel

VOCAB_SIZE = 24
RULE_STEP = 5  # token[t + 1] = (token[t] + RULE_STEP) % VOCAB_SIZE
SEQ_LEN = 16
BATCH_SIZE = 64
TRAIN_STEPS = 300
LEARNING_RATE = 3e-3


def make_batch(
    batch_size: int, seq_len: int, generator: torch.Generator
) -> torch.Tensor:
    """Sample sequences following the modular-arithmetic rule.

    Only the first token of each row is random; every later token is
    determined by the rule, so a model that learns it can drive the
    next-token cross-entropy to (near) zero.
    """
    ids = torch.empty(batch_size, seq_len, dtype=torch.long)
    ids[:, 0] = torch.randint(0, VOCAB_SIZE, (batch_size,), generator=generator)
    for position in range(1, seq_len):
        ids[:, position] = (ids[:, position - 1] + RULE_STEP) % VOCAB_SIZE
    return ids


@torch.no_grad()
def greedy_generate(
    model: CausalLMModel, prompt: torch.Tensor, num_new_tokens: int
) -> torch.Tensor:
    """Append ``num_new_tokens`` greedy (argmax) tokens to ``prompt``.

    This is a reference implementation: every step recomputes the full
    forward pass over the grown sequence because these teaching models wire
    no KV cache.
    """
    model.eval()
    ids = prompt
    for _ in range(num_new_tokens):
        logits = model(ids)
        assert isinstance(logits, torch.Tensor)
        ids = torch.cat([ids, logits[:, -1].argmax(dim=-1, keepdim=True)], dim=1)
    return ids


def main() -> None:
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)

    model = CausalLMModel(
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        intermediate_size=128,
        max_seq_len=SEQ_LEN + 8,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training on the rule: next token = (current + {RULE_STEP}) % {VOCAB_SIZE}")
    losses: list[float] = []
    model.train()
    for step in range(1, TRAIN_STEPS + 1):
        ids = make_batch(BATCH_SIZE, SEQ_LEN, generator)
        # Passing labels makes the model compute the shifted next-token
        # cross-entropy internally.
        output = model(ids, labels=ids)
        assert output.loss is not None
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        optimizer.step()
        losses.append(output.loss.item())
        if step == 1 or step % 25 == 0:
            print(f"step {step:4d}  loss {output.loss.item():.4f}")

    first = sum(losses[:25]) / 25
    last = sum(losses[-25:]) / 25
    # A uniform model scores ln(VOCAB_SIZE) ~= 3.18; the rule is
    # deterministic, so training should drive the loss close to zero.
    print(f"mean loss: first 25 steps {first:.4f} -> last 25 steps {last:.4f}")
    assert last < 0.25 * first, (
        f"loss did not decrease enough: {first:.4f} -> {last:.4f}"
    )

    prompt = torch.tensor([[3]])
    generated = greedy_generate(model, prompt, num_new_tokens=8)
    tokens = generated[0].tolist()
    print(f"prompt {prompt[0].tolist()} -> generated {tokens}")
    for previous, current in pairwise(tokens):
        assert current == (previous + RULE_STEP) % VOCAB_SIZE, (
            f"model broke the rule: {previous} -> {current}"
        )
    print("OK: loss decreased and greedy generation follows the rule.")


if __name__ == "__main__":
    main()
