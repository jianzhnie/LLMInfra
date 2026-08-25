"""Train a DFlash block-diffusion drafter against a small frozen target LM.

Teaching-grade tour of the DFlash training loop (arXiv:2602.06036):

1. Build a small frozen target LM whose per-position hidden states serve as
   the fused ``target_features`` the DFlash APIs consume (extracting and
   fusing multi-layer target features is the caller's job; here a single
   feature tensor keeps the data flow readable).
2. Train a ``BlockDiffusionDrafter`` with ``dflash_loss`` on synthetic
   batches: one random anchor block per sequence, the anchor token clean and
   the following ``block_size - 1`` positions masked and predicted in
   parallel.
3. Run ``DFlashDecoder`` for a few speculative cycles and report
   ``last_num_accepted`` / ``last_num_drafted`` (the acceptance rate), which
   climbs as the drafter learns to imitate the target.

Runs on CPU in a few seconds. Run from the repository root:

    python examples/train_dflash_drafter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

# The package is not pip-installed in this repo; make the script runnable
# directly from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llminfra.spec_decode import (
    BlockDiffusionDrafter,
    DFlashDecoder,
    dflash_loss,
)

VOCAB_SIZE = 16
HIDDEN_SIZE = 32
BLOCK_SIZE = 5  # one clean anchor + 4 drafted tokens per pass
SEQ_LEN = 24
BATCH_SIZE = 16
TRAIN_STEPS = 300
LEARNING_RATE = 3e-3
DECODE_CYCLES = 12


class TinyTargetLM(nn.Module):
    """Frozen stand-in target LM: embedding + MLP + LM head.

    The model is position-independent, so its next-token distribution is a
    deterministic (but nonlinear) function of the current token. That makes
    the imitation task learnable for the drafter while still going through
    the real DFlash APIs.
    """

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def features(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Per-position hidden states, consumed as fused target features."""
        return self.mlp(self.embedding(input_ids))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.features(input_ids))


@torch.no_grad()
def sample_from_target(
    target: TinyTargetLM,
    batch_size: int,
    seq_len: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Roll out greedy sequences from the target model.

    The drafter must learn the *target's* continuation, so the training
    sequences are drawn from the target itself (greedy decoding from random
    start tokens), mirroring how DFlash trains on target-distribution data.
    Uniform random sequences would be unpredictable and the loss would stall
    at ``log(vocab_size)``.
    """
    ids = torch.empty(batch_size, seq_len, dtype=torch.long)
    ids[:, 0] = torch.randint(0, VOCAB_SIZE, (batch_size,), generator=generator)
    for position in range(1, seq_len):
        ids[:, position] = target(ids[:, :position])[:, -1].argmax(dim=-1)
    return ids


def train_drafter(
    drafter: BlockDiffusionDrafter,
    target: TinyTargetLM,
    generator: torch.Generator,
) -> list[float]:
    """Optimize the drafter with ``dflash_loss``; the target stays frozen."""
    optimizer = torch.optim.Adam(drafter.parameters(), lr=LEARNING_RATE)
    losses: list[float] = []
    for step in range(1, TRAIN_STEPS + 1):
        ids = sample_from_target(target, BATCH_SIZE, SEQ_LEN, generator)
        with torch.no_grad():
            target_features = target.features(ids)
        loss = dflash_loss(drafter, ids, target_features)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step == 1 or step % 25 == 0:
            print(f"step {step:4d}  dflash loss {loss.item():.4f}")
    return losses


@torch.no_grad()
def decode_demo(drafter: BlockDiffusionDrafter, target: TinyTargetLM) -> float:
    """Run greedy speculative cycles and return the overall acceptance rate."""
    decoder = DFlashDecoder(drafter, target, append_bonus_token=True)
    prompt = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    ids = prompt
    accepted = drafted = 0
    for cycle in range(1, DECODE_CYCLES + 1):
        ids = decoder(ids, target.features(ids))
        accepted += decoder.last_num_accepted
        drafted += decoder.last_num_drafted
        print(
            f"cycle {cycle:2d}  accepted {decoder.last_num_accepted}/"
            f"{decoder.last_num_drafted} drafted  (seq len {ids.size(1)})"
        )
    return accepted / drafted


def main() -> None:
    torch.manual_seed(0)  # dflash_loss samples anchors from the global RNG
    generator = torch.Generator().manual_seed(0)

    target = TinyTargetLM(VOCAB_SIZE, HIDDEN_SIZE).eval()
    target.requires_grad_(False)
    drafter = BlockDiffusionDrafter(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=2,
        num_heads=2,
        block_size=BLOCK_SIZE,
        mask_token_id=0,
        target_feature_size=HIDDEN_SIZE,
    )

    print(f"Training a DFlash drafter (block_size={BLOCK_SIZE}) against the target")
    losses = train_drafter(drafter, target, generator)
    first = sum(losses[:25]) / 25
    last = sum(losses[-25:]) / 25
    print(f"mean loss: first 25 steps {first:.4f} -> last 25 steps {last:.4f}")
    assert last < 0.5 * first, (
        f"loss did not decrease enough: {first:.4f} -> {last:.4f}"
    )

    print("\nSpeculative decoding with the trained drafter")
    drafter.eval()
    acceptance_rate = decode_demo(drafter, target)
    print(f"acceptance rate over {DECODE_CYCLES} cycles: {acceptance_rate:.2%}")
    assert acceptance_rate > 0.5, f"acceptance rate too low: {acceptance_rate:.2%}"
    print("OK: drafter trained and most drafted tokens are accepted.")


if __name__ == "__main__":
    main()
