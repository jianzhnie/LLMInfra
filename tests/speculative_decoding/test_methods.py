"""Tests for speculative decoding strategies and draft heads."""

import pytest
import torch

from llminfra import (
    BlockDiffusionDrafter,
    DFlashDecoder,
    DSparkDecoder,
    DSparkScheduler,
    Eagle1Speculator,
    Eagle2Speculator,
    Eagle3Speculator,
    EagleSpeculator,
    MedusaHead,
    MTPDecoder,
    MultiTokenPredictionHead,
    NGramSpeculator,
    SpeculativeDecoder,
    dflash_loss,
    medusa_loss,
    mtp_loss,
)

HIDDEN_SIZE = 32


def _constant_model(vocab_size: int = 32):
    def model(input_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(input_ids.size(0), input_ids.size(1), vocab_size)

    return model


def _fixed_token_model(token_id: int, vocab_size: int = 16):
    def model(input_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(input_ids.size(0), input_ids.size(1), vocab_size)
        logits[..., token_id] = 1.0
        return logits

    return model


def _scripted_model(table: dict, default_token: int, vocab_size: int = 8):
    """Deterministic model with a per-row script of argmax predictions.

    Rows are identified by their first token; ``table`` maps that token to a
    cyclic list of predicted tokens per position.
    """

    def model(input_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(input_ids.size(0), input_ids.size(1), vocab_size)
        for row_index in range(input_ids.size(0)):
            script = table.get(int(input_ids[row_index, 0]))
            for position in range(input_ids.size(1)):
                token = (
                    default_token if script is None else script[position % len(script)]
                )
                logits[row_index, position, token] = 1.0
        return logits

    return model


def _row_by_row(decoder: SpeculativeDecoder, input_ids: torch.Tensor) -> torch.Tensor:
    """Reference path: decode each row independently (original behavior)."""
    rows = [decoder(row[None])[0] for row in input_ids]
    output = input_ids.new_full(
        (len(rows), max(row.numel() for row in rows)), decoder.pad_token_id
    )
    for index, row in enumerate(rows):
        output[index, : row.numel()] = row
    return output


def test_speculative_decoder_accepts_deterministic_tokens() -> None:
    model = _fixed_token_model(1)
    decoder = SpeculativeDecoder(model, model, num_speculative_tokens=3)
    output = decoder(torch.zeros(2, 4, dtype=torch.long))
    assert output.shape == (2, 7)
    assert (output[:, 4:] == 1).all()


def test_batched_draft_matches_row_by_row() -> None:
    """Batched drafting must be token-for-token identical to per-row decoding."""
    input_ids = torch.zeros(2, 4, dtype=torch.long)
    cases = [
        # Every draft accepted, bonus token appended.
        (_fixed_token_model(1), _fixed_token_model(1), True, 6),
        # Every draft rejected at the first step (correction appended).
        (_fixed_token_model(1), _fixed_token_model(2), True, 0),
    ]
    for draft_model, target_model, append_bonus, expected_accepted in cases:
        decoder = SpeculativeDecoder(
            draft_model,
            target_model,
            num_speculative_tokens=3,
            append_bonus_token=append_bonus,
        )
        batched = decoder(input_ids)
        assert decoder.last_num_accepted == expected_accepted
        assert torch.equal(batched, _row_by_row(decoder, input_ids))


def test_batched_draft_with_varying_acceptance_matches_row_by_row() -> None:
    """Rows accept different lengths (full, partial, none) with padding."""
    draft_model = _scripted_model({}, default_token=5)
    target_model = _scripted_model(
        {
            1: [5],  # accepts all drafts plus the bonus token
            2: [0, 0, 5, 5, 2],  # accepts two drafts, then a correction
            3: [0, 0, 7],  # rejects the first draft
        },
        default_token=5,
    )
    draft_calls = 0

    def counting_draft(input_ids: torch.Tensor) -> torch.Tensor:
        nonlocal draft_calls
        draft_calls += 1
        return draft_model(input_ids)

    decoder = SpeculativeDecoder(
        counting_draft,
        target_model,
        num_speculative_tokens=3,
        append_bonus_token=True,
    )
    input_ids = torch.tensor([[1, 0, 0], [2, 0, 0], [3, 0, 0]])
    batched = decoder(input_ids)
    assert decoder.last_num_accepted == 3 + 2 + 0
    # Batched drafting issues one draft call per speculative step,
    # not one per (row, step) pair.
    assert draft_calls == 3
    expected = torch.tensor(
        [
            [1, 0, 0, 5, 5, 5, 5],
            [2, 0, 0, 5, 5, 2, 0],
            [3, 0, 0, 7, 0, 0, 0],
        ]
    )
    assert torch.equal(batched, expected)
    assert torch.equal(batched, _row_by_row(decoder, input_ids))


def test_eagle_speculator_and_versioned_interfaces() -> None:
    target_model = _fixed_token_model(2)
    input_ids = torch.zeros(2, 4, dtype=torch.long)
    hidden_states = torch.randn(2, 4, HIDDEN_SIZE)

    for speculator_type in (
        EagleSpeculator,
        Eagle1Speculator,
        Eagle2Speculator,
        Eagle3Speculator,
    ):
        output = speculator_type(target_model, target_model, num_speculative_tokens=3)(
            input_ids, hidden_states
        )
        assert output.shape == (2, 7)
        assert (output[:, 4:] == 2).all()


def test_dspark_uses_dynamic_scheduler() -> None:
    model = _fixed_token_model(1, vocab_size=8)
    input_ids = torch.zeros(1, 3, dtype=torch.long)
    decoder = DSparkDecoder(model, model, DSparkScheduler((2,)))
    assert decoder(input_ids).size(1) >= input_ids.size(1)


def test_ngram_speculator_copies_prompt_continuation() -> None:
    prompt = torch.tensor([[1, 2, 1, 2, 0]])
    output = NGramSpeculator(
        _fixed_token_model(3, vocab_size=8),
        ngram_size=2,
        num_speculative_tokens=1,
    )(prompt)
    assert output.shape[0] == 1


def test_ngram_speculator_drafts_observed_continuation() -> None:
    def model(input_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(input_ids.size(0), input_ids.size(1), 8)
        # Accept the expected continuation at the four verified positions.
        for offset, token in enumerate((3, 1, 2, 2)):
            logits[:, 4 + offset, token] = 1.0
        return logits

    prompt = torch.tensor([[1, 2, 3, 1, 2]])
    output = NGramSpeculator(
        model,
        ngram_size=2,
        num_speculative_tokens=4,
        append_bonus_token=False,
    )(prompt)
    # The matched [1, 2] at index 0 is followed by [3, 1, 2]; the draft
    # copies that continuation and pads with its last token to length 4.
    assert output.tolist() == [[1, 2, 3, 1, 2, 3, 1, 2, 2]]


def test_multi_token_prediction_returns_multiple_logits() -> None:
    head = MultiTokenPredictionHead(HIDDEN_SIZE, vocab_size=64, num_predictions=3)
    logits = head(torch.randn(2, 7, HIDDEN_SIZE))
    assert len(logits) == 3
    assert all(item.shape == (2, 7, 64) for item in logits)


def test_medusa_head_candidates_loss_and_gradient() -> None:
    head = MedusaHead(HIDDEN_SIZE, vocab_size=64, num_heads=3)
    hidden = torch.randn(2, 7, HIDDEN_SIZE, requires_grad=True)
    labels = torch.randint(0, 64, (2, 7))
    logits = head(hidden)
    assert logits.shape == (2, 7, 3, 64)
    candidate_ids, candidate_scores = head.generate_candidates(hidden, top_k=4)
    assert candidate_ids.shape == candidate_scores.shape == (2, 3, 4)
    loss = medusa_loss(head, hidden, labels, weight_decay=0.8)
    loss.backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def test_speculative_decoder_validates_arguments() -> None:
    model = _constant_model()
    with pytest.raises(ValueError, match=">= 1"):
        SpeculativeDecoder(model, model, num_speculative_tokens=0)
    with pytest.raises(ValueError, match=">= 0"):
        SpeculativeDecoder(model, model, temperature=-0.5)


def test_speculative_decoder_accepts_short_prompt() -> None:
    model = _constant_model()
    output = SpeculativeDecoder(model, model, num_speculative_tokens=4)(
        torch.zeros(1, 2, dtype=torch.long)
    )
    assert output.size(1) >= 3


def test_residual_sampling_uses_probability_difference() -> None:
    model = _constant_model(vocab_size=4)
    decoder = SpeculativeDecoder(model, model, temperature=1.0)
    draft_logits = torch.tensor([[20.0, 0.0, 0.0, 0.0]])
    target_logits = torch.tensor([[0.0, 0.0, 20.0, 0.0]])
    samples = torch.cat(
        [decoder._sample_residual(draft_logits, target_logits) for _ in range(20)]
    )
    assert (samples == 2).all()


def test_eagle_speculator_validates_arguments() -> None:
    model = _constant_model()
    with pytest.raises(ValueError, match=">= 1"):
        EagleSpeculator(model, model, num_speculative_tokens=0)


def test_eagle_speculator_validates_hidden_states() -> None:
    model = _constant_model()
    speculator = EagleSpeculator(model, model, num_speculative_tokens=3)
    input_ids = torch.zeros(2, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="num_speculative_tokens"):
        speculator(input_ids, torch.randn(2, 2, HIDDEN_SIZE))
    with pytest.raises(ValueError, match="batch size"):
        speculator(input_ids, torch.randn(3, 4, HIDDEN_SIZE))


def test_mtp_decoder_validates_hidden_state() -> None:
    model = _constant_model()
    head = MultiTokenPredictionHead(HIDDEN_SIZE, vocab_size=32, num_predictions=2)
    decoder = MTPDecoder(head, model)
    input_ids = torch.zeros(2, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="batch size"):
        decoder(input_ids, torch.randn(3, 4, HIDDEN_SIZE))
    with pytest.raises(ValueError, match="batch, seq"):
        decoder(torch.zeros(4, dtype=torch.long), torch.randn(2, 4, HIDDEN_SIZE))


def test_mtp_loss_reuses_precomputed_logits(monkeypatch) -> None:
    head = MultiTokenPredictionHead(HIDDEN_SIZE, vocab_size=32, num_predictions=2)
    hidden = torch.randn(2, 7, HIDDEN_SIZE)
    labels = torch.randint(0, 32, (2, 7))
    expected = mtp_loss(head, hidden, labels)
    logits_list = head(hidden)
    monkeypatch.setattr(
        head,
        "forward",
        lambda *args, **kwargs: pytest.fail("head must not run again"),
    )
    actual = mtp_loss(head, hidden, labels, logits_list=logits_list)
    torch.testing.assert_close(actual, expected)


def test_mtp_loss_scores_each_head_against_its_shifted_labels() -> None:
    """Head k must be scored against labels[:, k + 1:] (off-by-one guard)."""
    head = MultiTokenPredictionHead(HIDDEN_SIZE, vocab_size=32, num_predictions=2)
    hidden = torch.randn(2, 7, HIDDEN_SIZE)
    labels = torch.randint(0, 32, (2, 7))
    logits_list = head(hidden)

    per_step = [
        torch.nn.functional.cross_entropy(
            logits_list[k][:, : -(k + 1)].reshape(-1, 32),
            labels[:, k + 1 :].reshape(-1),
        )
        for k in range(2)
    ]
    expected = torch.stack(per_step).mean()
    torch.testing.assert_close(mtp_loss(head, hidden, labels), expected)


def test_medusa_loss_matches_aligned_weighted_cross_entropy() -> None:
    """Head h predicts labels[:, h + 1:]; steps are weighted by decay**h."""
    head = MedusaHead(HIDDEN_SIZE, vocab_size=32, num_heads=3)
    hidden = torch.randn(2, 7, HIDDEN_SIZE)
    labels = torch.randint(0, 32, (2, 7))
    logits = head(hidden)

    total, weight_sum = 0.0, 0.0
    for h in range(3):
        offset = h + 1
        weight = 0.8**h
        total = total + weight * torch.nn.functional.cross_entropy(
            logits[:, :-offset, h].reshape(-1, 32), labels[:, offset:].reshape(-1)
        )
        weight_sum += weight
    expected = total / weight_sum
    torch.testing.assert_close(
        medusa_loss(head, hidden, labels, weight_decay=0.8), expected
    )

    # Branches whose targets are all -100 contribute no weight.
    masked_labels = labels.clone()
    masked_labels[:, 2:] = -100
    expected_head0 = torch.nn.functional.cross_entropy(
        logits[:, :-1, 0].reshape(-1, 32),
        masked_labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    torch.testing.assert_close(
        medusa_loss(head, hidden, masked_labels, weight_decay=0.8), expected_head0
    )


class _StubDrafter:
    """Drafter stub that always predicts a fixed token per draft slot."""

    def __init__(self, tokens, block_size=4, mask_token_id=0, vocab_size=8):
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.tokens = list(tokens)
        self.vocab_size = vocab_size

    def __call__(self, block, target_features, context_mask=None):
        logits = torch.zeros(block.size(0), self.block_size, self.vocab_size)
        for slot, token in enumerate(self.tokens):
            logits[:, slot + 1, token] = 1.0
        return logits


def _drafter(block_size=4, hidden_size=16, vocab_size=8):
    return BlockDiffusionDrafter(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=2,
        num_heads=2,
        block_size=block_size,
        mask_token_id=0,
    )


def test_dflash_drafter_output_shape_and_finite():
    drafter = _drafter()
    block = torch.zeros(3, 4, dtype=torch.long)
    features = torch.randn(3, 9, 16)
    logits = drafter(block, features)
    assert logits.shape == (3, 4, 8)
    assert torch.isfinite(logits).all()


def test_dflash_drafter_shares_embedding_and_head():
    embedding = torch.nn.Embedding(8, 16)
    head = torch.nn.Linear(16, 8, bias=False)
    drafter = BlockDiffusionDrafter(
        vocab_size=8,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        block_size=4,
        token_embedding=embedding,
        lm_head=head,
    )
    assert drafter.token_embedding is embedding
    assert drafter.lm_head is head


def test_dflash_drafter_kv_injection_changes_output():
    drafter = _drafter()
    block = torch.zeros(1, 4, dtype=torch.long)
    features_a = torch.randn(1, 6, 16)
    features_b = torch.randn(1, 6, 16)
    assert not torch.equal(drafter(block, features_a), drafter(block, features_b))


def test_dflash_drafter_context_mask_blocks_future_features():
    drafter = _drafter()
    block = torch.zeros(1, 4, dtype=torch.long)
    features = torch.randn(1, 6, 16)
    anchor = 2
    context_mask = torch.arange(6)[None, :] <= anchor
    perturbed = features.clone()
    perturbed[:, anchor + 1 :] += 100.0
    torch.testing.assert_close(
        drafter(block, features, context_mask),
        drafter(block, perturbed, context_mask),
    )


def test_dflash_drafter_validates_shapes():
    drafter = _drafter()
    with pytest.raises(ValueError, match=r"input_ids must have shape"):
        drafter(torch.zeros(1, 5, dtype=torch.long), torch.randn(1, 6, 16))
    with pytest.raises(ValueError, match="target_features"):
        drafter(torch.zeros(1, 4, dtype=torch.long), torch.randn(2, 6, 16))
    with pytest.raises(ValueError, match="context_mask"):
        drafter(
            torch.zeros(1, 4, dtype=torch.long),
            torch.randn(1, 6, 16),
            torch.ones(1, 5, dtype=torch.bool),
        )


def test_dflash_decoder_greedy_is_lossless():
    target = _fixed_token_model(3, vocab_size=8)
    decoder = DFlashDecoder(_drafter(), target, append_bonus_token=True)
    input_ids = torch.tensor([[1, 2, 2]])
    output = decoder(input_ids, torch.randn(1, 3, 16))
    # Whatever the drafter proposed, every emitted token is the target's
    # argmax: greedy speculative decoding is lossless by construction.
    assert (output[:, 3:] == 3).all()
    assert output.size(1) - 3 >= 1


def test_dflash_decoder_fully_accepted_block_appends_bonus():
    script = [5, 6, 7]
    target = _scripted_model({1: [4, *script, 2]}, default_token=0, vocab_size=8)
    drafter = _StubDrafter(script, block_size=4, vocab_size=8)
    decoder = DFlashDecoder(drafter, target, append_bonus_token=True)
    input_ids = torch.tensor([[1, 2]])
    output = decoder(input_ids, torch.randn(1, 2, 16))
    # All 3 drafts accepted, then the bonus token: input + 5,6,7 + bonus.
    assert output.size(1) == 2 + 4
    assert decoder.last_num_drafted == 3
    assert decoder.last_num_accepted == 3
    # Emitted tokens match the target's own greedy continuation.
    expected = [5, 6, 7, 2]  # script then bonus = target argmax after them
    assert output[0, 2:].tolist() == expected


def test_dflash_decoder_rejected_draft_is_corrected():
    target = _fixed_token_model(3, vocab_size=8)
    drafter = _StubDrafter([5, 6, 7], block_size=4, vocab_size=8)
    decoder = DFlashDecoder(drafter, target, append_bonus_token=True)
    input_ids = torch.tensor([[1, 2]])
    output = decoder(input_ids, torch.randn(1, 2, 16))
    # First draft (5) mismatches the target argmax (3): only the corrected
    # token is emitted and no bonus is appended.
    assert output[0].tolist() == [1, 2, 3]
    assert decoder.last_num_accepted == 0


def test_dflash_decoder_pads_shorter_rows():
    table = {1: [3, 3, 3, 3], 2: [4, 4, 4, 4]}
    target = _scripted_model(table, default_token=0, vocab_size=8)
    tokens = [[3, 3, 3], [0, 0, 0]]  # row 0 fully matches, row 1 mismatches
    logits = torch.zeros(2, 4, 8)
    for row, row_tokens in enumerate(tokens):
        for slot, token in enumerate(row_tokens):
            logits[row, slot + 1, token] = 1.0

    class _BatchStub:
        block_size = 4
        mask_token_id = 0

        def __call__(self, block, target_features, context_mask=None):
            return logits

    decoder = DFlashDecoder(_BatchStub(), target, append_bonus_token=True)
    input_ids = torch.tensor([[1, 9], [2, 9]])
    output = decoder(input_ids, torch.randn(2, 2, 16))
    assert output.size(1) == 2 + 4  # longest row: 3 accepted + bonus
    assert output[0, 2:].tolist() == [3, 3, 3, 3]
    assert output[1, 2].item() == 4  # corrected token
    assert output[1, 3:].eq(decoder.pad_token_id).all()


def test_dflash_decoder_sampling_mode_runs():
    torch.manual_seed(0)
    target = _fixed_token_model(3, vocab_size=8)
    decoder = DFlashDecoder(_drafter(), target, temperature=1.0)
    output = decoder(torch.tensor([[1, 2]]), torch.randn(1, 2, 16))
    assert output.size(1) >= 3
    assert output.size(1) <= 2 + 4


def test_dflash_decoder_validates_inputs():
    decoder = DFlashDecoder(_drafter(), _constant_model())
    with pytest.raises(ValueError, match="batch, seq"):
        decoder(torch.zeros(4, dtype=torch.long), torch.randn(1, 4, 16))
    with pytest.raises(ValueError, match="target_features"):
        decoder(torch.zeros(1, 4, dtype=torch.long), torch.randn(1, 5, 16))


def test_dflash_loss_matches_hand_computed_weighted_cross_entropy():
    drafter = _drafter(block_size=4, vocab_size=8)
    input_ids = torch.randint(1, 8, (2, 12))
    features = torch.randn(2, 12, 16)

    torch.manual_seed(0)
    loss = dflash_loss(drafter, input_ids, features)

    torch.manual_seed(0)
    anchor = torch.randint(0, 12 - 4 + 1, (2,))
    block = torch.zeros(2, 4, dtype=torch.long)
    block[:, 0] = input_ids[torch.arange(2), anchor]
    context_mask = torch.arange(12)[None, :] <= anchor[:, None]
    logits = drafter(block, features, context_mask)
    labels = input_ids.gather(1, anchor[:, None] + torch.arange(1, 4)[None, :])
    weights = torch.exp(-torch.arange(3, dtype=logits.dtype) / 4.0)
    per_token = torch.nn.functional.cross_entropy(
        logits[:, 1:].reshape(-1, 8), labels.reshape(-1), reduction="none"
    ).view(2, 3)
    expected = (per_token * weights).sum() / (weights.sum() * 2)
    torch.testing.assert_close(loss, expected)


def test_dflash_loss_requires_sequence_longer_than_block():
    drafter = _drafter(block_size=4, vocab_size=8)
    with pytest.raises(ValueError, match="block_size"):
        dflash_loss(drafter, torch.randint(0, 8, (1, 4)), torch.randn(1, 4, 16))


def test_dflash_loss_backward_reaches_draft_layers():
    drafter = _drafter(block_size=4, vocab_size=8)
    loss = dflash_loss(drafter, torch.randint(1, 8, (2, 10)), torch.randn(2, 10, 16))
    loss.backward()
    assert drafter.layers[0].q_proj.weight.grad is not None
    assert drafter.layers[0].k_context.weight.grad is not None
