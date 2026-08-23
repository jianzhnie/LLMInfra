"""Tests for speculative decoding strategies and draft heads."""

import pytest
import torch

from llminfra import (
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
