"""Tests for logits processors, beam search and generate() integration."""

import math

import pytest
import torch

from llminfra import CausalLMModel
from llminfra.generation import (
    LogitsProcessorList,
    MinPLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TopKLogitsProcessor,
    TopPLogitsProcessor,
    beam_search,
)


def test_top_k_keeps_exactly_k_tokens():
    processor = TopKLogitsProcessor(k=2)
    logits = torch.tensor([[1.0, 4.0, 2.0, 3.0]])
    out = processor(logits)
    assert torch.isfinite(out).sum().item() == 2
    assert out[0, 1] == 4.0 and out[0, 3] == 3.0
    assert torch.isneginf(out[0, 0]) and torch.isneginf(out[0, 2])
    # The input logits are not mutated.
    assert torch.isfinite(logits).all()


def test_top_k_larger_than_vocab_is_identity():
    logits = torch.randn(2, 8)
    out = TopKLogitsProcessor(k=16)(logits)
    assert torch.equal(out, logits)


def test_top_k_rejects_invalid_k():
    with pytest.raises(ValueError, match="k"):
        TopKLogitsProcessor(0)


def test_top_p_keeps_minimal_prefix_crossing_p():
    # Powers of two keep the cumulative sums exact in floating point.
    probs = torch.tensor([[0.5, 0.25, 0.125, 0.125]], dtype=torch.float64)
    logits = torch.log(probs)
    out = TopPLogitsProcessor(0.8)(logits)
    # Cumulative mass: 0.5, 0.75, 0.875, 1.0. The token reaching 0.875 is the
    # first to cross 0.8 and is kept; only the 0.125 tail token is removed.
    kept = torch.isfinite(out[0]).nonzero(as_tuple=True)[0].tolist()
    assert kept == [0, 1, 2]
    assert torch.isneginf(out[0, 3])


def test_top_p_boundary_keeps_token_at_exact_p():
    probs = torch.tensor([[0.5, 0.25, 0.125, 0.125]], dtype=torch.float64)
    logits = torch.log(probs)
    out = TopPLogitsProcessor(0.5)(logits)
    # 0.5 is not > 0.5, so removal starts at the token reaching 0.75; the
    # shift keeps that token, dropping only the two 0.125 tail tokens.
    kept = torch.isfinite(out[0]).nonzero(as_tuple=True)[0].tolist()
    assert kept == [0, 1]


def test_top_p_dominant_token_always_keeps_at_least_one():
    logits = torch.log(torch.tensor([[0.75, 0.25]], dtype=torch.float64))
    out = TopPLogitsProcessor(0.5)(logits)
    kept = torch.isfinite(out[0]).nonzero(as_tuple=True)[0].tolist()
    assert kept == [0]


def test_top_p_rejects_invalid_p():
    with pytest.raises(ValueError, match="p"):
        TopPLogitsProcessor(0.0)
    with pytest.raises(ValueError, match="p"):
        TopPLogitsProcessor(1.5)


def test_min_p_threshold_scales_with_max_prob():
    probs = torch.tensor([[0.5, 0.25, 0.125, 0.125]], dtype=torch.float64)
    logits = torch.log(probs)
    # min_p = 0.4: threshold = 0.4 * 0.5 = 0.2 -> drops the two 0.125 tokens.
    out = MinPLogitsProcessor(0.4)(logits)
    kept = torch.isfinite(out[0]).nonzero(as_tuple=True)[0].tolist()
    assert kept == [0, 1]
    # min_p = 0.9: threshold = 0.45 -> only the max-probability token stays.
    out = MinPLogitsProcessor(0.9)(logits)
    kept = torch.isfinite(out[0]).nonzero(as_tuple=True)[0].tolist()
    assert kept == [0]


def test_min_p_one_keeps_only_argmax():
    logits = torch.tensor([[1.0, 3.0, 3.0, 0.0]])
    out = MinPLogitsProcessor(1.0)(logits)
    kept = torch.isfinite(out[0]).nonzero(as_tuple=True)[0].tolist()
    assert kept == [1, 2]


def test_min_p_rejects_invalid_min_p():
    with pytest.raises(ValueError, match="min_p"):
        MinPLogitsProcessor(0.0)
    with pytest.raises(ValueError, match="min_p"):
        MinPLogitsProcessor(1.1)


def test_repetition_penalty_positive_and_negative_logits():
    processor = RepetitionPenaltyLogitsProcessor(2.0)
    logits = torch.tensor([[2.0, -2.0, 1.0, 0.0]])
    input_ids = torch.tensor([[0, 1, 3]])
    out = processor(logits, input_ids)
    assert out[0, 0] == 1.0  # positive logit is divided by the penalty
    assert out[0, 1] == -4.0  # negative logit is multiplied by the penalty
    assert out[0, 2] == 1.0  # unseen token is untouched
    assert out[0, 3] == 0.0  # zero logit stays zero in both directions


def test_repetition_penalty_requires_input_ids():
    processor = RepetitionPenaltyLogitsProcessor(1.5)
    with pytest.raises(ValueError, match="input_ids"):
        processor(torch.zeros(1, 4))


def test_repetition_penalty_rejects_invalid_penalty():
    with pytest.raises(ValueError, match="penalty"):
        RepetitionPenaltyLogitsProcessor(0.0)


def test_logits_processor_list_applies_in_order():
    logits = torch.tensor([[4.0, 3.0]])
    input_ids = torch.tensor([[0]])
    # Penalty first: token 0 drops to 1.0, so top-1 keeps token 1.
    penalty_first = LogitsProcessorList(
        [RepetitionPenaltyLogitsProcessor(4.0), TopKLogitsProcessor(1)]
    )
    out = penalty_first(logits, input_ids)
    assert torch.isneginf(out[0, 0]) and out[0, 1] == 3.0
    # Top-k first: only token 0 survives filtering, then it is penalized.
    topk_first = LogitsProcessorList(
        [TopKLogitsProcessor(1), RepetitionPenaltyLogitsProcessor(4.0)]
    )
    out = topk_first(logits, input_ids)
    assert out[0, 0] == 1.0 and torch.isneginf(out[0, 1])


def _toy_beam_logits(sequences: torch.Tensor) -> torch.Tensor:
    """Toy logits where greedy is globally suboptimal.

    From the prompt (token 0): token 1 with p=0.4, token 2 with p=0.6.
    After token 1: token 3 with p=0.9. After token 2: token 4 with p=0.55.
    Greedy picks [2, 4] (0.33); the beam-optimal path is [1, 3] (0.36).
    """
    logits = torch.full((sequences.size(0), 6), -20.0)
    for row, token in enumerate(sequences[:, -1].tolist()):
        if token == 0:
            logits[row, 1] = math.log(0.4)
            logits[row, 2] = math.log(0.6)
        elif token == 1:
            logits[row, 3] = math.log(0.9)
            logits[row, 4] = math.log(0.1)
        else:
            logits[row, 4] = math.log(0.55)
            logits[row, 5] = math.log(0.45)
    return logits


def test_beam_search_finds_global_optimum_greedy_misses():
    prompt = torch.tensor([[0]])
    greedy = beam_search(
        _toy_beam_logits,
        prompt,
        num_beams=1,
        max_new_tokens=2,
        length_penalty_alpha=0.0,
    )
    assert greedy.tolist() == [[0, 2, 4]]
    beamed = beam_search(
        _toy_beam_logits,
        prompt,
        num_beams=2,
        max_new_tokens=2,
        length_penalty_alpha=0.0,
    )
    assert beamed.tolist() == [[0, 1, 3]]


def test_beam_search_length_penalty_prefers_better_normalized_hypothesis():
    # Path A: [1, eos] with score log(0.51), length 2.
    # Path B: [2, 3, 3, 3, eos] with score log(0.49), length 5.
    # alpha=0 ranks A first (higher cumulative log-prob); alpha=1 ranks B
    # first (better GNMT-normalized score).
    def logits_fn(sequences: torch.Tensor) -> torch.Tensor:
        logits = torch.full((sequences.size(0), 6), -20.0)
        for row, token in enumerate(sequences[:, -1].tolist()):
            if token == 0:
                logits[row, 1] = math.log(0.51)
                logits[row, 2] = math.log(0.49)
            elif token == 1:
                logits[row, 4] = 1.0  # eos right away
            elif token == 2:
                logits[row, 3] = 1.0
            elif token == 3:
                if sequences.size(1) < 5:
                    logits[row, 3] = 1.0
                else:
                    logits[row, 4] = 1.0  # eos at length 5
        return logits

    prompt = torch.tensor([[0]])
    short_first = beam_search(
        logits_fn,
        prompt,
        num_beams=2,
        max_new_tokens=5,
        length_penalty_alpha=0.0,
        eos_token_id=4,
        pad_token_id=5,
    )
    assert short_first.tolist() == [[0, 1, 4]]
    long_first = beam_search(
        logits_fn,
        prompt,
        num_beams=2,
        max_new_tokens=5,
        length_penalty_alpha=1.0,
        eos_token_id=4,
        pad_token_id=5,
    )
    assert long_first.tolist() == [[0, 2, 3, 3, 3, 4]]


def test_beam_search_eos_heap_and_padding_per_row():
    # Batch row 0 emits eos (token 2) immediately; row 1 emits token 3
    # forever and runs to max_new_tokens.
    def logits_fn(sequences: torch.Tensor) -> torch.Tensor:
        logits = torch.full((sequences.size(0), 5), -20.0)
        for row in range(sequences.size(0)):
            if sequences[row, 0] == 1:
                logits[row, 2] = 1.0
            else:
                logits[row, 3] = 1.0
        return logits

    out = beam_search(
        logits_fn,
        torch.tensor([[1], [0]]),
        num_beams=2,
        max_new_tokens=3,
        length_penalty_alpha=0.0,
        eos_token_id=2,
        pad_token_id=4,
    )
    assert out.shape == (2, 4)
    assert out[0].tolist() == [1, 2, 4, 4]
    assert out[1].tolist() == [0, 3, 3, 3]


def test_beam_search_rejects_invalid_arguments():
    prompt = torch.tensor([[0]])
    with pytest.raises(ValueError, match="num_beams"):
        beam_search(_toy_beam_logits, prompt, num_beams=0, max_new_tokens=2)
    with pytest.raises(ValueError, match="max_new_tokens"):
        beam_search(_toy_beam_logits, prompt, num_beams=2, max_new_tokens=0)
    with pytest.raises(ValueError, match="length_penalty_alpha"):
        beam_search(
            _toy_beam_logits,
            prompt,
            num_beams=2,
            max_new_tokens=2,
            length_penalty_alpha=-1.0,
        )
    with pytest.raises(ValueError, match="batch, seq_len"):
        beam_search(_toy_beam_logits, torch.tensor([0]), num_beams=2, max_new_tokens=2)


def _make_generate_model(max_seq_len: int = 32) -> CausalLMModel:
    torch.manual_seed(0)
    model = CausalLMModel(
        vocab_size=32,
        hidden_size=16,
        num_layers=2,
        num_heads=2,
        intermediate_size=32,
        max_seq_len=max_seq_len,
        attention_name="mha",
    )
    model.eval()
    return model


def test_generate_top_k_one_matches_greedy():
    model = _make_generate_model()
    prompt = torch.randint(0, 32, (2, 4))
    greedy = model.generate(prompt, max_new_tokens=5)
    filtered = model.generate(prompt, max_new_tokens=5, top_k=1)
    assert torch.equal(filtered.sequences, greedy.sequences)


def test_generate_top_p_sampling_stays_in_kept_set():
    model = _make_generate_model()
    # Per-step distribution [0.5, 0.25, 0.125, 0.125, ~0...]: top_p=0.6 keeps
    # exactly tokens {0, 1} (the 0.75-cumulative token crosses 0.6).
    step_logits = torch.full((32,), -60.0)
    step_logits[:4] = torch.log(torch.tensor([0.5, 0.25, 0.125, 0.125]))

    def scripted_forward(input_ids, **kwargs):
        return step_logits[None, None, :].expand(
            input_ids.size(0), input_ids.size(1), 32
        )

    original_forward = model.forward
    model.forward = scripted_forward  # type: ignore[assignment]
    try:
        torch.manual_seed(0)
        prompt = torch.randint(0, 32, (2, 4))
        output = model.generate(prompt, max_new_tokens=20, temperature=1.0, top_p=0.6)
    finally:
        model.forward = original_forward  # type: ignore[assignment]

    generated = output.sequences[:, 4:]
    assert generated.shape == (2, 20)
    flat = set(generated.tolist()[0]) | set(generated.tolist()[1])
    assert flat <= {0, 1}
    # With 40 seeded draws from p = (2/3, 1/3) both tokens must show up.
    assert flat == {0, 1}


def test_generate_beam_search_finds_global_optimum_on_toy_model():
    model = _make_generate_model()

    def scripted_forward(input_ids, **kwargs):
        return _toy_beam_logits(input_ids)[:, None, :].expand(
            input_ids.size(0), input_ids.size(1), 6
        )

    original_forward = model.forward
    model.forward = scripted_forward  # type: ignore[assignment]
    try:
        greedy = model.generate(torch.tensor([[0]]), max_new_tokens=2)
        beamed = model.generate(torch.tensor([[0]]), max_new_tokens=2, num_beams=2)
    finally:
        model.forward = original_forward  # type: ignore[assignment]

    assert greedy.sequences.tolist() == [[0, 2, 4]]
    assert beamed.sequences.tolist() == [[0, 1, 3]]


def test_generate_beam_search_shape_and_eos_padding():
    model = _make_generate_model()

    def scripted_forward(input_ids, **kwargs):
        # Every row emits eos (token 3) with probability ~1 at every step.
        logits = torch.full((input_ids.size(0), input_ids.size(1), 32), -60.0)
        logits[:, :, 3] = 0.0
        return logits

    original_forward = model.forward
    model.forward = scripted_forward  # type: ignore[assignment]
    try:
        prompt = torch.randint(4, 32, (2, 4))
        output = model.generate(
            prompt, max_new_tokens=5, num_beams=3, eos_token_id=3, pad_token_id=9
        )
    finally:
        model.forward = original_forward  # type: ignore[assignment]

    # All beams emit eos at the first step, so the best hypothesis per row is
    # a single eos token right after the prompt.
    assert output.sequences.shape == (2, 4 + 1)
    assert torch.equal(output.sequences[:, :4], prompt)
    assert output.sequences[:, 4].tolist() == [3, 3]


def test_generate_beam_search_without_eos_runs_to_max_new_tokens():
    model = _make_generate_model()
    prompt = torch.randint(0, 32, (2, 4))
    output = model.generate(prompt, max_new_tokens=4, num_beams=2)
    assert output.sequences.shape == (2, 4 + 4)
    assert torch.equal(output.sequences[:, :4], prompt)


def test_generate_beam_search_top_k_one_collapses_to_greedy():
    model = _make_generate_model()
    prompt = torch.randint(0, 32, (2, 4))
    greedy = model.generate(prompt, max_new_tokens=5)
    beamed = model.generate(prompt, max_new_tokens=5, num_beams=2, top_k=1)
    assert torch.equal(beamed.sequences, greedy.sequences)


def test_generate_rejects_invalid_decoding_arguments():
    model = _make_generate_model()
    prompt = torch.randint(0, 32, (2, 4))
    with pytest.raises(ValueError, match="top_k"):
        model.generate(prompt, top_k=-1)
    with pytest.raises(ValueError, match="top_p"):
        model.generate(prompt, top_p=0.0)
    with pytest.raises(ValueError, match="top_p"):
        model.generate(prompt, top_p=1.5)
    with pytest.raises(ValueError, match="min_p"):
        model.generate(prompt, min_p=-0.1)
    with pytest.raises(ValueError, match="min_p"):
        model.generate(prompt, min_p=1.5)
    with pytest.raises(ValueError, match="repetition_penalty"):
        model.generate(prompt, repetition_penalty=0.0)
    with pytest.raises(ValueError, match="num_beams"):
        model.generate(prompt, num_beams=0)
    # Beam search is mutually exclusive with temperature sampling.
    with pytest.raises(ValueError, match="temperature"):
        model.generate(prompt, num_beams=2, temperature=0.7)
    # Beam search cannot be combined with speculative decoding.
    with pytest.raises(ValueError, match="draft_model"):
        model.generate(prompt, num_beams=2, draft_model=model)
    # Sampling filters are not applied by the speculative decoder.
    with pytest.raises(ValueError, match="draft_model"):
        model.generate(prompt, top_k=5, draft_model=model)
    with pytest.raises(ValueError, match="draft_model"):
        model.generate(prompt, repetition_penalty=1.1, draft_model=model)
