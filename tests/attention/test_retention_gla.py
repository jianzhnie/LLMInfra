"""Tests for Retention (RetNet) and Gated Linear Attention (GLA).

Covers output shapes, gradient flow, constructor validation, chunked-vs-
recurrent numerical agreement, causal masking, padding semantics and the
paper-level reference formulas:

- RetNet (arXiv:2307.08621): parallel form ``(Q K^T ⊙ D) V`` with
  ``D[n, m] = gamma_h ** (n - m)``, ``gamma_h = 1 - 2 ** (-5 - h)``.
- GLA (arXiv:2312.06635): recurrence ``S = S ⊙ g + k^T v`` with a
  data-dependent per-feature sigmoid gate. The gate multiplies the old
  state directly, so ``g = 1`` means no decay (ungated linear attention)
  and ``g = 0`` resets the state at every step — unlike the interpolating
  ``(1 - g) S + g · k^T v`` update of GatedDeltaNet, where the extremes are
  swapped.
"""

import math

import pytest
import torch
from helpers import make_hidden_state

from llminfra.attention.gated_linear_attention import GatedLinearAttention
from llminfra.attention.retention import Retention

HIDDEN = 64
HEADS = 4
BATCH = 2
SEQ = 7


# ---------------------------------------------------------------------------
# Retention (RetNet)
# ---------------------------------------------------------------------------


@pytest.fixture()
def retention():
    return Retention(HIDDEN, HEADS, feature_dim=16, chunk_size=4, dropout=0.0)


def test_retention_shape_and_gradient(retention):
    x = make_hidden_state(BATCH, SEQ, HIDDEN).requires_grad_(True)
    out = retention(x)
    assert out.shape == (BATCH, SEQ, HIDDEN)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_retention_recurrent_shape_and_gradient(retention):
    x = make_hidden_state(BATCH, SEQ, HIDDEN).requires_grad_(True)
    out = retention.recurrent_forward(x)
    assert out.shape == (BATCH, SEQ, HIDDEN)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_retention_empty_sequence(retention):
    x = make_hidden_state(BATCH, 0, HIDDEN)
    assert retention(x).shape == (BATCH, 0, HIDDEN)
    assert retention.recurrent_forward(x).shape == (BATCH, 0, HIDDEN)


def test_retention_does_not_return_weights(retention):
    with pytest.raises(ValueError, match="does not materialize"):
        retention(make_hidden_state(BATCH, SEQ, HIDDEN), return_attention_weights=True)
    with pytest.raises(ValueError, match="does not materialize"):
        retention.recurrent_forward(
            make_hidden_state(BATCH, SEQ, HIDDEN), return_attention_weights=True
        )


def test_retention_constructor_validation():
    with pytest.raises(ValueError, match="divisible"):
        Retention(30, HEADS)
    with pytest.raises(ValueError, match="feature_dim"):
        Retention(HIDDEN, HEADS, feature_dim=0)
    with pytest.raises(ValueError, match="chunk_size"):
        Retention(HIDDEN, HEADS, chunk_size=0)


def test_retention_decay_schedule_matches_paper():
    """Head h (1-indexed) decays by gamma_h = 1 - 2**(-5 - h), paper Eq. 5."""
    module = Retention(HIDDEN, HEADS)
    expected = torch.tensor(
        [1.0 - 2.0 ** (-5.0 - h) for h in range(1, HEADS + 1)],
        dtype=torch.float64,
    )
    torch.testing.assert_close(module.head_decay.double(), expected)


@pytest.mark.parametrize("seq_len", [1, 5, 16, 17])
@pytest.mark.parametrize("masked", [False, True])
def test_retention_chunked_matches_recurrent(seq_len, masked):
    """The chunked parallel form must equal the step-by-step recurrence."""
    module = Retention(HIDDEN, HEADS, feature_dim=16, chunk_size=8).double().eval()
    x = make_hidden_state(BATCH, seq_len, HIDDEN).double()
    mask = None
    if masked:
        mask = torch.ones(BATCH, 1, seq_len, dtype=torch.bool)
        mask[0, 0, -2:] = False  # trailing padding
        mask[1, 0, 0] = False  # leading padding: masked steps must not decay
    torch.testing.assert_close(
        module(x, attention_mask=mask),
        module.recurrent_forward(x, attention_mask=mask),
        rtol=1e-6,
        atol=1e-8,
    )


def _retention_paper_reference(x, num_heads, head_dim):
    """Naive parallel retention straight from paper Eq. 3-5.

    Assumes identity q/k/v/o projections: head h reads the slice
    ``x[..., h*head_dim:(h+1)*head_dim]``. The ``1/sqrt(head_dim)`` query
    scaling is the module's documented teaching addition.
    """
    seq_len = x.size(1)
    position = torch.arange(seq_len)
    distance = (position[:, None] - position[None, :]).clamp_min(0).to(x.dtype)
    outputs = []
    for h in range(num_heads):
        head_slice = x[..., h * head_dim : (h + 1) * head_dim]
        query = head_slice / math.sqrt(head_dim)
        gamma = 1.0 - 2.0 ** (-5.0 - (h + 1))
        decay_mask = torch.tril(gamma**distance)
        scores = query @ head_slice.transpose(-1, -2)
        outputs.append((scores * decay_mask) @ head_slice)
    return torch.cat(outputs, dim=-1)


def test_retention_matches_paper_parallel_reference():
    """Identity projections: both paths must equal the paper's formula."""
    module = Retention(HIDDEN, HEADS, bias=False).double().eval()
    with torch.no_grad():
        for proj in (module.q_proj, module.k_proj, module.v_proj, module.o_proj):
            proj.weight.copy_(torch.eye(HIDDEN, dtype=torch.float64))
    x = make_hidden_state(BATCH, SEQ, HIDDEN).double()
    reference = _retention_paper_reference(x, HEADS, HIDDEN // HEADS)
    torch.testing.assert_close(module(x), reference, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(
        module.recurrent_forward(x), reference, rtol=1e-6, atol=1e-8
    )


def test_retention_causal_does_not_see_future(retention):
    """Perturbing the last token must not change earlier outputs."""
    retention = retention.eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    perturbed = x.clone()
    perturbed[:, -1] += 10.0

    torch.testing.assert_close(retention(x)[:, :-1], retention(perturbed)[:, :-1])
    torch.testing.assert_close(
        retention.recurrent_forward(x)[:, :-1],
        retention.recurrent_forward(perturbed)[:, :-1],
    )


def test_retention_padding_does_not_update_state():
    """Outputs at valid positions must not depend on padded keys."""
    module = Retention(HIDDEN, HEADS, feature_dim=16, chunk_size=4).double().eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN).double()
    valid_len = SEQ - 2
    mask = torch.ones(BATCH, 1, SEQ, dtype=torch.bool)
    mask[:, :, valid_len:] = False

    torch.testing.assert_close(
        module(x, attention_mask=mask)[:, :valid_len],
        module(x[:, :valid_len]),
        rtol=1e-6,
        atol=1e-8,
    )


def test_retention_accepts_combined_causal_mask(retention):
    """A dense causal mask must reduce to its (empty) key-padding component."""
    retention = retention.eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    causal_mask = torch.tril(torch.ones(BATCH, 1, SEQ, SEQ, dtype=torch.bool))
    torch.testing.assert_close(retention(x, attention_mask=causal_mask), retention(x))


# ---------------------------------------------------------------------------
# Gated Linear Attention (GLA)
# ---------------------------------------------------------------------------


@pytest.fixture()
def gla():
    return GatedLinearAttention(HIDDEN, HEADS, feature_dim=16, chunk_size=4)


def test_gla_shape_and_gradient(gla):
    x = make_hidden_state(BATCH, SEQ, HIDDEN).requires_grad_(True)
    out = gla(x)
    assert out.shape == (BATCH, SEQ, HIDDEN)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_gla_low_rank_gate_shape_and_gradient():
    layer = GatedLinearAttention(HIDDEN, HEADS, feature_dim=8, gate_rank=4)
    x = make_hidden_state(BATCH, SEQ, HIDDEN).requires_grad_(True)
    out = layer(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_gla_empty_sequence(gla):
    x = make_hidden_state(BATCH, 0, HIDDEN)
    assert gla(x).shape == (BATCH, 0, HIDDEN)
    assert gla.recurrent_forward(x).shape == (BATCH, 0, HIDDEN)


def test_gla_does_not_return_weights(gla):
    with pytest.raises(ValueError, match="does not materialize"):
        gla(make_hidden_state(BATCH, SEQ, HIDDEN), return_attention_weights=True)
    with pytest.raises(ValueError, match="does not materialize"):
        gla.recurrent_forward(
            make_hidden_state(BATCH, SEQ, HIDDEN), return_attention_weights=True
        )


def test_gla_constructor_validation():
    with pytest.raises(ValueError, match="divisible"):
        GatedLinearAttention(30, HEADS)
    with pytest.raises(ValueError, match="feature_dim"):
        GatedLinearAttention(HIDDEN, HEADS, feature_dim=0)
    with pytest.raises(ValueError, match="gate_rank"):
        GatedLinearAttention(HIDDEN, HEADS, gate_rank=0)
    with pytest.raises(ValueError, match="chunk_size"):
        GatedLinearAttention(HIDDEN, HEADS, chunk_size=0)


@pytest.mark.parametrize("seq_len", [1, 5, 16, 17])
@pytest.mark.parametrize("gate_rank", [None, 4])
@pytest.mark.parametrize("masked", [False, True])
def test_gla_chunked_matches_recurrent(seq_len, gate_rank, masked):
    """The chunked parallel form must equal the step-by-step recurrence."""
    module = (
        GatedLinearAttention(
            HIDDEN, HEADS, feature_dim=16, gate_rank=gate_rank, chunk_size=8
        )
        .double()
        .eval()
    )
    x = make_hidden_state(BATCH, seq_len, HIDDEN).double()
    mask = None
    if masked:
        mask = torch.ones(BATCH, 1, seq_len, dtype=torch.bool)
        mask[0, 0, -2:] = False  # trailing padding
        mask[1, 0, 0] = False  # leading padding: masked steps must not decay
    torch.testing.assert_close(
        module(x, attention_mask=mask),
        module.recurrent_forward(x, attention_mask=mask),
        rtol=1e-6,
        atol=1e-8,
    )


def _gla_step_reference(module, x, attention_mask=None):
    """Slow reference for the GLA recurrence, written from the paper update.

    ``S_t = S_{t-1} ⊙ g_t + k_t^T v_t``, ``o_t = q_t S_t`` with
    ``g_t = sigmoid(gate projection)``; masked steps write nothing and keep
    the state (gate forced to 1).
    """
    query = module._split(module.q_proj(x)) * module.scale
    key = module._split(module.k_proj(x))
    value = module.split_head(module.v_proj(x))
    if module.g_proj is not None:
        gate_logits = module.g_proj(x)
    else:
        gate_logits = module.g_up_proj(module.g_down_proj(x))
    gate = torch.sigmoid(module._split(gate_logits))

    mask = module._key_padding_mask(attention_mask, x.size(0))
    if mask is not None:
        key = key * mask.unsqueeze(-1)
        value = value * mask.unsqueeze(-1)
        gate = torch.where(mask.unsqueeze(-1), gate, torch.ones_like(gate))

    state = torch.zeros(
        x.size(0), module.num_heads, module.feature_dim, module.head_dim, dtype=x.dtype
    )
    outputs = []
    for step in range(x.size(1)):
        state = state * gate[:, :, step].unsqueeze(-1) + (
            key[:, :, step].unsqueeze(-1) * value[:, :, step].unsqueeze(-2)
        )
        outputs.append(torch.matmul(query[:, :, step].unsqueeze(-2), state).squeeze(-2))
    return module.o_proj(module.combine_head(torch.stack(outputs, dim=2)))


@pytest.mark.parametrize("masked", [False, True])
def test_gla_matches_step_reference(masked):
    """Both paths must equal the handwritten per-step paper reference."""
    module = GatedLinearAttention(HIDDEN, HEADS, feature_dim=16).double().eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN).double()
    mask = None
    if masked:
        mask = torch.ones(BATCH, 1, SEQ, dtype=torch.bool)
        mask[0, 0, -2:] = False
    reference = _gla_step_reference(module, x, mask)
    torch.testing.assert_close(
        module(x, attention_mask=mask), reference, rtol=1e-6, atol=1e-8
    )
    torch.testing.assert_close(
        module.recurrent_forward(x, attention_mask=mask),
        reference,
        rtol=1e-6,
        atol=1e-8,
    )


def _force_gate(module, logit):
    """Make the gate input-independent by pinning it to ``sigmoid(logit)``."""
    with torch.no_grad():
        module.g_proj.weight.zero_()
        module.g_proj.bias.fill_(logit)


def test_gla_gate_one_is_ungated_linear_attention():
    """Gate = 1 multiplies the state by 1: no decay, plain cumulative sum.

    This is the exact behavior difference from GatedDeltaNet's interpolating
    update, where gate = 1 would instead *replace* the state.
    """
    module = GatedLinearAttention(HIDDEN, HEADS, feature_dim=16).double().eval()
    _force_gate(module, 50.0)  # sigmoid(50) == 1.0 exactly in float64
    x = make_hidden_state(BATCH, SEQ, HIDDEN).double()

    query = module._split(module.q_proj(x)) * module.scale
    key = module._split(module.k_proj(x))
    value = module.split_head(module.v_proj(x))
    kv_state = torch.einsum("bhsf,bhsd->bhsfd", key, value).cumsum(dim=2)
    reference = module.o_proj(
        module.combine_head(torch.einsum("bhsf,bhsfd->bhsd", query, kv_state))
    )

    torch.testing.assert_close(module(x), reference, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(
        module.recurrent_forward(x), reference, rtol=1e-6, atol=1e-8
    )


def test_gla_gate_zero_resets_state_every_step():
    """Gate = 0 erases the state before each write: only the current token
    contributes, ``o_t = (q_t · k_t) v_t``."""
    module = GatedLinearAttention(HIDDEN, HEADS, feature_dim=16).double().eval()
    _force_gate(module, -50.0)  # sigmoid(-50) ~ 2e-22, negligible residual
    x = make_hidden_state(BATCH, SEQ, HIDDEN).double()

    query = module._split(module.q_proj(x)) * module.scale
    key = module._split(module.k_proj(x))
    value = module.split_head(module.v_proj(x))
    current_only = (query * key).sum(dim=-1, keepdim=True) * value
    reference = module.o_proj(module.combine_head(current_only))

    torch.testing.assert_close(module(x), reference, rtol=1e-6, atol=1e-10)
    torch.testing.assert_close(
        module.recurrent_forward(x), reference, rtol=1e-6, atol=1e-10
    )


def test_gla_causal_does_not_see_future(gla):
    """Perturbing the last token must not change earlier outputs."""
    gla = gla.eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    perturbed = x.clone()
    perturbed[:, -1] += 10.0

    torch.testing.assert_close(gla(x)[:, :-1], gla(perturbed)[:, :-1])
    torch.testing.assert_close(
        gla.recurrent_forward(x)[:, :-1], gla.recurrent_forward(perturbed)[:, :-1]
    )


def test_gla_padding_does_not_update_state():
    """Outputs at valid positions must not depend on padded keys."""
    module = (
        GatedLinearAttention(HIDDEN, HEADS, feature_dim=16, chunk_size=4)
        .double()
        .eval()
    )
    x = make_hidden_state(BATCH, SEQ, HIDDEN).double()
    valid_len = SEQ - 2
    mask = torch.ones(BATCH, 1, SEQ, dtype=torch.bool)
    mask[:, :, valid_len:] = False

    torch.testing.assert_close(
        module(x, attention_mask=mask)[:, :valid_len],
        module(x[:, :valid_len]),
        rtol=1e-6,
        atol=1e-8,
    )


def test_gla_accepts_combined_causal_mask(gla):
    """A dense causal mask must reduce to its (empty) key-padding component."""
    gla = gla.eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    causal_mask = torch.tril(torch.ones(BATCH, 1, SEQ, SEQ, dtype=torch.bool))
    torch.testing.assert_close(gla(x, attention_mask=causal_mask), gla(x))
