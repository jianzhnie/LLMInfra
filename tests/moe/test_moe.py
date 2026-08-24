"""Tests for the mixture-of-experts modules.

Covers ExpertFFN, the top-k router, MixtureOfExperts, DeepSeekMoE, LatentMoE,
ExpertParallelMoE and the load-balancing auxiliary loss.
"""

import pytest
import torch
from helpers import make_hidden_state

from llminfra import (
    DeepSeekMoE,
    ExpertFFN,
    ExpertParallelMoE,
    LatentMoE,
    MixtureOfExperts,
    TopKRouter,
    load_balance_loss,
)
from llminfra.moe.mixture_of_experts import ExpertChoiceRouter, router_z_loss

HIDDEN = 32
HEADS = 4
SEQ = 8
BATCH = 2


def test_expert_ffn_shape_and_gradient():
    expert = ExpertFFN(hidden_size=16, intermediate_size=32)
    x = torch.randn(3, 5, 16, requires_grad=True)
    out = expert(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_topk_router_weights_sum_to_one():
    router = TopKRouter(hidden_size=16, num_experts=8, top_k=2)
    x = torch.randn(4, 16)
    weights, indices = router(x)
    assert weights.shape == (4, 2)
    assert indices.shape == (4, 2)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(4))
    assert indices.max().item() < 8


def test_mixture_of_experts_shape_and_gradient():
    moe = MixtureOfExperts(
        hidden_size=16,
        num_experts=8,
        intermediate_size=32,
        top_k=2,
    )
    x = torch.randn(2, 5, 16, requires_grad=True)
    out = moe(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_deepseek_moe_includes_shared_experts():
    moe = DeepSeekMoE(
        hidden_size=16,
        num_routed_experts=8,
        num_shared_experts=2,
        intermediate_size=32,
        top_k=2,
    )
    x = torch.randn(2, 3, 16, requires_grad=True)
    out = moe(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_latent_moe_shape_and_gradient():
    moe = LatentMoE(
        hidden_size=HIDDEN,
        latent_size=16,
        num_experts=4,
        intermediate_size=32,
        top_k=2,
    )
    x = torch.randn(BATCH, SEQ, HIDDEN, requires_grad=True)
    out = moe(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_expert_parallel_moe_shape_and_gradient():
    moe = ExpertParallelMoE(
        hidden_size=HIDDEN,
        num_experts=8,
        intermediate_size=64,
        top_k=2,
        world_size=2,
        rank=0,
    )
    x = torch.randn(BATCH, SEQ, HIDDEN, requires_grad=True)
    out = moe(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_load_balance_loss_is_finite():
    router = TopKRouter(HIDDEN, num_experts=8, top_k=2)
    x = torch.randn(16, HIDDEN)
    logits = router.routing_logits(x)
    _, indices = router(x)
    loss = load_balance_loss(logits, indices, num_experts=8)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_load_balance_loss_matches_hand_computed_value():
    """The loss must equal num_experts * sum(fraction * mean_softmax)."""
    logits = torch.tensor([[2.0, 0.0, -1.0, 0.5], [0.0, 1.0, 0.0, 2.0]])
    indices = torch.tensor([[0, 1], [1, 1]])  # fractions: 1/4, 3/4, 0, 0
    loss = load_balance_loss(logits, indices, num_experts=4)

    probabilities = torch.softmax(logits, dim=-1).mean(dim=0)
    fractions = torch.tensor([0.25, 0.75, 0.0, 0.0])
    expected = 4 * (fractions * probabilities).sum()
    torch.testing.assert_close(loss, expected)

    # Perfectly balanced routing with uniform probabilities gives exactly 1.
    uniform = torch.zeros(8, 4)
    balanced = torch.arange(4).repeat(2).reshape(4, 2).T.contiguous()
    torch.testing.assert_close(
        load_balance_loss(uniform, balanced, num_experts=4), torch.tensor(1.0)
    )


def test_load_balance_loss_empty_tokens():
    """Zero tokens must give a finite zero loss, not NaN."""
    logits = torch.empty(0, 8)
    indices = torch.empty(0, 2, dtype=torch.long)
    loss = load_balance_loss(logits, indices, num_experts=8)
    assert loss.ndim == 0
    assert loss.item() == 0.0
    assert torch.isfinite(loss)


def test_router_noise_epsilon_actually_perturbs_routing():
    """noise_epsilon must scale the noise (adding a constant would be a no-op)."""
    router = TopKRouter(
        HIDDEN, num_experts=8, top_k=2, add_noise=True, noise_epsilon=1.0
    )
    router.train()
    x = make_hidden_state(16, 1, HIDDEN)[:, 0]

    weights_1, indices_1 = router(x)
    weights_2, indices_2 = router(x)

    assert not torch.equal(indices_1, indices_2) or not torch.allclose(
        weights_1, weights_2
    )


def test_sigmoid_scoring_weights_finite_and_normalized():
    router = TopKRouter(HIDDEN, num_experts=8, top_k=2, scoring_func="sigmoid")
    x = torch.randn(16, HIDDEN)
    weights, indices = router(x)
    assert weights.shape == (16, 2)
    assert indices.shape == (16, 2)
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(16))


def test_router_z_loss_finite_scalar_and_backward():
    router = TopKRouter(HIDDEN, num_experts=8, top_k=2)
    x = torch.randn(16, HIDDEN, requires_grad=True)
    logits = router.routing_logits(x)
    loss = router_z_loss(logits)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_aux_free_balance_reduces_expert_count_variance():
    torch.manual_seed(0)
    num_experts = 8
    router = TopKRouter(
        HIDDEN,
        num_experts=num_experts,
        top_k=1,
        aux_free_balance=True,
        balance_update_rate=0.05,
    )
    router.train()
    assert router.router_bias is not None
    assert not router.router_bias.requires_grad

    # Skewed logits: every token strongly favors expert 0.
    logits = torch.randn(256, num_experts) * 0.1
    logits[:, 0] += 2.0
    x = (
        torch.linalg.solve(
            router.router.weight @ router.router.weight.T + torch.eye(num_experts),
            logits.T,
        ).T
        @ router.router.weight
    )

    def counts():
        _, indices = router(x)
        return torch.bincount(indices.flatten(), minlength=num_experts).float()

    initial_variance = counts().var(unbiased=False).item()
    for _ in range(200):
        router(x)
    final_variance = counts().var(unbiased=False).item()

    assert final_variance < initial_variance


def test_aux_free_balance_updates_only_in_training():
    router = TopKRouter(
        HIDDEN,
        num_experts=4,
        top_k=1,
        aux_free_balance=True,
        balance_update_rate=0.1,
    )
    x = torch.randn(8, HIDDEN)
    router.eval()
    router(x)
    torch.testing.assert_close(router.router_bias, torch.zeros(4))
    router.train()
    router(x)
    assert not torch.equal(router.router_bias, torch.zeros(4))


def test_gumbel_weights_ignore_router_bias():
    """Gumbel mode: router_bias steers selection but never the weights."""
    torch.manual_seed(0)
    num_experts = 8
    router = TopKRouter(
        HIDDEN,
        num_experts=num_experts,
        top_k=num_experts,
        routing_strategy="gumbel",
        aux_free_balance=True,
        gumbel_hard=False,
    )
    router.train()
    x = torch.randn(4, HIDDEN)
    # top_k == num_experts: the selection set is all experts either way,
    # so identical gumbel draws must give identical weights if unbiased.
    bias = torch.arange(num_experts, dtype=torch.float32) * 3.0

    torch.manual_seed(1)
    weights_clean, indices_clean = router(x)
    router.router_bias.data = bias
    torch.manual_seed(1)
    weights_biased, indices_biased = router(x)

    weights_clean = weights_clean.gather(-1, indices_clean.argsort(-1))
    weights_biased = weights_biased.gather(-1, indices_biased.argsort(-1))
    torch.testing.assert_close(weights_clean, weights_biased)


def test_gumbel_hard_weights_match_unbiased_scores():
    """Hard gumbel forward weights equal the unbiased top-k scores."""
    torch.manual_seed(0)
    router = TopKRouter(HIDDEN, num_experts=8, top_k=2, routing_strategy="gumbel")
    router.train()
    x = torch.randn(16, HIDDEN, requires_grad=True)
    weights, indices = router(x)
    expected = torch.softmax(router.router(x).gather(-1, indices), dim=-1)
    torch.testing.assert_close(weights, expected)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(16))

    # The straight-through gradient still reaches the router and the input.
    weights.sum().backward()
    assert router.router.weight.grad is not None
    assert torch.isfinite(router.router.weight.grad).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_expert_choice_router_shapes():
    num_experts, top_tokens = 8, 4
    router = ExpertChoiceRouter(
        hidden_size=HIDDEN, num_experts=num_experts, top_tokens=top_tokens
    )
    x = torch.randn(16, HIDDEN)
    weights, token_indices = router(x)
    assert weights.shape == (num_experts, top_tokens)
    assert token_indices.shape == (num_experts, top_tokens)
    assert torch.isfinite(weights).all()
    assert token_indices.max().item() < 16
    assert token_indices.min().item() >= 0


def test_expert_choice_router_rejects_too_few_tokens():
    router = ExpertChoiceRouter(hidden_size=HIDDEN, num_experts=4, top_tokens=8)
    with pytest.raises(ValueError, match="top_tokens"):
        router(torch.randn(4, HIDDEN))


def test_mixture_of_experts_expert_dropout_training_randomness():
    torch.manual_seed(0)
    moe = MixtureOfExperts(
        hidden_size=16,
        num_experts=8,
        intermediate_size=32,
        top_k=2,
        expert_dropout=0.5,
    )
    moe.train()
    x = torch.randn(4, 16)
    out_1 = moe(x)
    out_2 = moe(x)
    assert not torch.allclose(out_1, out_2)

    # Eval mode disables expert dropout: output is deterministic.
    moe.eval()
    torch.testing.assert_close(moe(x), moe(x))
