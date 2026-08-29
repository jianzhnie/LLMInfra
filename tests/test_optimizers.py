"""Tests for the Muon and MuonClip optimizers."""

import math

import pytest
import torch
from torch import nn

from llminfra.optimizers import Muon, MuonClip, zeropower_via_newtonschulz5


def _reference_newtonschulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Independent float32 Newton-Schulz reference for numerical checks."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.float32)
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


def _reference_muon_step(
    p: torch.Tensor,
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    lr: float,
    momentum: float,
    nesterov: bool,
    weight_decay: float,
) -> torch.Tensor:
    """Hand-computed single Muon step, independent of the optimizer code."""
    buf = momentum_buffer * momentum + grad
    update = grad + momentum * buf if nesterov else buf
    update_2d = update.reshape(update.size(0), -1)
    orthogonal = _reference_newtonschulz(update_2d)
    rows, cols = update_2d.shape
    scale = 0.2 * math.sqrt(max(rows, cols))
    out = p * (1.0 - lr * weight_decay)
    return out - lr * scale * orthogonal.reshape(p.shape)


def test_newtonschulz_orthogonalizes_tall_and_wide_matrices():
    generator = torch.Generator().manual_seed(0)
    for rows, cols in [(8, 16), (16, 8), (12, 12), (32, 64)]:
        G = torch.randn(rows, cols, generator=generator)
        X = zeropower_via_newtonschulz5(G).float()
        # Semi-orthogonal: all singular values cluster around 1 (the bf16
        # iteration only approximates, so allow a generous band).
        singular_values = torch.linalg.svdvals(X)
        assert singular_values.min() > 0.5
        assert singular_values.max() < 1.3
        # The smaller Gram matrix must be close to the identity.
        gram = X @ X.T if rows <= cols else X.T @ X
        eye = torch.eye(min(rows, cols))
        assert torch.allclose(gram, eye, atol=0.5), (rows, cols, gram)


def test_newtonschulz_is_scale_invariant():
    generator = torch.Generator().manual_seed(1)
    G = torch.randn(6, 10, generator=generator)
    small = zeropower_via_newtonschulz5(G).float()
    large = zeropower_via_newtonschulz5(1000.0 * G).float()
    # bf16 numerics: the two runs agree only up to bf16 rounding.
    assert torch.allclose(small, large, atol=0.1)


def test_muon_single_step_matches_hand_computed_reference():
    generator = torch.Generator().manual_seed(2)
    lr, momentum, weight_decay = 0.05, 0.9, 0.01
    p0 = torch.randn(6, 4, generator=generator)
    grad = torch.randn(6, 4, generator=generator)
    initial_buffer = torch.randn(6, 4, generator=generator)

    for nesterov in (False, True):
        p = p0.clone().requires_grad_(True)
        optimizer = Muon(
            [{"params": [p], "use_muon": True}],
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
        )
        optimizer.state[p]["momentum_buffer"] = initial_buffer.clone()
        p.grad = grad.clone()
        optimizer.step()

        expected = _reference_muon_step(
            p0, grad, initial_buffer, lr, momentum, nesterov, weight_decay
        )
        torch.testing.assert_close(p.detach(), expected, atol=2e-3, rtol=2e-2)


def test_non_matrix_params_fall_back_to_adamw():
    generator = torch.Generator().manual_seed(3)
    bias0 = torch.randn(8, generator=generator)
    grad = torch.randn(8, generator=generator)
    lr, betas, eps, weight_decay = 0.01, (0.9, 0.95), 1e-8, 0.01

    bias = bias0.clone().requires_grad_(True)
    # No explicit use_muon: a 1D-only group must default to AdamW.
    optimizer = Muon(
        [bias], lr=lr, adamw_betas=betas, adamw_eps=eps, weight_decay=weight_decay
    )
    assert optimizer.param_groups[0]["use_muon"] is False
    bias.grad = grad.clone()
    optimizer.step()

    assert "momentum_buffer" not in optimizer.state[bias]
    assert "exp_avg" in optimizer.state[bias]

    reference = bias0.clone().requires_grad_(True)
    ref_optimizer = torch.optim.AdamW(
        [reference], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
    )
    reference.grad = grad.clone()
    ref_optimizer.step()
    torch.testing.assert_close(bias.detach(), reference.detach())


def test_from_named_parameters_splits_matrix_and_excluded_params():
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    named = [
        *model.named_parameters(),
        ("embed.weight", nn.Parameter(torch.randn(4, 8))),
    ]
    optimizer = Muon.from_named_parameters(named, lr=0.02, adamw_lr=3e-3)

    muon_group, adamw_group = optimizer.param_groups
    assert muon_group["use_muon"] is True
    assert adamw_group["use_muon"] is False
    # Both Linear weights go to Muon; biases and the embedding go to AdamW.
    assert all(p.ndim == 2 for p in muon_group["params"])
    assert len(muon_group["params"]) == 2
    assert len(adamw_group["params"]) == 3
    assert adamw_group["lr"] == 3e-3


def test_muon_reduces_loss_on_toy_problem():
    torch.manual_seed(4)
    model = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 4))
    X = torch.randn(64, 8)
    true_w = torch.randn(8, 4)
    y = X @ true_w

    optimizer = Muon.from_named_parameters(
        model.named_parameters(), lr=0.05, adamw_lr=0.01
    )
    losses = []
    for _ in range(100):
        optimizer.zero_grad()
        loss = ((model(X) - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.1
    assert all(math.isfinite(value) for value in losses)


def _make_qk_pair(
    num_heads: int, head_dim: int, in_features: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(num_heads * head_dim, in_features, generator=generator) * 0.1
    k = torch.randn(num_heads * head_dim, in_features, generator=generator) * 0.1
    return nn.Parameter(q), nn.Parameter(k)


def test_qk_clip_rescales_only_heads_above_threshold():
    tau = 100.0
    q_param, k_param = _make_qk_pair(num_heads=2, head_dim=3, in_features=5, seed=5)
    q_before, k_before = q_param.detach().clone(), k_param.detach().clone()

    optimizer = MuonClip([q_param, k_param], lr=0.01)
    optimizer.register_qk_params("attn0", q_param, k_param, num_heads=2)

    # Head 0 exceeds tau by 4x (gamma = 1/2); head 1 stays below tau.
    applied = optimizer.qk_clip({"attn0": torch.tensor([4 * tau, 0.5 * tau])}, tau)

    assert applied["attn0"] == pytest.approx(0.5)
    torch.testing.assert_close(q_param[:3], q_before[:3] * 0.5)
    torch.testing.assert_close(k_param[:3], k_before[:3] * 0.5)
    torch.testing.assert_close(q_param[3:], q_before[3:])
    torch.testing.assert_close(k_param[3:], k_before[3:])


def test_qk_clip_leaves_weights_untouched_below_threshold():
    tau = 100.0
    q_param, k_param = _make_qk_pair(num_heads=2, head_dim=3, in_features=5, seed=6)
    q_before, k_before = q_param.detach().clone(), k_param.detach().clone()

    optimizer = MuonClip([q_param, k_param], lr=0.01)
    optimizer.register_qk_params("attn0", q_param, k_param, num_heads=2)
    applied = optimizer.qk_clip({"attn0": torch.tensor([0.9 * tau, 0.1 * tau])}, tau)

    assert applied["attn0"] == 1.0
    torch.testing.assert_close(q_param, q_before)
    torch.testing.assert_close(k_param, k_before)


def test_qk_clip_bounds_next_step_logits_by_threshold():
    torch.manual_seed(7)
    tau = 100.0
    num_heads, head_dim, in_features = 2, 4, 8
    q_param, k_param = _make_qk_pair(num_heads, head_dim, in_features, seed=8)
    optimizer = MuonClip([q_param, k_param], lr=0.01)
    optimizer.register_qk_params("attn0", q_param, k_param, num_heads)

    x = torch.randn(in_features)

    def head_logits() -> torch.Tensor:
        q = (q_param @ x).view(num_heads, head_dim)
        k = (k_param @ x).view(num_heads, head_dim)
        return (q * k).sum(-1) / math.sqrt(head_dim)

    # Scale up head 0's key weight so its logit exceeds tau, then clip.
    with torch.no_grad():
        logits = head_logits()
        k_param[:head_dim] *= 2 * tau / abs(logits[0])
    observed = head_logits()
    assert observed[0] > tau

    optimizer.qk_clip({"attn0": observed}, tau)
    clipped = head_logits()
    assert clipped.max().item() <= tau * (1 + 1e-5)


def test_state_dict_round_trip():
    generator = torch.Generator().manual_seed(9)
    matrix = nn.Parameter(torch.randn(4, 6, generator=generator))
    bias = nn.Parameter(torch.randn(4, generator=generator))
    optimizer = MuonClip(
        [
            {"params": [matrix], "use_muon": True},
            {"params": [bias], "use_muon": False},
        ],
        lr=0.02,
        weight_decay=0.01,
    )
    optimizer.register_qk_params("attn0", matrix, matrix, num_heads=1)
    matrix.grad = torch.randn_like(matrix)
    bias.grad = torch.randn_like(bias)
    optimizer.step()
    saved = optimizer.state_dict()

    new_matrix = nn.Parameter(matrix.detach().clone())
    new_bias = nn.Parameter(bias.detach().clone())
    reloaded = MuonClip(
        [
            {"params": [new_matrix], "use_muon": True},
            {"params": [new_bias], "use_muon": False},
        ],
        lr=0.02,
        weight_decay=0.01,
    )
    reloaded.load_state_dict(saved)

    torch.testing.assert_close(
        reloaded.state[new_matrix]["momentum_buffer"],
        optimizer.state[matrix]["momentum_buffer"],
    )
    torch.testing.assert_close(
        reloaded.state[new_bias]["exp_avg"], optimizer.state[bias]["exp_avg"]
    )
    assert reloaded.state[new_bias]["step"] == 1
    assert reloaded._loaded_qk_registry == {"attn0": 1}

    # A further step after reloading must stay finite and consistent.
    new_matrix.grad = torch.randn_like(new_matrix)
    reloaded.step()
    assert torch.isfinite(new_matrix).all()
