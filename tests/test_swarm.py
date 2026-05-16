"""Tests for src/swarm.py — swarm (PSO-style) consensus math.

Covers the four properties that make the convex combination safe:
- weights sum to 1 (row-stochasticity of the implicit W)
- β=0 collapses to FedAvg
- β→∞ collapses to winner-take-all selection
- α=0 disables consensus (params unchanged regardless of weights)
- α=1, β→∞ → every agent becomes the leader
"""
import math

import pytest
import torch

from src.swarm import compute_swarm_weights, swarm_consensus_step


def test_weights_sum_to_one():
    losses = torch.tensor([0.1, 0.5, 0.3, 0.9, 0.2])
    for beta in [0.0, 0.5, 1.0, 5.0, 100.0]:
        w = compute_swarm_weights(losses, beta)
        assert abs(w.sum().item() - 1.0) < 1e-5, f"β={beta}: sum={w.sum().item()}"
        assert (w >= 0).all()


def test_beta_zero_is_uniform():
    """β=0 → softmax of zeros → uniform 1/N."""
    losses = torch.tensor([0.1, 0.5, 0.3, 0.9])
    w = compute_swarm_weights(losses, beta=0.0)
    assert torch.allclose(w, torch.full((4,), 0.25), atol=1e-6)


def test_beta_large_is_winner_take_all():
    """β→∞ → softmax concentrates on argmin loss."""
    losses = torch.tensor([0.7, 0.3, 0.5, 0.9])   # argmin = index 1
    w = compute_swarm_weights(losses, beta=1000.0)
    assert w[1].item() > 0.99
    for i in [0, 2, 3]:
        assert w[i].item() < 0.01


def test_lower_loss_gets_higher_weight():
    """Weight is monotonically decreasing in loss for any β > 0."""
    losses = torch.tensor([0.9, 0.1, 0.5, 0.3])
    for beta in [0.5, 1.0, 2.0, 10.0]:
        w = compute_swarm_weights(losses, beta)
        # Sorted by loss: idx 1 (0.1) < idx 3 (0.3) < idx 2 (0.5) < idx 0 (0.9)
        assert w[1] > w[3] > w[2] > w[0], f"β={beta}: weights {w.tolist()}"


def test_alpha_zero_leaves_params_unchanged():
    """α=0 → θ_i ← θ_i — pure inertia, swarm signal ignored."""
    n = 4
    torch.manual_seed(0)
    params = {"w": torch.randn(n, 3, 5), "b": torch.randn(n, 7)}
    snapshot = {k: v.clone() for k, v in params.items()}
    losses = torch.tensor([0.1, 0.5, 0.3, 0.9])

    swarm_consensus_step(params, losses, alpha=0.0, beta=2.0)

    for k in params:
        assert torch.allclose(params[k], snapshot[k]), f"{k} changed at α=0"


def test_alpha_one_beta_inf_all_agents_become_leader():
    """α=1, β→∞ → every agent becomes a copy of argmin-loss agent."""
    n = 4
    torch.manual_seed(0)
    params = {"w": torch.randn(n, 3, 5), "b": torch.randn(n, 7)}
    leader_w = params["w"][1].clone()
    leader_b = params["b"][1].clone()
    losses = torch.tensor([0.7, 0.05, 0.5, 0.9])   # leader = idx 1

    swarm_consensus_step(params, losses, alpha=1.0, beta=10_000.0)

    for i in range(n):
        assert torch.allclose(params["w"][i], leader_w, atol=1e-4), \
            f"agent {i} not equal to leader after winner-take-all"
        assert torch.allclose(params["b"][i], leader_b, atol=1e-4)


def test_alpha_one_beta_zero_is_fedavg():
    """α=1, β=0 → θ_i ← mean across agents (exactly FedAvg)."""
    n = 4
    torch.manual_seed(0)
    params = {"w": torch.randn(n, 3, 5), "b": torch.randn(n, 7)}
    mean_w = params["w"].mean(dim=0).clone()
    mean_b = params["b"].mean(dim=0).clone()
    losses = torch.tensor([0.7, 0.3, 0.5, 0.9])   # doesn't matter at β=0

    swarm_consensus_step(params, losses, alpha=1.0, beta=0.0)

    for i in range(n):
        assert torch.allclose(params["w"][i], mean_w, atol=1e-6)
        assert torch.allclose(params["b"][i], mean_b, atol=1e-6)


def test_convex_combination_stays_in_hull():
    """No NaNs, no explosion: each new θ_i lies within max(|θ_j|) ball."""
    n = 8
    torch.manual_seed(0)
    params = {"w": torch.randn(n, 100) * 5.0}
    max_norm_before = params["w"].norm(dim=1).max().item()
    losses = torch.randn(n).abs()

    for alpha in [0.1, 0.5, 0.9, 1.0]:
        for beta in [0.0, 1.0, 100.0]:
            p = {"w": params["w"].clone()}
            swarm_consensus_step(p, losses, alpha, beta)
            assert torch.isfinite(p["w"]).all()
            # Convex combination of bounded points stays bounded.
            assert p["w"].norm(dim=1).max().item() <= max_norm_before + 1e-4


def test_swarm_step_matches_explicit_W():
    """Verify the implementation matches W = (1-α)I + α·1·wᵀ applied to stacked params."""
    n = 5
    torch.manual_seed(0)
    params = {"p": torch.randn(n, 13)}
    p_before = params["p"].clone()
    losses = torch.tensor([0.4, 0.1, 0.6, 0.3, 0.8])
    alpha, beta = 0.4, 1.5

    w = compute_swarm_weights(losses, beta)                       # (N,)
    W = (1.0 - alpha) * torch.eye(n) + alpha * torch.ones(n, 1) @ w.view(1, n)  # (N, N)
    expected = W @ p_before                                       # (N, 13)

    swarm_consensus_step(params, losses, alpha, beta)
    assert torch.allclose(params["p"], expected, atol=1e-6)


def test_row_stochastic_columns_not_in_general():
    """Document the asymmetry: rows of W sum to 1, columns generally do NOT."""
    n = 5
    losses = torch.tensor([0.1, 0.5, 0.3, 0.9, 0.2])
    alpha, beta = 0.6, 2.0
    w = compute_swarm_weights(losses, beta)
    W = (1.0 - alpha) * torch.eye(n) + alpha * torch.ones(n, 1) @ w.view(1, n)

    assert torch.allclose(W.sum(dim=1), torch.ones(n), atol=1e-6), "rows must sum to 1"
    col_sums = W.sum(dim=0)
    # Columns sum to (1-α) + Nα·w_j, which equals 1 only when w_j = 1/N (i.e. β=0).
    assert not torch.allclose(col_sums, torch.ones(n), atol=1e-3), \
        "columns should NOT all sum to 1 for β>0 (evolutionary drift)"
