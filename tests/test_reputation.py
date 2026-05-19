"""Tests for src/reputation.py — reputation-modulated consensus.

Covers the properties from теория/swarm-mezo.md §4:

- mixing weights are row-stochastic
- β=0 keeps reputations equal -> exact FedAvg averaging
- β large concentrates reputation on the loss winner
- reputation has memory: it carries forward across rounds
"""
import torch

from src.reputation import (
    reputation_consensus_step,
    reputation_weights,
    update_reputations,
)


def test_weights_sum_to_one():
    r = torch.tensor([0.1, 0.5, 0.3, 0.9, 0.2])
    w = reputation_weights(r)
    assert abs(w.sum().item() - 1.0) < 1e-6
    assert (w >= 0).all()


def test_beta_zero_keeps_reputations_uniform():
    r = torch.ones(4)
    losses = torch.tensor([0.1, 0.5, 0.3, 0.9])
    new = update_reputations(r, losses, beta=0.0)
    assert torch.allclose(new, torch.ones(4), atol=1e-6)


def test_consensus_step_with_beta_zero_is_fedavg():
    """β=0 -> reputations stay uniform -> every agent jumps to the mean."""
    N, M = 4, 3
    params = {"w": torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
        [4.0, 5.0, 6.0],
    ])}
    expected_mean = params["w"].mean(dim=0)
    reps = torch.ones(N)
    losses = torch.tensor([0.1, 0.5, 0.3, 0.9])
    reps, w = reputation_consensus_step(params, losses, reps, beta=0.0)
    for i in range(N):
        assert torch.allclose(params["w"][i], expected_mean, atol=1e-6)


def test_strong_beta_concentrates_on_winner():
    r = torch.ones(4)
    losses = torch.tensor([0.0, 1.0, 1.0, 1.0])
    new = update_reputations(r, losses, beta=100.0, gamma_r=1.0)
    w = reputation_weights(new)
    assert w[0].item() > 0.95, f"winner weight {w[0].item():.3f} too low"


def test_reputations_have_memory():
    """Re-applying update_reputations with the SAME losses keeps shrinking
    the losers — unlike memoryless softmax which is the same every call."""
    r = torch.ones(4)
    losses = torch.tensor([0.0, 1.0, 1.0, 1.0])

    r1 = update_reputations(r,  losses, beta=1.0)
    r2 = update_reputations(r1, losses, beta=1.0)

    # Winner share grew after the second application.
    w1 = reputation_weights(r1)
    w2 = reputation_weights(r2)
    assert w2[0].item() > w1[0].item()


def test_renormalisation_keeps_mean_one():
    r = torch.tensor([0.3, 0.4, 1.7, 0.9])
    losses = torch.tensor([0.1, 0.4, 0.2, 0.7])
    new = update_reputations(r, losses, beta=2.0)
    assert abs(new.mean().item() - 1.0) < 1e-6
