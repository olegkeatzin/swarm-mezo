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
    trim_weights,
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


# ── conformity control branch (лекция2.md слайд 11) ───────────────────────────

def test_conformity_equal_losses_is_fedavg():
    """All losses equal -> every agent sits on the consensus L̄ -> zero
    penalty -> reputations stay uniform, identical to FedAvg."""
    r = torch.ones(4)
    losses = torch.full((4,), 0.7)
    new = update_reputations(r, losses, beta=10.0, mode="conformity")
    assert torch.allclose(new, torch.ones(4), atol=1e-6)


def test_conformity_rewards_typical_not_best():
    """The key contrast with mode='loss': conformity crowns the agent closest
    to the weighted-mean loss, even demoting the actual loss winner when it is
    atypical. Here agent 0 has the best loss but is the farthest from L̄."""
    r = torch.ones(4)
    losses = torch.tensor([0.0, 0.5, 0.6, 0.6])     # agent 0 = loss winner

    w_loss = reputation_weights(update_reputations(r, losses, beta=5.0, mode="loss"))
    w_conf = reputation_weights(update_reputations(r, losses, beta=5.0, mode="conformity"))

    assert w_loss.argmax().item() == 0              # loss mode crowns the best
    assert w_conf.argmax().item() == 1              # conformity crowns the typical
    assert w_conf[0].item() < w_conf.mean().item()  # loss winner demoted as atypical


# ── trim_k modifier (robust aggregation — Yin et al. 2018) ────────────────────

def test_trim_k_zero_is_identity():
    """trim_k=0 leaves the mixing row untouched."""
    w   = torch.tensor([0.4, 0.1, 0.3, 0.2])
    out = trim_weights(w, losses=torch.tensor([0.1, 0.9, 0.3, 0.5]), trim_k=0)
    assert torch.allclose(out, w)


def test_trim_drops_worst_and_renormalises():
    """The trim_k highest-loss agents get weight 0; survivors keep their
    relative weights and are renormalised; the row stays row-stochastic."""
    w      = torch.tensor([0.4, 0.1, 0.3, 0.1, 0.1])
    losses = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.2])  # worst-2 = idx 1, 3
    out = trim_weights(w, losses, trim_k=2)
    assert out[1].item() == 0.0 and out[3].item() == 0.0
    assert abs(out.sum().item() - 1.0) < 1e-6
    assert (out >= 0).all()
    # survivors keep their relative proportions (0.4 : 0.3 : 0.1)
    assert torch.allclose(out[[0, 2, 4]], torch.tensor([0.5, 0.375, 0.125]), atol=1e-6)


def test_beta_zero_plus_trim_is_plain_trimmed_mean():
    """β=0 → uniform reputations → trim_k>0 gives a uniform mean over the
    N−trim_k survivors (the plain trimmed mean)."""
    params = {"w": torch.tensor([
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [99.0, 99.0],   # bad agent — would wreck a plain FedAvg mean
    ])}
    losses = torch.tensor([0.1, 0.2, 0.15, 5.0])  # agent 3 = worst
    reps = torch.ones(4)
    new_reps, w = reputation_consensus_step(
        params, losses, reps, beta=0.0, mode="loss", trim_k=1,
    )
    for i in range(4):
        assert torch.allclose(params["w"][i], torch.tensor([1.0, 1.0]), atol=1e-6)
    assert w[3].item() == 0.0
    assert torch.allclose(w[[0, 1, 2]], torch.full((3,), 1 / 3), atol=1e-6)


def test_trim_composes_with_reputation():
    """β>0 + trim_k>0: the worst agent is dropped, and among survivors the
    reputation weighting still favours the lowest loss."""
    reps   = torch.ones(4)
    losses = torch.tensor([0.0, 0.3, 0.5, 5.0])  # agent 3 = worst, agent 0 = best
    new_reps = update_reputations(reps, losses, beta=2.0, mode="loss")
    w = trim_weights(reputation_weights(new_reps), losses, trim_k=1)
    assert w[3].item() == 0.0                    # worst dropped
    assert w[0].item() > w[1].item() > w[2].item()  # reputation order preserved


def test_trim_rejects_bad_k():
    w      = torch.tensor([0.25, 0.25, 0.25, 0.25])
    losses = torch.tensor([0.1, 0.2, 0.3, 0.4])
    for bad in (4, 5):
        try:
            trim_weights(w, losses, trim_k=bad)
            assert False, f"trim_k={bad} should have raised"
        except ValueError:
            pass
