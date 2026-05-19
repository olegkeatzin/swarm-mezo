"""Tests for E3: consensus matrices, reputation update, and multi-well swarm."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from swarm_mezo.consensus import (
    assert_doubly_stochastic,
    assert_row_stochastic,
    doubly_stochastic_W,
    reputation_W,
    update_reputations,
)
from swarm_mezo.objectives import MultiWell
from swarm_mezo.swarm import run_swarm


def test_full_graph_W_is_doubly_stochastic():
    W = doubly_stochastic_W(5, topology="full")
    assert_doubly_stochastic(W)
    assert np.allclose(W, 1.0 / 5)


def test_ring_W_is_doubly_stochastic():
    W = doubly_stochastic_W(6, topology="ring")
    assert_doubly_stochastic(W)


def test_reputation_W_is_row_stochastic_but_not_doubly():
    r = np.array([1.0, 2.0, 3.0, 4.0])
    W = reputation_W(r)
    assert_row_stochastic(W)
    # Column sums equal r_j / sum(r) * N — not 1.
    col_sums = W.sum(axis=0)
    assert not np.allclose(col_sums, 1.0)


def test_reputation_update_concentrates_on_best():
    # Strong beta and a clear loss winner -> winner keeps reputation,
    # others shrink towards zero (relative to winner).
    r = np.ones(4)
    losses = np.array([0.0, 1.0, 1.0, 1.0])
    r_new = update_reputations(r, losses, beta=100.0, gamma_r=1.0)
    assert r_new[0] > 10 * r_new[1]


def test_symmetric_full_W_is_invariant_to_beta():
    """Control claim: under symmetric mode, the converged point doesn't
    depend on beta. We hit the same theta_mean for two very different beta
    values (beta is only consumed by reputation_W in symmetric mode it is
    ignored)."""
    obj = MultiWell()
    out_a = run_swarm(obj, N=6, n_steps=100, eta=0.05, eps=5e-3,
                      consensus_mode="symmetric", beta=0.0, seed=42,
                      init_center=np.zeros(2), init_spread=1.5)
    out_b = run_swarm(obj, N=6, n_steps=100, eta=0.05, eps=5e-3,
                      consensus_mode="symmetric", beta=100.0, seed=42,
                      init_center=np.zeros(2), init_spread=1.5)
    np.testing.assert_allclose(out_a["theta_mean"], out_b["theta_mean"], atol=1e-12)


def test_swarm_descends_loss():
    obj = MultiWell()
    out = run_swarm(obj, N=8, n_steps=200, eta=0.05, eps=5e-3,
                    consensus_mode="reputation", beta=1.0, seed=0,
                    init_center=np.zeros(2), init_spread=0.5)
    hist = out["history"]
    assert hist.loss_mean[-1] < hist.loss_mean[0], "swarm did not reduce loss"
