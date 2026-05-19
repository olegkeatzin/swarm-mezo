"""Tests for E2: shared seed bank breaks the 1/N variance reduction."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from swarm_mezo.experiments import run_e2


def test_e2_K1_has_no_reduction():
    """K=1: every agent uses the same z, so consensus = single estimate.
    Variance must be flat across N."""
    res = run_e2(M=50, cond_number=10.0, reps=400,
                 Ns=(1, 2, 4, 8, 16, 32), Ks=(1,), seed=0)
    v = res.variance[0]
    rel = v.max() / v.min()
    assert rel < 1.05, f"K=1 variance should be flat, got ratio {rel:.3f}"


def test_e2_independent_recovers_1_over_N():
    """K=None branch must reproduce the E1 slope ~ -1."""
    res = run_e2(M=50, cond_number=10.0, reps=400,
                 Ns=(1, 2, 4, 8, 16, 32), Ks=(None,), seed=0)
    log_N = np.log(res.Ns.astype(float))
    log_v = np.log(res.variance[0])
    slope, _ = np.polyfit(log_N, log_v, 1)
    assert -1.15 < slope < -0.85, f"slope {slope:.3f} not near -1"


def test_e2_plateau_height_orders_by_K():
    """For N >> K the variance plateau scales like Var(N=1)/K, so larger K
    must produce a lower plateau."""
    res = run_e2(M=50, cond_number=10.0, reps=600,
                 Ns=(1, 2, 4, 8, 16, 32, 64, 128),
                 Ks=(1, 4, 16), seed=0)
    plateaus = res.variance[:, -1]   # variance at N=128 for each K
    # K=1 plateau highest, K=16 lowest
    assert plateaus[0] > plateaus[1] > plateaus[2], (
        f"plateaus not ordered by K: {plateaus}"
    )
    # ratio approximately 4x between K=1 and K=4
    ratio = plateaus[0] / plateaus[1]
    assert 3.0 < ratio < 5.5, f"K=1/K=4 plateau ratio {ratio:.2f} not near 4"
