"""Sanity tests for the E1 sub-pipeline (SPSA unbiasedness, 1/N law)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from swarm_mezo.experiments import run_e1, run_e2
from swarm_mezo.mezo import spsa_estimate
from swarm_mezo.objectives import Quadratic


def test_spsa_mean_matches_true_grad():
    """On a quadratic, E[ĝ] = ∇f exactly. Empirical mean over many reps
    should be close."""
    obj = Quadratic(M=20, cond_number=10.0, seed=0)
    rng = np.random.default_rng(123)
    theta = rng.standard_normal(20)
    true_g = obj.true_grad(theta)

    reps = 5000
    avg = np.zeros_like(theta)
    for _ in range(reps):
        avg += spsa_estimate(obj, theta, eps=1e-3, rng=rng)
    avg /= reps

    # Tolerance ~ sqrt(Var/reps). For M=20 the per-rep variance is moderate.
    np.testing.assert_allclose(avg, true_g, atol=0.5, rtol=0.1)


def test_e1_slope_near_minus_one():
    """The whole point of H1: log-log slope of variance vs N is ≈ −1."""
    result = run_e1(M=50, cond_number=10.0, reps=400,
                    Ns=(1, 2, 4, 8, 16, 32), seed=0)
    assert -1.15 < result.slope < -0.85, (
        f"slope {result.slope:.3f} is not consistent with the 1/N law"
    )


def test_consensus_variance_strictly_decreases():
    result = run_e1(M=50, cond_number=10.0, reps=400,
                    Ns=(1, 2, 4, 8, 16, 32), seed=0)
    diffs = np.diff(result.variance)
    assert np.all(diffs < 0), "variance must decrease monotonically in N"
