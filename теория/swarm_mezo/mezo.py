"""SPSA / MeZO gradient estimator (two forward passes, no autograd)."""
from __future__ import annotations

import numpy as np


def spsa_estimate(
    objective,
    theta: np.ndarray,
    eps: float = 1e-3,
    rng: np.random.Generator | None = None,
    z: np.ndarray | None = None,
    perturbation: str = "gaussian",
) -> np.ndarray:
    """One SPSA gradient estimate ĝ = (f(θ+εz) − f(θ−εz)) / (2ε) · z.

    If ``z`` is provided it is used as-is — this is what E2 (FedKSeed seed-bank
    simulation) needs to feed identical perturbations to several agents.
    Otherwise ``z`` is drawn from ``rng`` according to ``perturbation``.
    """
    if z is None:
        if rng is None:
            raise ValueError("either rng or z must be provided")
        if perturbation == "gaussian":
            z = rng.standard_normal(theta.shape)
        elif perturbation == "rademacher":
            z = rng.choice([-1.0, 1.0], size=theta.shape)
        else:
            raise ValueError(f"unknown perturbation type: {perturbation}")
    f_plus = objective.value(theta + eps * z)
    f_minus = objective.value(theta - eps * z)
    return (f_plus - f_minus) / (2.0 * eps) * z
