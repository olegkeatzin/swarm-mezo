"""Multi-agent MeZO optimisation loop with consensus mixing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .consensus import (
    doubly_stochastic_W,
    reputation_W,
    update_reputations,
)
from .mezo import spsa_estimate


@dataclass
class SwarmHistory:
    loss_mean: list[float] = field(default_factory=list)
    loss_min: list[float] = field(default_factory=list)
    theta_mean: list[np.ndarray] = field(default_factory=list)
    disagreement: list[float] = field(default_factory=list)   # Σ‖θ_i − θ̄‖²


def run_swarm(
    objective,
    N: int,
    n_steps: int,
    eta: float,
    eps: float,
    consensus_mode: str,
    beta: float,
    seed: int,
    init_center: Optional[np.ndarray] = None,
    init_spread: float = 0.3,
    gamma_r: float = 1.0,
    topology: str = "full",
    seed_bank: Optional[np.ndarray] = None,
) -> dict:
    """Run ``N`` agents for ``n_steps`` of (local MeZO step → consensus).

    ``consensus_mode`` is either ``"symmetric"`` (doubly-stochastic W per the
    chosen ``topology``) or ``"reputation"`` (row-stochastic W rebuilt from
    reputations updated by the loss vector each step). ``beta`` enters the
    reputation update only.

    If ``seed_bank`` is provided, perturbations z_i are drawn from that finite
    pool of seeds (FedKSeed-style); otherwise each agent samples its own z.
    """
    rng = np.random.default_rng(seed)
    M = int(objective.M)
    if init_center is None:
        init_center = np.zeros(M)
    theta = init_center[None, :] + rng.standard_normal((N, M)) * init_spread
    reputations = np.ones(N, dtype=np.float64)

    history = SwarmHistory()

    for step in range(n_steps):
        # 1) Local MeZO step for each agent.
        for i in range(N):
            if seed_bank is not None:
                chosen = int(seed_bank[rng.integers(0, len(seed_bank))])
                z = np.random.default_rng(chosen).standard_normal(M)
            else:
                z = rng.standard_normal(M)
            g = spsa_estimate(objective, theta[i], eps=eps, z=z)
            theta[i] = theta[i] - eta * g

        # 2) Compute losses for reputation update and logging.
        losses = np.array([objective.value(theta[i]) for i in range(N)])

        # 3) Build mixing matrix and apply consensus step.
        if consensus_mode == "symmetric":
            W = doubly_stochastic_W(N, topology=topology)
        elif consensus_mode == "reputation":
            reputations = update_reputations(reputations, losses, beta, gamma_r)
            W = reputation_W(reputations)
        else:
            raise ValueError(f"unknown consensus_mode: {consensus_mode}")
        theta = W @ theta

        # 4) Log.
        theta_bar = theta.mean(axis=0)
        history.loss_mean.append(float(np.mean(losses)))
        history.loss_min.append(float(np.min(losses)))
        history.theta_mean.append(theta_bar.copy())
        history.disagreement.append(float(np.sum((theta - theta_bar) ** 2)))

    return {
        "theta": theta,
        "theta_mean": theta.mean(axis=0),
        "reputations": reputations,
        "history": history,
    }
