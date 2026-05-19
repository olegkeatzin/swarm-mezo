"""Consensus matrices: symmetric doubly-stochastic and row-stochastic reputational."""
from __future__ import annotations

import numpy as np


def doubly_stochastic_W(N: int, topology: str = "full", alpha: float | None = None) -> np.ndarray:
    """Build a symmetric doubly-stochastic mixing matrix via W = I − α·L(G).

    Currently supports two topologies:
    - ``"full"``  — complete graph, α = 1/N gives W = (1/N)·J (uniform average).
    - ``"ring"``  — cycle graph, Metropolis-style weights (W_ii = W_ij = 1/3).
    """
    if topology == "full":
        return np.full((N, N), 1.0 / N, dtype=np.float64)

    if topology == "ring":
        # Each node has exactly 2 neighbours on the cycle; Metropolis gives 1/3.
        W = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            W[i, i] = 1.0 / 3.0
            W[i, (i - 1) % N] = 1.0 / 3.0
            W[i, (i + 1) % N] = 1.0 / 3.0
        return W

    raise ValueError(f"unknown topology: {topology}")


def reputation_W(reputations: np.ndarray) -> np.ndarray:
    """Row-stochastic matrix with W_ij = r_j / Σ_l r_l (every row identical)."""
    r = np.asarray(reputations, dtype=np.float64)
    if np.any(r < 0):
        raise ValueError("reputations must be non-negative")
    row = r / r.sum()
    return np.tile(row, (r.shape[0], 1))


def update_reputations(
    reputations: np.ndarray,
    losses: np.ndarray,
    beta: float,
    gamma_r: float = 1.0,
) -> np.ndarray:
    """r_i ← r_i / (γ_r + β · |L_i − L_min|), then renormalise to mean 1.

    Renormalisation preserves scale across steps without changing the resulting
    mixing weights (which only depend on ratios r_j / Σ r).
    """
    L_min = float(np.min(losses))
    new = reputations / (gamma_r + beta * np.abs(losses - L_min))
    new = new * (len(new) / new.sum())
    return new


def assert_doubly_stochastic(W: np.ndarray, tol: float = 1e-9) -> None:
    N = W.shape[0]
    assert W.shape == (N, N)
    assert np.all(W >= -tol), "negative entries"
    assert np.allclose(W.sum(axis=1), 1.0, atol=tol), "row sums != 1"
    assert np.allclose(W.sum(axis=0), 1.0, atol=tol), "column sums != 1"


def assert_row_stochastic(W: np.ndarray, tol: float = 1e-9) -> None:
    N = W.shape[0]
    assert W.shape == (N, N)
    assert np.all(W >= -tol), "negative entries"
    assert np.allclose(W.sum(axis=1), 1.0, atol=tol), "row sums != 1"
