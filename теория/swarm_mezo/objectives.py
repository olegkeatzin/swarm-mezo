"""Synthetic objectives for Swarm-MeZO sanity simulation."""
from __future__ import annotations

import numpy as np


class Quadratic:
    """f(θ) = 0.5 · θᵀ A θ with diagonal A (cheap, controllable conditioning).

    Diagonal eigenvalues are spaced log-linearly from 1 to ``cond_number``,
    so the condition number of A equals exactly ``cond_number``.
    """

    def __init__(self, M: int = 200, cond_number: float = 100.0, seed: int = 0):
        if cond_number < 1.0:
            raise ValueError("cond_number must be >= 1")
        self.M = M
        self.cond_number = cond_number
        # log-spaced eigenvalues — bad-but-not-terrible conditioning
        self.a = np.geomspace(1.0, cond_number, M)
        self._seed = seed

    def value(self, theta: np.ndarray) -> float:
        return 0.5 * float(np.sum(self.a * theta * theta))

    def true_grad(self, theta: np.ndarray) -> np.ndarray:
        return self.a * theta


class MultiWell:
    """f(θ) = − Σ_k d_k · exp(−‖θ − c_k‖² / (2σ²)).

    A sum of inverted Gaussian wells. One well is deeper (global minimum),
    the rest are shallower (local minima). Designed for H3: a non-convex
    landscape in which we can unambiguously tell which basin each agent
    ended up in by nearest-center assignment.
    """

    def __init__(
        self,
        centers: np.ndarray | None = None,
        depths: np.ndarray | None = None,
        sigma: float = 0.6,
    ):
        if centers is None:
            centers = np.array([[0.0, 0.0], [2.2, 0.0], [-2.0, 0.4]])
            depths = np.array([1.0, 0.6, 0.6])
        self.centers = np.asarray(centers, dtype=float)
        self.depths = np.asarray(depths, dtype=float)
        self.sigma = float(sigma)
        self.M = int(self.centers.shape[1])
        self.global_idx = int(np.argmax(self.depths))
        self.global_center = self.centers[self.global_idx]

    def value(self, theta: np.ndarray) -> float:
        diff = theta[None, :] - self.centers
        sq = np.sum(diff * diff, axis=1)
        return float(-np.sum(self.depths * np.exp(-sq / (2.0 * self.sigma ** 2))))

    def true_grad(self, theta: np.ndarray) -> np.ndarray:
        diff = theta[None, :] - self.centers
        sq = np.sum(diff * diff, axis=1)
        coef = self.depths * np.exp(-sq / (2.0 * self.sigma ** 2))
        # d/dθ_j [−Σ d_k e^{−‖θ−c_k‖²/2σ²}] = Σ d_k e^{...} · (θ_j − c_kj)/σ²
        return np.sum(coef[:, None] * diff, axis=0) / (self.sigma ** 2)

    def nearest_well(self, theta: np.ndarray) -> int:
        d = np.linalg.norm(self.centers - theta, axis=1)
        return int(np.argmin(d))
