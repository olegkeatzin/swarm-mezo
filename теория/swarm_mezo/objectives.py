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


class QuadraticWithWells:
    """Smooth quadratic with a deep global well at origin and a few shallower
    local wells at offset points. Designed to mirror the *spirit* of a
    prompt-based fine-tuning loss surface: dominantly convex, with a small
    number of distinct local attractors — not a dense forest of minima.

        f(θ) = 0.5·‖θ‖² − d_g·exp(−‖θ‖²/(2σ²))
               − Σ_k d_k·exp(−‖θ − c_k‖²/(2σ²))

    Default config in dim M=10:
      - Global well at origin, depth 2.0  → f(0) ≈ −2.0
      - Local well 1 at (+3, 0, ..., 0), depth 1.0  → f(c_1) ≈ +3.5
      - Local well 2 at (−2.5, 0, ..., 0), depth 0.8  → f(c_2) ≈ +2.3
      - σ = 0.8

    With these numbers the global well is unambiguously deepest, but the
    local wells are deep enough to trap agents that initialise on the
    wrong side of the origin.
    """

    def __init__(
        self,
        M: int = 10,
        sigma: float = 0.8,
        global_depth: float = 2.0,
        local_centers: np.ndarray | None = None,
        local_depths: np.ndarray | None = None,
    ):
        self.M = int(M)
        self.sigma = float(sigma)
        self.global_depth = float(global_depth)
        if local_centers is None:
            c1 = np.zeros(M); c1[0] = 3.0
            c2 = np.zeros(M); c2[0] = -2.5
            local_centers = np.stack([c1, c2])
            local_depths = np.array([1.0, 0.8])
        self.local_centers = np.asarray(local_centers, dtype=float)
        self.local_depths = np.asarray(local_depths, dtype=float)

    def value(self, theta: np.ndarray) -> float:
        quad = 0.5 * float(theta @ theta)
        global_w = self.global_depth * float(np.exp(-(theta @ theta) / (2 * self.sigma ** 2)))
        diff = theta[None, :] - self.local_centers
        sq = np.sum(diff * diff, axis=1)
        local_w = float(np.sum(self.local_depths * np.exp(-sq / (2 * self.sigma ** 2))))
        return quad - global_w - local_w

    def true_grad(self, theta: np.ndarray) -> np.ndarray:
        g = theta.copy()
        # d/dθ [ −d_g·exp(−‖θ‖²/2σ²) ] = d_g · (θ/σ²) · exp(−‖θ‖²/2σ²)
        g += self.global_depth * (theta / self.sigma ** 2) * np.exp(-(theta @ theta) / (2 * self.sigma ** 2))
        diff = theta[None, :] - self.local_centers
        sq = np.sum(diff * diff, axis=1)
        coef = self.local_depths * np.exp(-sq / (2 * self.sigma ** 2)) / (self.sigma ** 2)
        g += np.sum(coef[:, None] * diff, axis=0)
        return g

    def is_in_global_basin(self, theta: np.ndarray, radius: float = 1.5) -> bool:
        """Centroid is in the global basin iff ‖θ‖ < radius. Default 1.5 sits
        comfortably between origin and the nearest local center (‖c_2‖=2.5)."""
        return float(np.linalg.norm(theta)) < radius
