"""Consensus matrices for decentralized averaging over different graph topologies.

A consensus matrix W is N x N, doubly stochastic (rows and columns each sum to
1), and encodes "after one consensus round, agent i's new parameters are
sum_j W_ij * θ_j". Iterating W contracts deviations from the centroid at rate
|λ₂(W)| — the second-largest-magnitude eigenvalue. The **spectral gap**
1 − |λ₂| is the universal currency for "how fast does this topology mix".

Topologies provided:
  - full:  W = (1/N) * 1·1ᵀ (FedAvg).            gap = 1     (one-shot mixing)
  - star:  one hub + (N-1) leaves, Metropolis.    gap = 1/N   (slow, scales O(1/N))
  - ring:  each agent ↔ two neighbors, cyclic.    gap ≈ π²/(3N²) (very slow)

The point of Day 3 is to show these gaps empirically: log‖θ_t − θ̄‖ vs round
should have slope log|λ₂|, matching the constructed W's spectrum.
"""
from __future__ import annotations

import math

import torch


def build_full(n: int) -> torch.Tensor:
    """Complete graph consensus = FedAvg. W_ij = 1/N. Rank 1. λ₂ = 0, gap = 1."""
    return torch.full((n, n), 1.0 / n)


def build_ring(n: int) -> torch.Tensor:
    """Cyclic ring: each agent averages itself with its two neighbors (weights 1/3).

    W is symmetric circulant, doubly stochastic by construction. Its eigenvalues are
    (1 + 2·cos(2π·k/N)) / 3 for k = 0, ..., N-1; λ₁ = 1 (k=0), λ₂ = (1 + 2·cos(2π/N))/3.
    For large N, gap ≈ π² / (3·N²) — slow mixing, scales like 1/N².
    """
    if n < 3:
        raise ValueError(f"ring topology requires n ≥ 3, got n={n}")
    W = torch.zeros(n, n)
    for i in range(n):
        W[i, i] = 1.0 / 3.0
        W[i, (i - 1) % n] = 1.0 / 3.0
        W[i, (i + 1) % n] = 1.0 / 3.0
    return W


def build_star(n: int) -> torch.Tensor:
    """Star: agent 0 is hub, 1..N-1 are leaves; only hub-leaf edges.

    Using uniform Metropolis weights for this graph:
      - W[hub, hub]   = 1/N
      - W[hub, leaf]  = 1/N   (hub averages with each leaf)
      - W[leaf, hub]  = 1/N
      - W[leaf, leaf] = 1 - 1/N
      - W[leaf_i, leaf_j] = 0 for i != j

    Doubly stochastic: hub-row sums to 1/N + (N-1)/N = 1; leaf-row to 1/N + (N-1)/N = 1.
    Spectrum: λ₁ = 1, λ₂ = (N-1)/N (with multiplicity N-2 from the unmixed leaves).
    Gap = 1/N — better than ring, far worse than full.
    """
    if n < 2:
        raise ValueError(f"star topology requires n ≥ 2, got n={n}")
    W = torch.zeros(n, n)
    W[0, 0] = 1.0 / n
    W[0, 1:] = 1.0 / n
    W[1:, 0] = 1.0 / n
    for i in range(1, n):
        W[i, i] = 1.0 - 1.0 / n
    return W


TOPOLOGIES = {
    "full": build_full,
    "ring": build_ring,
    "star": build_star,
}


def build_topology(name: str, n: int) -> torch.Tensor:
    if name not in TOPOLOGIES:
        raise KeyError(f"unknown topology {name!r}; pick from {sorted(TOPOLOGIES)}")
    return TOPOLOGIES[name](n)


def spectral_gap(W: torch.Tensor) -> float:
    """1 − |λ₂(W)|, the contraction rate per consensus round.

    Eigenvalues may be complex for non-symmetric W; we sort by magnitude.
    """
    eigs = torch.linalg.eigvals(W).abs().sort(descending=True).values
    return 1.0 - eigs[1].item()


def second_eigenvalue(W: torch.Tensor) -> float:
    """|λ₂(W)| — the per-round consensus residual contraction factor."""
    eigs = torch.linalg.eigvals(W).abs().sort(descending=True).values
    return eigs[1].item()


def apply_consensus(params: dict[str, torch.Tensor], W: torch.Tensor) -> None:
    """In-place: replace each stacked param tensor p of shape (N, *) with W @ p.

    Per parameter we do  p_new[i] = Σ_j W[i, j] · p[j]  — the consensus update.
    W is moved to the param's device for matmul. params dict is modified in place.
    """
    for p in params.values():
        if W.device != p.device:
            W = W.to(p.device)
        flat  = p.reshape(p.shape[0], -1)   # (N, d)
        mixed = W @ flat                     # (N, d)
        p.copy_(mixed.reshape(p.shape))


def consensus_distance(params: dict[str, torch.Tensor]) -> float:
    """Scalar measuring how far agents are from their common centroid.

    Computes sqrt(Σ_param Σ_i ‖p_i − p̄‖²), i.e., the Frobenius distance of the
    stacked params from the rank-1 "all-equal" subspace. After a consensus round
    this should multiply by ≈ |λ₂(W)|.
    """
    total_sq = 0.0
    for p in params.values():
        mean = p.mean(dim=0, keepdim=True)        # (1, *)
        diff = (p - mean).reshape(p.shape[0], -1) # (N, d)
        total_sq += (diff.float() ** 2).sum().item()
    return math.sqrt(total_sq)
