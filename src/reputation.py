"""Reputation-modulated consensus for federated MeZO.

Implements the reputational mixing rule from теория/swarm-mezo.md §4:

      r_i ← r_i / (γ_r + β · |L_i − L_min|)              (renormalised to mean 1)
      W_ij = r_j / Σ_l r_l         (row-stochastic, every row identical)
      θ_i ← Σ_j W_ij · θ_j

  Validated by experiment E3 in `теория/swarm_mezo/`. Reputation has memory:
  an agent that was the worst once keeps a lower weight even after its loss
  equalises. That feedback loop is what causes the cascade-into-local-minimum
  failure mode observed at β=100 in E3.

Predicted regimes (from E3 on 2D multi-well):
- β = 0       → all reputations stay equal → W = (1/N)·J → exact FedAvg.
- β ∈ [1, 10] → measurable improvement over FedAvg (working window).
- β = 100     → information cascade: one lucky agent monopolises reputation,
                swarm collapses into its basin.

This module exposes the math; `src/federated.py` wires the eval-batch loss
computation to it via `reputation_config`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ReputationConfig:
    """Per-round configuration for reputational consensus.

    eval_batch is a (input_ids, attention_mask, labels) tuple on the target
    device, used at every consensus round to score every agent. Should be
    class-balanced and disjoint from training data.

    γ_r damps reputation movement: the canonical theory choice γ_r = 1 makes
    β·|ΔL| dimensionless against it. Reputations are renormalised to mean 1
    each round so the scale doesn't drift.
    """
    eval_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    beta:    float = 1.0
    gamma_r: float = 1.0


@torch.no_grad()
def update_reputations(
    reputations: torch.Tensor,
    losses: torch.Tensor,
    beta: float,
    gamma_r: float = 1.0,
) -> torch.Tensor:
    """r_i ← r_i / (γ_r + β · |L_i − L_min|), renormalised to mean 1.

    Vectorised counterpart of swarm_mezo.consensus.update_reputations.
    Renormalisation is a numerical convenience — W = r / Σr is invariant
    under multiplicative rescaling, so the consensus dynamics are unchanged.
    Returns a fresh tensor; caller is expected to overwrite the stored
    reputations.
    """
    L_min = losses.min()
    denom = gamma_r + beta * (losses - L_min).abs()
    new = reputations / denom
    new = new * (new.numel() / new.sum())
    return new


@torch.no_grad()
def reputation_weights(reputations: torch.Tensor) -> torch.Tensor:
    """w_j = r_j / Σ_l r_l. Row of the reputational mixing matrix W."""
    return reputations / reputations.sum()


@torch.no_grad()
def reputation_consensus_step(
    params: dict[str, torch.Tensor],
    losses: torch.Tensor,
    reputations: torch.Tensor,
    beta: float,
    gamma_r: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one reputational consensus step in-place on stacked params.

    Mathematically:
        r ← update_reputations(r, L, β, γ_r)
        w_j = r_j / Σ r
        θ_i ← Σ_j w_j · θ_j      (every agent jumps to the weighted centroid)

    Returns the (updated reputations, mixing weights) for logging.
    """
    new_reps = update_reputations(reputations, losses, beta, gamma_r)
    w = reputation_weights(new_reps)
    for p in params.values():
        bcast = w.view(-1, *([1] * (p.ndim - 1)))               # (N, 1, ..., 1)
        theta_bar = (bcast * p).sum(dim=0, keepdim=True)        # (1, *)
        p.copy_(theta_bar.expand_as(p))
    return new_reps, w
