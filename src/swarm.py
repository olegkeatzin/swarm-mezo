"""Swarm-MeZO: PSO-flavored consensus that biases mixing toward better agents.

Standard consensus (`src/consensus.py`) uses a fixed doubly-stochastic W so the
mean over agents is preserved across rounds. Swarm consensus replaces that
with a *data-dependent* mixing rule:

    θ_i ← (1 − α) · θ_i + α · θ_swarm,
    θ_swarm = Σ_j w_j · θ_j,    w_j = softmax(−β · L_j)_j

Two hyperparameters:
  - β (selectivity / inverse temperature): β=0 → uniform w_j = 1/N (FedAvg);
    β→∞ → w_j collapses onto argmin_j L_j (winner-take-all selection, ES).
  - α (social coefficient / inertia): α=1 → every agent jumps to the swarm
    center (and if β→∞, they all become the leader); α<1 → inertia preserves
    per-agent diversity. α=0 disables consensus entirely.

Interpretation as a matrix: W = (1−α)·I + α·1·wᵀ, with wᵀ summing to 1.
Rows sum to 1 (row-stochastic) but columns do NOT in general — the column sum
is (1−α) + Nα·w_j, which depends on j. This is the deliberate departure from
classical consensus: the mean of the parameters is no longer preserved, the
swarm drifts toward low-loss agents. That asymmetry IS the evolutionary
signal — it's also why this is not a Granichin-style consensus method and
should be presented as a bridge to ES / PSO on the defense.

Convex-combination math guarantees no NaN explosion: since w_j ∈ [0,1] sums
to 1 and α ∈ [0,1], the new θ_i is a convex combination of all current θ_j,
bounded inside their hull.

The implementation is vmap-friendly: it accepts the stacked params dict
already produced by `_stack_models` in src/federated.py and operates entirely
on tensors of shape (N, *param_shape). It needs an evaluation batch and a
loss function (`vmapped_loss`) injected — those live inside the training loop
in src/federated.py, so this module exposes the math and a config object;
src/federated.py wires them together.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SwarmConfig:
    """Per-round configuration for swarm-style consensus.

    eval_batch is a (input_ids, attention_mask, labels) tuple of shape (B, L)
    each, already on the target device. The same batch is shown to every
    agent at each consensus round — fitness is "loss on this fixed probe set".
    Use a held-out batch (NOT seen during training) to avoid overfitting the
    selection signal to per-agent training data.

    For non-IID experiments, the eval_batch should be class-balanced; otherwise
    the selection penalizes specialized agents who happen to see a class-skewed
    probe. See README / CLAUDE.md Day-4 notes.
    """
    eval_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    alpha: float = 0.5
    beta:  float = 1.0


@torch.no_grad()
def compute_swarm_weights(losses: torch.Tensor, beta: float) -> torch.Tensor:
    """w_j = softmax(−β · L_j)_j.

    β=0 → uniform. β large → concentrated on argmin. Always non-negative,
    sums to 1, never NaN provided losses are finite (softmax is numerically
    stable in torch).
    """
    return torch.softmax(-beta * losses, dim=0)


@torch.no_grad()
def swarm_consensus_step(
    params: dict[str, torch.Tensor],
    losses: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """In-place swarm mixing of stacked params, given pre-computed agent losses.

    For each parameter tensor of shape (N, *):
        θ_swarm = Σ_j softmax(−β·L)_j · θ_j         # (1, *)
        θ_i    ← (1 − α) · θ_i + α · θ_swarm        # (N, *)

    Returns the softmax weights (N,) for logging — useful to see, on the
    defense plot, which agents "led" each round.
    """
    weights = compute_swarm_weights(losses, beta)                    # (N,)
    for p in params.values():
        bcast = weights.view(-1, *([1] * (p.ndim - 1)))              # (N, 1, ..., 1)
        theta_swarm = (bcast * p).sum(dim=0, keepdim=True)           # (1, *)
        p.copy_((1.0 - alpha) * p + alpha * theta_swarm)
    return weights
