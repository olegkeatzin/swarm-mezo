"""MeZO optimizer: memory-efficient zeroth-order SPSA for fine-tuning.

Reference: Malladi et al., "Fine-Tuning Language Models with Just Forward Passes",
NeurIPS 2023. https://arxiv.org/abs/2305.17333

The same `seed` is reused across all three perturbations within a single step,
so the random direction `z` is regenerated identically on each pass. This is
what keeps memory at inference-level — `z` is never materialized for all
parameters at once.
"""
from __future__ import annotations

from typing import Callable

import torch


class MeZOOptimizer:
    def __init__(self, model: torch.nn.Module, lr: float = 1e-6, eps: float = 1e-3) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps
        self.params = [p for p in model.parameters() if p.requires_grad]
        # Per-instance generator, lazily placed on the model's device.
        # Avoids both the global-RNG race condition and CPU→GPU copies.
        self._rng: torch.Generator | None = None

    def _get_rng(self) -> torch.Generator:
        if self._rng is None:
            device = self.params[0].device if self.params else torch.device('cpu')
            self._rng = torch.Generator(device=device)
        return self._rng

    def _perturb(self, scaling: float, seed: int) -> None:
        """In-place: p ← p + scaling · z, with z ~ N(0, I) regenerated from `seed`."""
        rng = self._get_rng()
        rng.manual_seed(seed)
        for p in self.params:
            z = torch.empty_like(p).normal_(generator=rng)
            p.data.add_(z, alpha=scaling)

    @torch.no_grad()
    def step(self, loss_fn: Callable[[], torch.Tensor]) -> float:
        """One MeZO step. `loss_fn` returns a scalar loss tensor on the same batch.

        Returns the loss at θ+εz (the first forward), as a Python float.
        """
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

        # θ → θ + ε·z
        self._perturb(scaling=self.eps, seed=seed)
        loss_plus = loss_fn().item()

        # θ → θ − ε·z   (step back by 2ε from θ+εz)
        self._perturb(scaling=-2.0 * self.eps, seed=seed)
        loss_minus = loss_fn().item()

        # θ → θ          (restore)
        self._perturb(scaling=self.eps, seed=seed)

        projected_grad = (loss_plus - loss_minus) / (2.0 * self.eps)

        # θ ← θ − η · projected_grad · z
        self._perturb(scaling=-self.lr * projected_grad, seed=seed)

        return loss_plus
