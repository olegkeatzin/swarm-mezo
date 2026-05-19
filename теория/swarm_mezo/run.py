"""Entry point: runs experiment E1 and writes PNG + CSV into results/.

Run from the `теория/` directory:

    python -m swarm_mezo.run

or directly:

    python теория/swarm_mezo/run.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow running as a plain script (no package context).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarm_mezo.experiments import run_e1, run_e2, run_e3
from swarm_mezo.plots import (
    plot_e1, plot_e2, plot_e3,
    save_e1_csv, save_e2_csv, save_e3_csv,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    print("Running E1 (H1: 1/N variance reduction)...")
    result = run_e1(
        M=200,
        cond_number=100.0,
        eps=1e-3,
        reps=1000,
        Ns=(1, 2, 4, 8, 16, 32, 64, 128),
        seed=0,
    )
    print(f"  log-log slope = {result.slope:.4f} (expected ~ -1)")
    print(f"  variance(N=1)   = {result.variance[0]:.4e}")
    print(f"  variance(N=128) = {result.variance[-1]:.4e}")

    plot_e1(result, RESULTS_DIR / "e1_variance_vs_N.png")
    save_e1_csv(result, RESULTS_DIR / "e1_variance_vs_N.csv")
    print(f"Saved results to {RESULTS_DIR}")

    verdict = "CONFIRMED" if -1.1 <= result.slope <= -0.9 else "INCONCLUSIVE"
    print(f"H1 verdict: {verdict}")

    print()
    print("Running E2 (H2: shared seed bank breaks the 1/N law)...")
    e2 = run_e2(
        M=200,
        cond_number=100.0,
        eps=1e-3,
        reps=1000,
        Ns=(1, 2, 4, 8, 16, 32, 64, 128),
        Ks=(1, 4, 16, 64, None),
        seed=0,
    )
    for ki, K in enumerate(e2.Ks):
        tag = "inf" if K is None else str(K)
        v1, vN = e2.variance[ki, 0], e2.variance[ki, -1]
        ratio = v1 / vN if vN > 0 else float("inf")
        print(f"  K={tag:>3}:  Var(N=1)={v1:.3e}  Var(N=128)={vN:.3e}  ratio={ratio:.2f}")
    plot_e2(e2, RESULTS_DIR / "e2_variance_seedbank.png")
    save_e2_csv(e2, RESULTS_DIR / "e2_variance_seedbank.csv")

    print()
    print("Running E3 (H3: reputational beta on a multi-well landscape)...")
    e3 = run_e3(
        N=8, n_steps=250, eta=0.05, eps=5e-3, n_runs=50,
        betas=(0.0, 0.1, 1.0, 10.0, 100.0),
        init_spread=2.0, seed=0,
    )
    for bi, beta in enumerate(e3.betas):
        print(f"  beta={beta:7.2f}  rep_hit={e3.rep_hit_rate[bi]:.2f} "
              f"sym_hit={e3.sym_hit_rate[bi]:.2f}  "
              f"rep_loss={e3.rep_mean_loss[bi]:+.4f} "
              f"sym_loss={e3.sym_mean_loss[bi]:+.4f}")
    plot_e3(e3, RESULTS_DIR / "e3_convergence_vs_beta.png")
    save_e3_csv(e3, RESULTS_DIR / "e3_convergence_vs_beta.csv")

    best_beta = e3.betas[int(np.argmax(e3.rep_hit_rate))]
    worst_beta = e3.betas[int(np.argmin(e3.rep_hit_rate))]
    sym_spread = e3.sym_hit_rate.max() - e3.sym_hit_rate.min()
    print(f"H3 reputational: best beta = {best_beta}, worst beta = {worst_beta}")
    print(f"H3 symmetric control: hit-rate spread across beta = {sym_spread:.3f} "
          f"(should be ~0)")


if __name__ == "__main__":
    main()
