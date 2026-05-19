"""Plot helpers for Swarm-MeZO experiments."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .experiments import E1Result, E2Result, E3Result


def plot_e1(result: E1Result, out_path: Path) -> None:
    """Log-log plot of consensus-variance vs N with a slope −1 reference line."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(result.Ns, result.variance, "o-", label="empirical Var(ĝ_avg)")

    # reference line of slope −1 anchored at the first data point
    ref_y = result.variance[0] * (result.Ns / result.Ns[0]) ** -1.0
    ax.loglog(result.Ns, ref_y, "--", color="gray", label="slope −1 reference")

    ax.set_xlabel("number of agents N")
    ax.set_ylabel("E‖(1/N) Σ ĝ_i − ∇f‖²")
    ax.set_title(
        f"E1: consensus-SPSA variance vs N "
        f"(fitted slope = {result.slope:.3f})"
    )
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_e2(result: E2Result, out_path: Path) -> None:
    """Family of Var(N) curves for different seed-bank sizes K."""
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    cmap = plt.get_cmap("viridis")
    for ki, K in enumerate(result.Ks):
        label = "K = inf (independent)" if K is None else f"K = {K}"
        color = cmap(ki / max(len(result.Ks) - 1, 1))
        ax.loglog(result.Ns, result.variance[ki], "o-", color=color, label=label)

    # slope -1 reference anchored at the independent-baseline N=1 point
    if None in result.Ks:
        base_idx = result.Ks.index(None)
        ref_y = result.variance[base_idx, 0] * (result.Ns / result.Ns[0]) ** -1.0
        ax.loglog(result.Ns, ref_y, "--", color="gray", alpha=0.6,
                  label="slope -1 reference")

    ax.set_xlabel("number of agents N")
    ax.set_ylabel("E|| (1/N) Sum g_i - grad f ||^2")
    ax.set_title("E2: consensus-SPSA variance vs N for shared seed banks")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_e2_csv(result: E2Result, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["N"] + [
        "K_inf" if K is None else f"K_{K}" for K in result.Ks
    ]
    arr = np.column_stack([result.Ns] + [result.variance[ki] for ki in range(len(result.Ks))])
    fmt = ["%d"] + ["%.10e"] * len(result.Ks)
    np.savetxt(out_path, arr, delimiter=",", header=",".join(header),
               comments="", fmt=fmt)


def plot_e3(result: E3Result, out_path: Path) -> None:
    """Two-panel plot: (a) global-min hit rate vs β, (b) final mean loss vs β."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Use symlog so β=0 is visible alongside log-spaced positive values.
    for ax in (ax1, ax2):
        ax.set_xscale("symlog", linthresh=0.05)
        ax.set_xlabel("beta (reputational selection strength)")
        ax.grid(True, which="both", ls=":", alpha=0.5)

    ax1.plot(result.betas, result.rep_hit_rate, "o-", label="reputational W")
    ax1.plot(result.betas, result.sym_hit_rate, "s--", label="symmetric (control)")
    ax1.set_ylabel("fraction of runs in the global minimum")
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_title("E3: hit rate vs beta")
    ax1.legend()

    ax2.plot(result.betas, result.rep_mean_loss, "o-", label="reputational W")
    ax2.plot(result.betas, result.sym_mean_loss, "s--", label="symmetric (control)")
    ax2.set_ylabel("final mean loss of swarm centroid")
    ax2.set_title("E3: final loss vs beta (lower is better)")
    ax2.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_e3_csv(result: E3Result, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.column_stack([
        result.betas,
        result.rep_hit_rate,
        result.sym_hit_rate,
        result.rep_mean_loss,
        result.sym_mean_loss,
    ])
    np.savetxt(
        out_path, arr, delimiter=",",
        header="beta,rep_hit_rate,sym_hit_rate,rep_mean_loss,sym_mean_loss",
        comments="",
        fmt=["%.4f", "%.6f", "%.6f", "%.6e", "%.6e"],
    )


def save_e1_csv(result: E1Result, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.column_stack([result.Ns, result.variance])
    header = "N,variance"
    np.savetxt(out_path, arr, delimiter=",", header=header, comments="",
               fmt=["%d", "%.10e"])
