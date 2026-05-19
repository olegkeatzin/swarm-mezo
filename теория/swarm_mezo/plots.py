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
    """Mean loss trajectory per β + FedAvg baseline.

    Each reputational β gets its own curve; FedAvg is β-independent, so we
    plot a single dashed reference line (using the β=0 FedAvg run, which
    equals every other β's FedAvg run up to numerical noise).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    steps = np.arange(1, result.n_steps + 1)

    for bi, beta in enumerate(result.betas):
        color = cmap(bi / max(len(result.betas) - 1, 1))
        ax.plot(steps, result.rep_loss_curve[bi], color=color, lw=1.5,
                label=f"β = {beta:g}")

    ax.plot(steps, result.fedavg_loss_curve[0], "k--", lw=2,
            label="FedAvg-MeZO (W = (1/N)·J)")

    ax.set_xlabel("step")
    ax.set_ylabel("mean swarm loss  (averaged across runs)")
    ax.set_title(
        f"E3: loss trajectories, QuadraticWithWells M={result.M}  "
        f"(N={result.N}, {result.n_runs} runs)"
    )
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_e3_csv(result: E3Result, out_path: Path) -> None:
    """Summary table: per-β hit-rate and final loss for both modes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.column_stack([
        result.betas,
        result.rep_hit_rate,
        result.fedavg_hit_rate,
        result.rep_final_loss,
        result.fedavg_final_loss,
    ])
    np.savetxt(
        out_path, arr, delimiter=",",
        header="beta,rep_hit_rate,fedavg_hit_rate,rep_final_loss,fedavg_final_loss",
        comments="",
        fmt=["%.4f", "%.6f", "%.6f", "%.6e", "%.6e"],
    )


def save_e1_csv(result: E1Result, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.column_stack([result.Ns, result.variance])
    header = "N,variance"
    np.savetxt(out_path, arr, delimiter=",", header=header, comments="",
               fmt=["%d", "%.10e"])
