"""Experiments verifying the three Swarm-MeZO hypotheses (E1 only so far)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .mezo import spsa_estimate
from .objectives import MultiWell, Quadratic, QuadraticWithWells
from .swarm import run_swarm


@dataclass
class E1Result:
    Ns: np.ndarray              # agent counts probed
    variance: np.ndarray        # E‖(1/N)Σĝ_i − ∇f‖²  per N
    slope: float                # log-log regression slope (expected ≈ −1)
    intercept: float
    reps: int
    M: int
    cond_number: float
    eps: float


def run_e1(
    M: int = 200,
    cond_number: float = 100.0,
    eps: float = 1e-3,
    reps: int = 1000,
    Ns: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
    seed: int = 0,
) -> E1Result:
    """E1: variance of the consensus SPSA estimate vs N (H1, the 1/N law).

    Uses *independent* perturbations across agents (no seed-bank) and a fixed θ.
    For each rep we draw ``max(Ns)`` independent ĝ_i; for every N in ``Ns`` the
    consensus estimate is the mean of the first N of them. Variance is reported
    as MSE against the true gradient (which, for a pure quadratic, equals the
    expected value of ĝ — SPSA is unbiased there).
    """
    Ns_arr = np.asarray(sorted(set(Ns)), dtype=int)
    max_N = int(Ns_arr.max())

    obj = Quadratic(M=M, cond_number=cond_number, seed=seed)
    rng = np.random.default_rng(seed + 1)
    theta = rng.standard_normal(M)
    true_g = obj.true_grad(theta)

    # Generate (reps, max_N) independent SPSA estimates once, reuse for all N.
    ests = np.empty((reps, max_N, M), dtype=np.float64)
    for r in range(reps):
        for i in range(max_N):
            ests[r, i] = spsa_estimate(obj, theta, eps=eps, rng=rng)

    variances = np.empty(len(Ns_arr), dtype=np.float64)
    for k, N in enumerate(Ns_arr):
        avg = ests[:, :N, :].mean(axis=1)         # (reps, M)
        err = avg - true_g                         # bias is ~0 for quadratic
        variances[k] = float(np.mean(np.sum(err * err, axis=1)))

    log_N = np.log(Ns_arr.astype(float))
    log_v = np.log(variances)
    slope, intercept = np.polyfit(log_N, log_v, 1)

    return E1Result(
        Ns=Ns_arr,
        variance=variances,
        slope=float(slope),
        intercept=float(intercept),
        reps=reps,
        M=M,
        cond_number=cond_number,
        eps=eps,
    )


@dataclass
class E2Result:
    Ns: np.ndarray                       # agent counts probed
    Ks: tuple[Optional[int], ...]        # seed-bank sizes (None = independent)
    variance: np.ndarray                 # shape (len(Ks), len(Ns))
    reps: int
    M: int
    cond_number: float
    eps: float


def _spsa_batch(objective, theta, eps, zs):
    """Apply SPSA to a stack of perturbations ``zs`` of shape (k, M)."""
    out = np.empty_like(zs)
    for i in range(zs.shape[0]):
        out[i] = spsa_estimate(objective, theta, eps=eps, z=zs[i])
    return out


def run_e2(
    M: int = 200,
    cond_number: float = 100.0,
    eps: float = 1e-3,
    reps: int = 1000,
    Ns: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
    Ks: tuple[Optional[int], ...] = (1, 4, 16, 64, None),
    seed: int = 0,
) -> E2Result:
    """E2: variance of the consensus SPSA estimate when agents share a finite
    seed bank of size K (FedKSeed-style). K=None means each agent draws an
    independent perturbation — baseline equivalent to E1.

    Per rep we sample K fresh perturbations; each of ``max(Ns)`` agents picks
    one of those K uniformly with replacement. Small K forces collisions among
    agents, which prevents the 1/N variance reduction.
    """
    Ns_arr = np.asarray(sorted(set(Ns)), dtype=int)
    max_N = int(Ns_arr.max())
    cum_divisor = np.arange(1, max_N + 1, dtype=np.float64)[:, None]

    obj = Quadratic(M=M, cond_number=cond_number, seed=seed)
    base_rng = np.random.default_rng(seed + 1)
    theta = base_rng.standard_normal(M)
    true_g = obj.true_grad(theta)

    variance = np.empty((len(Ks), len(Ns_arr)), dtype=np.float64)

    for ki, K in enumerate(Ks):
        rng_k = np.random.default_rng(seed + 1000 + (ki + 1) * 17)
        sq_sum = np.zeros(len(Ns_arr), dtype=np.float64)
        for _ in range(reps):
            if K is None:
                zs = rng_k.standard_normal((max_N, M))
                ests = _spsa_batch(obj, theta, eps, zs)
            else:
                zs = rng_k.standard_normal((K, M))
                bank = _spsa_batch(obj, theta, eps, zs)
                picks = rng_k.integers(0, K, size=max_N)
                ests = bank[picks]
            cum = np.cumsum(ests, axis=0) / cum_divisor   # (max_N, M)
            for j, N in enumerate(Ns_arr):
                err = cum[N - 1] - true_g
                sq_sum[j] += float(err @ err)
        variance[ki] = sq_sum / reps

    return E2Result(
        Ns=Ns_arr,
        Ks=tuple(Ks),
        variance=variance,
        reps=reps,
        M=M,
        cond_number=cond_number,
        eps=eps,
    )


@dataclass
class E3Result:
    betas: np.ndarray
    # Mean loss trajectories (averaged across n_runs), shape (len(betas), n_steps)
    rep_loss_curve: np.ndarray
    fedavg_loss_curve: np.ndarray
    # Summary scalars at the end of training
    rep_hit_rate: np.ndarray
    fedavg_hit_rate: np.ndarray
    rep_final_loss: np.ndarray
    fedavg_final_loss: np.ndarray
    n_runs: int
    N: int
    M: int
    n_steps: int
    hit_radius: float


def run_e3(
    N: int = 8,
    M: int = 10,
    n_steps: int = 300,
    eta: float = 0.01,
    eps: float = 1e-3,
    n_runs: int = 50,
    betas: tuple[float, ...] = (0.0, 0.05, 0.1, 0.5, 1.0, 5.0, 50.0),
    init_spread: float = 1.5,
    hit_radius: float = 1.5,
    seed: int = 0,
) -> E3Result:
    """E3: under reputational consensus, sweep β on a mostly-convex landscape
    with a deep global well at origin and two shallower local wells off-axis
    (QuadraticWithWells in dim M; mirrors the locally-smooth structure of a
    prompt-based fine-tuning loss). Track both final hit-rate and the full
    mean-loss trajectory, since the curve shape — speed of descent vs plateau
    vs cascade-induced bump — is what tells us *how* β acts.

    Baseline is FedAvg-MeZO (W = (1/N)·J) — β-independent by construction.
    Same seed per run for both modes, so the difference isolates the matrix.
    """
    obj = QuadraticWithWells(M=M)
    betas_arr = np.asarray(betas, dtype=float)
    init_center_arr = np.zeros(M)

    rep_curve     = np.zeros((len(betas_arr), n_steps))
    fedavg_curve  = np.zeros((len(betas_arr), n_steps))
    rep_hits      = np.zeros(len(betas_arr))
    fedavg_hits   = np.zeros(len(betas_arr))
    rep_final     = np.zeros(len(betas_arr))
    fedavg_final  = np.zeros(len(betas_arr))

    for bi, beta in enumerate(betas_arr):
        rep_finals = []
        fa_finals  = []
        for r in range(n_runs):
            run_seed = seed + 1 + r            # same seed across modes -> same init
            out_rep = run_swarm(
                obj, N=N, n_steps=n_steps, eta=eta, eps=eps,
                consensus_mode="reputation", beta=beta, seed=run_seed,
                init_center=init_center_arr, init_spread=init_spread,
            )
            out_fa = run_swarm(
                obj, N=N, n_steps=n_steps, eta=eta, eps=eps,
                consensus_mode="symmetric", beta=beta, seed=run_seed,
                init_center=init_center_arr, init_spread=init_spread,
            )
            rep_curve[bi]    += np.asarray(out_rep["history"].loss_mean)
            fedavg_curve[bi] += np.asarray(out_fa["history"].loss_mean)
            tb_rep = out_rep["theta_mean"]
            tb_fa  = out_fa["theta_mean"]
            if obj.is_in_global_basin(tb_rep, radius=hit_radius):
                rep_hits[bi] += 1
            if obj.is_in_global_basin(tb_fa, radius=hit_radius):
                fedavg_hits[bi] += 1
            rep_finals.append(obj.value(tb_rep))
            fa_finals.append(obj.value(tb_fa))
        rep_curve[bi]    /= n_runs
        fedavg_curve[bi] /= n_runs
        rep_final[bi]    = float(np.mean(rep_finals))
        fedavg_final[bi] = float(np.mean(fa_finals))

    return E3Result(
        betas=betas_arr,
        rep_loss_curve=rep_curve,
        fedavg_loss_curve=fedavg_curve,
        rep_hit_rate=rep_hits / n_runs,
        fedavg_hit_rate=fedavg_hits / n_runs,
        rep_final_loss=rep_final,
        fedavg_final_loss=fedavg_final,
        n_runs=n_runs,
        N=N,
        M=M,
        n_steps=n_steps,
        hit_radius=hit_radius,
    )
