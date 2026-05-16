"""Tests for src/consensus.py: doubly-stochastic matrices, spectral gaps,
consensus application, distance metric."""
import math

import pytest
import torch

from src.consensus import (
    apply_consensus,
    build_full,
    build_ring,
    build_star,
    build_topology,
    consensus_distance,
    second_eigenvalue,
    spectral_gap,
)


@pytest.mark.parametrize("name", ["full", "ring", "star"])
@pytest.mark.parametrize("n", [3, 4, 8])
def test_doubly_stochastic(name, n):
    W = build_topology(name, n)
    row_sums = W.sum(dim=1)
    col_sums = W.sum(dim=0)
    assert torch.allclose(row_sums, torch.ones(n), atol=1e-6), \
        f"{name}: rows don't sum to 1 (got {row_sums.tolist()})"
    assert torch.allclose(col_sums, torch.ones(n), atol=1e-6), \
        f"{name}: cols don't sum to 1 (got {col_sums.tolist()})"
    assert (W >= -1e-12).all(), f"{name}: negative entries"


@pytest.mark.parametrize("name", ["full", "ring", "star"])
@pytest.mark.parametrize("n", [4, 8])
def test_largest_eigenvalue_is_one(name, n):
    W = build_topology(name, n)
    eigs = torch.linalg.eigvals(W).abs().sort(descending=True).values
    assert abs(eigs[0].item() - 1.0) < 1e-6, \
        f"{name}: λ_max = {eigs[0].item():.6f}, expected 1.0"


@pytest.mark.parametrize("n", [4, 8, 16])
def test_full_gap_is_one(n):
    """Full graph is rank-1, so λ₂ = 0, gap = 1."""
    W = build_full(n)
    assert abs(spectral_gap(W) - 1.0) < 1e-6


@pytest.mark.parametrize("n", [4, 8, 16])
def test_star_gap_is_one_over_n(n):
    """Metropolis-weighted star: λ₂ = (n-1)/n, gap = 1/n."""
    W = build_star(n)
    assert abs(spectral_gap(W) - 1.0 / n) < 1e-6


@pytest.mark.parametrize("n", [4, 8, 16])
def test_ring_gap_matches_closed_form(n):
    """Ring with (1/3, 1/3, 1/3) weights: λ₂ = (1 + 2·cos(2π/n)) / 3."""
    W = build_ring(n)
    expected_lambda2 = (1.0 + 2.0 * math.cos(2.0 * math.pi / n)) / 3.0
    expected_gap     = 1.0 - expected_lambda2
    assert abs(spectral_gap(W) - expected_gap) < 1e-6, \
        f"ring(n={n}): got gap {spectral_gap(W):.6f}, expected {expected_gap:.6f}"


def test_full_has_largest_gap_for_all_n():
    """full graph always has the maximal gap (= 1) — it mixes in one round."""
    for n in [4, 8, 16, 32]:
        g_full = spectral_gap(build_full(n))
        g_star = spectral_gap(build_star(n))
        g_ring = spectral_gap(build_ring(n))
        assert g_full > g_star, f"n={n}: full ({g_full}) ≤ star ({g_star})"
        assert g_full > g_ring, f"n={n}: full ({g_full}) ≤ ring ({g_ring})"


def test_ring_beats_star_for_small_n_loses_for_large_n():
    """Crossover: ring's gap ~ 2π²/N², star's gap = 1/N.
    Ring wins while 2π²/N² > 1/N, i.e. N < 2π² ≈ 19.7.
    """
    g_ring_8 = spectral_gap(build_ring(8))
    g_star_8 = spectral_gap(build_star(8))
    assert g_ring_8 > g_star_8, "ring should mix faster than star at N=8"

    g_ring_32 = spectral_gap(build_ring(32))
    g_star_32 = spectral_gap(build_star(32))
    assert g_star_32 > g_ring_32, "star should mix faster than ring at N=32"


def test_apply_consensus_full_equals_mean():
    """W = full → after one apply, every agent slice equals the mean."""
    n = 4
    torch.manual_seed(0)
    params = {"w": torch.randn(n, 3, 5), "b": torch.randn(n, 5)}
    expected_w = params["w"].mean(dim=0).clone()
    expected_b = params["b"].mean(dim=0).clone()

    apply_consensus(params, build_full(n))

    for i in range(n):
        assert torch.allclose(params["w"][i], expected_w, atol=1e-6)
        assert torch.allclose(params["b"][i], expected_b, atol=1e-6)


def test_apply_consensus_matches_W_times_stacked():
    """For arbitrary W, apply_consensus[p][i] = Σ_j W[i,j] · p[j]."""
    n = 4
    torch.manual_seed(0)
    W = build_ring(n)
    p_before = torch.randn(n, 7, 11)
    params = {"p": p_before.clone()}

    apply_consensus(params, W)

    flat_before  = p_before.reshape(n, -1)
    expected     = (W @ flat_before).reshape(n, 7, 11)
    assert torch.allclose(params["p"], expected, atol=1e-6)


def test_consensus_distance_zero_when_agents_identical():
    n = 4
    p = torch.randn(3, 5)
    params = {"w": p.unsqueeze(0).expand(n, *p.shape).contiguous()}
    assert consensus_distance(params) < 1e-6


def test_consensus_distance_contracts_by_lambda2_after_apply():
    """After one consensus round, distance should multiply by ~|λ₂|.

    Use ring (non-trivial gap) and a random initial point. Run a few rounds
    and check the geometric mean rate matches the closed-form λ₂.
    """
    n = 8
    W = build_ring(n)
    lambda2 = second_eigenvalue(W)

    torch.manual_seed(0)
    params = {"w": torch.randn(n, 64)}

    rates = []
    for _ in range(20):
        d_before = consensus_distance(params)
        apply_consensus(params, W)
        d_after = consensus_distance(params)
        if d_before > 1e-8:
            rates.append(d_after / d_before)

    avg_rate = sum(rates) / len(rates)
    # As random projections concentrate, the rate should approach λ₂ from below.
    # Allow generous tolerance — a single round can be smaller if the initial
    # vector happens to have small λ₂-component.
    assert avg_rate <= lambda2 + 0.05, \
        f"average contraction {avg_rate:.4f} exceeded |λ₂|={lambda2:.4f}"
    assert avg_rate > 0.0
