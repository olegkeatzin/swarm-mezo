"""Unit tests for the vmap-based stacked MeZO helpers in src/federated.py.

These cover the math-critical pieces (perturbation reversibility, update
direction, consensus averaging, weight untying) without spinning up a full
language model. Integration with HuggingFace via vmap+functional_call is
exercised by `scripts/run_fedavg.py`.
"""
import torch

from src.federated import (
    _apply_mezo_update,
    _perturb_stacked,
    _untie_weights_inplace,
    fedavg_consensus,
)


def _stacked_params(n_agents: int = 4, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    return {
        "a.weight": torch.randn(n_agents, 8, 16),
        "a.bias":   torch.randn(n_agents, 8),
        "b.weight": torch.randn(n_agents, 4, 8),
    }


def test_perturb_stacked_round_trip_restores_params():
    """+ε·z then −ε·z with the same seed must return params to their starting values."""
    params = _stacked_params()
    before = {k: v.clone() for k, v in params.items()}
    rng = torch.Generator()

    _perturb_stacked(params, scaling=1e-3, seed=42, rng=rng)
    _perturb_stacked(params, scaling=-1e-3, seed=42, rng=rng)

    for k, v in params.items():
        diff = (v - before[k]).abs().max().item()
        assert diff < 1e-7, f"{k}: not reversible (max abs diff {diff})"


def test_perturb_stacked_three_pass_pattern_restores():
    """+ε, −2ε, +ε with the same seed must net to zero — the actual MeZO pattern."""
    params = _stacked_params()
    before = {k: v.clone() for k, v in params.items()}
    rng = torch.Generator()
    seed = 7

    _perturb_stacked(params,  1e-3, seed, rng)
    _perturb_stacked(params, -2e-3, seed, rng)
    _perturb_stacked(params,  1e-3, seed, rng)

    for k, v in params.items():
        diff = (v - before[k]).abs().max().item()
        assert diff < 1e-6, f"{k}: three-pass not net-zero (diff={diff})"


def test_perturb_stacked_each_agent_gets_independent_z():
    """Different agent slices must get distinct random draws, not identical ones."""
    n = 4
    params = {"w": torch.zeros(n, 1024)}
    rng = torch.Generator()

    _perturb_stacked(params, scaling=1.0, seed=42, rng=rng)

    for i in range(1, n):
        diff = (params["w"][0] - params["w"][i]).abs().max().item()
        assert diff > 0.1, (
            f"agents 0 and {i} got nearly identical z (max abs diff={diff:.2e}); "
            "this would make per-agent perturbation degenerate to a single direction"
        )


def test_apply_mezo_update_matches_explicit_formula():
    """Apply the update, then independently reconstruct -lr·pg·z and compare."""
    n = 3
    params = {"w": torch.zeros(n, 8)}
    rng = torch.Generator()
    pg  = torch.tensor([1.0, -2.0, 0.5])
    lr  = 0.01
    seed = 99

    before = params["w"].clone()
    _apply_mezo_update(params, pg, lr=lr, seed=seed, rng=rng)
    delta = params["w"] - before     # actual delta

    # Reconstruct expected delta: same seed → same z.
    rng.manual_seed(seed)
    z = torch.empty_like(params["w"]).normal_(generator=rng)
    expected = -lr * pg.view(-1, 1) * z

    diff = (delta - expected).abs().max().item()
    assert diff < 1e-7, f"update doesn't match -lr·pg·z (diff={diff})"


def test_apply_mezo_update_broadcasts_over_higher_rank_params():
    """Projected_grad of shape (N,) must broadcast over (N, *param_shape) correctly."""
    n = 2
    params = {"conv": torch.zeros(n, 3, 4, 5)}    # 4-D param
    rng = torch.Generator()
    pg = torch.tensor([2.0, -1.0])
    seed = 1

    before = params["conv"].clone()
    _apply_mezo_update(params, pg, lr=0.1, seed=seed, rng=rng)
    delta = params["conv"] - before   # (n, 3, 4, 5)

    # Per-agent delta should be proportional to its scalar projected_grad.
    rng.manual_seed(seed)
    z = torch.empty_like(params["conv"]).normal_(generator=rng)
    for i in range(n):
        expected_i = -0.1 * pg[i].item() * z[i]
        assert torch.allclose(delta[i], expected_i, atol=1e-7)


def test_fedavg_consensus_replaces_with_mean():
    n = 4
    torch.manual_seed(0)
    params = {"w": torch.randn(n, 3, 5), "b": torch.randn(n, 5)}
    expected_w = params["w"].mean(dim=0).clone()
    expected_b = params["b"].mean(dim=0).clone()

    fedavg_consensus(params)

    for i in range(n):
        assert torch.allclose(params["w"][i], expected_w, atol=1e-6)
        assert torch.allclose(params["b"][i], expected_b, atol=1e-6)


def test_untie_weights_separates_tied_parameters():
    """After untying, two tied parameters must be independent tensors with equal values."""
    m = torch.nn.Module()
    m.a = torch.nn.Linear(4, 4, bias=False)
    m.b = torch.nn.Linear(4, 4, bias=False)
    m.b.weight = m.a.weight       # tie
    assert m.a.weight.data_ptr() == m.b.weight.data_ptr()

    _untie_weights_inplace(m)

    assert m.a.weight.data_ptr() != m.b.weight.data_ptr(), "ties not broken"
    assert torch.equal(m.a.weight, m.b.weight), "values diverged during untie"

    # Mutating one must no longer affect the other.
    m.a.weight.data.fill_(99.0)
    assert not torch.equal(m.a.weight, m.b.weight)


def test_untie_weights_preserves_named_parameters_count():
    """After untying, both copies must show up in named_parameters() (deduplicated)."""
    m = torch.nn.Module()
    m.a = torch.nn.Linear(4, 4, bias=False)
    m.b = torch.nn.Linear(4, 4, bias=False)
    m.b.weight = m.a.weight

    n_before = sum(1 for _ in m.named_parameters())
    _untie_weights_inplace(m)
    n_after  = sum(1 for _ in m.named_parameters())

    assert n_after == n_before + 1, (
        f"expected one extra parameter after untying (was {n_before}, now {n_after})"
    )
