"""Test A: perturbation reversibility.

If we apply +eps*z then -eps*z with the SAME seed, parameters must return to
exactly their starting values (up to floating-point noise). This is the load-
bearing invariant of MeZO — if it fails, the seed-based z regeneration is
broken and the optimizer is mathematically nonsense.
"""
import torch

from src.mezo import MeZOOptimizer


def _make_model(seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    )


def test_perturb_round_trip_restores_params():
    model = _make_model()
    opt = MeZOOptimizer(model, lr=1e-6, eps=1e-3)

    before = [p.data.clone() for p in opt.params]

    opt._perturb(scaling=opt.eps, seed=42)
    opt._perturb(scaling=-opt.eps, seed=42)

    for p, b in zip(opt.params, before):
        assert torch.allclose(p.data, b, atol=1e-7), \
            f"perturbation not reversible: max abs diff {(p.data - b).abs().max().item()}"


def test_step_restores_then_updates_along_z():
    """After step(): θ_new == θ_old − lr * g_hat * z, where z is the seed's draw.

    We verify this by reproducing the expected update manually and comparing.
    """
    model = _make_model()
    opt = MeZOOptimizer(model, lr=1e-3, eps=1e-2)

    x = torch.randn(4, 8)
    y = torch.randn(4, 4)

    def loss_fn():
        return ((model(x) - y) ** 2).mean()

    before = [p.data.clone() for p in opt.params]

    # MeZOOptimizer.step() draws its per-step seed via torch.randint;
    # patch that to deterministically control it.
    fixed_seed = 12345
    orig_randint = torch.randint

    def fake_randint(*args, **kwargs):
        return torch.tensor([fixed_seed])

    torch.randint = fake_randint
    try:
        loss_plus_reported = opt.step(loss_fn)
    finally:
        torch.randint = orig_randint

    # Independently compute what step() should have done.
    for p, b in zip(opt.params, before):
        p.data.copy_(b)

    opt._perturb(scaling=opt.eps, seed=fixed_seed)
    lp = loss_fn().item()
    opt._perturb(scaling=-2.0 * opt.eps, seed=fixed_seed)
    lm = loss_fn().item()
    opt._perturb(scaling=opt.eps, seed=fixed_seed)
    g_hat = (lp - lm) / (2.0 * opt.eps)
    opt._perturb(scaling=-opt.lr * g_hat, seed=fixed_seed)

    expected = [p.data.clone() for p in opt.params]

    # Reset and re-run actual step() with the patched seed; result must match.
    for p, b in zip(opt.params, before):
        p.data.copy_(b)

    torch.randint = fake_randint
    try:
        opt.step(loss_fn)
    finally:
        torch.randint = orig_randint

    for p, e in zip(opt.params, expected):
        assert torch.allclose(p.data, e, atol=1e-7)

    assert isinstance(loss_plus_reported, float)
