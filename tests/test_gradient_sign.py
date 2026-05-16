"""Test C: SPSA estimate must point in roughly the same direction as the true gradient.

For each seed:
    projected_grad = (L(θ+εz) − L(θ−εz)) / (2ε)
    SPSA gradient estimate = projected_grad · z
    True gradient g = ∇_θ L(θ)

E[<projected_grad · z, g>] should be > 0 (and in expectation equals ||g||^2
in the SPSA limit ε → 0).

If this average comes out negative, there is a sign bug somewhere — most
likely in the update direction or in projected_grad.
"""
import torch

from src.mezo import MeZOOptimizer


def test_spsa_dot_true_grad_is_positive_in_expectation():
    torch.manual_seed(0)

    d_in, d_out, n = 8, 4, 32
    model = torch.nn.Linear(d_in, d_out, bias=False)
    model.eval()

    X = torch.randn(n, d_in)
    Y = torch.randn(n, d_out)

    def loss_with_grad():
        return ((model(X) - Y) ** 2).mean()

    # Compute the true gradient once at the current θ.
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss_with_grad().backward()
    true_grads = [p.grad.detach().clone() for p in model.parameters()]

    # Snapshot parameters so we can restore between SPSA samples.
    theta0 = [p.data.clone() for p in model.parameters()]

    opt = MeZOOptimizer(model, lr=0.0, eps=1e-4)  # lr=0: don't update during the test

    n_samples = 200
    dots = []
    with torch.no_grad():
        for k in range(n_samples):
            seed = 1000 + k

            opt._perturb(scaling=opt.eps, seed=seed)
            lp = ((model(X) - Y) ** 2).mean().item()
            opt._perturb(scaling=-2.0 * opt.eps, seed=seed)
            lm = ((model(X) - Y) ** 2).mean().item()
            opt._perturb(scaling=opt.eps, seed=seed)  # restore

            g_hat = (lp - lm) / (2.0 * opt.eps)

            # Reconstruct z from the same seed and dot it (scaled by g_hat) with the true grad.
            torch.manual_seed(seed)
            dot = 0.0
            for p, tg in zip(model.parameters(), true_grads):
                z = torch.normal(0.0, 1.0, p.shape, device=p.device, dtype=p.dtype)
                dot += float((g_hat * z * tg).sum().item())
            dots.append(dot)

            # Sanity: params un-touched between samples.
            for p, t0 in zip(model.parameters(), theta0):
                assert torch.allclose(p.data, t0, atol=1e-6)

    mean_dot = sum(dots) / len(dots)
    assert mean_dot > 0, f"<SPSA est, true grad> not positive in expectation: {mean_dot:.4e}"
