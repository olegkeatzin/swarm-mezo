"""Test B: MeZO must drive a tiny model's loss down on a synthetic task.

This is the most important end-to-end sanity check. If MeZO can't even fit
a 1-layer linear regression, something fundamental is broken (sign, scaling,
seed-reuse, anything).
"""
import torch

from src.mezo import MeZOOptimizer


def test_mezo_converges_on_linear_regression():
    torch.manual_seed(0)

    # Synthetic: y = X W_true + noise. Model: a single Linear layer with no bias.
    n, d_in, d_out = 64, 8, 4
    W_true = torch.randn(d_in, d_out)
    X = torch.randn(n, d_in)
    Y = X @ W_true + 0.01 * torch.randn(n, d_out)

    model = torch.nn.Linear(d_in, d_out, bias=False)
    model.eval()  # no dropout/BN, but explicit for honesty
    opt = MeZOOptimizer(model, lr=1e-2, eps=1e-3)

    def loss_fn():
        with torch.no_grad():
            return ((model(X) - Y) ** 2).mean()

    initial_loss = loss_fn().item()
    final_loss = initial_loss
    for _ in range(3000):
        final_loss = opt.step(loss_fn)

    # MeZO is noisy, so the bar is loose: an order-of-magnitude reduction.
    assert final_loss < 0.5 * initial_loss, \
        f"MeZO did not converge: initial={initial_loss:.4f}, final={final_loss:.4f}"
    # And it shouldn't be wildly bad in absolute terms either.
    assert final_loss < 5.0, f"final loss {final_loss:.4f} too high"
