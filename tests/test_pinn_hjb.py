"""Tests for the value-network PINN (src/mfg/pinn/): formulas against exact solutions, and a short
training run against the lam = 0 closed form."""

import numpy as np
import pytest
import torch

from mfg.benchmarks import merton_pi, merton_value
from mfg.params import Params
from mfg.pinn.hjb import (
    ValueNetSettings,
    edge_targets,
    hjb_residual,
    optimal_position,
    terminal_targets,
    train_value_net,
)
from mfg.pinn.nets import derivative
from mfg.utility import pi_terminal

P = Params()  # base parameters, lam = 2
MBAR = 1.1


def columns(t, x):
    """Column tensors in float64 that track gradients, as the PINN functions expect."""
    t, x = np.broadcast_arrays(np.asarray(t, dtype=float), np.asarray(x, dtype=float))

    def as_column(a):
        return torch.tensor(a.reshape(-1, 1), dtype=torch.float64).requires_grad_()

    return as_column(t), as_column(x)


def merton_v(t, x):
    """Exact v = log(-V) without penalty (lam = 0): -gamma x e^{r tau} - kappa^2 tau / 2."""
    tau = P.T - t
    return -P.gamma * x * torch.exp(P.r * tau) - 0.5 * P.kappa**2 * tau


def quadratic_target_v(t, x):
    """Exact v for the pure penalty utility -(lam/2)(mbar - x)^2, up to a constant.

    By hand: V = -(lam/2) e^{-kappa^2 tau} (mbar - x e^{r tau})^2 solves the HJB, with
    pi* = (mu - r)/sigma^2 (mbar e^{-r tau} - x). Valid below x = mbar e^{-r tau}. Unlike Merton,
    its v_xx is not zero, so it checks the curvature term v_xx + v_x^2.
    """
    tau = P.T - t
    return -(P.kappa**2) * tau + 2 * torch.log(MBAR - x * torch.exp(P.r * tau))


# Points well below mbar e^{-r tau}, where both exact solutions are valid.
T_GRID, X_GRID = np.meshgrid(np.linspace(0.0, 0.9, 4), np.linspace(-1.0, 0.5, 7))


def test_derivative_can_be_taken_twice():
    z = torch.linspace(-2.0, 2.0, 9, dtype=torch.float64).reshape(-1, 1).requires_grad_()
    dy = derivative(torch.sin(z), z)
    d2y = derivative(dy, z)
    torch.testing.assert_close(dy, torch.cos(z))
    torch.testing.assert_close(d2y, -torch.sin(z))


@pytest.mark.parametrize("v_fn", [merton_v, quadratic_target_v])
def test_exact_solutions_have_zero_residual(v_fn):
    t, x = columns(T_GRID, X_GRID)
    residual = hjb_residual(v_fn, t, x, P)
    assert residual.abs().max().item() < 1e-8


def test_exact_solutions_give_their_known_positions():
    t, x = columns(T_GRID, X_GRID)
    tau = P.T - T_GRID.ravel()
    merton = optimal_position(merton_v, t, x, P).detach().numpy().ravel()
    target = optimal_position(quadratic_target_v, t, x, P).detach().numpy().ravel()
    # By hand: (mu - r)/sigma^2 = 1.5, and gamma = 1 in the Merton position.
    np.testing.assert_allclose(merton, 1.5 * np.exp(-P.r * tau), rtol=1e-10)
    target_by_hand = 1.5 * (MBAR * np.exp(-P.r * tau) - X_GRID.ravel())
    np.testing.assert_allclose(target, target_by_hand, rtol=1e-10)


def test_terminal_targets_give_the_exact_final_position():
    x = np.concatenate([np.linspace(-1.0, 1.0, 11), np.linspace(1.2, 3.0, 10)])
    v, v_x, v_xx = terminal_targets(x, MBAR, P)
    position = -(P.mu - P.r) / P.sigma**2 * v_x / (v_xx + v_x**2)
    np.testing.assert_allclose(position, pi_terminal(x, MBAR, P), rtol=1e-10)
    # v_T = log(e^{-x} + (mbar - x)^2) by hand: log(e + 2.1^2) at x = -1, -x above the mean.
    v_hand, _, _ = terminal_targets(np.array([-1.0, MBAR, 3.0]), MBAR, P)
    np.testing.assert_allclose(v_hand, [1.96407, -1.1, -3.0], atol=1e-5)


def test_edge_targets_by_hand_and_at_the_corners():
    low, high = edge_targets(np.array([0.0, P.T]), MBAR, P)
    # t = 0: log(6.86342), the Merton-strategy value from x = -1; -3 e^{0.02} - 0.045 at x = 3.
    # t = T: the final utility, so the edges meet the terminal condition at the corners.
    np.testing.assert_allclose(low, [1.92621, 1.96407], atol=1e-5)
    np.testing.assert_allclose(high, [-3.10560, -3.0], atol=1e-5)


SMALL = dict(width=32, depth=3, n_inside=1000, n_kink=200, n_terminal=200, n_edge=50, device="cpu")


def test_value_net_learns_merton_without_penalty():
    # A small network on the CPU (about 15 s). Checked where the population lives, x in [0, 2.2];
    # observed errors are about 0.02 for pi* and 0.3% for V, so the margins are about three times.
    p = P.with_(lam=0.0)
    sol = train_value_net(p, MBAR, ValueNetSettings(adam_steps=2000, lbfgs_steps=200, **SMALL))
    t, x = np.meshgrid([0.0, 0.5, 0.9], np.linspace(0.0, 2.2, 12))
    assert sol.value(t, x).shape == t.shape
    np.testing.assert_allclose(sol.value(t, x), merton_value(t, x, p), rtol=0.01)
    np.testing.assert_allclose(sol.position(t, x), merton_pi(t, p), atol=0.05)
    # Merton: v_xx + v_x^2 = gamma^2 e^{2 r tau} exactly (observed error 0.007), residual 0 (0.03).
    np.testing.assert_allclose(sol.concavity(t, x), np.exp(2 * p.r * (p.T - t)), atol=0.03)
    np.testing.assert_allclose(sol.residual(t, x), 0.0, atol=0.1)


def test_warm_start_leaves_the_earlier_solution_untouched():
    first = train_value_net(P, MBAR, ValueNetSettings(adam_steps=0, lbfgs_steps=0, **SMALL))
    before = first.value(0.0, np.linspace(-1.0, 3.0, 5))
    train_value_net(P, 1.4, ValueNetSettings(adam_steps=20, lbfgs_steps=0, **SMALL), net=first.net)
    np.testing.assert_array_equal(first.value(0.0, np.linspace(-1.0, 3.0, 5)), before)
