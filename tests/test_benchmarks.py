import numpy as np
import pytest

from mfg.benchmarks import (
    merton_pi,
    merton_strategy_value,
    merton_value,
    population_mean,
    population_var,
)
from mfg.params import Params
from mfg.utility import utility

P = Params(lam=0.0)


def test_merton_numbers_match_proposal():
    assert merton_pi(0.0, P) == pytest.approx(1.47, abs=0.005)
    assert population_mean(P.T, P) == pytest.approx(1.110, abs=0.001)


def test_merton_value_solves_hjb():
    # Residual of V_t + r x V_x - (mu - r)^2 V_x^2 / (2 sigma^2 V_xx), by central differences.
    t, x = np.meshgrid(np.linspace(0.1, 0.9, 9), np.linspace(-0.5, 2.5, 13))
    h = 1e-4
    V = lambda t, x: merton_value(t, x, P)  # noqa: E731
    V_t = (V(t + h, x) - V(t - h, x)) / (2 * h)
    V_x = (V(t, x + h) - V(t, x - h)) / (2 * h)
    V_xx = (V(t, x + h) - 2 * V(t, x) + V(t, x - h)) / h**2
    residual = V_t + P.r * x * V_x - (P.mu - P.r) ** 2 * V_x**2 / (2 * P.sigma**2 * V_xx)
    assert np.max(np.abs(residual)) < 1e-5


def test_population_moments_match_simulation():
    rng = np.random.default_rng(0)
    n, steps = 200_000, 200
    dt = P.T / steps
    X = rng.normal(P.m0_mean, P.m0_std, n)
    for k in range(steps):
        pi = merton_pi(k * dt, P)
        dW = np.sqrt(dt) * rng.standard_normal(n)
        X += (P.r * X + pi * (P.mu - P.r)) * dt + P.sigma * pi * dW
    assert X.mean() == pytest.approx(population_mean(P.T, P), abs=0.005)
    assert X.var() == pytest.approx(population_var(P.T, P), rel=0.02)


def test_merton_strategy_value_reduces_to_merton_value_at_lam_0():
    t, x = np.meshgrid(np.linspace(0.0, 0.9, 4), np.linspace(-1.0, 3.0, 9))
    assert merton_strategy_value(t, x, 1.1, P) == pytest.approx(merton_value(t, x, P))


def test_merton_strategy_value_at_T_is_the_utility():
    p = Params()
    x = np.linspace(-1.0, 3.0, 9)
    assert merton_strategy_value(p.T, x, 1.1, p) == pytest.approx(utility(x, 1.1, p))


@pytest.mark.parametrize("t0, x0", [(0.0, -1.0), (0.5, -1.0), (0.0, 0.8)])
def test_merton_strategy_value_matches_simulation(t0, x0):
    p, mbar = Params(), 1.1
    rng = np.random.default_rng(1)
    n, steps = 200_000, 200
    dt = (p.T - t0) / steps
    X = np.full(n, x0)
    for k in range(steps):
        pi = merton_pi(t0 + k * dt, p)
        dW = np.sqrt(dt) * rng.standard_normal(n)
        X += (p.r * X + pi * (p.mu - p.r)) * dt + p.sigma * pi * dW
    U = utility(X, mbar, p)
    stderr = U.std() / np.sqrt(n)
    assert merton_strategy_value(t0, x0, mbar, p) == pytest.approx(U.mean(), abs=4 * stderr)
