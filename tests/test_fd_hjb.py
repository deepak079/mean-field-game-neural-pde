import numpy as np
import pytest

from mfg.benchmarks import merton_pi, merton_strategy_value, merton_value
from mfg.fd import solve_hjb
from mfg.params import Params
from mfg.utility import pi_terminal, utility

P = Params()
P0 = P.with_(lam=0.0)
MBAR = 1.1


@pytest.fixture(scope="module")
def sol():
    return solve_hjb(P, MBAR)


def merton_errors(nx, nt):
    s = solve_hjb(P0, MBAR, nx, nt)
    t, x = np.meshgrid(s.t, s.x, indexing="ij")
    return np.max(np.abs(s.V - merton_value(t, x, P0))), np.max(np.abs(s.pi - merton_pi(t, P0)))


def test_lam_0_matches_merton():
    err_V, err_pi = merton_errors(800, 400)
    assert err_V < 5e-6
    assert err_pi < 5e-4


def test_lam_0_converges_at_first_order():
    # Halving dx and dt together: implicit Euler's O(dt) error dominates, so errors halve.
    errors = np.array([merton_errors(nx, nx // 2) for nx in (400, 800, 1600)])
    ratios = errors[:-1] / errors[1:]
    assert np.all((ratios > 1.5) & (ratios < 2.5))


def test_value_lies_between_merton_strategy_and_cara(sol):
    # Optimal beats holding Merton; the penalty only lowers utility. Tolerance: O(dt) error.
    t, x = np.meshgrid(sol.t, sol.x, indexing="ij")
    assert np.all(sol.V >= merton_strategy_value(t, x, MBAR, P) - 2e-6)
    assert np.all(sol.V <= merton_value(t, x, P))


def test_value_is_increasing_and_concave_with_no_guards_triggered(sol):
    assert np.all(np.diff(sol.V, axis=1) > 0)
    assert np.all(sol.V[:, 2:] - 2 * sol.V[:, 1:-1] + sol.V[:, :-2] < 0)
    assert sol.n_clipped == 0
    assert sol.n_upwind == 0


def test_policy_iteration_converges_every_step(sol):
    assert sol.iterations.max() < 20


def test_position_one_step_before_T_matches_terminal_formula(sol):
    # Away from the kink and from the edge node, where pi is copied and V is pinned to V_fix.
    keep = (np.abs(sol.x - MBAR) > 0.3) & (sol.x > P.x_min + 0.1) & (sol.x < P.x_max)
    assert np.max(np.abs(sol.pi[-2] - pi_terminal(sol.x, MBAR, P))[keep]) < 2e-3


def test_lagged_policy_agrees_with_full_policy_iteration(sol):
    lagged = solve_hjb(P, MBAR, max_iter=1)
    assert np.max(np.abs(lagged.V - sol.V)) < 2e-5
    # The lag is O(1) only at the kink in the last steps before T, where pi jumps.
    early = sol.t <= 0.9
    assert np.max(np.abs(lagged.pi - sol.pi)[early]) < 2e-3


def test_moving_the_lower_boundary_leaves_the_population_region_unchanged(sol):
    wider = solve_hjb(P.with_(x_min=-2.0), MBAR, nx=1000)  # same dx
    x = np.linspace(0.0, 2.2, 111)
    assert np.max(np.abs(wider.value(0.0, x) - sol.value(0.0, x))) < 1e-4
    assert np.max(np.abs(wider.position(0.0, x) - sol.position(0.0, x))) < 5e-3


def test_value_and_position_broadcast_like_t_plus_x(sol):
    t = np.array([[0.0], [0.5]])
    x = np.array([0.0, 1.0, 2.0])
    assert sol.value(t, x).shape == (2, 3)
    assert sol.position(0.5, x).shape == (3,)
    assert sol.value(P.T, 0.5) == pytest.approx(utility(0.5, MBAR, P), abs=1e-6)
