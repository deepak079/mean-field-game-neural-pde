import numpy as np
import pytest

from mfg.params import Params
from mfg.utility import pi_terminal, utility, utility_x, utility_xx

P = Params(gamma=1.0, lam=2.0)
MBAR = 1.1


def test_derivatives_match_finite_differences_away_from_kink():
    x = np.concatenate([np.linspace(-0.9, 1.0, 20), np.linspace(1.2, 2.9, 20)])
    h = 1e-5
    fd_x = (utility(x + h, MBAR, P) - utility(x - h, MBAR, P)) / (2 * h)
    fd_xx = (utility(x + h, MBAR, P) - 2 * utility(x, MBAR, P) + utility(x - h, MBAR, P)) / h**2
    np.testing.assert_allclose(utility_x(x, MBAR, P), fd_x, rtol=1e-6)
    np.testing.assert_allclose(utility_xx(x, MBAR, P), fd_xx, rtol=1e-4)


def test_terminal_position_drops_just_below_average():
    assert pi_terminal(MBAR + 1e-9, MBAR, P) == pytest.approx(1.5)
    assert pi_terminal(MBAR - 1e-9, MBAR, P) == pytest.approx(0.214, abs=0.001)


def test_terminal_position_crosses_merton_at_gap_one_over_gamma():
    for gamma in (0.5, 1.0, 3.0):
        p = P.with_(gamma=gamma)
        merton = (p.mu - p.r) / (p.gamma * p.sigma**2)
        assert pi_terminal(MBAR - 1 / gamma, MBAR, p) == pytest.approx(merton)
