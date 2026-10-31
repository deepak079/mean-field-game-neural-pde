import numpy as np
import pytest

from mfg.benchmarks import population_mean
from mfg.fd import phi, solve_equilibrium
from mfg.params import Params

P = Params()
P0 = P.with_(lam=0.0)


@pytest.fixture(scope="module")
def eq():
    return solve_equilibrium(P)


def test_lam_0_equilibrium_is_the_merton_mean():
    # Without the penalty Phi doesn't depend on m̄, so g is a line and the secant is exact.
    eq0 = solve_equilibrium(P0)
    assert eq0.mbar == pytest.approx(population_mean(P0.T, P0), abs=1e-5)
    assert len(eq0.history) == 2


def test_lam_2_equilibrium_is_a_fixed_point(eq):
    assert abs(eq.fp.mean[-1] - eq.mbar) < 1e-10
    assert 1.06 < eq.mbar < 1.07  # below the lam = 0 value 1.110
    assert len(eq.history) <= 6


def test_equilibrium_is_unique_on_a_wide_range():
    # g(m̄) = Phi(m̄) - m̄ on a coarse grid: strictly decreasing, so one zero crossing.
    mbars = np.linspace(0.0, 2.2, 12)
    g = np.array([phi(P, m, nx=200, nt=100) for m in mbars]) - mbars
    assert np.all(np.diff(g) < 0)
    assert g[0] > 0 > g[-1]


def test_equilibrium_converges_under_refinement(eq):
    coarse = [solve_equilibrium(P, nx, nx // 2).mbar for nx in (200, 400)]
    changes = np.abs(np.diff([*coarse, eq.mbar]))
    assert changes[1] < changes[0]
    assert changes[1] < 1e-5
