import numpy as np
import pytest
from scipy.stats import norm

from mfg.benchmarks import population_mean, population_var
from mfg.fd import solve_fokker_planck, solve_hjb
from mfg.params import Params

P = Params()
P0 = P.with_(lam=0.0)
MBAR = 1.1


@pytest.fixture(scope="module")
def hjb():
    return solve_hjb(P, MBAR)


@pytest.fixture(scope="module")
def fp(hjb):
    return solve_fokker_planck(P, hjb.position)


def gaussian_error(nx, nt):
    fp = solve_fokker_planck(P0, solve_hjb(P0, MBAR, nx, nt).position, nx, nt)
    t, x = np.meshgrid(fp.t, fp.x, indexing="ij")
    exact = norm.pdf(x, population_mean(t, P0), np.sqrt(population_var(t, P0)))
    return np.max(np.abs(fp.m - exact)), fp


def test_lam_0_density_is_the_gaussian():
    err, fp = gaussian_error(800, 400)
    var = fp.dx * np.sum(fp.m * (fp.x - fp.mean[:, None]) ** 2, axis=1)
    assert err < 2e-3  # peak density is about 2
    assert np.max(np.abs(fp.mean - population_mean(fp.t, P0))) < 1e-5
    assert np.max(np.abs(var - population_var(fp.t, P0))) < 1e-4


def test_lam_0_converges_at_first_order():
    errors = np.array([gaussian_error(nx, nx // 2)[0] for nx in (400, 800, 1600)])
    ratios = errors[:-1] / errors[1:]
    assert np.all((ratios > 1.5) & (ratios < 2.5))


def test_mass_is_conserved_and_density_is_nonnegative(fp):
    assert np.max(np.abs(fp.mass - 1.0)) < 1e-12
    assert np.all(fp.m >= 0.0)
    assert fp.n_upwind == 0


def test_mean_follows_the_mean_ode(hjb, fp):
    # Implicit Euler: m̄^{n+1} - m̄^n = dt * dx * sum_j b_j(pi^n) m_j^{n+1}, exactly.
    dt = fp.t[1] - fp.t[0]
    b = P.r * fp.x + hjb.pi[:-1] * (P.mu - P.r)
    rhs = dt * fp.dx * np.sum(b * fp.m[1:], axis=1)
    assert np.max(np.abs(np.diff(fp.mean) - rhs)) < 1e-8


def test_expected_value_is_constant_in_time(hjb, fp):
    # The FP matrix is the transpose of the HJB's, so dx * sum_j m_j^n V_j^n doesn't depend on n.
    expected_value = fp.dx * np.sum(fp.m * hjb.V, axis=1)
    assert np.ptp(expected_value) < 1e-6


def test_mass_stays_away_from_the_edges(fp):
    edge_cells = 20  # 0.1 at each end
    assert np.max(fp.dx * fp.m[:, :edge_cells].sum(axis=1)) < 1e-5
    assert np.max(fp.dx * fp.m[:, -edge_cells:].sum(axis=1)) < 1e-5


def test_penalty_bunches_wealth_just_below_the_average(fp):
    cara = solve_fokker_planck(P0, solve_hjb(P0, MBAR).position)
    mode = fp.x[np.argmax(fp.m[-1])]
    assert MBAR - 0.05 < mode < MBAR
    assert fp.m[-1].max() > 3 * cara.m[-1].max()
    # Fewer investors fall far behind: the left tail is thinner, not fatter.
    first_percentile = [s.x[np.searchsorted(s.dx * np.cumsum(s.m[-1]), 0.01)] for s in (fp, cara)]
    assert first_percentile[0] > first_percentile[1] + 0.1


def test_density_broadcasts_and_is_zero_outside_the_grid(fp):
    t = np.array([[0.0], [1.0]])
    x = np.array([-5.0, 1.0, 5.0])
    d = fp.density(t, x)
    assert d.shape == (2, 3)
    assert d[0, 0] == 0.0 and d[1, 2] == 0.0
    assert d[0, 1] == pytest.approx(norm.pdf(1.0, P.m0_mean, P.m0_std), rel=1e-3)
