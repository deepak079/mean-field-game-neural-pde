"""Mean-field equilibrium with the FD solvers: the m̄ with Phi(m̄) = m̄.

Phi(m̄) is the average terminal wealth when everyone best-responds to m̄: one HJB solve, then one
Fokker-Planck solve under the resulting position. The root of Phi(m̄) - m̄ is found by the
secant method. Appendix C of DS's notes on the HJB and the Fokker-Planck equation (kept outside
the repo).
"""

from dataclasses import dataclass

from mfg.benchmarks import population_mean
from mfg.fd.fokker_planck import FPSolution, solve_fokker_planck
from mfg.fd.hjb import HJBSolution, solve_hjb
from mfg.params import Params


@dataclass
class Equilibrium:
    mbar: float  # equilibrium average terminal wealth m̄_T
    hjb: HJBSolution  # best response to mbar
    fp: FPSolution  # population under that best response; fp.mean[-1] equals mbar
    history: list[tuple[float, float]]  # (m̄, Phi(m̄)) for every evaluation, in order


def best_response(p: Params, mbar: float, nx: int = 800, nt: int = 400):
    """HJB solve for a given m̄, then the population's evolution under the optimal position."""
    hjb = solve_hjb(p, mbar, nx, nt)
    return hjb, solve_fokker_planck(p, hjb.position, nx, nt)


def phi(p: Params, mbar: float, nx: int = 800, nt: int = 400) -> float:
    """Average terminal wealth when everyone best-responds to m̄."""
    return best_response(p, mbar, nx, nt)[1].mean[-1]


def solve_equilibrium(
    p: Params, nx: int = 800, nt: int = 400, tol: float = 1e-10, max_iter: int = 20
) -> Equilibrium:
    """Secant method on g(m̄) = Phi(m̄) - m̄, starting from the lam = 0 equilibrium and Phi of it."""
    history = []

    def g(mbar):
        hjb, fp = best_response(p, mbar, nx, nt)
        history.append((mbar, fp.mean[-1]))
        return fp.mean[-1] - mbar, hjb, fp

    a = population_mean(p.T, p)
    g_a, _, _ = g(a)
    b = a + g_a  # = Phi(a): one step of plain iteration gives the second point
    g_b, hjb, fp = g(b)

    for _ in range(max_iter):
        if abs(g_b) < tol:
            return Equilibrium(b, hjb, fp, history)
        a, g_a, b = b, g_b, b - g_b * (b - a) / (g_b - g_a)
        g_b, hjb, fp = g(b)

    raise RuntimeError(f"no equilibrium after {len(history)} evaluations: {history}")
