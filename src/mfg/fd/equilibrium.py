"""Mean-field equilibrium with the FD solvers: the m̄ with Phi(m̄) = m̄.

Phi(m̄) is the average terminal wealth when everyone best-responds to m̄: one HJB solve, then one
Fokker-Planck solve under the resulting position. The root of Phi(m̄) - m̄ is found by the
secant method in mfg.equilibrium, shared with the PINN solver. Appendix C of DS's notes on the HJB
and the Fokker-Planck equation (kept outside the repo).
"""

from dataclasses import dataclass

from mfg.benchmarks import population_mean
from mfg.equilibrium import secant_fixed_point
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

    def phi_with_solutions(mbar):
        hjb, fp = best_response(p, mbar, nx, nt)
        return fp.mean[-1], (hjb, fp)

    fixed = secant_fixed_point(phi_with_solutions, population_mean(p.T, p), tol, max_iter)
    hjb, fp = fixed.solution
    return Equilibrium(fixed.mbar, hjb, fp, fixed.history)
