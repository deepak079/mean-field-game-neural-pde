"""The search for the equilibrium m̄, shared by the FD and the PINN solvers.

An equilibrium is a fixed point of Phi, the average terminal wealth when everyone best-responds to
m̄: Phi(m̄) = m̄. The search only needs Phi as a function, so it does not care how Phi is
computed (FD solves or trained networks). Appendix C of DS's notes on the HJB and the
Fokker-Planck equation (kept outside the repo).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class FixedPoint:
    mbar: float  # the fixed point: Phi(mbar) = mbar within the tolerance
    solution: Any  # what phi returned alongside Phi at mbar (e.g. the solutions there)
    history: list[tuple[float, float]]  # (m̄, Phi(m̄)) for every evaluation, in order


def secant_fixed_point(
    phi: Callable[[float], tuple[float, Any]],
    start: float,
    tol: float = 1e-10,
    max_iter: int = 20,
) -> FixedPoint:
    """Secant method on g(m̄) = Phi(m̄) - m̄.

    phi(m̄) returns (Phi(m̄), solution), where solution is anything the caller wants back for the
    final m̄. The first two points are start and Phi(start), i.e. one step of plain iteration.
    The search stops when |g| < tol; for a noisy Phi (a PINN), tol should be about the noise.
    """
    history = []

    def g(mbar):
        value, solution = phi(mbar)
        history.append((mbar, value))
        return value - mbar, solution

    a = start
    g_a, _ = g(a)
    b = a + g_a  # = Phi(a)
    g_b, solution = g(b)

    for _ in range(max_iter):
        if abs(g_b) < tol:
            return FixedPoint(b, solution, history)
        if g_b == g_a:  # flat secant: no next point (can happen when Phi is noisy)
            break
        a, g_a, b = b, g_b, b - g_b * (b - a) / (g_b - g_a)
        g_b, solution = g(b)

    raise RuntimeError(f"no fixed point after {len(history)} evaluations: {history}")
