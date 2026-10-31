"""Finite-difference solvers: the reference solutions the PINNs are compared against."""

from mfg.fd.hjb import HJBSolution, solve_hjb

__all__ = ["HJBSolution", "solve_hjb"]
