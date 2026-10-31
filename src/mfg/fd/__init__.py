"""Finite-difference solvers: the reference solutions the PINNs are compared against."""

from mfg.fd.fokker_planck import FPSolution, solve_fokker_planck
from mfg.fd.hjb import HJBSolution, solve_hjb

__all__ = ["FPSolution", "HJBSolution", "solve_fokker_planck", "solve_hjb"]
