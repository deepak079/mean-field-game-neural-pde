"""Mean-field equilibrium with the PINNs: the m̄ with Phi(m̄) = m̄.

One evaluation of Phi (pinn_best_response) trains three networks for the given m̄:
    1. the value network (hjb.py), which gives pi*;
    2. the position network, fitted to that pi* (fokker_planck.py);
    3. the density network, under the position network;
and reads Phi from the mean equation dM/dt = r M + (mu - r) E_m[pi*], with the value network's pi*
(DensityNetSolution.mean_by_equation): about ten times more accurate than the density's own mean.

The search is the secant method shared with the FD equilibrium (mfg.equilibrium), from the lam = 0
mean. The first evaluation trains from scratch with the full settings; each later one starts from
the networks of the one before (warm start) and trains briefly, since m̄ moves only a little.
Phi from networks carries training noise, so the search stops at tol, about that noise, and not at
the FD solver's 1e-10.

Result at lam = 2 with the default settings: m̄ = 1.064982 against FD's 1.065128 (error -1.5e-4),
after 3 evaluations in 60 min (43 min cold, then about 9 min each warm). The PINN Phi is low by a
steady 1.3e-4 to 1.8e-4 at every point visited, a bias rather than noise, which moves the fixed
point by about bias / (1 - Phi'), Phi' = -0.065. Joint training (one loss over both networks and
m̄) is left out.
"""

import time
from dataclasses import dataclass, field

from mfg.benchmarks import population_mean
from mfg.equilibrium import secant_fixed_point
from mfg.params import Params
from mfg.pinn.fokker_planck import (
    DensityNetSettings,
    DensityNetSolution,
    PositionNetSettings,
    fit_position_net,
    train_density_net,
)
from mfg.pinn.hjb import ValueNetSettings, ValueNetSolution, train_value_net


@dataclass
class Stage:
    """Settings of the three networks for one evaluation of Phi."""

    value: ValueNetSettings = field(default_factory=ValueNetSettings)
    position: PositionNetSettings = field(default_factory=PositionNetSettings)
    density: DensityNetSettings = field(default_factory=DensityNetSettings)


def warm_stage() -> Stage:
    """Short training from the previous evaluation's networks, at a smaller step size."""
    return Stage(
        value=ValueNetSettings(adam_steps=2000, lr_start=3e-4, lr_end=1e-4, lbfgs_steps=200),
        position=PositionNetSettings(adam_steps=5000, lr_start=3e-4, lbfgs_steps=200),
        density=DensityNetSettings(adam_steps=2000, lr_start=3e-4, lr_end=1e-4, lbfgs_steps=300),
    )


@dataclass
class PinnEquilibriumSettings:
    first: Stage = field(default_factory=Stage)  # the first evaluation, from scratch
    warm: Stage = field(default_factory=warm_stage)  # every later one
    tol: float = 2e-4  # stop when |Phi(m̄) - m̄| < tol
    max_iter: int = 8
    verbose: bool = False


@dataclass
class BestResponse:
    """Everything one evaluation of Phi produced."""

    mbar: float  # the assumed final mean
    phi: float  # the population's final mean when everyone best-responds to mbar
    value: ValueNetSolution
    position_net: object  # the frozen network fitted to value.position
    position_rms: float  # its misfit
    density: DensityNetSolution
    seconds: float


def pinn_best_response(p: Params, mbar, stage: Stage, previous: BestResponse | None = None):
    """Train the three networks for mbar (from previous's networks, if given) and read Phi."""
    start = time.time()
    value = train_value_net(p, mbar, stage.value, net=previous and previous.value.net)
    position_net, rms = fit_position_net(
        value.position, p, mbar, stage.position, net=previous and previous.position_net
    )
    density = train_density_net(
        p, position_net, stage.density, net=previous and previous.density.net
    )
    phi = density.mean_by_equation(value.position)
    return BestResponse(mbar, phi, value, position_net, rms, density, time.time() - start)


@dataclass
class PinnEquilibrium:
    mbar: float  # equilibrium average terminal wealth m̄_T
    response: BestResponse  # the networks at mbar
    history: list[tuple[float, float]]  # (m̄, Phi(m̄)) for every evaluation, in order
    seconds: float


def solve_pinn_equilibrium(
    p: Params,
    settings: PinnEquilibriumSettings | None = None,
    previous: BestResponse | None = None,
) -> PinnEquilibrium:
    """Secant method on Phi(m̄) - m̄ with PINN evaluations of Phi.

    previous: networks to start from (for example the equilibrium at a nearby lam); then every
    evaluation, the first one included, uses the warm settings.
    """
    s = settings or PinnEquilibriumSettings()
    start = time.time()
    last = [previous]

    def phi(mbar):
        stage = s.first if last[0] is None else s.warm
        response = pinn_best_response(p, mbar, stage, last[0])
        last[0] = response
        if s.verbose:
            print(f"m̄ = {mbar:.6f}   Phi = {response.phi:.6f}   ({response.seconds:.0f} s)")
        return response.phi, response

    fixed = secant_fixed_point(phi, population_mean(p.T, p), s.tol, s.max_iter)
    return PinnEquilibrium(fixed.mbar, fixed.solution, fixed.history, time.time() - start)
