"""PINN for the finite-N game ("league of N investors"): N investors who see each other's wealth.

The game
    Investor i has wealth X^i (its own Brownian motion) and maximises E[U(X^i_T, xbar_T)], with
    xbar_T the final mean of all N wealths, i included, and U the shortfall utility of utility.py.
    Written with the mean ybar of my N - 1 rivals, the penalty is the 1-D one with
    lam_N = lam (1 - 1/N)^2 (benchmarks.lambda_n), so the 1-D utility functions are reused.
    Everyone uses the same feedback rule pi(t, x, y) from their own seat: a symmetric Markov Nash
    equilibrium.

What the network learns
    v(t, x, y) = log(-V), with x my wealth and y the N - 1 rivals' wealths, by one SymmetricNet
    (nets.py) per N. Its HJB in v-form is the 1-D one plus two terms per rival:

        v_t + r x v_x - (kappa^2/2) v_x^2 / (v_xx + v_x^2)
            + sum_j [ (r y_j + pi_j (mu - r)) v_{y_j}
                      + (sigma^2/2) pi_j^2 (v_{y_j y_j} + v_{y_j}^2) ] = 0
        v(T, x, y) = log( exp(-gamma x) + (lam_N/2) ((ybar - x)^+)^2 )

    where pi_j = pi(t, y_j, y^(j)) is the rule read from rival j's seat (its own wealth in the
    first slot, my x and the other rivals in the rivals' slots). My best response is, as in 1-D,
    pi* = -(mu - r)/sigma^2 v_x / (v_xx + v_x^2). There is no Fokker-Planck equation: the rivals
    are in the state, and the group is measured by Monte Carlo (montecarlo.py).

Rounds (fictitious play, as in Han and Hu 2020)
    During a round the rivals' rule is frozen: Merton in round 1, then a frozen copy of the
    network of the round before. The round trains the network to be the best response (round 1
    from scratch, later rounds briefly and warm-started). The change of the rule between rounds,
    measured on fixed check states, decides when to stop: from round 3, as soon as it no longer
    halves. The result is accepted if the last change is below accept_delta.

Loss (three parts, each a mean of squares, coefficients 1)
    equation  the residual above divided by kappa^2 / 2, on group points (type A, a realistic
              group), far points (type B, one wealth far away) and kink points (near T, my wealth
              near ybar); the rivals' positions are data during the round.
    terminal  at t = T, v equals the final utility.
    slopes    at t = T, my own v_x and v_xx equal their exact values, away from the kink at ybar.
    No box and no edges: the points are drawn around the lam = 0 population instead (sample_batch),
    and results are read only on states like the trained ones (check_states, quantile rivals).

Checks: tests/test_nplayer.py (exact C2 solution, seats, symmetry), scripts/run_nplayer.py
--check c2 (trained C2 at N = 5) and the C3 numbers saved in each round's history.
"""

import copy
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace

import numpy as np
import torch

from mfg.benchmarks import (
    c2_k,
    c2_position,
    lambda_n,
    merton_pi,
    population_mean,
    population_var,
)
from mfg.params import Params
from mfg.pinn.hjb import MIN_CURVATURE, terminal_targets
from mfg.pinn.nets import SymmetricNet, pick_device
from mfg.utility import pi_terminal

# ---------------------------------------------------------------- settings


@dataclass
class PointSettings:
    """Points drawn at every Adam step (reasons: module docstring and the guide, section 8)."""

    n_group: int = 2000  # type A: a realistic group, all N wealths from one N(centre, width^2)
    n_far: int = 2000  # type B: a group with one wealth (mine with probability 1/2) far away
    n_kink: int = 1000  # near T, my wealth near the rivals' mean, where pi changes fastest
    n_terminal: int = 1000  # at t = T, half of type A and half of type B
    centre_shift: float = 0.3  # the group's centre is M(t) + U[-shift, shift]...
    spread: tuple[float, float] = (0.5, 1.5)  # ...and its width S(t) times U[spread]
    far: tuple[float, float] = (-1.5, 3.5)  # the far wealth of type B is uniform here
    kink_dt: float = 0.1  # kink points have t in [T - kink_dt, T]...
    kink_dx: float = 0.3  # ...and x in ybar + [-kink_dx, kink_dx]
    kink_band: float = 0.05  # at t = T, slopes are not matched closer than this to ybar

    def scaled(self, factor):
        """The same with `factor` times the points per step (plan B for a slow N)."""
        counts = {
            name: max(1, round(getattr(self, name) * factor))
            for name in ("n_group", "n_far", "n_kink", "n_terminal")
        }
        return replace(self, **counts)


@dataclass
class StageSettings:
    """Training of one round: Adam with a step size that shrinks exponentially, then L-BFGS."""

    adam_steps: int = 10_000
    lr_start: float = 1e-3
    lr_end: float = 1e-4
    lbfgs_steps: int = 500  # on one fixed batch, with the guard of train_round (0 to skip)


def warm_stage():
    """Later rounds: a short training from the network of the round before, as in 1-D."""
    return StageSettings(adam_steps=2000, lr_start=3e-4, lr_end=1e-4, lbfgs_steps=200)


@dataclass
class NetSettings:
    """Size of the SymmetricNet (nets.py)."""

    features: int = 32  # numbers that summarise one rival
    summary_width: int = 32  # width of the summary network's two hidden layers
    width: int = 64  # final network: neurons per hidden layer...
    depth: int = 4  # ...and hidden layers


@dataclass
class NPlayerSettings:
    """Everything that is chosen for a run; configs/nplayer.yaml fills it (settings_from_dict)."""

    net: NetSettings = field(default_factory=NetSettings)
    points: PointSettings = field(default_factory=PointSettings)
    first_round: StageSettings = field(default_factory=StageSettings)
    later_rounds: StageSettings = field(default_factory=warm_stage)
    max_rounds: int = 6
    min_rounds: int = 3  # the stop rule needs two changes to compare
    accept_delta: float = 0.02  # accepted if the last change of the rule is below this
    n_check: int = 20_000  # check states: how many...
    check_seed: int = 1  # ...drawn with this seed...
    check_t_max: float = 0.9  # ...with t in [0, check_t_max]
    coef_equation: float = 1.0  # coefficients of the three parts of the loss
    coef_terminal: float = 1.0
    coef_slopes: float = 1.0
    rival_clip: tuple[float, float] = (0.0, 10.0)  # the rivals' positions are kept in this range
    seed: int = 0  # fixes the initial network and the points
    device: str | None = None  # None: the GPU if there is one
    print_every: int = 0  # print the loss every this many Adam steps (0: silent)


def settings_from_dict(d, cls=NPlayerSettings):
    """Rebuild a settings dataclass from a plain dict (a YAML section or dataclasses.asdict)."""
    values = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        value = d[f.name]
        if is_dataclass(f.type) and isinstance(value, dict):
            value = settings_from_dict(value, f.type)
        elif isinstance(value, list):
            value = tuple(value)
        values[f.name] = value
    return cls(**values)


# ---------------------------------------------------------------- seats


def others_table(N):
    """Row i: the indices of the other N - 1 wealths, in order. Built once per N."""
    return np.array([[k for k in range(N) if k != i] for i in range(N)])


def seats(w):
    """The seat of each of the N investors whose wealths are the columns of w (n, N).

    Returns their own wealth (n, N) and their rivals' wealths (n, N, N - 1), seat by seat, in the
    column order of w. Works on NumPy arrays and on torch tensors. This is the one place that
    builds seats; training and the Monte Carlo both use it.
    """
    idx = others_table(w.shape[1])
    if torch.is_tensor(w):
        idx = torch.as_tensor(idx, device=w.device)
    return w, w[:, idx]


def rival_seats(x, y):
    """The seats of my N - 1 rivals, from my wealth x (n, 1) and theirs y (n, N - 1).

    Rival j's own wealth is y_j; its rivals are my x and the other N - 2 rivals. Returns own
    (n, N - 1) and others (n, N - 1, N - 1).
    """
    cat = torch.cat if torch.is_tensor(x) else np.concatenate
    own, others = seats(cat([x, y], 1))
    return own[:, 1:], others[:, 1:]


# ---------------------------------------------------------------- derivatives and the residual


def _grad(y, z, create_graph):
    """dy/dz (sum over the outputs), or 0 if y does not depend on z."""
    if not y.requires_grad:
        return torch.zeros_like(z)
    grad = torch.autograd.grad(
        y, z, torch.ones_like(y), create_graph=create_graph, allow_unused=True
    )[0]
    return torch.zeros_like(z) if grad is None else grad


def v_derivatives(v_fn, t, x, y):
    """v and its derivatives v_t, v_x, v_xx, v_y (n, N - 1) and v_yy (n, N - 1) at (t, x, y).

    v_yy holds the diagonal second derivatives v_{y_j y_j} only; the cross derivatives do not
    appear in the equation (each wealth has its own noise). v_fn is the network or any torch
    function of (t, x, y); t, x, y are tensors with requires_grad=True. One pass gives v_t, v_x
    and all the v_{y_j}; v_xx takes another; each v_{y_j y_j} one more. These N - 1 passes are
    what grows with N.
    """
    v = v_fn(t, x, y)
    v_t, v_x, v_y = (_grad(v, z, create_graph=True) for z in (t, x, y))
    v_xx = _grad(v_x, x, create_graph=True)
    v_yy = torch.stack(
        [_grad(v_y[:, j], y, create_graph=True)[:, j] for j in range(y.shape[1])], dim=1
    )
    return v, v_t, v_x, v_xx, v_y, v_yy


def residual_formula(x, y, pi_rivals, v_t, v_x, v_xx, v_y, v_yy, p: Params):
    """The HJB residual in v-form, divided by kappa^2 / 2, from the derivatives of v.

    R = [v_t + r x v_x + sum_j ((r y_j + pi_j (mu - r)) v_{y_j} + (sigma^2/2) pi_j^2 (v_{y_j y_j}
    + v_{y_j}^2))] / (kappa^2/2) - v_x^2 / max(v_xx + v_x^2, MIN_CURVATURE). pi_rivals (n, N - 1)
    are the rivals' positions, data during a round. The division and the curvature floor are
    those of the 1-D residual (hjb.residual_formula).
    """
    drift = (p.r * y + (p.mu - p.r) * pi_rivals) * v_y
    noise = 0.5 * p.sigma**2 * pi_rivals**2 * (v_yy + v_y**2)
    rivals = (drift + noise).sum(dim=1, keepdim=True)
    curvature = torch.clamp(v_xx + v_x**2, min=MIN_CURVATURE)
    return (v_t + p.r * x * v_x + rivals) / (0.5 * p.kappa**2) - v_x**2 / curvature


def hjb_residual(v_fn, pi_rivals, t, x, y, p: Params):
    """HJB residual (see residual_formula) of v_fn at (t, x, y) with the rivals at pi_rivals."""
    _, v_t, v_x, v_xx, v_y, v_yy = v_derivatives(v_fn, t, x, y)
    return residual_formula(x, y, pi_rivals, v_t, v_x, v_xx, v_y, v_yy, p)


def position_formula(v_x, v_xx, p: Params):
    """pi* = -(mu - r) / sigma^2 * v_x / (v_xx + v_x^2), the dollars in the stock (not clipped)."""
    return -(p.mu - p.r) / p.sigma**2 * v_x / (v_xx + v_x**2)


# ---------------------------------------------------------------- what the rivals invest


def merton_position(t, p: Params):
    """The Merton position (benchmarks.merton_pi) for a torch tensor t."""
    return (p.mu - p.r) / (p.gamma * p.sigma**2) * torch.exp(-p.r * (p.T - t))


def rival_positions(v_fn, t, x, y, p: Params, clip=(0.0, 10.0)):
    """What each rival invests at (t, x, y): the rule of v_fn read from every rival's seat.

    v_fn(t, own, others) is a frozen network or an exact formula. At a seat only the own wealth
    carries a gradient; the others are data. pi_j = clip(-(mu - r)/sigma^2 v_x / max(v_xx + v_x^2,
    MIN_CURVATURE)): the floor is the 1-D safety net and the clip the FD one, and both should
    never act. Returns pi (n, N - 1) without gradient, and how many positions hit the floor and
    the cap of the clip.
    """
    n, n_rivals = y.shape
    own, others = rival_seats(x.detach(), y.detach())
    own = own.reshape(-1, 1).clone().requires_grad_(True)
    others = others.reshape(n * n_rivals, n_rivals)
    t_seat = t.detach().repeat_interleave(n_rivals, dim=0)
    with torch.enable_grad():
        v = v_fn(t_seat, own, others)
        v_x = _grad(v, own, create_graph=True)
        v_xx = _grad(v_x, own, create_graph=False)
    curvature = torch.clamp(v_xx + v_x**2, min=MIN_CURVATURE)
    raw = (-(p.mu - p.r) / p.sigma**2 * v_x / curvature).detach()
    n_floor, n_cap = int((raw < clip[0]).sum()), int((raw > clip[1]).sum())
    return raw.clamp(*clip).reshape(n, n_rivals), n_floor, n_cap


class MertonRivals:
    """Round 1: every rival holds the Merton position (exact, smooth, nothing trained)."""

    n_floor = n_cap = 0

    def __init__(self, p: Params):
        self.p = p

    def __call__(self, t, x, y):
        return merton_position(t, self.p) * torch.ones_like(y)


class FrozenRivals:
    """Rivals who play the rule of a frozen copy of a network (the round before).

    The copy's weights carry no gradient, so a round trains the best response to fixed rivals.
    n_floor and n_cap count the clips of rival_positions over the round (they should stay 0).
    """

    def __init__(self, net, p: Params, clip=(0.0, 10.0)):
        self.net = copy.deepcopy(net)
        for w in self.net.parameters():
            w.requires_grad_(False)
        self.p, self.clip = p, clip
        self.n_floor = self.n_cap = 0

    def __call__(self, t, x, y):
        pi, n_floor, n_cap = rival_positions(self.net, t, x, y, self.p, self.clip)
        self.n_floor += n_floor
        self.n_cap += n_cap
        return pi


# ---------------------------------------------------------------- the final condition


class ShortfallTerminal:
    """The final condition of our game: v_T = log(-U(x, ybar)) with lam_N, and its slopes.

    targets(x, y) gives v_T, v_T', v_T'' (NumPy, from hjb.terminal_targets) and `away`, the
    points further than kink_band from ybar, where the slopes are matched (at ybar U has a kink
    and v_xx jumps). position(x, y) is the exact position at t = T (utility.pi_terminal), used by
    check C3. The C2 check replaces this object by its own (scripts/run_nplayer.py).
    """

    def __init__(self, p: Params, N, kink_band=0.05):
        self.p_n = p.with_(lam=lambda_n(p.lam, N))
        self.kink_band = kink_band

    def targets(self, x, y):
        ybar = y.mean(axis=1, keepdims=True)
        v_T, v_T_x, v_T_xx = terminal_targets(x, ybar, self.p_n)
        return v_T, v_T_x, v_T_xx, np.abs(x - ybar) > self.kink_band

    def position(self, x, y):
        return pi_terminal(x, y.mean(axis=1, keepdims=True), self.p_n)


class C2Terminal:
    """The final condition of check C2: the original utility -exp(-gamma (x - eta xbar)).

    v_T = -gamma [(1 - eta/N) x - (eta/N) sum_j y_j] is linear, so v_T' = -gamma (1 - eta/N),
    v_T'' = 0 and there is no kink: slopes are matched everywhere. The exact solution is in
    benchmarks.py (c2_value, c2_position).
    """

    def __init__(self, p: Params, N, eta=0.5):
        self.p, self.N, self.eta = p, N, eta

    def targets(self, x, y):
        a, b = 1.0 - self.eta / self.N, self.eta / self.N
        v_T = -self.p.gamma * (a * x - b * y.sum(axis=1, keepdims=True))
        slope = np.full_like(v_T, -self.p.gamma * a)
        return v_T, slope, np.zeros_like(v_T), np.ones_like(v_T, bool)

    def position(self, x, y):
        return np.full_like(x, c2_position(self.p.T, self.N, self.eta, self.p))

    def value(self, t, x, y):
        """The exact v = log(-V) in torch, for tests and for the C2 error (benchmarks.c2_value)."""
        a, b = 1.0 - self.eta / self.N, self.eta / self.N
        tau = self.p.T - t
        linear = a * x - b * y.sum(dim=1, keepdim=True)
        return (
            -self.p.gamma * torch.exp(self.p.r * tau) * linear
            + self.p.kappa**2 * c2_k(self.N, self.eta) * tau
        )


# ---------------------------------------------------------------- the points


def _population(t, p: Params):
    """Mean M(t) and standard deviation S(t) of the lam = 0 population (benchmarks.py)."""
    return population_mean(t, p), np.sqrt(population_var(t, p))


def _group(rng, t, N, p: Params, s: PointSettings):
    """Type A: the N wealths of a realistic group at the times t (n, 1), mine first.

    The centre and the width of the group are drawn first, around M(t) and S(t), then every
    wealth from N(centre, width^2). Drawing each wealth widely on its own would almost never
    give a realistic group (19 rivals within 0.3 of a centre: probability 2e-10).
    """
    n = len(t)
    M, S = _population(t, p)
    centre = M + rng.uniform(-s.centre_shift, s.centre_shift, (n, 1))
    width = S * rng.uniform(*s.spread, (n, 1))
    return centre + width * rng.standard_normal((n, N))


def _far(rng, w, s: PointSettings):
    """Type B: in each group one wealth is replaced by a uniform draw on s.far: mine with
    probability 1/2, otherwise a random rival's (each rival sees me as its far rival)."""
    n, N = w.shape
    column = np.where(rng.random(n) < 0.5, 0, rng.integers(1, N, n))
    w = w.copy()
    w[np.arange(n), column] = rng.uniform(*s.far, n)
    return w


def sample_batch(p: Params, N, s: PointSettings, rng, device, terminal, rivals, dtype=None):
    """Fresh points and their data: equation points (types A and B, then the kink points), and
    terminal points with their targets from `terminal.targets` (ShortfallTerminal or C2Terminal).

    rivals(t, x, y) gives the rivals' positions (n, N - 1) on device tensors; they are computed
    here once, since they are data during a round. Drawn with the NumPy generator rng (so that
    the lam = 0 population formulas of benchmarks.py are reused) and moved to the device.
    """
    dtype = dtype or torch.float32
    t_group = rng.uniform(0.0, p.T, (s.n_group, 1))
    w_group = _group(rng, t_group, N, p, s)
    t_far = rng.uniform(0.0, p.T, (s.n_far, 1))
    w_far = _far(rng, _group(rng, t_far, N, p, s), s)
    t_kink = rng.uniform(p.T - s.kink_dt, p.T, (s.n_kink, 1))
    w_kink = _group(rng, t_kink, N, p, s)
    w_kink[:, :1] = w_kink[:, 1:].mean(axis=1, keepdims=True) + rng.uniform(
        -s.kink_dx, s.kink_dx, (s.n_kink, 1)
    )
    n_a = s.n_terminal // 2
    t_T = np.full((s.n_terminal, 1), p.T)
    w_T = np.concatenate(
        [_group(rng, t_T[:n_a], N, p, s), _far(rng, _group(rng, t_T[n_a:], N, p, s), s)]
    )
    v_T, v_T_x, v_T_xx, away = terminal.targets(w_T[:, :1], w_T[:, 1:])

    def to_device(a, kind=dtype):
        return torch.as_tensor(np.asarray(a), dtype=kind, device=device)

    w_eq = np.concatenate([w_group, w_far, w_kink])
    batch = {
        "t": to_device(np.concatenate([t_group, t_far, t_kink])),
        "x": to_device(w_eq[:, :1]),
        "y": to_device(w_eq[:, 1:]),
        "t_T": to_device(t_T),
        "x_T": to_device(w_T[:, :1]),
        "y_T": to_device(w_T[:, 1:]),
        "v_T": to_device(v_T),
        "v_T_x": to_device(v_T_x),
        "v_T_xx": to_device(v_T_xx),
        "away": to_device(away, torch.bool),
    }
    batch["pi_rivals"] = rivals(batch["t"], batch["x"], batch["y"])
    return batch


def check_states(p: Params, N, n=20_000, seed=1, t_max=0.9):
    """Fixed states on which the rule is compared between rounds and with exact solutions.

    t uniform on [0, t_max] (near T the rule at the kink is not reliable, as in 1-D) and the N
    wealths drawn independently from N(M(t), S(t)^2), the lam = 0 population. NumPy float64
    arrays t, x of shape (n,) and y of shape (n, N - 1), ready for NPlayerSolution.position; the
    same seed gives the same states.
    """
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, t_max, (n, 1))
    M, S = _population(t, p)
    w = M + S * rng.standard_normal((n, N))
    return {"t": t.ravel(), "x": w[:, 0], "y": w[:, 1:]}


# ---------------------------------------------------------------- the loss


def nplayer_loss(v_fn, batch, p: Params, s: NPlayerSettings):
    """The loss (the three parts times their coefficients) and the parts themselves.

    v_fn is the network (or an exact function for tests). The terminal points only need v, v_x
    and v_xx, so they go through the network apart, without the rivals' second derivatives.
    """
    t, x, y = (batch[k].requires_grad_() for k in ("t", "x", "y"))
    _, v_t, v_x, v_xx, v_y, v_yy = v_derivatives(v_fn, t, x, y)
    residual = residual_formula(x, y, batch["pi_rivals"], v_t, v_x, v_xx, v_y, v_yy, p)
    equation = residual.pow(2).mean()

    x_T = batch["x_T"].requires_grad_()
    v = v_fn(batch["t_T"], x_T, batch["y_T"])
    v_T_x = _grad(v, x_T, create_graph=True)
    v_T_xx = _grad(v_T_x, x_T, create_graph=True)
    terminal = (v - batch["v_T"]).pow(2).mean()
    away = batch["away"]
    slopes = ((v_T_x - batch["v_T_x"])[away] ** 2 + (v_T_xx - batch["v_T_xx"])[away] ** 2).mean()

    parts = {"equation": equation, "terminal": terminal, "slopes": slopes}
    total = s.coef_equation * equation + s.coef_terminal * terminal + s.coef_slopes * slopes
    return total, parts


# ---------------------------------------------------------------- training one round


@dataclass
class RoundTraining:
    """What one round's training recorded."""

    history: dict  # the loss and its parts every 100 Adam steps, and after L-BFGS
    guard_winner: str  # "lbfgs" or "adam": which weights the L-BFGS guard kept ("none": no L-BFGS)
    n_floor: int  # rivals' positions clipped at the floor during the round (should be 0)
    n_cap: int  # ...and at the cap
    seconds: float


def train_round(
    net, rivals, p: Params, N, s: NPlayerSettings, stage: StageSettings, terminal, seed
):
    """Train `net` in place to be the best response to `rivals` (MertonRivals or FrozenRivals).

    Adam with new points at every step and a step size shrinking from lr_start to lr_end, then
    L-BFGS on one fixed batch with a guard: in many dimensions L-BFGS can fit that batch and do
    worse off it, so the weights before and after are compared on one fresh batch and the better
    ones are kept. `seed` fixes the points of the round.
    """
    start = time.time()
    device = next(net.parameters()).device
    rng = np.random.default_rng(seed)
    floor_before, cap_before = rivals.n_floor, rivals.n_cap
    history = {"step": [], "total": [], "equation": [], "terminal": [], "slopes": []}

    def batch():
        return sample_batch(p, N, s.points, rng, device, terminal, rivals)

    def record(step, total, parts):
        history["step"].append(step)
        history["total"].append(total.item())
        for name, value in parts.items():
            history[name].append(value.item())

    adam = torch.optim.Adam(net.parameters(), lr=stage.lr_start)
    gamma = (stage.lr_end / stage.lr_start) ** (1 / max(stage.adam_steps, 1))
    shrink = torch.optim.lr_scheduler.ExponentialLR(adam, gamma=gamma)
    for step in range(stage.adam_steps):
        total, parts = nplayer_loss(net, batch(), p, s)
        adam.zero_grad()
        total.backward()
        adam.step()
        shrink.step()
        if step % 100 == 0:
            record(step, total, parts)
        if s.print_every and step % s.print_every == 0:
            print(f"  Adam step {step:6d}   loss {total.item():.2e}")

    guard_winner = "none"
    if stage.lbfgs_steps > 0:
        before = copy.deepcopy(net.state_dict())
        fixed = batch()
        lbfgs = torch.optim.LBFGS(
            net.parameters(),
            max_iter=stage.lbfgs_steps,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure():
            lbfgs.zero_grad()
            total, _ = nplayer_loss(net, fixed, p, s)
            total.backward()
            return total

        lbfgs.step(closure)
        # The guard: both sets of weights on the same fresh batch; keep the better one.
        fresh = batch()
        after = copy.deepcopy(net.state_dict())
        loss_after = nplayer_loss(net, fresh, p, s)[0].item()
        net.load_state_dict(before)
        loss_before = nplayer_loss(net, fresh, p, s)[0].item()
        if loss_after <= loss_before:
            net.load_state_dict(after)
            guard_winner = "lbfgs"
        else:
            guard_winner = "adam"

    total, parts = nplayer_loss(net, batch(), p, s)
    record(stage.adam_steps + stage.lbfgs_steps, total, parts)
    if s.print_every:
        print(f"  final loss {total.item():.2e}   L-BFGS guard kept: {guard_winner}")
    return RoundTraining(
        history,
        guard_winner,
        rivals.n_floor - floor_before,
        rivals.n_cap - cap_before,
        time.time() - start,
    )


# ---------------------------------------------------------------- the rounds


def should_stop(deltas, min_rounds):
    """Stop when the change of the rule no longer halves: Delta_k >= Delta_(k-1) / 2, from round
    min_rounds on. Each short round alone moves pi by the training noise, which is not known in
    advance; while the rule approaches the equilibrium Delta falls fast, and when only noise is
    left it stops falling."""
    return len(deltas) >= max(min_rounds, 2) and deltas[-1] >= 0.5 * deltas[-2]


def is_accepted(deltas, min_rounds, accept_delta, max_rounds=None):
    """Accepted if the rounds ended, by the rule above or by reaching max_rounds while the change
    was still halving, with a last change below accept_delta."""
    ended = should_stop(deltas, min_rounds) or (
        max_rounds is not None and len(deltas) >= max_rounds
    )
    return bool(ended and deltas[-1] < accept_delta)


def terminal_position_errors(solution, states, terminal):
    """Check C3: the rule at t = T against the exact one (terminal.position) on the wealths of the
    check states, outside the kink band ("away") and further than 0.3 from ybar ("far")."""
    x, y = states["x"], states["y"]
    x_col = x[:, None]  # the terminal objects take x as a column, like the batch
    error = np.abs(solution.position(solution.p.T, x, y) - terminal.position(x_col, y).ravel())
    _, _, _, away = terminal.targets(x_col, y)
    far = np.abs(x - y.mean(axis=1)) > 0.3
    out = {}
    for name, mask in (("away", away.ravel()), ("far", far)):
        chosen = error[mask]
        out[f"mean_{name}"] = float(chosen.mean()) if chosen.size else float("nan")
        out[f"max_{name}"] = float(chosen.max()) if chosen.size else float("nan")
    return out


@dataclass
class RoundsStart:
    """Where the rounds continue from: a saved round (resume) or a network of another run
    (--init-from). The default is a fresh network with the rivals at Merton."""

    net: SymmetricNet | None = None
    rounds_done: int = 0
    deltas: list[float] = field(default_factory=list)
    pi_before: np.ndarray | None = None  # the rule of `net` on the check states (computed if None)


@dataclass
class RoundResult:
    round: int
    delta: float  # mean |pi_k - pi_(k-1)| on the check states
    pi_check: np.ndarray  # the rule on the check states after this round
    training: RoundTraining
    c3: dict  # terminal_position_errors after this round
    stage: StageSettings


@dataclass
class NPlayerEquilibrium:
    solution: "NPlayerSolution"
    rounds: list[RoundResult]  # the rounds of this call (after a resume, only the new ones)
    deltas: list[float]  # of every round, those before a resume included
    accepted: bool
    finished: bool  # False when stopped early by stop_after (for the smoke test)
    seconds: float


def solve_nplayer_equilibrium(
    p: Params,
    N,
    settings: NPlayerSettings | None = None,
    terminal=None,
    start: RoundsStart | None = None,
    on_round: Callable | None = None,
    stop_after: int | None = None,
) -> NPlayerEquilibrium:
    """The symmetric equilibrium of the N-player game by rounds (fictitious play).

        rivals <- Merton;  net <- new network;  pi_before <- Merton
        for k = 1, ..., max_rounds:
            train net to be the best response to rivals (round 1 full, the others warm)
            pi_k <- the rule of net on the check states;  Delta_k <- mean |pi_k - pi_before|
            on_round(result, net)                          # the script saves the round here
            stop if k >= min_rounds and Delta_k >= Delta_(k-1) / 2
            rivals <- frozen copy of net;  pi_before <- pi_k
        accepted <- ended (by that rule or at max_rounds) and the last Delta < accept_delta

    terminal: the final condition (ShortfallTerminal by default; C2Terminal for check C2). start:
    continue from a network instead (then every round is warm and the rivals start at its rule).
    stop_after: return after that many new rounds, as if interrupted.
    """
    s = settings or NPlayerSettings()
    device = s.device or pick_device()
    terminal = terminal or ShortfallTerminal(p, N, s.points.kink_band)
    states = check_states(p, N, s.n_check, s.check_seed, s.check_t_max)
    start = start or RoundsStart()
    began = time.time()

    torch.manual_seed(s.seed)
    if start.net is None:
        net = SymmetricNet(p, **asdict(s.net)).to(device)
        rivals = MertonRivals(p)
        pi_before = merton_pi(states["t"], p).ravel()
    else:
        net = start.net.to(device)
        rivals = FrozenRivals(net, p, s.rival_clip)
        pi_before = start.pi_before
        if pi_before is None:
            pi_before = NPlayerSolution(net, p, N).position(states["t"], states["x"], states["y"])
    fresh_start = start.net is None and start.rounds_done == 0

    deltas, rounds, finished = list(start.deltas), [], True
    k = start.rounds_done
    while k < s.max_rounds:
        k += 1
        stage = s.first_round if (k == 1 and fresh_start) else s.later_rounds
        if s.print_every:
            print(f"N = {N}, round {k}: {stage.adam_steps} Adam + {stage.lbfgs_steps} L-BFGS steps")
        training = train_round(net, rivals, p, N, s, stage, terminal, seed=1000 * s.seed + k)
        solution = NPlayerSolution(net, p, N)
        pi_k = solution.position(states["t"], states["x"], states["y"])
        deltas.append(float(np.mean(np.abs(pi_k - pi_before))))
        c3 = terminal_position_errors(solution, states, terminal)
        result = RoundResult(k, deltas[-1], pi_k, training, c3, stage)
        rounds.append(result)
        if on_round is not None:
            on_round(result, net)
        if s.print_every:
            print(f"  Delta_{k} = {deltas[-1]:.4f}   ({training.seconds:.0f} s)")
        if stop_after is not None and len(rounds) >= stop_after:
            finished = False
            break
        if should_stop(deltas, s.min_rounds):
            break
        rivals = FrozenRivals(net, p, s.rival_clip)
        pi_before = pi_k

    accepted = finished and is_accepted(deltas, s.min_rounds, s.accept_delta, s.max_rounds)
    return NPlayerEquilibrium(
        NPlayerSolution(net, p, N), rounds, deltas, accepted, finished, time.time() - began
    )


# ---------------------------------------------------------------- the interface


def _broadcast(t, x, y, N):
    """NumPy inputs to columns: t and x broadcast like t + x, y carries the N - 1 rivals on its
    last axis and broadcasts with them on the others. Returns t, x (n, 1), y (n, N - 1) and the
    shape of the result."""
    t, x = np.broadcast_arrays(np.asarray(t, dtype=float), np.asarray(x, dtype=float))
    y = np.asarray(y, dtype=float)
    if y.ndim == 0 or y.shape[-1] != N - 1:
        raise ValueError(f"y must have the {N - 1} rivals' wealths on its last axis, got {y.shape}")
    shape = np.broadcast_shapes(t.shape, y.shape[:-1])
    t, x = np.broadcast_to(t, shape), np.broadcast_to(x, shape)
    y = np.broadcast_to(y, (*shape, N - 1))
    return t.reshape(-1, 1), x.reshape(-1, 1), y.reshape(-1, N - 1), shape


def own_derivatives(v_fn, t, x, y):
    """v, v_x and v_xx only (what the rule needs): cheaper than v_derivatives."""
    v = v_fn(t, x, y)
    v_x = _grad(v, x, create_graph=True)
    return v, v_x, _grad(v_x, x, create_graph=True)


@dataclass
class NPlayerSolution:
    """A trained network for N players, read with NumPy arrays: t and x broadcast like t + x, and y
    has the N - 1 rivals' wealths on its last axis, broadcasting with t + x on the others (the
    CONTRIBUTING interface with one more argument). For n different states pass t and x of shape
    (n,) and y of shape (n, N - 1); a column x (n, 1) with y (n, N - 1) would give n x n results.
    Points are evaluated in chunks, so a Monte Carlo can pass millions at once."""

    net: SymmetricNet
    p: Params
    N: int
    chunk: int = 20_000

    def _evaluate(self, fn, t, x, y):
        t, x, y, shape = _broadcast(t, x, y, self.N)
        device = next(self.net.parameters()).device
        out = []
        for i in range(0, len(t), self.chunk):
            columns = [
                torch.tensor(a[i : i + self.chunk], dtype=torch.float32, device=device)
                for a in (t, x, y)
            ]
            for c in columns:
                c.requires_grad_()
            out.append(fn(*columns).detach().cpu().numpy().astype(float).ravel())
        return np.concatenate(out).reshape(shape)

    def value(self, t, x, y):
        """V(t, x, y) = -exp(v)."""
        return -np.exp(self._evaluate(self.net, t, x, y))

    def position(self, t, x, y):
        """pi*(t, x, y), the optimal dollars in the stock (not clipped)."""

        def rule(tt, xx, yy):
            _, v_x, v_xx = own_derivatives(self.net, tt, xx, yy)
            return position_formula(v_x, v_xx, self.p)

        return self._evaluate(rule, t, x, y)

    def concavity(self, t, x, y):
        """v_xx + v_x^2 = V_xx / V: positive where V is concave, which pi* needs."""

        def curvature(tt, xx, yy):
            _, v_x, v_xx = own_derivatives(self.net, tt, xx, yy)
            return v_xx + v_x**2

        return self._evaluate(curvature, t, x, y)

    def residual(self, t, x, y):
        """Scaled HJB residual with the rivals playing this same rule: near 0 at an equilibrium
        where the network solves its equation."""
        rivals = FrozenRivals(self.net, self.p)

        def equilibrium_residual(tt, xx, yy):
            return hjb_residual(self.net, rivals(tt, xx, yy), tt, xx, yy, self.p)

        return self._evaluate(equilibrium_residual, t, x, y)
