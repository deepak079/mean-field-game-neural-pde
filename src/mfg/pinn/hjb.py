"""PINN for the value function V of one investor, for a fixed assumed final mean mbar.

What the network learns
    The network gives v(t, x) = log(-V(t, x)). Then V = -exp(v) is always negative, as it must be
    (the utility is), and v is of size one and almost straight (with lam = 0 it is exactly
    straight in x).

What it is trained on: a loss with four parts, each a mean of squares
    equation  inside the domain, the HJB residual in v-form (hjb_residual), divided by
              kappa^2 / 2 so that it is of size one like the other parts.
    terminal  at t = T, v equals the final utility exactly: v_T = log(-U(x, mbar)).
    slopes    at t = T, v_x and v_xx also equal their exact values, except within kink_band of
              mbar, where U has a kink and v_xx jumps. pi* depends on v_xx, which the value
              alone does not pin down well.
    edges     at x_min and x_max, the values shared with the FD solver (benchmarks.py):
              - x_min = -1: merton_strategy_value, the expected utility of holding the Merton
                position until T. It is exact at t = T and when lam = 0, never above the true V
                (Merton is a possible strategy, not the best one), and close to it, because the
                value barely changes near the best strategy. Its small error fades inward, away
                from where the population lives.
              - x_max = 3: merton_value. From x = 3 nobody ends below the mean, so the true V lies
                between merton_value (above) and merton_strategy_value (below), and these two
                differ by about 1e-14 there.
              At t = T both edge values equal the final utility, so all the conditions agree at
              the corners of the domain.

Reading the strategy
    pi* = -(mu - r) / sigma^2 * v_x / (v_xx + v_x^2), where v_xx + v_x^2 = V_xx / V is positive
    because V is concave.
"""

import copy
from dataclasses import dataclass

import numpy as np
import torch

from mfg.benchmarks import merton_strategy_value, merton_value
from mfg.params import Params
from mfg.pinn.nets import TanhNet, derivative, pick_device
from mfg.utility import utility, utility_x, utility_xx

# Smallest value of v_xx + v_x^2 that the loss divides by. Only a safety net for the first
# training steps, while the network is still random. At the solution this quantity stays far
# above it (its smallest value is about 0.66, at x = -1, for lam = 2 and mbar = 1.1), so the bound
# plays no role in the final result; pi* is always read without it.
MIN_CURVATURE = 0.05


@dataclass
class ValueNetSettings:
    """Size of the network, points per step and training choices (reasons in the task summary,
    docs/tasks/V_PINN task/)."""

    width: int = 64  # neurons per hidden layer
    depth: int = 4  # hidden layers
    n_inside: int = 4000  # random points in the whole domain, new at every step
    n_kink: int = 1000  # extra points near (T, mbar), where pi* changes fastest...
    kink_dt: float = 0.1  # ...with t in [T - kink_dt, T]
    kink_dx: float = 0.3  # ...and x in [mbar - kink_dx, mbar + kink_dx]
    n_terminal: int = 1000  # points at t = T
    kink_band: float = 0.05  # at t = T, slopes are not matched closer than this to mbar
    n_edge: int = 200  # points on each edge
    coef_equation: float = 1.0  # coefficients of the four parts of the loss
    coef_terminal: float = 1.0
    coef_slopes: float = 1.0
    coef_edges: float = 1.0
    adam_steps: int = 10_000
    lr_start: float = 1e-3  # Adam step size at the first step, shrinking smoothly...
    lr_end: float = 1e-4  # ...to this one at the last step
    lbfgs_steps: int = 500  # final polish on one fixed set of points (0 to skip it)
    seed: int = 0
    device: str | None = None  # None: the GPU if there is one
    print_every: int = 0  # print the loss every this many Adam steps (0: silent)


def v_derivatives(v_fn, t, x):
    """v and its derivatives v_t, v_x, v_xx at the points (t, x), by automatic differentiation.

    v_fn is the network or any torch function of (t, x); t and x are column tensors with
    requires_grad=True.
    """
    v = v_fn(t, x)
    # v_t and v_x in one pass back through the network, which is faster than two.
    v_t, v_x = torch.autograd.grad(v, (t, x), torch.ones_like(v), create_graph=True)
    return v, v_t, v_x, derivative(v_x, x)


def residual_formula(x, v_t, v_x, v_xx, p: Params):
    """The HJB residual in v-form, divided by kappa^2 / 2, from the derivatives of v.

    R = (v_t + r x v_x - kappa^2 v_x^2 / (2 (v_xx + v_x^2))) / (kappa^2 / 2). The terms of the
    equation are of size kappa^2 / 2 = 0.045, so without the division an error in the curvature
    (which sets pi*) would hardly show in the loss.
    """
    curvature = torch.clamp(v_xx + v_x**2, min=MIN_CURVATURE)
    return (v_t + p.r * x * v_x) / (0.5 * p.kappa**2) - v_x**2 / curvature


def hjb_residual(v_fn, t, x, p: Params):
    """HJB residual (see residual_formula) of v_fn at the points (t, x); 0 where v solves it."""
    _, v_t, v_x, v_xx = v_derivatives(v_fn, t, x)
    return residual_formula(x, v_t, v_x, v_xx, p)


def optimal_position(v_fn, t, x, p: Params):
    """pi* = -(mu - r) / sigma^2 * v_x / (v_xx + v_x^2), the dollars held in the stock."""
    _, _, v_x, v_xx = v_derivatives(v_fn, t, x)
    return -(p.mu - p.r) / p.sigma**2 * v_x / (v_xx + v_x**2)


def terminal_targets(x, mbar, p: Params):
    """Exact v, v_x and v_xx at t = T, from the final utility U (NumPy in and out).

    v_T = log(-U), v_T' = U_x / U and v_T'' = U_xx / U - (U_x / U)^2; v_T'' jumps at x = mbar.
    """
    U = utility(x, mbar, p)
    slope = utility_x(x, mbar, p) / U
    return np.log(-U), slope, utility_xx(x, mbar, p) / U - slope**2


def edge_targets(t, mbar, p: Params):
    """v at the lower and at the upper edge, from the values shared with the FD solver (NumPy)."""
    low = merton_strategy_value(t, p.x_min, mbar, p)
    high = merton_value(t, p.x_max, p)
    return np.log(-low), np.log(-high)


def sample_batch(p: Params, mbar, s: ValueNetSettings, device):
    """Fresh random points and their targets: inside the domain (with extra points near
    (T, mbar)), at t = T and on the two edges. Targets are computed with NumPy.

    All the points go in one column, in that order, so that the network runs once per step
    (much faster on a GPU than one run per group).
    """

    def uniform(n, low, high):
        return low + (high - low) * torch.rand(n, 1, device=device)

    def to_device(a):
        return torch.as_tensor(a, dtype=torch.float32, device=device)

    t_kink = uniform(s.n_kink, p.T - s.kink_dt, p.T)
    x_kink = uniform(s.n_kink, mbar - s.kink_dx, mbar + s.kink_dx).clamp(p.x_min, p.x_max)
    t_in = torch.cat([uniform(s.n_inside, 0.0, p.T), t_kink])
    x_in = torch.cat([uniform(s.n_inside, p.x_min, p.x_max), x_kink])
    x_T = uniform(s.n_terminal, p.x_min, p.x_max)
    t_edge = uniform(s.n_edge, 0.0, p.T)

    v_T, v_T_x, v_T_xx = terminal_targets(x_T.cpu().numpy().astype(float), mbar, p)
    v_low, v_high = edge_targets(t_edge.cpu().numpy().astype(float), mbar, p)
    return {
        "t": torch.cat([t_in, torch.full_like(x_T, p.T), t_edge, t_edge]),
        "x": torch.cat(
            [x_in, x_T, torch.full_like(t_edge, p.x_min), torch.full_like(t_edge, p.x_max)]
        ),
        "n_inside": len(t_in),
        "n_terminal": len(x_T),
        "v_T": to_device(v_T),
        "v_T_x": to_device(v_T_x),
        "v_T_xx": to_device(v_T_xx),
        "away": (x_T - mbar).abs() > s.kink_band,  # far enough from the kink to match slopes
        "v_edge": to_device(np.concatenate([v_low, v_high])),
    }


def value_net_loss(net, batch, p: Params, s: ValueNetSettings):
    """The loss (the four parts times their coefficients) and the parts themselves."""
    t, x = batch["t"].requires_grad_(), batch["x"].requires_grad_()
    v, v_t, v_x, v_xx = v_derivatives(net, t, x)
    # Split the stacked points back into their groups (see sample_batch).
    n_in, n_T = batch["n_inside"], batch["n_terminal"]
    inside, final, edge = slice(0, n_in), slice(n_in, n_in + n_T), slice(n_in + n_T, None)

    residual = residual_formula(x[inside], v_t[inside], v_x[inside], v_xx[inside], p)
    equation = residual.pow(2).mean()
    terminal = (v[final] - batch["v_T"]).pow(2).mean()
    away = batch["away"]
    slope_error = (v_x[final] - batch["v_T_x"])[away] ** 2
    curvature_error = (v_xx[final] - batch["v_T_xx"])[away] ** 2
    slopes = (slope_error + curvature_error).mean()
    edges = (v[edge] - batch["v_edge"]).pow(2).mean()

    parts = {"equation": equation, "terminal": terminal, "slopes": slopes, "edges": edges}
    total = (
        s.coef_equation * equation
        + s.coef_terminal * terminal
        + s.coef_slopes * slopes
        + s.coef_edges * edges
    )
    return total, parts


def train_value_net(p: Params, mbar, settings: ValueNetSettings | None = None, net=None):
    """Train the value network for an assumed final mean mbar; returns a ValueNetSolution.

    net: a trained network to start from (for example the one for a nearby mbar). It is copied
    first, so the solution it came from does not change.
    """
    s = settings or ValueNetSettings()
    device = s.device or pick_device()
    torch.manual_seed(s.seed)
    net = TanhNet(p, s.width, s.depth) if net is None else copy.deepcopy(net)
    net = net.to(device)
    history = {"step": [], "total": [], "equation": [], "terminal": [], "slopes": [], "edges": []}

    def record(step, total, parts):
        """Keep the loss and its parts, for plotting how training went."""
        history["step"].append(step)
        history["total"].append(total.item())
        for name, value in parts.items():
            history[name].append(value.item())

    # Stage 1, Adam: new points at every step; the step size shrinks from lr_start to lr_end.
    adam = torch.optim.Adam(net.parameters(), lr=s.lr_start)
    gamma = (s.lr_end / s.lr_start) ** (1 / max(s.adam_steps, 1))
    shrink = torch.optim.lr_scheduler.ExponentialLR(adam, gamma=gamma)
    for step in range(s.adam_steps):
        total, parts = value_net_loss(net, sample_batch(p, mbar, s, device), p, s)
        adam.zero_grad()
        total.backward()
        adam.step()
        shrink.step()
        if step % 100 == 0:
            record(step, total, parts)
        if s.print_every and step % s.print_every == 0:
            print(f"Adam step {step:6d}   loss {total.item():.2e}")

    # Stage 2, L-BFGS: polishes on one fixed set of points (it needs the same loss at every call).
    if s.lbfgs_steps > 0:
        batch = sample_batch(p, mbar, s, device)
        lbfgs = torch.optim.LBFGS(
            net.parameters(), max_iter=s.lbfgs_steps, history_size=50, line_search_fn="strong_wolfe"
        )

        def closure():
            lbfgs.zero_grad()
            total, _ = value_net_loss(net, batch, p, s)
            total.backward()
            return total

        lbfgs.step(closure)

    # Final loss on fresh points, which the network has not seen.
    total, parts = value_net_loss(net, sample_batch(p, mbar, s, device), p, s)
    record(s.adam_steps + s.lbfgs_steps, total, parts)
    if s.print_every:
        print(f"final loss {total.item():.2e}")
    return ValueNetSolution(net, p, mbar, history)


@dataclass
class ValueNetSolution:
    """A trained value network, read with NumPy arrays that broadcast like t + x.

    value and position are the interface shared with the FD solver (CONTRIBUTING.md); residual
    and concavity help to see where the network can be trusted.
    """

    net: TanhNet
    p: Params
    mbar: float
    history: dict

    def _evaluate(self, fn, t, x):
        """Run fn on column tensors made from the NumPy inputs, and return a NumPy array."""
        t, x = np.broadcast_arrays(np.asarray(t, dtype=float), np.asarray(x, dtype=float))
        device = next(self.net.parameters()).device

        def column(a):
            return torch.tensor(
                a.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True
            )

        out = fn(column(t), column(x))
        return out.detach().cpu().numpy().astype(float).reshape(t.shape)

    def value(self, t, x):
        """V(t, x) = -exp(v(t, x))."""
        return -np.exp(self._evaluate(self.net, t, x))

    def position(self, t, x):
        """pi*(t, x), the optimal dollars in the stock."""
        return self._evaluate(lambda tt, xx: optimal_position(self.net, tt, xx, self.p), t, x)

    def residual(self, t, x):
        """Scaled HJB residual (see hjb_residual): near 0 where the network solves the equation."""
        return self._evaluate(lambda tt, xx: hjb_residual(self.net, tt, xx, self.p), t, x)

    def concavity(self, t, x):
        """v_xx + v_x^2 = V_xx / V: positive where V is concave, which pi* needs."""

        def curvature(tt, xx):
            _, _, v_x, v_xx = v_derivatives(self.net, tt, xx)
            return v_xx + v_x**2

        return self._evaluate(curvature, t, x)
