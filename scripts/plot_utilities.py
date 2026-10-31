"""Compare the original relative-performance utility with the shortfall-penalty utility.

Usage:  python scripts/plot_utilities.py [--out results/figures/utility_comparison.png]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfg import Params
from mfg.utility import utility, utility_x, utility_xx

THETA = 0.5  # peer weight in the original utility, for illustration only

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
CARA = "#8f8e88"
ORIGINAL = "#2a78d6"
PROPOSED = "#eb6834"


def original(x, mbar, p):
    """-exp(-gamma (x - theta mbar)): plain CARA times the constant exp(gamma theta mbar)."""
    return -np.exp(-p.gamma * (x - THETA * mbar))


def original_x(x, mbar, p):
    return p.gamma * np.exp(-p.gamma * (x - THETA * mbar))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/figures/utility_comparison.png")
    parser.add_argument(
        "--stacked", action="store_true", help="panels one above another, for a printed page"
    )
    args = parser.parse_args()

    p = Params()
    mbar = 1.1  # lambda = 0 equilibrium mean m̄_T from the proposal
    cara = p.with_(lam=0.0)
    x = np.linspace(-1.0, 3.0, 801)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    if args.stacked:
        fig, axes = plt.subplots(3, 1, figsize=(6.5, 8.0), constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)

    panels = [
        (
            "Utility U(x)",
            utility(x, mbar, cara),
            original(x, mbar, p),
            utility(x, mbar, p),
        ),
        (
            "Marginal utility U′(x)",
            utility_x(x, mbar, cara),
            original_x(x, mbar, p),
            utility_x(x, mbar, p),
        ),
        (
            "Risk aversion A(x) = −U″/U′",
            np.full_like(x, p.gamma),
            np.full_like(x, p.gamma),
            -utility_xx(x, mbar, p) / utility_x(x, mbar, p),
        ),
    ]

    for ax, (title, u_cara, u_orig, u_prop) in zip(axes, panels, strict=True):
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.axvspan(x[0], mbar, color="#f4f3f0", zorder=0)
        ax.axvline(mbar, color=INK_2, linewidth=0.8)
        ax.plot(x, u_cara, color=CARA, linewidth=2, label="Plain CARA (λ = 0)")
        ax.plot(x, u_orig, color=ORIGINAL, linewidth=2, label=f"Original (θ = {THETA})")
        ax.plot(x, u_prop, color=PROPOSED, linewidth=2, label=f"Proposed (λ = {p.lam:g})")
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_xlabel("terminal wealth x")
        ax.set_xlim(x[0], x[-1])

    axes[0].set_ylim(-5, 0.2)
    axes[1].set_ylim(0, 6)
    axes[2].set_ylim(0, 7.8)
    for ax in axes:
        top = ax.get_ylim()[1]
        ax.text(mbar - 0.05, top, "below m̄", ha="right", va="top", color=INK_2, fontsize=9)
        ax.text(mbar + 0.05, top, f"m̄ = {mbar}", ha="left", va="top", color=INK_2, fontsize=9)

    # Risk-aversion panel: the original coincides with plain CARA (A = γ everywhere).
    ax = axes[2]
    ax.lines[1].set_linestyle((0, (4, 3)))
    a_kink = p.gamma + p.lam * np.exp(p.gamma * mbar) / p.gamma
    crossover = mbar - 1.0 / p.gamma
    marks = [
        (mbar, a_kink, "just below m̄: 7.0\n(protect your place)", (-0.9, 5.6)),
        (crossover, p.gamma, "gap = 1/γ: back to γ", (0.25, 0.3)),
        (
            -0.5,
            -utility_xx(-0.5, mbar, p) / utility_x(-0.5, mbar, p),
            "far behind: 0.75\n(gamble to catch up)",
            (-0.9, 3.4),
        ),
    ]
    for mx, my, text, (tx, ty) in marks:
        ax.plot(mx, my, "o", color=PROPOSED, markersize=7, markeredgecolor="white", mew=2)
        ax.annotate(
            text,
            (mx, my),
            (tx, ty),
            color=INK,
            fontsize=9,
            arrowprops={"arrowstyle": "-", "color": INK_2, "linewidth": 0.8},
        )
    ax.text(2.95, p.gamma + 0.15, "original = CARA: A = γ", ha="right", color=ORIGINAL, fontsize=9)

    axes[0].legend(loc="lower right", frameon=False, labelcolor=INK)
    fig.suptitle(
        f"Original vs shortfall-penalty utility  (γ = {p.gamma:g}, m̄ = {mbar})",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=12,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
