"""Copy the numbers of Deepak's slides into docs/presentation/slides.html.

The density, strategy and equilibrium slides (9 and 10) draw from window.DSDATA, a copy of part of
docs/presentation/guides/figure_data.js: the density and the strategy at the equilibrium on
wealth in [0, 2.2] for every saved time, and FD's Phi with both searches. Rerunning replaces the
block between the markers "ds data: start" and "ds data: end".

    python scripts/embed_slide_data.py
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "docs" / "presentation"
START, END = "/* ds data: start */", "/* ds data: end */"


def window(xs, rows, lo, hi):
    """The points of xs in [lo, hi], and every row on them, rounded to 4 decimals."""
    keep = [i for i, x in enumerate(xs) if lo - 1e-9 <= x <= hi + 1e-9]
    return [xs[i] for i in keep], [[round(row[i], 4) for i in keep] for row in rows]


def main():
    text = (ROOT / "guides" / "figure_data.js").read_text(encoding="utf-8")
    d = json.loads(text[text.index("{") : text.rindex("}") + 1])
    den, st, eq = d["density"], d["strategy"], d["equilibrium"]
    dx, dfd = window(den["x"], den["fd"], 0.0, 2.2)
    _, dpn = window(den["x"], den["pinn"], 0.0, 2.2)
    sx, sfd = window(st["x"], st["fd"], 0.0, 2.2)
    _, spn = window(st["x"], st["pinn"], 0.0, 2.2)
    keys = (
        "phi_grid",
        "phi_fd",
        "pinn_steps",
        "fd_history",
        "mbar_pinn",
        "mbar_fd",
        "mbar_free",
        "tol",
    )
    out = {
        "source": "docs/presentation/guides/figure_data.js (made " + d["made"] + ")",
        "t": st["t"],
        "density": {
            "x": dx,
            "fd": dfd,
            "pinn": dpn,
            "free_mean": den["free_mean"],
            "free_sd": den["free_sd"],
        },
        "strategy": {"x": sx, "fd": sfd, "pinn": spn, "merton": st["merton"]},
        "equilibrium": {k: eq[k] for k in keys},
        "summary": d["summary"],
    }
    block = START + "\nwindow.DSDATA = " + json.dumps(out, separators=(",", ":")) + ";\n" + END
    path = ROOT / "slides.html"
    html = path.read_text(encoding="utf-8")
    if START not in html:
        raise SystemExit(f"no '{START}' marker in {path}")
    html = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda m: block, html, flags=re.S)
    path.write_text(html, encoding="utf-8")
    print(f"embedded {len(block) // 1024} KB into {path}")


if __name__ == "__main__":
    main()
