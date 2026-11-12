# How we work

## The loop

1. **Pick one of your tasks from the roadmap** in `README.md`. For an unassigned task, agree in
   chat first, then add your initials in the PR.
2. **Branch from an up-to-date `main`:**
   ```bash
   git switch main && git pull
   git switch -c feat/fd-hjb        # or fix/..., exp/..., docs/...
   ```
3. **Commit small and often.** Message: imperative, says what changed, e.g.
   `Add upwind FD solver for the HJB`. Optionally run `pre-commit run --all-files` to format
   the code before pushing.
4. **Run the tests locally** with `pytest`. They must all pass.
5. **Push and open a pull request** into `main`. Fill in the template, and if the PR finishes a
   roadmap task, tick its box in `README.md` in the same PR.
6. **The other person reviews.** Reviewer checks the maths as well as the code: does the
   discretisation match the equation, and does the test compare against a real known answer?
7. **Squash-merge**, delete the branch, pull `main`.

Nobody pushes directly to `main`.

## Branch prefixes

| Prefix  | For |
|---------|-----|
| `feat/` | new solver or functionality |
| `fix/`  | bug fixes |
| `exp/`  | experiments and sweeps (code that produces results/figures) |
| `docs/` | notes, report, README |

## How the FD and PINN solvers fit together

The finite-difference solvers (`src/mfg/fd/`, DS) and the PINNs (`src/mfg/pinn/`, AI) solve the
same equations, so their answers must be directly comparable.

- **Same interface.** For given `Params` and a fixed m̄, every HJB solver returns an object with
  `value(t, x)` and `position(t, x)`; every Fokker–Planck solver returns one with
  `density(t, x)`. They take and return NumPy arrays (broadcasting like `t + x`), so a PINN
  wrapper converts to and from tensors inside.
- **Same boundary conditions.** Both use the functions in `benchmarks.py`: `merton_value` at
  x = 3, and the lower-boundary value at x = −1 once it exists (at λ = 0, `merton_value` at both
  ends). Different boundary choices would show up as solver error.
- **Same test case.** Before the equilibrium is known, compare at λ = 2 with m̄ = 1.1 fixed,
  on the domain and parameters of `configs/base.yaml`.
- **Separate folders.** Each side edits its own folder; shared code (`params.py`, `utility.py`,
  `benchmarks.py`) changes only in small PRs of its own.

## Rules that keep results trustworthy

- **Every numerical method gets a test against a known answer** (λ = 0 closed form, FD reference,
  Monte Carlo, or a hand-derived formula such as the t = T position).
- **Results come from configs.** A figure in the report must be reproducible from a file in
  `configs/` plus a commit hash. Save the config alongside outputs in `results/`.
- **Don't commit outputs.** `results/`, checkpoints and arrays are git-ignored; only final report
  figures are committed, under `report/figures/`.
- **Notebooks are for exploring.** Once code is reused, move it into `src/mfg` and test it.
  Outputs may be committed, so results can be read without rerunning; run the notebook top to
  bottom first, so the outputs match the code. Pre-commit rejects files over 5 MB.
- **Fix the seed** for anything random, and record it.

## Avoiding merge conflicts

- Pull `main` before starting a branch, and rebase or merge `main` into long-lived branches often.
- Split work by file/module where possible (e.g. one person on `fd/`, the other on `pinn/`).
- Keep pull requests small: one task, one PR.
