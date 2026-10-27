# How we work

## The loop

1. **Pick a task from the roadmap** in `README.md` and put your initials next to it, so we don't
   overlap.
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

## Rules that keep results trustworthy

- **Every numerical method gets a test against a known answer** (λ = 0 closed form, FD reference,
  Monte Carlo, or a hand-derived formula such as the t = T position).
- **Results come from configs.** A figure in the report must be reproducible from a file in
  `configs/` plus a commit hash. Save the config alongside outputs in `results/`.
- **Don't commit outputs.** `results/`, checkpoints and arrays are git-ignored; only final report
  figures are committed, under `report/figures/`.
- **Notebooks are for exploring.** Once code is reused, move it into `src/mfg` and test it.
  Clear notebook outputs before committing (Kernel → Restart & Clear Output, or
  `pre-commit run --all-files`) to keep diffs readable.
- **Fix the seed** for anything random, and record it.

## Avoiding merge conflicts

- Pull `main` before starting a branch, and rebase or merge `main` into long-lived branches often.
- Split work by file/module where possible (e.g. one person on `fd/`, the other on `pinn/`).
- Keep pull requests small: one task, one PR.
