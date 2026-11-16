# Presentation

The class presentation of Problem 21: the slides made so far, the guides they are built from, and an
example of how they were made with Claude Code.

| Path | What it is |
|---|---|
| `slides.html` | The whole deck, 19 slides. **Deepak's part (1–10):** cover, what a mean-field game is (drivers on two roads), the investors, the utility, the two equations, the equilibrium loop, two ways to solve the equations, the training, the density network, the results. **Ander's part (11–19):** the league of N investors (11–17), a real case, Chile's pension funds and why the penalty is a square (18), and "Thank you · Questions?" (19). |
| `prompt_example.txt` | The prompt Ander used to start the slides with Claude Code (translated), and the tools to install. A template for your sections. |
| `guides/` | The HTML guides that explain the project, all in English: the source material for the slides. |
| `reference/AMS520_attention_factors_slides.html` | Ander's AMS 520 deck. The look and the slide engine of `slides.html` come from it. |

## The guides

| File | What it explains |
|---|---|
| `study_guide_hjb_fokker_planck.html` | Where the HJB and Fokker–Planck equations come from, term by term, the equilibrium, and the PINN plan. Written before the code: its section 9 and task list are the plan, not the code as it is now. |
| `implementation_guide.html` | How the equilibrium is built in the code: the V network, the copy of π*, the density network, the final mean, the secant loop and the finite-difference check. |
| `pinn_value_network_guide.html` | The V network in detail: the equation for v = log(−V), the edges, the loss part by part, the training and its results. Slide 8 (the training) comes from it. |
| `presentation_figures.html` | Nine interactive figures at slide size, rebuilt from the results notebook (equilibrium, strategy, density, investors, λ sweep, accuracy, PINN vs FD). Each one says what it shows and what to say. It needs `figure_data.js` next to it. |
| `../tasks/N_investors task/N_investors_guide.html` | The finite-N extension (kept with its task in `docs/tasks/`). |

The study guide and the implementation guide are translations of Ander's Spanish originals; the
other three were written in English.

## How to open them

- GitHub shows HTML as code: download the folder (or check out this branch) and open the files in a
  browser. They need an internet connection for the fonts and the formulas (MathJax).
- In `slides.html`: ← → or a clicker to move, F for full screen, `#7` at the end of the address to
  jump to slide 7. Some slides build up step by step; the buttons and sliders on them are live.
- **`slides.html` in this repository is the master copy.** Edit it here, on a branch, so both parts
  stay in one file; a copy kept elsewhere (such as a private claude.ai page) would overwrite the
  other part's changes.
- Every number is inside the file: nothing else is needed to run it. The numbers come from saved
  results: the Monte Carlo investors of the finite-difference equilibrium (slide 3), the training
  of the V network at m̄ = 1.1 (slide 8), the density and the strategy at the equilibrium (slides 9
  and 10, copied from `guides/figure_data.js` by `scripts/embed_slide_data.py`; slide 16 also draws
  that strategy as the N → ∞ limit), the league runs in `results/nplayer/main` (slides 13, 16 and
  17; slide 16 plays one game per N with them), and a second training of each N that saved photos
  of the network as it learned (slide 15). Slide 18 is a sketch, not data; slide 2 (two roads) is an
  illustration with made-up travel times.
