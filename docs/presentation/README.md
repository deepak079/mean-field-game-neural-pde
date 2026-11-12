# Presentation

The class presentation of Problem 21: the slides made so far, the guides they are built from, and an
example of how they were made with Claude Code.

| Path | What it is |
|---|---|
| `slides.html` | Ander's part, 18 slides: cover, the game, the equations, the loop and the V network (1–9), the league of N investors (10–16), a real case, Chile's pension funds and why the penalty is a square (17), and "Thank you · Questions?" (18). Deepak's Fokker–Planck slides are not in it. |
| `prompt_example.txt` | The prompt Ander used to start the slides with Claude Code (translated), and the tools to install. A template for your sections. |
| `guides/` | The HTML guides that explain the project, all in English: the source material for the slides. |
| `reference/AMS520_attention_factors_slides.html` | Ander's AMS 520 deck. The look and the slide engine of `slides.html` come from it. |

## The guides

| File | What it explains |
|---|---|
| `study_guide_hjb_fokker_planck.html` | Where the HJB and Fokker–Planck equations come from, term by term, the equilibrium, and the PINN plan. Written before the code: its section 9 and task list are the plan, not the code as it is now. |
| `implementation_guide.html` | How the equilibrium is built in the code: the V network, the copy of π*, the density network, the final mean, the secant loop and the finite-difference check. |
| `pinn_value_network_guide.html` | The V network in detail: the equation for v = log(−V), the edges, the loss part by part, the training and its results. Slides 7 to 9 come from it. |
| `presentation_figures.html` | Nine interactive figures at slide size, rebuilt from the results notebook (equilibrium, strategy, density, investors, λ sweep, accuracy, PINN vs FD). Each one says what it shows and what to say. It needs `figure_data.js` next to it. |
| `../tasks/N_investors task/N_investors_guide.html` | The finite-N extension (kept with its task in `docs/tasks/`). |

The study guide and the implementation guide are translations of Ander's Spanish originals; the
other three were written in English.

## How to open them

- GitHub shows HTML as code: download the folder (or check out this branch) and open the files in a
  browser. They need an internet connection for the fonts and the formulas (MathJax).
- In `slides.html`: ← → or a clicker to move, F for full screen, `#7` at the end of the address to
  jump to slide 7. Some slides build up step by step; the buttons and sliders on them are live.
- `slides.html` is a copy of the deck Ander keeps as a private page on claude.ai, with every number
  inside it: nothing else is needed to run it. Its numbers come from saved results: the V network
  trained at m̄ = 1.1 (slides 7–9), the training shown in the V network guide (slide 9), the Monte
  Carlo investors of the finite-difference equilibrium (slides 1–2), the league runs in
  `results/nplayer/main` (slides 12–14), and a second training of each N that saved photos of the
  network as it learned (slide 16). Slide 17 is a sketch, not data.
