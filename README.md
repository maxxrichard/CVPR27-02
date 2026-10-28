# GRAFT: Graph-Regularized Anchor-displacement Field Transduction for Vision-Language Models

Research project targeting CVPR 2027: label-free, training-free adaptation of a frozen vision-language
classifier from an unlabeled test batch or stream under **realistic** conditions (few effective classes per
batch, correlated streams, several backbones).

**Core idea.** Adapted prototypes are text anchors plus a *displacement field* over the class vocabulary,
`mu_k = t_k + d_k`. The field has a Gaussian Markov random field prior on a semantic class graph built from
the text encoder, so that rare and absent classes borrow the displacement of their well-observed neighbours,
and class presence is estimated explicitly (label shift). All updates are closed form on cached features.
Anchor-only shrinkage (StatA, MOON) and a single global shift are the two extreme settings of the prior.

Paper draft: `paper/main.tex` (official CVPR author-kit style, unmodified). Reports: `docs/`.

## Status (read this first)
* Implemented and unit tested (`tests/`, 17 tests, synthetic data only): closed-form M-step, special cases,
  harmonic extension of absent classes, monotone objective decrease, online/batch equivalence, EB, samplers,
  official-harness adapter.
* **No benchmark experiment has been run.** Every result table in the paper contains `TBD`. The experiment
  plan and commands are in `docs/experiment_plan.md`; features must first be extracted with
  `scripts/extract_features.py` (requires the datasets and a CLIP/OpenCLIP install).

## Layout
```
graft/                 method (solver.py), class graph, baselines/controls, samplers, metrics, adapter
scripts/               extract_features.py, run_batch.py, run_online.py, analyze_smoothness.py
tests/                 pytest unit tests (synthetic)
visualization/         draw_architecture.py (figures 1-3), plot_results.py, generate_tables.py
configs/               JSON configs for the main method, controls and ablations
paper/                 CVPR LaTeX manuscript (main.tex, sec/, figures/, tables/, references.bib, supplementary.tex)
docs/                  deliverables: trend analysis, blueprint report, proposal, experiment plan, novelty audit, review
results/raw|processed  result JSON files (empty until experiments are run)
```

## Quick start
```bash
pip install -r requirements.txt
python -m pytest tests -q                       # unit tests (seconds, CPU)
python visualization/draw_architecture.py       # regenerate figures 1-3
# extract features once per dataset/backbone (needs data + CLIP):
python scripts/extract_features.py --dataset dtd --root /data --family clip --backbone ViT-B/16 --out features
# realistic batch protocol, "Low" setting:
python scripts/run_batch.py --features features/dtd__ViT-B-16.pt --method graft --batch_size 64 --kmin 2 --kmax 10 --n_tasks 1000 --out results/raw/dtd_B64_K2-10_graft.json
# online protocol:
python scripts/run_online.py --features features/dtd__ViT-B-16.pt --batch_size 128 --gamma 0.01 --n_runs 100 --out results/raw/dtd_online_g0.01_graft.json
# tables and plots from results:
python visualization/generate_tables.py && python visualization/plot_results.py
```

## Using the official StatA / MOON harness (recommended for the paper tables)
See `scripts/official_harness_patch.md`: copy `graft/official_adapter.py` into the official repository's
`solvers/` folder; the solver signature matches `StatA_solver`, so all official samplers, features and baselines
are shared. The official repositories are AGPL-3.0; nothing from them is vendored here.

## Compile the paper
```bash
cd paper && latexmk -pdf main.tex && latexmk -pdf supplementary.tex
```
