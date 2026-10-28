# Deliverable H: Research Integrity and Novelty Audit

## Genuinely novel (to the best of a literature search on 2021 to 2026 work)
1. Modelling VLM prototype adaptation as a **displacement field over the class vocabulary** with a **Gaussian
   Markov random field prior on the semantic class graph** (partial pooling across classes), with the closed-form
   graph-filter M-step `D = (N_z + s(alpha I + lambda L_T))^{-1} N_z E`.
2. The analysis: StatA/MOON shrinkage and the global shift as the two extreme special cases; absent classes as
   harmonic extensions via Kron reduction; posterior-variance dominance; monotone block updates.
3. Empirical Bayes selection of the pooling strength from the unlabeled batch via the Schur complement onto present
   classes.
4. Joint treatment with an explicit presence / label-shift estimate inside the same objective.

## Inherited from earlier work (cited in the paper)
* Generative transduction with KL anchor to the zero-shot posterior and sample-graph Laplacian: TransCLIP (NeurIPS
  2024); tempered likelihood (1/50) and 10-iteration block scheme: TransCLIP/StatA.
* Count-dependent anchoring (the alpha term): StatA (CVPR 2025), MOON (ICML 2026).
* Label-shift EM for class priors: Saerens et al. 2002; for VLMs: Frolic, OnZeta, LSA (2026 preprint).
* Realistic evaluation protocols and the eleven-dataset suite: StatA; the Dirichlet-correlated stream sampler
  description: StatA appendix. Our samplers are re-implemented from the description (no code copied; official
  repos are AGPL-3.0).
* Class graphs in zero-shot learning (DGP, CVPR 2019; PPN, IJCAI 2019) and in CLIP TTA (COSMIC, CVPR 2025) use
  graph propagation for weights / clique selection, not a GMRF prior on displacements.

## Closest prior work and the difference
| Work | Shared with GRAFT | Different in GRAFT |
|---|---|---|
| StatA (CVPR 2025) | anchor shrinkage with soft counts, protocols | cross-class coupling lambda L_T, presence pi, EB, shared covariance, exact MM |
| MOON (ICML 2026) | dynamic per-class shrinkage | pooling across classes (MOON is still per class), presence |
| TransCLIP (NeurIPS 2024) | objective skeleton | field prior, presence |
| LATA (CVPR 2026 oral) | Laplacian smoothing | over samples (LATA) vs over classes (GRAFT); conformal focus |
| COSMIC (CVPR 2025) | class graph from text | used for clique selection in caches, not as a prior on prototypes |
| TPS (WACV 2025) | prototype shifts | per-image gradient-based shifts vs batch-level closed-form field |
| IMA/TMA (CVPRW 2026) | shared transform in embedding space | gradient-based per-image adapters vs closed-form pooled field |
| LSA (arXiv 2606.15169) | label-shift correction for CLIP | GRAFT combines it with prototype adaptation |

## What has been implemented
`graft/` package: solver (batch + online), class graph, baselines/controls, samplers, metrics, feature I/O,
official-harness adapter; scripts for feature extraction, batch/online evaluation, smoothness analysis;
visualisation scripts; 23 configs; LaTeX manuscript and supplementary; figures 1 to 3.

## What has actually been tested
17 unit tests on synthetic data only (`docs/tests_passed.txt`): shapes/probabilities; Laplacian properties; closed
form vs L-BFGS and CG vs exact solve; lambda = 0 reduces to count shrinkage; complete pooling reduces to global
shift; harmonic-extension property for absent classes; monotone objective decrease; improvement over zero-shot and
suppression of absent classes on synthetic shifted mixtures; graph pooling lowers prototype error for rare classes
on a synthetic smooth field; online equals batch on one batch; stream runs; EB selects lambda > 0 on a smooth
synthetic field; determinism; sampler protocol properties; metrics; adapter shapes. LaTeX compiles.

## What remains unverified
* Every benchmark number for GRAFT, MOON, OGA and the controls. At the authors' request the manuscript tables and
  plots now contain **placeholder values** (grey entries, watermarked plots, produced by
  `scripts/make_placeholder_results.py`) so that layout and pipeline can be exercised; they are synthetic, not
  measured, and must be replaced before any use. Baseline numbers for CLIP, MTA, EM-Dirichlet, ZLaP, TransCLIP,
  StatA, TENT, TDA and DMN are the published 11-dataset averages from the official StatA README. No dataset was downloaded and no VLM was run in this session, by
  the user's instruction.
* H1 on real data (whether oracle displacement fields are smooth on the text graph).
* Whether EB improves over fixed strengths on real data.
* Runtime figures.
* The claim that the method is backbone-agnostic.

## Placeholder data in figures
`plot_keff.pdf`, `plot_sensitivity.pdf`, `plot_smoothness.pdf` and `plot_qualitative.pdf` are drawn from
`results/placeholder/` and carry a red PLACEHOLDER watermark; `plot_embedding_synthetic.pdf` is real solver output
on the synthetic unit-test generator and is labelled synthetic in its caption. The Results and Ablation text
describes the placeholder numbers and must be rewritten to the measured ones.

## Experiments essential before submission (in order)
E1 (smoothness), E2 (main table with official baselines through the StatA/MOON harness), E6/E7 (controls including
random graph), E3 (online), E5 (calibration), E4 (backbones), E8 (cost).

## Limitations that could threaten acceptance
* If H1 fails, the graph term brings nothing and the contribution reduces to presence + EB on top of StatA.
* Reviewers may view the method as "StatA plus a Laplacian"; the paper must show the random-graph control and the
  K_eff curve to demonstrate that pooling along semantics is what helps.
* MOON (ICML 2026) is a strong contemporaneous baseline; it must be included with its official code.
* The inherited tempered Gaussian likelihood is heuristic; a vMF version of GRAFT is a natural reviewer request.
* The 2025 trend statistics come from a list that includes workshop papers.
