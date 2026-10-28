# Deliverable C: Novel Research Proposal (GRAFT)

## Problem statement
Given a frozen vision-language model, unit-norm image features `X in R^{N x d}` of an unlabeled test batch (or a
stream of batches) and text anchors `T in R^{K x d}` for a vocabulary of K classes, produce posteriors `z_i` that
are more accurate and at least as well calibrated as the zero-shot posterior `y_hat`, **without labels, training
or gradients**, and without collapsing when (i) only K_eff << K classes are present in the batch, (ii) the stream is
correlated, (iii) the backbone is not CLIP.

## Precise research gap
All existing transductive / online methods estimate **per-class parameters from that class's own test samples**:
GMM means (TransCLIP), anchored GMM means and covariances (StatA), vMF mean directions (MOON), per-class caches
(TDA, DMN, BoostAdapter), per-class Gaussians (OGA), or sample-graph propagation seeded at the text prompts (ZLaP,
ECALP, LATA). Consequences, all documented or directly implied by the official code:
1. A class with no samples is left at its text anchor (anchored methods) or fitted to foreign samples (unanchored
   methods), although the anchor is displaced from the image distribution by a shift that is largely shared across
   classes (modality gap, domain shift).
2. A class with one or two samples is adapted from those samples alone; no other class informs it.
3. Mixing weights are implicitly uniform, so absent classes keep competing for samples.
4. The anchor strength is a fixed constant tuned once; the NeurIPS 2025 benchmark criticises such tuning.

## Scientific insight and hypothesis
**Insight.** The cross-modal displacement field `e*_k = E[x | y = k] - t_k` is not an independent per-class
quantity: it has a global component and a component that varies smoothly over the semantics of the classes. The
text encoder already provides the geometry of the vocabulary; a graph over text anchors is therefore a prior over
how displacements should co-vary.

**Hypothesis H1 (class-level smoothness).** The Dirichlet energy of the oracle displacement field on the text kNN
class graph is substantially smaller than on degree-matched random graphs. (Falsifiable with labels, independent of
any adaptation method.)

**Hypothesis H2 (consequence).** Partial pooling of the field over the class graph, with explicit presence
inference, gives (a) no degradation below zero-shot in any realistic setting, (b) gains over anchor-only shrinkage
that grow as K_eff decreases, (c) a flat accuracy curve versus K_eff, and (d) no loss in the all-classes setting.
Refutation: if the random-graph control matches the text graph, or if gains vanish at low K_eff, H2 is refuted.

## Method summary (GRAFT: Graph-Regularized Anchor-displacement Field Transduction)
* Prototypes `mu_k = t_k + d_k`; displacement field `D in R^{K x d}`.
* GMRF prior `D_{:,j} ~ N(0, Q^{-1})`, `Q = (alpha I + lambda L_T) / (kappa sigma_bar^2)`; `L_T` is the normalised
  Laplacian of the text kNN graph. alpha: anchor (StatA/MOON mechanism); lambda: cross-class coupling (new).
* Class presence `pi_k = (n_k + c) / (N + K c)` (Dirichlet-regularised label-shift estimate).
* Shared diagonal covariance, tempered Gaussian likelihood, zero-shot KL anchor and sample Laplacian inherited.
* Closed-form M-step: `d_{:,j} = (N_z + s_j (alpha I + lambda L_T))^{-1} N_z e_{:,j}`, solved by batched CG.
* Empirical Bayes rule for (alpha, lambda) from the marginal likelihood of present-class displacements (Schur
  complement onto present classes).
* Online: sufficient statistics (n, S, q, N) with forgetting; same M-steps.

## Expected benefits and failure modes
Benefits: no collapse (anchor), correct placement of absent/rare classes under shared shift (graph), suppression of
absent classes (presence), hyper-parameter-free strength (EB), backbone-agnostic (cosine geometry only), cheap.
Failure modes: polysemous class names (text neighbours visually unrelated); multi-domain batches (one field per
class); fully disconnected graph components with no present class (reduces to anchor only); tiny batches with
< 3 present classes (EB disabled).

## Novelty evidence (see H_novelty_audit.md)
No prior work regularises VLM prototype displacements with a GMRF over the class graph or analyses absent classes
as harmonic extensions; StatA/MOON are exact lambda = 0 special cases; COSMIC uses class graphs for clique
selection, PPN/DGP propagate prototypes in supervised zero-shot learning with trained propagation.

## Required validation experiments (priority order)
1. Main realistic batch table (7 settings x 11 datasets, 1000 tasks) with all baselines and the four controls.
2. H1 smoothness measurement with labels on all datasets (cheap; decides the narrative).
3. Online streams (4 settings x 11 datasets, 100 runs).
4. Backbone transfer (OpenCLIP, SigLIP, ViT-L/14, RN50) and ImageNet shifts.
5. Component and prior ablations including the random-graph control.
6. Calibration (ECE, NLL) and worst-decile accuracy.
7. Runtime table.
