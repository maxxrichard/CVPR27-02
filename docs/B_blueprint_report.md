# Deliverable B: Blueprint Paper Report

## Candidate blueprint papers (read through official code, abstracts and documented results)
| # | Paper | Venue | Code | Why considered |
|---|---|---|---|---|
| 1 | Zanella, Fuchs, De Vleeschouwer, Ben Ayed. *Realistic Test-Time Adaptation of Vision-Language Models* (StatA) | CVPR 2025 (Highlight), arXiv:2501.03729 | github.com/MaxZanella/StatA (AGPL-3.0) | defines the realistic protocols, documents the collapse, principled anchor regulariser |
| 2 | Zanella, Gérin, Ben Ayed. *Boosting Vision-Language Models with Transduction* (TransCLIP) | NeurIPS 2024 (Spotlight) | github.com/MaxZanella/transduction-for-vlms | the GMM + KL + Laplacian objective and block-MM solver that StatA extends |
| 3 | Huang, Liu, Wang, Ju, Qiao, Luo. *Von Mises-Fisher Mixture Model with Dynamic Shrinkage for Realistic Test-Time Transduction* (MOON) | ICML 2026, arXiv:2607.15851 | github.com/walawalagoose/MOON | penalised-likelihood view of anchoring; dynamic per-class/instance shrinkage |
| 4 | Sheng, Liang, He, Wang, Tan. *The Illusion of Progress? A Critical Look at TTA for VLMs* | NeurIPS 2025 | github.com/TomSheng21/tta-vlm | unified benchmark; SigLIP and calibration findings |
| 5 | Bozorgtabar et al. *LATA: Laplacian-Assisted Transductive Adaptation for Conformal Uncertainty in Medical VLMs* | CVPR 2026 (Oral), arXiv:2602.17535 | — | transductive kNN smoothing with conformal validity |
| 6 | Martin et al. *Transductive Zero-Shot and Few-Shot CLIP* | CVPR 2024 | github.com/SegoleneMartin/transductive-CLIP | Dirichlet mixture on the simplex |
| 7 | Kalantidis, Tolias. *Label Propagation for Zero-shot Classification with VLMs* (ZLaP) | CVPR 2024 | — | graph propagation over samples |
| 8 | Fuchs, Zanella, De Vleeschouwer. *Online Gaussian Test-Time Adaptation of VLMs* (OGA) | CVPRW 2025 | github.com/cfuchs2023/OGA | online Gaussian MAP, expected tail accuracy |

**Selected blueprint: StatA (CVPR 2025).** It frames the problem, defines the evaluation protocols that expose the
gap, provides an official harness with all baselines, and its mechanism is simple enough to be analysed exactly.

## Blueprint analysis (StatA)
1. **Research question.** Can unlabeled transductive / online adaptation of VLMs be made robust when batches contain
   a variable, possibly very small number of effective classes and when streams are correlated?
2. **Motivation.** Existing transductive methods assume all classes are present and balanced; under realistic
   batches they fall far below zero-shot (reported: TransCLIP -26.3, ZLaP -37.7 points at "very low" K_eff).
3. **Formulation (from the official solver).** Gaussian mixture on unit-norm CLIP features with per-class means
   `mu_k` and per-class diagonal covariances; assignments `z` regularised by the zero-shot posterior `y_hat^lambda`
   (KL anchor) and a kNN Laplacian over samples; the likelihood is tempered by 1/50.
   Anchor: `beta_k = n_k / (n_k + alpha)` with `alpha = 1` (hard counts by default);
   `mu_k = normalize(beta_k * xbar_k + (1 - beta_k) * t_k)`;
   `cov_k = beta_k * cov_hat_k + (1 - beta_k) * (cov_init + (t_k - mu_k)^2)`.
4. **Architecture.** Frozen CLIP; cached features; 10 outer iterations; E-step with 5 inner sweeps.
5. **Algorithmic innovation.** The count-dependent convex combination between empirical statistics and text-derived
   statistics (a KL-to-anchor penalty in the exponential family), which prevents collapse when n_k is small.
6. **Objectives.** Penalised negative log-likelihood = GMM NLL + KL(z || y_hat) + Laplacian + anchor penalty.
7. **Experiments.** 11 datasets; batch protocol with K_eff ranges at B=64 and B=1000 plus full set; online protocol
   with Dirichlet-correlated streams; CLIP ViT-B/16 main; 1000 tasks / 100 runs.
8. **Strongest results (official README).** Averages over 11 datasets: StatA 70.4 / 69.3 / 67.4 / 69.7 / 69.8 /
   69.0 / 69.9 across the seven batch settings vs CLIP 65.2; online 67.0 / 68.9 / 69.5 / 69.1.
9. **Ablations.** Anchor weight alpha, soft vs hard counts, Laplacian weight; the paper studies the ratio of batch
   size to K_eff.
10. **Assumptions and limitations.** (a) Per-class independence: the anchor of class k is only its own text
    embedding; no information flows between classes. (b) Absent classes keep mu_k = t_k even under a shared domain
    shift and keep competing with uniform implicit mixing weights. (c) alpha is a fixed global constant (one
    pseudo-count). (d) Gaussian with per-class covariance on normalised features; prototype re-normalisation breaks
    the exact MM structure. (e) Evaluated with CLIP only.
11. **Open challenges.** Cross-class information sharing; explicit label-shift modelling; hyper-parameter-free
    strength selection; backbone transfer; calibration.
12. **Opportunity.** Replace independent per-class anchoring with a structured prior over the class vocabulary.

## Reverse engineering: essential vs optional design decisions
* Essential: zero-shot prior in the E-step (KL anchor); count-dependent anchoring of the means; cached features.
* Optional / inherited: sample Laplacian (helps in the all-classes setting), per-class covariance anchoring,
  prototype re-normalisation, hard vs soft counts, temperature 50.
* Why it works: with few samples the posterior mean stays near the text anchor, so the model never hallucinates
  prototypes for classes without evidence.
* When it fails: whenever the text anchor is itself displaced from the image distribution (modality gap, domain
  shift) and the class has no or few samples; and when absent classes with anchors near present-class data keep
  stealing responsibility because mixing weights are uniform.

## Closest competitor: MOON (ICML 2026)
From the official solver: vMF mixture on the sphere, same E-step structure (tempered, Laplacian, y_hat prior),
mean update `normalize(beta_k * v_k + (1 - beta_k) * t_k)` with `beta_k = n_k / (alpha_k + n_k)`,
`alpha_k = alpha / class_weight_k` where class weights are the geometric mean of mean and max zero-shot confidence
per class, and sample weights `1 - normalised entropy` of z. Concentration kappa_k via the Banerjee approximation.
Still per-class and anchored at the text embedding; no cross-class pooling, no mixing proportions.

Both StatA and MOON correspond exactly to the lambda = 0 (graph-free) special case of the GRAFT prior.
