# Mock CVPR review (strict main-track reviewer)

**Summary.** The paper proposes GRAFT, a transductive / online adaptation method for VLMs that models adapted
prototypes as text anchors plus a displacement field with a GMRF prior over a text-derived class graph, adds a
label-shift estimate of class presence, and provides closed-form inference plus an empirical Bayes rule.

**Strengths.**
1. The diagnosis is precise and tied to a documented failure (StatA, Illusion of Progress): per-class independence
   of adaptation parameters. The proposed prior is a clean generalisation whose extremes are the existing anchor
   mechanism and a global shift; Proposition 1 makes the relation to StatA/MOON exact.
2. The mechanism is analysable: absent classes are harmonic extensions (Kron reduction), posterior variance never
   increases, block updates are monotone; the hypothesis (class-level smoothness) is explicit and testable with
   labels independently of the method.
3. Practicality: gradient free, cached features, drop-in for the official harness, EB removes the main tuned
   constant; the paper pre-registers success criteria and ships controls (random graph, complete pooling,
   anchor-only) that could falsify the claim.

**Weaknesses.**
1. **No experimental evidence in the current draft.** All tables are TBD. Without the main table, the online table
   and especially the random-graph control, the contribution cannot be assessed. (Blocking.)
2. The novelty could be perceived as incremental: "StatA with a Laplacian over classes and a Dirichlet on pi".
   The authors must show that pooling along *semantics* (not just any pooling) helps, and that gains are not
   explained by the presence term alone.
3. The likelihood is the inherited tempered Gaussian (kappa = 50) rather than a principled hyperspherical model;
   MOON argues for vMF. A vMF variant of the field or a justification of the temperature is needed. EB uses an
   isotropic approximation whose effect is unquantified.

**Possible rejection reasons.** Missing results; failure of H1 on most datasets; gains < 1 point over StatA/MOON;
no improvement at low K_eff where the method is motivated; omission of MOON as a baseline.

**Experiments needed.** E1 smoothness analysis; E2 full main table with official baselines; random-graph and
complete-graph controls; K_eff curve; online table; SigLIP/OpenCLIP; calibration; per-dataset tables in the
supplement; runtime.

**Claims requiring stronger evidence.** "Backbone-agnostic" (needs SigLIP numbers); "removes hyper-parameter
tuning" (needs EB vs fixed comparison across datasets); "calibration" (needs ECE table).

**Potential overlaps.** StatA / MOON (anchoring), COSMIC (class graph), DGP / PPN (prototype propagation on class
graphs), LSA / OnZeta / Frolic (label shift), IMA&TMA (shared transform), LATA (Laplacian smoothing, over samples).

**Revisions already made in response to this review.** The special-case propositions and the random-graph /
complete-pooling / presence-only controls were added to the ablation design; MOON was added as a baseline; the
vMF variant and the isotropic EB approximation are listed as limitations; the claims in the abstract and
conclusion are conditioned on pending evidence.
