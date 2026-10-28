# Deliverable A: Research Trend Analysis, CVPR 2025 vs CVPR 2026

## 1. Sources and method
Official pages (cvpr.thecvf.com, openaccess.thecvf.com, arxiv.org) could not be fetched from this session's
network, so the analysis combines (i) official statistics quoted in press releases and the CVPR account,
retrieved through web search, (ii) two machine-readable accepted-paper lists hosted on GitHub, and (iii) reading
of the official code repositories of the key papers.

| Item | CVPR 2025 | CVPR 2026 | Source |
|---|---|---|---|
| Submissions reviewed | 13,008 | 16,092 | CVPR 2025 wrap-up; CVPR 2026 technical-program release |
| Accepted (main) | 2,872 (22.1%) | 4,090 (25.4%); 4,089 presentations | official wrap-up / @CVPR; one tracker reports 4,071 |
| Findings track | n/a | 1,717 recommended | CVPR 2026 release |
| Best paper | VGGT (feed-forward 3D) | D4RT (dynamic 4D scene reconstruction) | award pages |
| Best student paper | Neural inverse rendering from propagating light | Native and compact structured latents for 3D generation | award pages |
| Honourable mentions | MegaSaM, Navigation World Models, others | SAM 3D; NitroGen; ChordEdit (student) | award pages (sources disagree on the exact list) |
| Highest-submission areas (official) | not retrieved | image/video synthesis and generation; vision, language and reasoning; multimodal learning; 3D from multi-view and sensors; medical and biological vision | CVPR 2026 release |

Machine-readable lists used for counting:
* CVPR 2026: `gisbi-kim/cvpr2026-explorer` (4,070 main-conference papers with abstracts and oral/highlight/poster
  type; 141 orals, 575 highlights, 3,354 posters). The explorer also ships a semantic topic taxonomy.
* CVPR 2025: `gisbi-kim/cvml-paper-phylogeny` (`papers.json`, entries with venue CVPR and year 2025: 3,532
  titles). **Caveat:** this count exceeds the 2,872 accepted main-conference papers, so the list also contains
  workshop papers (for example the CVPRW paper OGA appears in it). The 2025 percentages below are therefore
  slightly diluted; growth factors should be read as indicative, not as official statistics.

Reproducible script: `docs/trends/keyword_stats.py` (title-only regular expressions applied identically to both
years); raw output in `docs/trends/keyword_stats_output.txt`.

## 2. Title-keyword statistics (identical regexes for both years)

| Area (title keywords) | 2025 (n, %) | 2026 (n, %) | share growth |
|---|---|---|---|
| Vision-language / multimodal (VLM, MLLM, CLIP) | 435 (12.3) | 686 (16.9) | 1.37x |
| Diffusion / flow matching | 387 (11.0) | 378 (9.3) | 0.85x |
| Video generation / world models | 114 (3.2) | 149 (3.7) | 1.13x |
| 3D reconstruction / NVS / Gaussian splatting / NeRF | 167 (4.7) | 208 (5.1) | 1.08x |
| Embodied AI / robotics / VLA | 84 (2.4) | 166 (4.1) | 1.71x |
| Open-vocabulary | 34 (1.0) | 51 (1.3) | 1.30x |
| Test-time adaptation / training / transductive | 38 (1.1) | 66 (1.6) | 1.51x |
| Zero-/few-shot / training-free | 155 (4.4) | 172 (4.2) | 0.96x |
| Segmentation / dense prediction | 233 (6.6) | 230 (5.7) | 0.86x |
| Self-supervised / representation / foundation features | 136 (3.9) | 119 (2.9) | 0.76x |
| Medical imaging | 63 (1.8) | 90 (2.2) | 1.24x |
| Robustness / OOD / domain shift | 178 (5.0) | 170 (4.2) | 0.83x |
| Efficiency / compression / distillation | 283 (8.0) | 359 (8.8) | 1.10x |
| Model merging / task vectors | 10 (0.3) | 23 (0.6) | 2.00x |
| Calibration / uncertainty / conformal | 62 (1.8) | 79 (1.9) | 1.11x |
| Agents | 36 (1.0) | 100 (2.5) | 2.41x |
| "test-time scaling/compute/training" | 3 (0.1) | 12 (0.3) | 3.47x |

CVPR 2026 semantic taxonomy (explorer labels, papers may carry several): Generative Models 969, Multimodal &
Language 967 (Vision-Language Models 825), Learning Algorithms 829, 3D Vision & Geometry 788 (Reconstruction 457),
Robustness & Safety 486, Video & Motion 431, Segmentation & Dense Prediction 381, Recognition 343, Data &
Evaluation 319, Low-level 298, Detection & Tracking 286, Robotics & Embodied 267, Medical 213.

Abstract-level counts for CVPR 2026 (explorer abstracts): "test-time adaptation" 39; "vision-language model" 378;
"feed-forward 3D / VGGT" 63; "model merging / task vector" 16; "feature upsampling" 3; "register/artifact
tokens" 4; "label shift / class imbalance" 14; "calibration / conformal" 183.

## 3. Ranked research areas (evidence-based, with opportunity assessment)

| Rank | Area | Representative CVPR 2025 | Representative CVPR 2026 | Importance | Saturation | Open gap | Data/code | Compute | Novelty potential |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Feed-forward 3D / 4D reconstruction | VGGT (best paper), MegaSaM | D4RT (best paper), SAM 3D, tttLRM, MuM, VGGT-Segmentor | very high | medium | long-context, dynamic, outlier views | good | very high (multi-GPU pretraining) | high but infeasible here |
| 2 | VLM / MLLM reasoning and grounding | many | TIPSv2, Molmo2, GR3D | very high | high | patch-text alignment, hallucination | good | very high | medium |
| 3 | Generative video / world models | many | Back to Basics, MacTok, EgoX | very high | high | efficiency, physics | good | very high | medium |
| 4 | Realistic unlabeled adaptation of VLMs (TTA / transduction) | StatA (highlight), COSMIC, Bayesian TTA, O-TPT, Conf-OT, OGA (W) | LATA (oral), STAR, Dynamic Logits Adjustment, BOLT, IMA&TMA (W), Vocabulary Scaling Law (highlight) | high | medium | documented collapse under few effective classes, non-iid streams, backbone transfer, calibration (StatA; "Illusion of Progress", NeurIPS 2025) | excellent (11 datasets, official harness, cached features) | low (no training) | high: principled fix of a documented failure |
| 5 | Training-free open-vocabulary segmentation | CASS, LPOSS, ResCLIP | GLA-CLIP, VR-CLIP, Retrieve&Segment, Power of Prior | high | medium-high | heuristic last-layer edits, window inconsistency | good | medium | medium |
| 6 | Model merging / task vectors | TSV-M | DuetMerging, DC-Merge, ACE-Merging, AlphaMerging (F) | medium-high | high (many SVD variants) | data-free interference | good | low | medium-low |
| 7 | ViT token aggregation / registers / artifacts | (ECCV/NeurIPS lineage) | Vision Transformers Need More Than Registers | high | low | root cause of artifacts | good | very high (pretraining) | high but infeasible here |
| 8 | Feature upsampling for VFMs | (ICLR/ICCV/NeurIPS lineage) | Upsample Anything, (AnyUp ICLR 2026) | medium | low-medium | principle behind upsampling | good | medium | medium |
| 9 | Efficiency / token reduction for VLMs | many | many | high | very high | — | good | medium | low |
| 10 | Medical VLM adaptation and uncertainty | Conf-OT | LATA (oral) | high | medium | coverage under shift | medium | low | medium |

Popularity and opportunity diverge: areas 1 to 3 dominate the proceedings but demand pretraining-scale compute
and are saturated by large labs; area 4 is smaller (about 40 to 66 papers per year) yet grew 1.5x in share,
has an officially documented, unresolved failure mode, cheap and fully reproducible experiments on cached
features, an official harness, a CVPR 2025 highlight paper as blueprint and an ICML 2026 follow-up that leaves the
cross-class structure untouched.

## 4. Selected direction and justification
**Unlabeled (transductive and online) adaptation of vision-language models under realistic class presence.**
* Methodological weakness: all methods (GMM, vMF, caches, label propagation) estimate per-class parameters
  independently from each class's own test samples; absent and rare classes collapse or stay at biased anchors.
* Measurable gap: StatA reports drops of 26 to 38 points for TransCLIP/ZLaP at very low effective classes; the
  NeurIPS 2025 benchmark finds marginal gains, poor SigLIP transfer and worse calibration.
* Benchmarks: 11 datasets + 4 ImageNet shifts, official samplers, official baselines, cached features.
* Principled innovation: a graph-structured prior over the class vocabulary (partial pooling) with closed-form
  inference and provable properties; the dominant anchor-based methods are exact special cases.
* Falsifiable: class-level smoothness of displacement fields can be measured with labels; the accuracy curve
  versus effective classes is the pre-registered test.
