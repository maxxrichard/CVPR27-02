# Deliverable D: Implementation Notes

## Modules
| File | Content | Status |
|---|---|---|
| `graft/config.py` | `GRAFTConfig` dataclass (all hyper-parameters, pseudo-count units) | done |
| `graft/class_graph.py` | text kNN class graph, normalised Laplacian, Dirichlet energy, smoothness ratio | done, tested |
| `graft/solver.py` | tempered Gaussian likelihood; sample graph; closed-form field solve (exact / batched CG); EB evidence; E-step; M-steps; objective; `graft_batch`; `GRAFTOnline` | done, tested |
| `graft/baselines.py` | zero-shot; anchor-only control; complete pooling; presence only | done |
| `graft/samplers.py` | realistic batch and Dirichlet-stream samplers (re-implemented) | done, tested |
| `graft/metrics.py` | accuracy, ECE, NLL, per-class accuracy | done, tested |
| `graft/data.py` | cached feature I/O | done |
| `graft/official_adapter.py` | drop-in `GRAFT_solver` for the StatA/MOON harness | done, tested (shapes) |
| `scripts/extract_features.py`, `scripts/datasets_registry.py` | CLIP / OpenCLIP / SigLIP feature caching, CoOp dataset layout | written, **not executed** (no data / model access in this session) |
| `scripts/run_batch.py`, `scripts/run_online.py`, `scripts/analyze_smoothness.py` | evaluation harness | written, **not executed on real data** |
| `visualization/*` | figures 1 to 3 (rendered), result plots and tables (run on empty results: produce TBD tables, draw nothing) | done |
| `tests/` | 17 pytest tests on synthetic data | all pass |

## Tensor shapes
`X: N x d` (unit rows), `T: K x d` (unit rows), `y_hat, z: N x K`, `D: K x d`, `var: d`, `pi: K`,
`L_T: K x K` dense (K <= a few thousand), `W_s: N x N` sparse (knn entries per row), online statistics
`n: K`, `S: K x d`, `q: d`, `N: scalar`.

## Numerical notes
* The field solve is SPD; CG with Jacobi preconditioning converges in < 30 iterations in tests (tolerance 1e-6).
  Exact batched solve is used for K <= 256 and verified against CG.
* The likelihood temperature kappa = 50 keeps responsibilities soft; variances are clamped at 1e-8.
* EB uses an isotropic approximation and is disabled when fewer than 3 classes are present.
* Prototype re-normalisation is off by default to keep the exact MM structure (ablation flag provided).

## Known deviations from the blueprint
* Shared diagonal covariance (TransCLIP) instead of per-class (StatA); soft counts instead of hard counts; no
  prototype re-normalisation; mixing proportions estimated. Each is an ablation row.
