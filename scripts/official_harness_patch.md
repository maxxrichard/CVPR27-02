# Running GRAFT inside the official StatA / MOON evaluation harness

The official repositories (AGPL-3.0, not vendored here):
* https://github.com/MaxZanella/StatA   (CVPR 2025; batch + online samplers, TransCLIP/StatA/ZLaP/Dirichlet/TDA/DMN/TENT)
* https://github.com/walawalagoose/MOON (ICML 2026; same harness + MOON, OGA, GDA, OpenCLIP support)

Steps (identical for both):
1. `pip install -e /path/to/this/repo` (or add it to `PYTHONPATH`).
2. Copy `graft/official_adapter.py` to `<repo>/solvers/GRAFT.py`.
3. In `<repo>/solvers/__init__.py` add: `from .GRAFT import GRAFT_solver`.
4. In `<repo>/main.py`:
   * add `'GRAFT'` to the `choices` of `--method`;
   * in `get_hp(...)` add
     ```python
     elif method_name == 'GRAFT':
         return GRAFT_solver, {'alpha': 1.0, 'lam': 4.0, 'graph_knn': 5, 'empirical_bayes': False,
                               'presence': True, 'lambda_y_hat': 1, 'lambda_laplacian': 1, 'n_neighbors': 3}
     ```
   * the online loop calls the solver per batch; for the stateful online variant use `graft.GRAFTOnline`
     directly (see `scripts/run_online.py`) or run the batch solver per batch (transductive-within-batch).
5. Run the official scripts, e.g. `bash scripts/StatA_batch.sh /data vit_b16` with `--method GRAFT`.

The solver signature `GRAFT_solver(query_features, query_labels, clip_prototypes, **hp) -> (y_hat, z)` matches
`StatA_solver`; `clip_prototypes` may be `(1, d, K)` or `(d, K)`.
