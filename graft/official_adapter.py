"""Drop-in solver for the official StatA / MOON evaluation harnesses (github.com/MaxZanella/StatA,
github.com/walawalagoose/MOON).  Copy this file to `<repo>/solvers/GRAFT.py`, add
`from .GRAFT import GRAFT_solver` to `<repo>/solvers/__init__.py` and register the method name in `main.py`
(see scripts/official_harness_patch.md).  The signature matches `StatA_solver`.
"""
import torch
from graft import GRAFTConfig, graft_batch


def GRAFT_solver(query_features, query_labels, clip_prototypes, alpha=1.0, lam=4.0, graph_knn=5,
                 empirical_bayes=False, presence=True, lambda_y_hat=1.0, lambda_laplacian=1.0, n_neighbors=3, max_iter=10, **kw):
    T = clip_prototypes
    if T.dim() == 3:                      # (1, d, K) layout used by the official code
        T = T.squeeze(0)
    if T.shape[0] != query_features.shape[1]:
        T = T.T                            # make it (d, K)
    T = T.T.contiguous()                   # (K, d)
    cfg = GRAFTConfig(alpha=alpha, lam=lam, graph_knn=graph_knn, empirical_bayes=empirical_bayes, presence=presence,
                      lambda_y=lambda_y_hat, lambda_laplacian=lambda_laplacian, sample_knn=n_neighbors, n_outer=max_iter)
    dev = query_features.device
    y_hat, z = graft_batch(query_features.to(dev), T.to(dev), cfg)
    return y_hat.cpu(), z.cpu()
