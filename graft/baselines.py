"""Simple baselines and controls expressed in the same framework.

* zero_shot          : the frozen VLM classifier.
* count_shrinkage    : GRAFT with lam = 0 (independent shrinkage of every class towards its own anchor,
                       the mechanism of StatA/MOON in our likelihood; the architecture-matched control).
* global_shift       : GRAFT on the complete class graph with alpha = 0 and very large lam (complete pooling: one
                       displacement shared by all classes = modality/domain offset).
* label_shift_only   : no displacement at all, only the class-presence (pi) correction (Saerens-style EM).
For the published baselines (TransCLIP, StatA, MOON, ZLaP, ECALP, TDA, DMN, OGA, ...) use their official code
with the adapter in graft/official_adapter.py so that protocols match exactly.
"""
import dataclasses
import torch
from .config import GRAFTConfig
from .solver import graft_batch
from .class_graph import complete_graph


def zero_shot(X, T, logit_scale: float = 100.0):
    X = torch.nn.functional.normalize(X.float(), dim=-1)
    T = torch.nn.functional.normalize(T.float(), dim=-1)
    y = torch.softmax(logit_scale * X @ T.T, dim=1)
    return y, y


def count_shrinkage(X, T, cfg: GRAFTConfig = None, alpha: float = 1.0):
    cfg = dataclasses.replace(cfg or GRAFTConfig(), lam=0.0, alpha=alpha, empirical_bayes=False)
    return graft_batch(X, T, cfg)


def global_shift(X, T, cfg: GRAFTConfig = None, lam: float = 1e4):
    cfg = dataclasses.replace(cfg or GRAFTConfig(), alpha=0.0, lam=lam, empirical_bayes=False)
    W = complete_graph(T.shape[0], dtype=torch.float32, device=T.device)
    return graft_batch(X, T, cfg, W_c=W)


def label_shift_only(X, T, cfg: GRAFTConfig = None):
    cfg = dataclasses.replace(cfg or GRAFTConfig(), alpha=1e9, lam=0.0, empirical_bayes=False, presence=True)
    return graft_batch(X, T, cfg)
