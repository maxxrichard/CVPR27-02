"""GRAFT: Graph-Regularized Anchor-displacement Field Transduction for vision-language models.

Training-free, label-free adaptation of a frozen VLM classifier from an unlabeled test batch or
stream.  The class prototypes are modelled as text anchors plus a displacement field that is
partially pooled over a semantic class graph (a Gaussian Markov random field prior), together with
an explicit class-presence (label-shift) model.  See docs/ and paper/ for the derivation.
"""
from .config import GRAFTConfig
from .class_graph import build_class_graph, normalized_laplacian, dirichlet_energy
from .solver import graft_batch, GRAFTOnline, GRAFTState
from .baselines import zero_shot, count_shrinkage, global_shift

__all__ = [
    "GRAFTConfig", "build_class_graph", "normalized_laplacian", "dirichlet_energy",
    "graft_batch", "GRAFTOnline", "GRAFTState", "zero_shot", "count_shrinkage", "global_shift",
]
__version__ = "0.1.0"
