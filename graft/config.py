from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence
import json


@dataclass
class GRAFTConfig:
    """All hyper-parameters of GRAFT.  Defaults are used for every dataset and backbone.

    Pseudo-count parameterisation: the prior precision on the displacement field is
        Q = (1 / (kappa * sigma_bar^2)) * (alpha * I_K + lam * L_T),
    so that `alpha` and `lam` are expressed in units of (soft) sample counts.  With lam = 0 and
    alpha = 1 the M-step reduces exactly to the count-based shrinkage beta_k = n_k / (n_k + 1).
    """
    # Zero-shot prior
    logit_scale: float = 100.0          # CLIP logit scale tau
    lambda_y: float = 1.0               # exponent of the zero-shot posterior in the E-step (KL anchor)
    # Class-conditional likelihood
    cov: str = "diag"                   # "diag" (shared across classes) or "iso"
    kappa: float = 50.0                 # likelihood temperature (covariance inflation), inherited from TransCLIP
    # Displacement-field prior (pseudo-count units)
    alpha: float = 1.0                  # pull towards the text anchor (zero displacement)
    lam: float = 4.0                    # graph smoothness across semantically related classes
    graph_knn: int = 5                  # neighbours per class in the text kNN graph
    graph_rho: float = 0.05             # kernel bandwidth on text cosine similarity
    empirical_bayes: bool = False       # select (alpha, lam) by type-II maximum likelihood
    eb_alpha_grid: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    eb_lam_grid: Sequence[float] = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
    eb_min_present: int = 3             # minimum number of present classes before EB is trusted
    # Class presence / label shift
    presence: bool = True               # estimate mixing proportions pi (label-shift correction)
    dirichlet_pseudo_count: float = 0.5 # c = a - 1 of the symmetric Dirichlet prior on pi
    # Sample graph (manifold) term, inherited from TransCLIP/StatA
    sample_knn: int = 3                 # 0 disables the sample Laplacian term
    lambda_laplacian: float = 1.0
    # Optimisation
    n_outer: int = 10
    n_inner_z: int = 3
    normalize_prototypes: bool = False  # project mu_k back to the unit sphere after the M-step
    # Online
    forgetting: float = 1.0             # exponential forgetting factor of sufficient statistics (1 = none)
    # Numerics
    linear_solver: str = "auto"         # "exact", "cg" or "auto"
    cg_iters: int = 30
    eps: float = 1e-8

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "GRAFTConfig":
        return cls(**d)
