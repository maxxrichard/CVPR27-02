"""GRAFT solver: batch (transductive) and online variants.

Model
-----
    x_i | y_i = k  ~  N(mu_k, Sigma)            (shared diagonal or isotropic covariance, tempered by kappa)
    mu_k = t_k + d_k                              (text anchor + displacement)
    D = [d_1; ...; d_K] ~ N(0, Q^{-1}),  Q = (alpha I + lam L_T) / (kappa sigma_bar^2)     (GMRF over the class graph)
    y_i ~ Cat(pi),  pi ~ Dir(c + 1)              (class presence / label shift)
    plus the zero-shot KL anchor (y_hat^lambda_y) and the sample-graph Laplacian term inherited from TransCLIP.

The objective J(z, D, Sigma, pi) is minimised by block-coordinate descent; every block has a closed form
except the z-block, which uses the majorise-minimise linearisation of the Laplacian term.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import math
import torch
import torch.nn.functional as F

from .config import GRAFTConfig
from .class_graph import build_class_graph, normalized_laplacian


# ----------------------------------------------------------------------------------------------
# Sample graph (inherited from TransCLIP / StatA): sparse kNN cosine affinities
# ----------------------------------------------------------------------------------------------
def build_sample_affinity(X: torch.Tensor, knn: int) -> Optional[torch.Tensor]:
    if knn <= 0 or X.shape[0] <= 1:
        return None
    N = X.shape[0]
    knn = min(knn, N - 1)
    S = X @ X.T
    S.fill_diagonal_(-float("inf"))
    idx = S.topk(knn, dim=1).indices
    vals = S.gather(1, idx)
    rows = torch.arange(N, device=X.device).unsqueeze(1).expand(-1, knn).reshape(-1)
    W = torch.sparse_coo_tensor(torch.stack([rows, idx.reshape(-1)]), vals.reshape(-1), (N, N)).coalesce()
    return W


# ----------------------------------------------------------------------------------------------
# Tempered Gaussian log-likelihood with shared (diagonal) covariance
# ----------------------------------------------------------------------------------------------
def gaussian_loglik(X: torch.Tensor, Mu: torch.Tensor, var: torch.Tensor, kappa: float, chunk: int = 4096) -> torch.Tensor:
    """ell_ik = -(1/(2 kappa)) [ sum_j (x_ij - mu_kj)^2 / var_j + sum_j log var_j ].  Returns N x K."""
    inv = 1.0 / var                                           # (d,)
    logdet = torch.log(var).sum()
    # ||x - mu||^2_{inv} = x^2.inv - 2 x.(mu inv) + mu^2.inv
    mu_inv = Mu * inv                                        # K x d
    mu_term = (Mu * mu_inv).sum(1)                           # K
    out = torch.empty(X.shape[0], Mu.shape[0], dtype=X.dtype, device=X.device)
    for s in range(0, X.shape[0], chunk):
        xb = X[s:s + chunk]
        x_term = (xb * xb * inv).sum(1, keepdim=True)        # n x 1
        out[s:s + chunk] = x_term - 2.0 * xb @ mu_inv.T + mu_term[None, :]
    return -(out + logdet) / (2.0 * kappa)


# ----------------------------------------------------------------------------------------------
# Displacement-field M-step: solve (N + s_j (alpha I + lam L)) d_{:,j} = N e_{:,j} for every dimension j
# ----------------------------------------------------------------------------------------------
def solve_displacement(E: torch.Tensor, n: torch.Tensor, L: torch.Tensor, alpha: float, lam: float,
                       scale: torch.Tensor, method: str = "auto", cg_iters: int = 30, tol: float = 1e-6) -> torch.Tensor:
    """E: K x d empirical displacements (x_bar_k - t_k, zero where n_k = 0); n: K soft counts;
    L: K x K Laplacian; scale: d-vector s_j = var_j / var_bar (1 for isotropic).
    Returns D = argmin_D sum_k n_k ||d_k - e_k||^2 / s  + alpha ||D||^2 + lam tr(D^T L D) column-wise."""
    K, d = E.shape
    rhs = n[:, None] * E
    if method == "auto":
        method = "exact" if K <= 256 else "cg"
    if method == "exact":
        # batched over dimensions in chunks to bound memory
        D = torch.empty_like(E)
        base = alpha * torch.eye(K, dtype=E.dtype, device=E.device) + lam * L
        chunk = max(1, int(2e8 // (K * K)))
        for s in range(0, d, chunk):
            sc = scale[s:s + chunk]                                  # c
            A = torch.diag(n)[None] + sc[:, None, None] * base[None]  # c x K x K
            D[:, s:s + chunk] = torch.linalg.solve(A, rhs[:, s:s + chunk].T.unsqueeze(-1)).squeeze(-1).T
        return D
    # Conjugate gradients, all columns at once (operators differ per column only by the scalar s_j)
    def matvec(V):
        return n[:, None] * V + scale[None, :] * (alpha * V + lam * (L @ V))
    # Jacobi preconditioner
    diagA = n[:, None] + scale[None, :] * (alpha + lam * torch.diagonal(L)[:, None])
    Minv = 1.0 / diagA.clamp_min(1e-12)
    X = torch.zeros_like(E)
    R = rhs - matvec(X)
    Z = Minv * R
    P = Z.clone()
    rz = (R * Z).sum(0)
    rhs_norm = rhs.norm(dim=0).clamp_min(1e-30)
    for _ in range(cg_iters):
        AP = matvec(P)
        denom = (P * AP).sum(0).clamp_min(1e-30)
        a = rz / denom
        X = X + a[None, :] * P
        R = R - a[None, :] * AP
        if (R.norm(dim=0) / rhs_norm).max() < tol:
            break
        Z = Minv * R
        rz_new = (R * Z).sum(0)
        b = rz_new / rz.clamp_min(1e-30)
        P = Z + b[None, :] * P
        rz = rz_new
    return X


# ----------------------------------------------------------------------------------------------
# Empirical Bayes (type-II ML) selection of the prior strengths (alpha, lam) in pseudo-count units
# ----------------------------------------------------------------------------------------------
def log_evidence(E: torch.Tensor, n: torch.Tensor, L: torch.Tensor, alpha: float, lam: float,
                 present: torch.Tensor, ridge: float = 1e-6) -> float:
    """Isotropic-approximation marginal log-likelihood of the empirical displacements of present classes.

    e_P ~ N(0, Schur(alpha I + lam L)_P^{-1} + diag(1/n_P)) in pseudo-count units (common factor dropped).
    """
    K = L.shape[0]
    Qf = alpha * torch.eye(K, dtype=L.dtype, device=L.device) + lam * L
    P = present
    A = ~present
    Qpp = Qf[P][:, P]
    if A.any():
        Qaa = Qf[A][:, A] + ridge * torch.eye(int(A.sum()), dtype=L.dtype, device=L.device)
        Qpa = Qf[P][:, A]
        S = Qpp - Qpa @ torch.linalg.solve(Qaa, Qpa.T)
    else:
        S = Qpp
    S = S + ridge * torch.eye(S.shape[0], dtype=S.dtype, device=S.device)
    Cov = torch.linalg.inv(S) + torch.diag(1.0 / n[P])
    Lc, info = torch.linalg.cholesky_ex(Cov)
    if info.item() != 0:
        return -float("inf")
    Ep = E[P]
    sol = torch.cholesky_solve(Ep, Lc)
    quad = (Ep * sol).sum()
    logdet = 2.0 * torch.log(torch.diagonal(Lc)).sum()
    d = E.shape[1]
    return float(-0.5 * quad - 0.5 * d * logdet)


def select_prior_strength(E: torch.Tensor, n: torch.Tensor, L: torch.Tensor, kappa: float, var_bar: float,
                          cfg: GRAFTConfig) -> Tuple[float, float, dict]:
    present = n >= 1.0
    if int(present.sum()) < cfg.eb_min_present:
        return cfg.alpha, cfg.lam, {"used_default": True}
    Es = E / math.sqrt(kappa * var_bar)   # whiten to pseudo-count units
    best = (-float("inf"), cfg.alpha, cfg.lam)
    table = {}
    for a in cfg.eb_alpha_grid:
        for l in cfg.eb_lam_grid:
            if a == 0 and l == 0:
                continue
            ev = log_evidence(Es, n, L, float(a), float(l), present)
            table[(float(a), float(l))] = ev
            if ev > best[0]:
                best = (ev, float(a), float(l))
    return best[1], best[2], {"used_default": False, "evidence": table}


# ----------------------------------------------------------------------------------------------
# State and the batch solver
# ----------------------------------------------------------------------------------------------
@dataclass
class GRAFTState:
    T: torch.Tensor            # K x d text anchors (unit norm)
    L: torch.Tensor            # K x K class Laplacian
    D: torch.Tensor            # K x d displacement field
    var: torch.Tensor          # d shared variances
    pi: torch.Tensor           # K mixing proportions
    alpha: float
    lam: float
    # running sufficient statistics (online)
    n: Optional[torch.Tensor] = None       # K soft counts
    s: Optional[torch.Tensor] = None       # K x d  sum_i z_ik x_i
    q: Optional[torch.Tensor] = None       # d      sum_i x_ij^2
    N: float = 0.0

    @property
    def Mu(self) -> torch.Tensor:
        return self.T + self.D


def _prepare(X, T, cfg):
    X = F.normalize(X.float(), dim=-1)
    T = F.normalize(T.float(), dim=-1)
    return X, T


def _init_state(X, T, cfg, W_c=None) -> Tuple[GRAFTState, torch.Tensor]:
    K, d = T.shape
    y_hat = torch.softmax(cfg.logit_scale * X @ T.T, dim=1)
    if W_c is None:
        W_c = build_class_graph(T, cfg.graph_knn, cfg.graph_rho)
    L = normalized_laplacian(W_c)
    var = _variance_from_stats(X, T, y_hat, cfg)
    pi = torch.full((K,), 1.0 / K, dtype=X.dtype, device=X.device)
    st = GRAFTState(T=T, L=L, D=torch.zeros_like(T), var=var, pi=pi, alpha=cfg.alpha, lam=cfg.lam)
    return st, y_hat


def _variance_from_stats(X, Mu, z, cfg):
    N = X.shape[0]
    # var_j = (1/N) [ sum_i x_ij^2 - 2 sum_k mu_kj s_kj + sum_k n_k mu_kj^2 ]
    n = z.sum(0)
    s = z.T @ X
    q = (X * X).sum(0)
    var = (q - 2.0 * (Mu * s).sum(0) + (n[:, None] * Mu * Mu).sum(0)) / max(N, 1)
    var = var.clamp_min(cfg.eps)
    if cfg.cov == "iso":
        var = torch.full_like(var, var.mean())
    return var


def _z_step(X, y_hat, st: GRAFTState, cfg: GRAFTConfig, W_s, z, log_pi):
    ll = gaussian_loglik(X, st.Mu, st.var, cfg.kappa)
    base = ll + cfg.lambda_y * torch.log(y_hat.clamp_min(1e-30)) + log_pi[None, :]
    for _ in range(cfg.n_inner_z):
        logits = base
        if W_s is not None and cfg.lambda_laplacian > 0:
            coupling = torch.sparse.mm(W_s, z) + torch.sparse.mm(W_s.t(), z)
            logits = logits + cfg.lambda_laplacian / (2.0 * cfg.sample_knn) * coupling
        z = torch.softmax(logits, dim=1)
    return z, ll


def _m_step_from_stats(st: GRAFTState, cfg: GRAFTConfig, n, s, q, N, select_eb: bool):
    K, d = st.T.shape
    # presence / label shift
    if cfg.presence:
        c = cfg.dirichlet_pseudo_count
        st.pi = (n + c) / (N + K * c)
    # empirical displacements of present classes
    nz = n > cfg.eps
    xbar = torch.where(nz[:, None], s / n.clamp_min(cfg.eps)[:, None], st.T)
    E = torch.where(nz[:, None], xbar - st.T, torch.zeros_like(st.T))
    var_bar = float(st.var.mean())
    if select_eb:
        st.alpha, st.lam, _ = select_prior_strength(E, n, st.L, cfg.kappa, var_bar, cfg)
    scale = st.var / var_bar if cfg.cov == "diag" else torch.ones_like(st.var)
    st.D = solve_displacement(E, n, st.L, st.alpha, st.lam, scale, cfg.linear_solver, cfg.cg_iters)
    if cfg.normalize_prototypes:
        st.D = F.normalize(st.T + st.D, dim=-1) - st.T
    # shared variance with the new prototypes
    Mu = st.Mu
    var = (q - 2.0 * (Mu * s).sum(0) + (n[:, None] * Mu * Mu).sum(0)) / max(N, 1)
    st.var = var.clamp_min(cfg.eps)
    if cfg.cov == "iso":
        st.var = torch.full_like(st.var, st.var.mean())
    return st


def objective(X, y_hat, z, st: GRAFTState, cfg: GRAFTConfig, W_s) -> float:
    """Penalised negative log-likelihood J (up to additive constants), used for monitoring and tests.

    J = sum_ik z_ik [log z_ik - log pi_k - lambda_y log yhat_ik - ell_ik]
        - lambda_L/(2 knn) sum_ij W_ij z_i^T z_j
        + (1/(2 kappa var_bar)) tr(D^T (alpha I + lam L) D)  - c sum_k log pi_k
    """
    ll = gaussian_loglik(X, st.Mu, st.var, cfg.kappa)
    log_pi = torch.log(st.pi.clamp_min(1e-30))
    J = (z * (torch.log(z.clamp_min(1e-30)) - log_pi[None, :] - cfg.lambda_y * torch.log(y_hat.clamp_min(1e-30)) - ll)).sum()
    if W_s is not None and cfg.lambda_laplacian > 0:
        J = J - cfg.lambda_laplacian / (2.0 * cfg.sample_knn) * (z * torch.sparse.mm(W_s, z)).sum()
    var_bar = st.var.mean()
    Qd = st.alpha * st.D + st.lam * (st.L @ st.D)
    J = J + 0.5 * (st.D * Qd).sum() / (cfg.kappa * var_bar)
    if cfg.presence:
        J = J - cfg.dirichlet_pseudo_count * log_pi.sum()
    return float(J)


def graft_batch(X: torch.Tensor, T: torch.Tensor, cfg: Optional[GRAFTConfig] = None, W_c: Optional[torch.Tensor] = None,
                return_state: bool = False, track_objective: bool = False):
    """Transductive GRAFT on one unlabeled batch.

    X: N x d image features, T: K x d text prototypes (any scale; normalised inside).
    Returns (y_hat, z) [, state] [, objective trace]: zero-shot posterior and adapted posterior (N x K).
    """
    cfg = cfg or GRAFTConfig()
    X, T = _prepare(X, T, cfg)
    st, y_hat = _init_state(X, T, cfg, W_c)
    W_s = build_sample_affinity(X, cfg.sample_knn)
    z = y_hat.clone()
    q = (X * X).sum(0)
    N = X.shape[0]
    trace = []
    for it in range(cfg.n_outer + 1):
        log_pi = torch.log(st.pi.clamp_min(1e-30))
        z, _ = _z_step(X, y_hat, st, cfg, W_s, z, log_pi)
        if track_objective:
            trace.append(objective(X, y_hat, z, st, cfg, W_s))
        if it == cfg.n_outer:
            break
        n = z.sum(0)
        s = z.T @ X
        st = _m_step_from_stats(st, cfg, n, s, q, float(N), select_eb=cfg.empirical_bayes)
        if track_objective:
            trace.append(objective(X, y_hat, z, st, cfg, W_s))
    out = [y_hat, z]
    if return_state:
        out.append(st)
    if track_objective:
        out.append(trace)
    return tuple(out)


# ----------------------------------------------------------------------------------------------
# Online variant: running sufficient statistics with optional forgetting
# ----------------------------------------------------------------------------------------------
class GRAFTOnline:
    """Stream of batches.  Each batch is first assigned with the current parameters (optionally refined
    transductively within the batch), then the sufficient statistics are updated and the closed-form
    M-step is re-run.  `predict_before_update=True` gives strictly inductive predictions."""

    def __init__(self, T: torch.Tensor, cfg: Optional[GRAFTConfig] = None, W_c: Optional[torch.Tensor] = None,
                 predict_before_update: bool = False, n_outer_online: int = 2):
        self.cfg = cfg or GRAFTConfig()
        T = F.normalize(T.float(), dim=-1)
        K, d = T.shape
        if W_c is None:
            W_c = build_class_graph(T, self.cfg.graph_knn, self.cfg.graph_rho)
        L = normalized_laplacian(W_c)
        self.st = GRAFTState(T=T, L=L, D=torch.zeros_like(T), var=None, pi=torch.full((K,), 1.0 / K, dtype=T.dtype, device=T.device),
                             alpha=self.cfg.alpha, lam=self.cfg.lam,
                             n=torch.zeros(K, dtype=T.dtype, device=T.device), s=torch.zeros_like(T),
                             q=torch.zeros(d, dtype=T.dtype, device=T.device), N=0.0)
        self.predict_before_update = predict_before_update
        self.n_outer_online = n_outer_online

    @torch.no_grad()
    def step(self, X: torch.Tensor):
        cfg, st = self.cfg, self.st
        X = F.normalize(X.float(), dim=-1)
        y_hat = torch.softmax(cfg.logit_scale * X @ st.T.T, dim=1)
        if st.var is None:                       # first batch: initialise variance from the zero-shot assignment
            st.var = _variance_from_stats(X, st.T, y_hat, cfg)
        W_s = build_sample_affinity(X, cfg.sample_knn)
        z = y_hat.clone()
        log_pi = torch.log(st.pi.clamp_min(1e-30))
        z, _ = _z_step(X, y_hat, st, cfg, W_s, z, log_pi)
        pred = z.clone() if self.predict_before_update else None
        rho = cfg.forgetting
        base = (st.n.clone(), st.s.clone(), st.q.clone(), st.N)
        for _ in range(self.n_outer_online):
            n = rho * base[0] + z.sum(0)
            s = rho * base[1] + z.T @ X
            q = rho * base[2] + (X * X).sum(0)
            N = rho * base[3] + X.shape[0]
            st = _m_step_from_stats(st, cfg, n, s, q, N, select_eb=cfg.empirical_bayes)
            log_pi = torch.log(st.pi.clamp_min(1e-30))
            z, _ = _z_step(X, y_hat, st, cfg, W_s, z, log_pi)
        st.n, st.s, st.q, st.N = n, s, q, N
        self.st = st
        return y_hat, (pred if pred is not None else z)
