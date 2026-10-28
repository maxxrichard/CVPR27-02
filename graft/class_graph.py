"""Semantic class graph built from the text prototypes and Laplacian utilities."""
import torch


def build_class_graph(T: torch.Tensor, knn: int = 5, rho: float = 0.05, mutual: bool = False) -> torch.Tensor:
    """Dense symmetric affinity W (K x K) over classes from unit-norm text prototypes T (K x D).

    w_kl = exp((cos(t_k, t_l) - 1) / rho) on the union (or mutual) kNN graph, zero diagonal.
    """
    T = torch.nn.functional.normalize(T, dim=-1)
    K = T.shape[0]
    S = T @ T.T
    S.fill_diagonal_(-float("inf"))
    knn = max(0, min(knn, K - 1))
    W = torch.zeros(K, K, dtype=T.dtype, device=T.device)
    if knn == 0:
        return W
    idx = S.topk(knn, dim=1).indices                       # K x knn
    vals = torch.exp((S.gather(1, idx) - 1.0) / rho)
    W.scatter_(1, idx, vals)
    if mutual:
        W = torch.minimum(W, W.T)
    else:
        W = torch.maximum(W, W.T)
    W.fill_diagonal_(0.0)
    return W


def complete_graph(K: int, dtype=torch.float32, device=None) -> torch.Tensor:
    W = torch.ones(K, K, dtype=dtype, device=device)
    W.fill_diagonal_(0.0)
    return W


def normalized_laplacian(W: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Symmetric normalised Laplacian L = I - D^{-1/2} W D^{-1/2} (isolated nodes get L_kk = 0)."""
    d = W.sum(1)
    dinv = torch.where(d > eps, d.clamp_min(eps).rsqrt(), torch.zeros_like(d))
    L = -(dinv[:, None] * W * dinv[None, :])
    L.diagonal().add_(torch.where(d > eps, torch.ones_like(d), torch.zeros_like(d)))
    return L


def dirichlet_energy(D: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """tr(D^T L D) for a displacement field D (K x d)."""
    return torch.einsum("kd,kl,ld->", D, L, D)


def smoothness_ratio(D: torch.Tensor, L: torch.Tensor, n_random: int = 20, seed: int = 0) -> dict:
    """Ratio of the Dirichlet energy of D on L to its energy on degree-preserving random relabellings.

    A ratio well below 1 supports the class-level smoothness hypothesis (H1 in the paper).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    e = dirichlet_energy(D, L).item()
    rand = []
    K = D.shape[0]
    for _ in range(n_random):
        perm = torch.randperm(K, generator=g).to(D.device)
        rand.append(dirichlet_energy(D[perm], L).item())
    rand_t = torch.tensor(rand)
    return {"energy": e, "random_mean": rand_t.mean().item(), "random_std": rand_t.std().item(),
            "ratio": e / max(rand_t.mean().item(), 1e-12)}
