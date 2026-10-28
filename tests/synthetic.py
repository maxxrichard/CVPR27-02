"""Synthetic generator used by the unit tests: a hyperspherical mixture whose per-class displacement field is
smooth over a text-derived class graph.  No real data, no pretrained model."""
import torch
import torch.nn.functional as F


def make_problem(K=30, d=64, n_per_class=20, present=None, noise=0.15, smooth=0.6, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    # text anchors: K points organised in clusters so that the kNN text graph is meaningful
    centers = F.normalize(torch.randn(6, d, generator=g), dim=-1)
    T = F.normalize(centers[torch.arange(K) % 6] + 0.35 * torch.randn(K, d, generator=g), dim=-1)
    # smooth displacement field: a global shift plus a cluster-wise component plus small class noise
    shift = 0.25 * F.normalize(torch.randn(1, d, generator=g), dim=-1)
    cluster_shift = 0.2 * F.normalize(torch.randn(6, d, generator=g), dim=-1)[torch.arange(K) % 6]
    Dtrue = shift + smooth * cluster_shift + 0.05 * torch.randn(K, d, generator=g)
    Mu = F.normalize(T + Dtrue, dim=-1)
    present = list(range(K)) if present is None else present
    X, Y = [], []
    for k in present:
        X.append(F.normalize(Mu[k][None] + noise * torch.randn(n_per_class, d, generator=g), dim=-1))
        Y.append(torch.full((n_per_class,), k, dtype=torch.long))
    X, Y = torch.cat(X), torch.cat(Y)
    perm = torch.randperm(len(Y), generator=g)
    return X[perm].to(device), Y[perm].to(device), T.to(device), Dtrue.to(device)
