import dataclasses, math
import torch
import pytest

from graft import GRAFTConfig, graft_batch, GRAFTOnline, zero_shot, count_shrinkage, global_shift
from graft.class_graph import build_class_graph, normalized_laplacian, complete_graph, dirichlet_energy
from graft.solver import solve_displacement, objective, build_sample_affinity, log_evidence, select_prior_strength
from graft.metrics import accuracy, expected_calibration_error
from graft.samplers import BatchSampler, OnlineSampler
from tests.synthetic import make_problem

torch.manual_seed(0)


def test_shapes_and_probabilities():
    X, Y, T, _ = make_problem(K=20, d=32, n_per_class=8)
    y_hat, z, st = graft_batch(X, T, GRAFTConfig(), return_state=True)
    assert y_hat.shape == z.shape == (X.shape[0], T.shape[0])
    assert torch.allclose(z.sum(1), torch.ones(X.shape[0]), atol=1e-5)
    assert torch.isfinite(z).all() and (z >= 0).all()
    assert st.D.shape == T.shape and st.var.shape == (T.shape[1],) and st.pi.shape == (T.shape[0],)
    assert abs(float(st.pi.sum()) - 1.0) < 1e-5


def test_laplacian_properties():
    T = torch.nn.functional.normalize(torch.randn(25, 16), dim=-1)
    W = build_class_graph(T, knn=4, rho=0.05)
    assert torch.allclose(W, W.T) and float(W.diagonal().abs().sum()) == 0.0
    L = normalized_laplacian(W)
    ev = torch.linalg.eigvalsh(L)
    assert ev.min() > -1e-5 and ev.max() < 2 + 1e-5          # PSD, spectrum in [0, 2]
    const = torch.ones(25, 3) * W.sum(1).sqrt()[:, None]       # D^{1/2} 1 is in the null space
    assert dirichlet_energy(const, L).abs() < 1e-4


def test_closed_form_m_step_matches_gradient_descent():
    K, d = 12, 7
    T = torch.nn.functional.normalize(torch.randn(K, d), dim=-1)
    L = normalized_laplacian(build_class_graph(T, 3, 0.05))
    E = torch.randn(K, d); n = torch.rand(K) * 10; n[:3] = 0.0; E[:3] = 0.0
    scale = torch.rand(d) + 0.5
    alpha, lam = 0.7, 2.5
    D_exact = solve_displacement(E, n, L, alpha, lam, scale, method="exact")
    D_cg = solve_displacement(E, n, L, alpha, lam, scale, method="cg", cg_iters=200, tol=1e-10)
    assert torch.allclose(D_exact, D_cg, atol=1e-4)
    # objective: sum_k n_k ||d_k - e_k||^2 / s_j + s_j^{-1}... -> we solve columnwise (N + s_j(alpha I + lam L)) d = N e
    Dv = torch.zeros(K, d, requires_grad=True)
    opt = torch.optim.LBFGS([Dv], lr=1, max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-14)
    def closure():
        opt.zero_grad()
        f = 0.5 * (n[:, None] * (Dv - E) ** 2).sum() + 0.5 * (scale[None, :] * (alpha * Dv * Dv + lam * Dv * (L @ Dv))).sum()
        f.backward(); return f
    opt.step(closure)
    assert torch.allclose(Dv.detach(), D_exact, atol=2e-3)  # L-BFGS precision


def test_special_case_count_shrinkage():
    """lam = 0: d_k = n_k / (n_k + alpha s_j) e_k  (StatA/MOON-style independent shrinkage)."""
    K, d = 9, 5
    L = torch.eye(K); E = torch.randn(K, d); n = torch.rand(K) * 5; scale = torch.ones(d)
    D = solve_displacement(E, n, L, alpha=1.0, lam=0.0, scale=scale, method="exact")
    beta = n / (n + 1.0)
    assert torch.allclose(D, beta[:, None] * E, atol=1e-6)


def test_special_case_complete_pooling_is_global_shift():
    """alpha = 0, lam -> inf on the complete graph: every class receives the count-weighted mean displacement."""
    K, d = 10, 6
    L = normalized_laplacian(complete_graph(K, dtype=torch.float64)); E = torch.randn(K, d, dtype=torch.float64)
    n = torch.rand(K, dtype=torch.float64) * 5 + 0.1
    D = solve_displacement(E, n, L, alpha=0.0, lam=1e5, scale=torch.ones(d, dtype=torch.float64), method="exact")
    g = (n[:, None] * E).sum(0) / n.sum()
    assert torch.allclose(D, g[None, :].expand(K, d), atol=1e-3)


def test_absent_classes_are_harmonic_extensions():
    """Rows with n_k = 0 satisfy (alpha I + lam L)_{A,:} D = 0, i.e. the Kron-reduction property."""
    K, d = 14, 4
    T = torch.nn.functional.normalize(torch.randn(K, d), dim=-1)
    L = normalized_laplacian(build_class_graph(T, 4, 0.05))
    n = torch.rand(K) * 5; absent = torch.tensor([1, 5, 9]); n[absent] = 0.0
    E = torch.randn(K, d); E[absent] = 0.0
    alpha, lam = 0.5, 3.0
    D = solve_displacement(E, n, L, alpha, lam, torch.ones(d), method="exact")
    Q = alpha * torch.eye(K) + lam * L
    assert (Q[absent] @ D).abs().max() < 1e-4


def test_objective_decreases_monotonically():
    X, Y, T, _ = make_problem(K=15, d=24, n_per_class=10, seed=1)
    cfg = GRAFTConfig(n_outer=8, empirical_bayes=False)
    _, _, trace = graft_batch(X, T, cfg, track_objective=True)
    diffs = [b - a for a, b in zip(trace[:-1], trace[1:])]
    assert max(diffs) < 1e-3 * abs(trace[0]), f"objective increased: {diffs}"


def test_improves_over_zero_shot_on_synthetic_shift():
    X, Y, T, _ = make_problem(K=30, d=64, n_per_class=25, seed=2)
    y_hat, _ = zero_shot(X, T)
    _, z = graft_batch(X, T, GRAFTConfig())
    assert accuracy(z, Y) >= accuracy(y_hat, Y)


def test_absent_classes_not_hallucinated():
    """Only 4 of 40 classes are present: the presence model must concentrate the posterior mass on them."""
    present = [3, 11, 20, 35]
    X, Y, T, _ = make_problem(K=40, d=48, n_per_class=16, present=present, seed=3)
    _, z, st = graft_batch(X, T, GRAFTConfig(), return_state=True)
    absent = [k for k in range(40) if k not in present]
    # every present class must receive far more prior mass than any absent class (label-shift inference)
    assert float(st.pi[present].min()) > 10.0 * float(st.pi[absent].max())
    assert accuracy(z, Y) >= 90.0
    assert set(z.argmax(1).tolist()) <= set(present)


def test_graph_pooling_helps_rare_classes():
    """Classes with a single sample: the graph-pooled displacement must be closer to the true one than
    independent shrinkage (synthetic field is smooth by construction)."""
    K, d = 36, 48
    present = list(range(K))
    X, Y, T, Dtrue = make_problem(K=K, d=d, n_per_class=12, present=present, seed=4)
    keep = torch.ones(len(Y), dtype=torch.bool)
    rare = list(range(0, K, 3))
    for k in rare:                       # keep one sample for every third class
        idx = (Y == k).nonzero().squeeze(1)
        keep[idx[1:]] = False
    X, Y = X[keep], Y[keep]
    cfg = GRAFTConfig(sample_knn=0)
    _, _, st_pool = graft_batch(X, T, cfg, return_state=True)
    _, _, st_ind = graft_batch(X, T, dataclasses.replace(cfg, lam=0.0), return_state=True)
    Mu_true = torch.nn.functional.normalize(T + Dtrue, dim=-1)
    err_pool = (torch.nn.functional.normalize(st_pool.Mu[rare], dim=-1) - Mu_true[rare]).norm(dim=1).mean()
    err_ind = (torch.nn.functional.normalize(st_ind.Mu[rare], dim=-1) - Mu_true[rare]).norm(dim=1).mean()
    assert err_pool < err_ind, (err_pool, err_ind)


def test_online_equals_batch_on_single_batch_without_forgetting():
    X, Y, T, _ = make_problem(K=12, d=20, n_per_class=10, seed=5)
    cfg = GRAFTConfig(n_outer=2, sample_knn=0)
    _, z_b = graft_batch(X, T, cfg)
    on = GRAFTOnline(T, cfg, n_outer_online=2)
    _, z_o = on.step(X)
    assert torch.allclose(z_b, z_o, atol=1e-5)


def test_online_stream_runs_and_tracks_statistics():
    X, Y, T, _ = make_problem(K=12, d=20, n_per_class=30, seed=6)
    on = GRAFTOnline(T, GRAFTConfig(forgetting=0.9))
    s = OnlineSampler(Y, batch_size=32, gamma=0.01, seed=0)
    total = 0
    for idx in s:
        _, z = on.step(X[idx]); total += len(idx)
        assert torch.isfinite(z).all()
    assert total == len(Y) and on.st.N > 0


def test_empirical_bayes_prefers_pooling_on_smooth_field():
    X, Y, T, _ = make_problem(K=30, d=48, n_per_class=12, seed=7)
    _, _, st = graft_batch(X, T, GRAFTConfig(empirical_bayes=True, sample_knn=0), return_state=True)
    assert st.lam > 0.0


def test_determinism():
    X, Y, T, _ = make_problem(K=10, d=16, n_per_class=8, seed=8)
    _, z1 = graft_batch(X, T, GRAFTConfig()); _, z2 = graft_batch(X, T, GRAFTConfig())
    assert torch.equal(z1, z2)


def test_samplers_protocol():
    Y = torch.arange(50).repeat_interleave(7)
    bs = BatchSampler(Y, batch_size=64, kmin=2, kmax=10, seed=0)
    for _ in range(20):
        idx = bs.sample(); k = len(set(Y[idx].tolist()))
        assert 2 <= k <= 10 and len(idx) <= 64
    bs_all = BatchSampler(Y, batch_size=-1, seed=0)
    assert len(bs_all.sample()) == len(Y)
    os_ = OnlineSampler(Y, batch_size=64, gamma=-1, seed=0)
    order = [i for b in os_ for i in b]
    assert sorted(order) == list(range(len(Y)))


def test_metrics():
    p = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]); y = torch.tensor([0, 1, 1])
    assert abs(accuracy(p, y) - 66.666) < 0.01
    assert 0 <= expected_calibration_error(p, y) <= 100


def test_official_adapter_shapes():
    from graft.official_adapter import GRAFT_solver
    X, Y, T, _ = make_problem(K=8, d=12, n_per_class=5, seed=9)
    y_hat, z = GRAFT_solver(X, Y, T.T.unsqueeze(0))     # (1, d, K) layout of the official harness
    assert z.shape == (X.shape[0], 8)
