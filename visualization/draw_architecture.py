"""Publication figures drawn from the method specification (vector PDF).  No experimental data is used.
  Fig. 1 overall framework      -> paper/figures/fig1_framework.pdf      (double column)
  Fig. 2 pooling module detail  -> paper/figures/fig2_module.pdf         (single column)
  Fig. 3 conceptual comparison  -> paper/figures/fig3_concept.pdf        (single column, conceptual)
Run: python visualization/draw_architecture.py
"""
import os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon
import matplotlib.patheffects as pe

OUT = [os.path.join(os.path.dirname(__file__), "..", "paper", "figures"),
       os.path.join(os.path.dirname(__file__), "..", "figures", "architecture")]
for o in OUT:
    os.makedirs(o, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42})
C_FROZEN = "#d9dde3"     # frozen / inherited components
C_FROZEN_E = "#5f6b7a"
C_NEW = "#ffd59e"        # novel components
C_NEW_E = "#c2610a"
C_DATA = "#dbeafe"
C_DATA_E = "#1d4ed8"
C_PRIOR = "#e9d5ff"
C_PRIOR_E = "#6d28d9"
C_TRAIN = "#9ca3af"


def box(ax, x, y, w, h, text, fc, ec, lw=1.0, fs=7.2, bold=False, rounding=0.04, style="round"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"{style},pad=0.004,rounding_size={rounding}", fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal",
            linespacing=1.15)
    return p


def arrow(ax, p0, p1, color="#374151", lw=1.0, ls="-", shrink=1.5, rad=0.0, text=None, tpos=0.5, toff=(0, 0.02), fs=6.5):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, color=color, lw=lw, linestyle=ls,
                        shrinkA=shrink, shrinkB=shrink, connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    if text:
        x = p0[0] + tpos * (p1[0] - p0[0]) + toff[0]; y = p0[1] + tpos * (p1[1] - p0[1]) + toff[1]
        ax.text(x, y, text, ha="center", va="bottom", fontsize=fs, color=color)


def save(fig, name):
    for o in OUT:
        fig.savefig(os.path.join(o, name), bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------
# Figure 1: overall framework
# ------------------------------------------------------------------------------------------------
def fig1():
    fig, ax = plt.subplots(figsize=(7.1, 3.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    FS = 6.3
    # column A: inputs
    box(ax, 0.005, 0.60, 0.11, 0.27, "unlabeled test\nbatch or stream\n$\\{x_i\\}_{i=1}^{N}$", C_DATA, C_DATA_E, fs=5.9)
    box(ax, 0.005, 0.17, 0.11, 0.27, "class names\n$\\{c_k\\}_{k=1}^{K}$\n(full vocabulary)", C_DATA, C_DATA_E, fs=5.9)
    # column B: frozen encoders
    box(ax, 0.13, 0.63, 0.105, 0.21, "frozen image\nencoder $f_v$", C_FROZEN, C_FROZEN_E, fs=FS)
    box(ax, 0.13, 0.20, 0.105, 0.21, "frozen text\nencoder $f_t$\n(prompt ensemble)", C_FROZEN, C_FROZEN_E, fs=5.9)
    arrow(ax, (0.11, 0.735), (0.13, 0.735)); arrow(ax, (0.11, 0.305), (0.13, 0.305))
    # column C: features
    box(ax, 0.255, 0.65, 0.085, 0.17, "$X\\in\\mathbb{R}^{N\\times d}$\nunit norm", "white", C_DATA_E, fs=FS)
    box(ax, 0.255, 0.22, 0.085, 0.17, "anchors\n$T\\in\\mathbb{R}^{K\\times d}$", "white", C_DATA_E, fs=FS)
    arrow(ax, (0.235, 0.735), (0.255, 0.735)); arrow(ax, (0.235, 0.305), (0.255, 0.305))
    # column D: derived quantities
    box(ax, 0.365, 0.74, 0.155, 0.16, "zero-shot prior\n$\\hat{y}=\\mathrm{softmax}(\\tau XT^{\\!\\top})$", C_FROZEN, C_FROZEN_E, fs=5.8)
    box(ax, 0.365, 0.49, 0.155, 0.15, "sample kNN graph $W_s$\n(inherited)", C_FROZEN, C_FROZEN_E, fs=5.8)
    box(ax, 0.365, 0.17, 0.155, 0.19, "semantic class graph\n$L_T$ from $TT^{\\!\\top}$\n(normalised Laplacian)", C_NEW, C_NEW_E, fs=5.8)
    arrow(ax, (0.34, 0.77), (0.365, 0.82)); arrow(ax, (0.34, 0.70), (0.365, 0.565))
    arrow(ax, (0.34, 0.305), (0.365, 0.265)); arrow(ax, (0.34, 0.36), (0.365, 0.78), color="#9ca3af", lw=0.7)
    # GRAFT block
    gx, gy, gw, gh = 0.555, 0.11, 0.32, 0.86
    ax.add_patch(FancyBboxPatch((gx, gy), gw, gh, boxstyle="round,pad=0.004,rounding_size=0.02", fc="#fff8ee", ec=C_NEW_E, lw=1.2, ls="--"))
    ax.text(gx + 0.01, gy + gh - 0.012, "GRAFT inference (closed form, no gradients, $\\times 10$ iterations)", ha="left", va="top", fontsize=5.7, weight="bold", color=C_NEW_E)
    ex, ew = gx + 0.012, 0.135
    mx, mw = gx + 0.172, 0.135
    box(ax, ex, 0.52, ew, 0.26, "E-step\n$z_{ik}\\propto\\pi_k\\hat{y}_{ik}^{\\lambda_y}e^{\\ell_{ik}(\\mu,\\sigma)}$\n+ sample-graph term", C_FROZEN, C_FROZEN_E, fs=5.7)
    box(ax, ex, 0.15, ew, 0.24, "sufficient statistics\n$n_k=\\sum_i z_{ik}$\n$E=\\bar{X}_z-T$\n$q=\\sum_i x_i^{2}$", "white", C_NEW_E, fs=5.8)
    box(ax, mx, 0.67, mw, 0.13, "presence / label shift\n$\\pi_k\\propto n_k+c$", C_NEW, C_NEW_E, fs=5.8)
    box(ax, mx, 0.41, mw, 0.21, "graph-pooled\ndisplacement field\n$D=(N_z+sQ)^{-1}N_zE$", C_NEW, C_NEW_E, fs=5.9, bold=True)
    box(ax, mx, 0.17, mw, 0.13, "shared covariance\n$\\sigma^2_j$ (inherited)", C_FROZEN, C_FROZEN_E, fs=5.8)
    ax.text(mx + mw / 2, 0.87, "prior $Q=\\alpha I+\\lambda L_T$", ha="center", va="center", fontsize=5.6, color=C_PRIOR_E,
            bbox=dict(boxstyle="round,pad=0.2", fc=C_PRIOR, ec=C_PRIOR_E, lw=0.8))
    arrow(ax, (mx + mw / 2, 0.84), (mx + mw / 2, 0.80), color=C_PRIOR_E)
    arrow(ax, (ex + 0.03, 0.52), (ex + 0.03, 0.39), text="$z$", toff=(0.014, -0.03))
    arrow(ax, (ex + ew, 0.36), (mx, 0.72), rad=-0.3)
    arrow(ax, (ex + ew, 0.30), (mx, 0.48))
    arrow(ax, (ex + ew, 0.22), (mx, 0.235))
    arrow(ax, (mx, 0.60), (ex + ew, 0.60), color=C_NEW_E)
    ax.text(ex + ew + 0.002, 0.47, "feedback $\\mu,\\pi,\\sigma^2$", ha="right", va="center", fontsize=4.8, color=C_NEW_E)
    # inputs to block
    arrow(ax, (0.52, 0.82), (ex, 0.74), text="$\\hat{y}$", tpos=0.5, toff=(0, 0.012), fs=5.8)
    arrow(ax, (0.52, 0.565), (ex, 0.60), text="$W_s$", tpos=0.5, toff=(0, 0.012), fs=5.8)
    arrow(ax, (0.52, 0.34), (mx, 0.47), color=C_NEW_E, rad=0.18, text="$L_T$", tpos=0.12, toff=(0, 0.02), fs=5.8)
    arrow(ax, (0.30, 0.65), (ex, 0.27), color="#9ca3af", lw=0.7, rad=0.3, text="$X$", tpos=0.85, toff=(-0.012, 0.012), fs=5.8)
    # outputs
    box(ax, 0.9, 0.58, 0.095, 0.26, "adapted\nposterior $z$\n(prediction,\ncalibration)", C_DATA, C_DATA_E, fs=FS)
    box(ax, 0.9, 0.18, 0.095, 0.26, "adapted state\n$\\mu=T+D$, $\\pi$\n(reused\nonline)", C_DATA, C_DATA_E, fs=FS)
    arrow(ax, (gx + gw, 0.71), (0.9, 0.71)); arrow(ax, (gx + gw, 0.31), (0.9, 0.31))
    # legend
    for i, (c, e, t) in enumerate([(C_DATA, C_DATA_E, "data / outputs"), (C_FROZEN, C_FROZEN_E, "frozen or inherited"), (C_NEW, C_NEW_E, "novel (this work)"), (C_PRIOR, C_PRIOR_E, "prior")]):
        ax.add_patch(FancyBboxPatch((0.01 + i * 0.165, 0.012), 0.02, 0.045, boxstyle="round,pad=0.002", fc=c, ec=e, lw=0.8))
        ax.text(0.036 + i * 0.165, 0.035, t, va="center", fontsize=5.9)
    save(fig, "fig1_framework.pdf")


# ------------------------------------------------------------------------------------------------
# Figure 2: the graph-pooled displacement field (module detail)
# ------------------------------------------------------------------------------------------------
def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(3.4, 1.6))
    rng = np.random.default_rng(3)
    # a small class graph: 9 classes in 3 semantic clusters
    centers = np.array([[0.25, 0.7], [0.72, 0.72], [0.5, 0.22]])
    pts = np.concatenate([c + 0.09 * rng.standard_normal((3, 2)) for c in centers])
    clusters = np.repeat(np.arange(3), 3)
    edges = [(i, j) for i in range(9) for j in range(i + 1, 9) if clusters[i] == clusters[j]] + [(0, 3), (4, 6), (2, 7)]
    true_field = np.array([[0.12, 0.05], [0.12, 0.05], [0.12, 0.05], [-0.02, 0.12], [-0.02, 0.12], [-0.02, 0.12], [0.08, -0.1], [0.08, -0.1], [0.08, -0.1]])
    counts = np.array([30, 1, 0, 25, 0, 2, 40, 0, 1])
    noise = 0.08 * rng.standard_normal((9, 2)) / np.sqrt(np.maximum(counts, 1))[:, None]
    emp = true_field + noise
    titles = ["(a) empirical $e_k=\\bar{x}_k-t_k$", "(b) $\\lambda=0$: independent\nshrinkage to anchors", "(c) GRAFT: graph-pooled\nfield $D$"]
    # closed form on the toy graph
    W = np.zeros((9, 9))
    for i, j in edges:
        W[i, j] = W[j, i] = 1.0
    dinv = 1 / np.sqrt(W.sum(1)); L = np.eye(9) - dinv[:, None] * W * dinv[None, :]
    def solve(alpha, lam):
        N = np.diag(counts.astype(float)); Q = alpha * np.eye(9) + lam * L
        return np.linalg.solve(N + Q, N @ np.where(counts[:, None] > 0, emp, 0))
    fields = [np.where(counts[:, None] > 0, emp, np.nan), solve(1.0, 0.0), solve(1.0, 6.0)]
    for ax, title, F in zip(axes, titles, fields):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=6.0, pad=2)
        for i, j in edges:
            ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], color="#cbd5e1", lw=0.8, zorder=1)
        for k in range(9):
            s = 10 + 1.2 * counts[k]
            ax.scatter(pts[k, 0], pts[k, 1], s=s, color="white", edgecolor=C_NEW_E if counts[k] == 0 else C_FROZEN_E, lw=0.9, zorder=3)
            if counts[k] == 0:
                ax.text(pts[k, 0] + (0.07 if pts[k, 1] < 0.75 else -0.02), pts[k, 1] + (0.05 if pts[k, 1] < 0.75 else -0.11), "absent", fontsize=4.3, ha="left" if pts[k, 1] < 0.75 else "center", color=C_NEW_E)
            v = F[k]
            if np.all(np.isfinite(v)):
                ax.annotate("", xy=(pts[k, 0] + 1.6 * v[0], pts[k, 1] + 1.6 * v[1]), xytext=(pts[k, 0], pts[k, 1]),
                            arrowprops=dict(arrowstyle="-|>", color=C_NEW_E if ax is axes[2] else "#374151", lw=0.9, mutation_scale=6), zorder=4)
    axes[0].text(0.02, 0.02, "node size $\\propto n_k$", fontsize=4.8, transform=axes[0].transAxes)
    axes[2].text(1.0, -0.16, "$D=(N_z+\\alpha I+\\lambda L_T)^{-1}N_z E$ ; absent/rare classes borrow their neighbours' displacement", fontsize=4.9,
                 transform=axes[2].transAxes, ha="right")
    plt.subplots_adjust(wspace=0.05, left=0.01, right=0.99, top=0.82, bottom=0.08)
    save(fig, "fig2_module.pdf")


# ------------------------------------------------------------------------------------------------
# Figure 3: conceptual comparison (labelled as conceptual)
# ------------------------------------------------------------------------------------------------
def fig3():
    fig, axes = plt.subplots(1, 3, figsize=(3.4, 1.5))
    rng = np.random.default_rng(1)
    # two present classes (A, B) and one absent class (C) whose anchor is near B's images
    tA, tB, tC = np.array([0.22, 0.75]), np.array([0.62, 0.72]), np.array([0.82, 0.62])
    shift = np.array([0.08, -0.32])
    xA = tA + shift + 0.05 * rng.standard_normal((25, 2))
    xB = tB + shift + 0.05 * rng.standard_normal((25, 2))
    titles = ["(a) per-class fitting\n(no anchor)", "(b) anchor-only shrinkage", "(c) GRAFT"]
    for ax, title in zip(axes, titles):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=6.0, pad=2)
        ax.scatter(xA[:, 0], xA[:, 1], s=4, color="#60a5fa", lw=0); ax.scatter(xB[:, 0], xB[:, 1], s=4, color="#34d399", lw=0)
        for t, c, lab in [(tA, "#1d4ed8", "A"), (tB, "#047857", "B"), (tC, C_NEW_E, "C (absent)")]:
            ax.scatter(*t, marker="*", s=28, color=c, zorder=5); ax.text(t[0] + 0.03, t[1] + 0.03, lab, fontsize=5, color=c)
    # (a): C's prototype drifts to B's data and steals it
    a = axes[0]
    muC = xB.mean(0) + np.array([0.05, 0.02])
    a.annotate("", xy=muC, xytext=tC, arrowprops=dict(arrowstyle="-|>", color=C_NEW_E, lw=0.9, mutation_scale=6))
    a.scatter(*muC, marker="X", s=22, color=C_NEW_E, zorder=6); a.text(0.02, 0.04, "absent C captures B's\nsamples (collapse)", fontsize=4.8, color=C_NEW_E)
    a.annotate("", xy=xA.mean(0), xytext=tA, arrowprops=dict(arrowstyle="-|>", color="#1d4ed8", lw=0.8, mutation_scale=6))
    # (b): anchored: A, B move partially; C stays at its (biased) anchor; rare classes under-adapted
    b = axes[1]
    for t, x, c in [(tA, xA, "#1d4ed8"), (tB, xB, "#047857")]:
        m = 0.6 * x.mean(0) + 0.4 * t
        b.annotate("", xy=m, xytext=t, arrowprops=dict(arrowstyle="-|>", color=c, lw=0.8, mutation_scale=6)); b.scatter(*m, marker="X", s=18, color=c, zorder=6)
    b.scatter(*tC, marker="X", s=18, color=C_NEW_E, zorder=6); b.text(0.02, 0.04, "C frozen at a biased anchor;\nshift ignored for absent/rare", fontsize=4.8, color=C_NEW_E)
    # (c): GRAFT: field shared through the graph, C moves with the domain shift, and presence suppresses it
    c_ = axes[2]
    for t, x, c in [(tA, xA, "#1d4ed8"), (tB, xB, "#047857")]:
        m = x.mean(0)
        c_.annotate("", xy=m, xytext=t, arrowprops=dict(arrowstyle="-|>", color=c, lw=0.8, mutation_scale=6)); c_.scatter(*m, marker="X", s=18, color=c, zorder=6)
    mC = tC + shift
    c_.plot([tB[0], tC[0]], [tB[1], tC[1]], color="#cbd5e1", lw=0.8, zorder=1)
    c_.annotate("", xy=mC, xytext=tC, arrowprops=dict(arrowstyle="-|>", color=C_NEW_E, lw=0.9, mutation_scale=6, ls="--"))
    c_.scatter(*mC, marker="X", s=18, color=C_NEW_E, zorder=6, alpha=0.5)
    c_.text(0.02, 0.04, "C borrows B's displacement;\n$\\pi_C\\!\\to\\!0$ (presence)", fontsize=4.8, color=C_NEW_E)
    fig.text(0.5, -0.02, "Conceptual illustration (2-D cartoon of the embedding space), not experimental data.", ha="center", fontsize=4.8, style="italic")
    plt.subplots_adjust(wspace=0.05, left=0.01, right=0.99, top=0.82, bottom=0.02)
    save(fig, "fig3_concept.pdf")


if __name__ == "__main__":
    fig1(); fig2(); fig3()
    print("figures written to", OUT)
