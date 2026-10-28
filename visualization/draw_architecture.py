"""Publication-style method figures drawn from the method specification (vector PDF, Okabe-Ito palette,
orthogonal connectors, consistent 7 pt sans-serif labels).  No experimental data is used.
  Fig. 1 overall framework      -> paper/figures/fig1_framework.pdf      (double column)
  Fig. 2 pooling module detail  -> paper/figures/fig2_module.pdf         (single column)
  Fig. 3 conceptual comparison  -> paper/figures/fig3_concept.pdf        (single column, conceptual)
Run: python visualization/draw_architecture.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pubstyle import apply, OKABE_ITO, SINGLE, DOUBLE
apply()
OUT = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "figures"),
       os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "architecture")]
for o in OUT:
    os.makedirs(o, exist_ok=True)

# role colours (fill, edge)
DATA = ("#eef3fb", "#0072B2")        # blue
FROZEN = ("#f1f1f1", "#6b6b6b")      # grey
NEW = ("#fdebd3", "#D55E00")         # vermillion / orange
PRIOR = ("#f3e5f5", "#CC79A7")       # purple
INK = "#222222"


def box(ax, x, y, w, h, text, role, fs=6.6, bold=False, lw=0.9):
    fc, ec = role
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.002,rounding_size=0.012", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, weight="bold" if bold else "normal", linespacing=1.2)
    return (x, y, w, h)


def edge(ax, p0, p1, color=INK, lw=0.7, text=None, tpos=(0, 0.015), fs=6, rad=0.0, ls="-"):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, color=color, lw=lw, shrinkA=1, shrinkB=1,
                        connectionstyle=f"arc3,rad={rad}", linestyle=ls)
    ax.add_patch(a)
    if text:
        ax.text((p0[0] + p1[0]) / 2 + tpos[0], (p0[1] + p1[1]) / 2 + tpos[1], text, ha="center", va="bottom", fontsize=fs, color=color)


def elbow(ax, p0, p1, color=INK, lw=0.7, text=None, fs=6, via="h"):
    """Orthogonal connector: horizontal then vertical (via='h') or vertical then horizontal (via='v')."""
    mid = (p1[0], p0[1]) if via == "h" else (p0[0], p1[1])
    ax.plot([p0[0], mid[0]], [p0[1], mid[1]], color=color, lw=lw, solid_capstyle="round")
    a = FancyArrowPatch(mid, p1, arrowstyle="-|>", mutation_scale=7, color=color, lw=lw, shrinkA=0, shrinkB=1)
    ax.add_patch(a)
    if text:
        ax.text(mid[0], mid[1] + 0.015, text, ha="center", va="bottom", fontsize=fs, color=color)


def right(b): return (b[0] + b[2], b[1] + b[3] / 2)
def left(b): return (b[0], b[1] + b[3] / 2)
def top(b): return (b[0] + b[2] / 2, b[1] + b[3])
def bottom(b): return (b[0] + b[2] / 2, b[1])


def save(fig, name):
    for o in OUT:
        fig.savefig(os.path.join(o, name))
    plt.close(fig)


def fig1():
    fig, ax = plt.subplots(figsize=(DOUBLE, 3.1))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    H, Y1, Y2 = 0.17, 0.70, 0.20
    i1 = box(ax, 0.005, Y1, 0.10, H, "Unlabeled test\nbatch or stream\n$\\{x_i\\}_{i=1}^{N}$", DATA, fs=5.9)
    i2 = box(ax, 0.005, Y2, 0.10, H, "Class names\n$\\{c_k\\}_{k=1}^{K}$\n(full vocabulary)", DATA, fs=5.9)
    e1 = box(ax, 0.125, Y1, 0.10, H, "Frozen image\nencoder $f_v$", FROZEN)
    e2 = box(ax, 0.125, Y2, 0.10, H, "Frozen text\nencoder $f_t$\n(prompt ensemble)", FROZEN, fs=5.7)
    f1 = box(ax, 0.245, Y1, 0.09, H, "Features\n$X\\in\\mathbb{R}^{N\\times d}$", ("white", DATA[1]))
    f2 = box(ax, 0.245, Y2, 0.09, H, "Anchors\n$T\\in\\mathbb{R}^{K\\times d}$", ("white", DATA[1]))
    for a_, b_ in [(i1, e1), (i2, e2), (e1, f1), (e2, f2)]:
        edge(ax, right(a_), left(b_))
    d1 = box(ax, 0.36, 0.74, 0.155, 0.15, "Zero-shot prior\n$\\hat{y}=\\mathrm{softmax}(\\tau XT^{\\!\\top})$", FROZEN, fs=5.8)
    d2 = box(ax, 0.36, 0.50, 0.155, 0.14, "Sample kNN graph\n$W_s$ (inherited)", FROZEN, fs=6.1)
    d3 = box(ax, 0.36, 0.16, 0.155, 0.18, "Semantic class graph\n$L_T$ from $TT^{\\!\\top}$\n(normalised Laplacian)", NEW, fs=5.8)
    edge(ax, right(f1), left(d1)); elbow(ax, right(f1), left(d2), via="h")
    edge(ax, right(f2), left(d3)); elbow(ax, (f2[0] + f2[2] * 0.7, top(f2)[1]), (d1[0] + 0.025, d1[1]), color="#9a9a9a", via="v")
    # GRAFT block
    gx, gy, gw, gh = 0.545, 0.06, 0.335, 0.90
    ax.add_patch(FancyBboxPatch((gx, gy), gw, gh, boxstyle="round,pad=0.002,rounding_size=0.015", fc="#fffaf3", ec=NEW[1], lw=1.0, ls=(0, (4, 2))))
    ax.text(gx + 0.01, gy + gh - 0.012, "GRAFT inference (closed form, no gradients)", ha="left", va="top", fontsize=6.6, weight="bold", color=NEW[1])
    ex, ew, mx, mw = gx + 0.012, 0.135, gx + 0.182, 0.142
    E = box(ax, ex, 0.56, ew, 0.26, "E-step\n$z_{ik}\\propto\\pi_k\\,\\hat{y}_{ik}^{\\lambda_y}\\,e^{\\ell_{ik}(\\mu,\\sigma)}$\n+ sample-graph term", FROZEN, fs=6.0)
    St = box(ax, ex, 0.16, ew, 0.26, "Sufficient statistics\nof $(X, z)$:\n$n_k=\\sum_i z_{ik}$\n$E=\\bar{X}_z-T,\\ q$", ("white", NEW[1]), fs=6.0)
    M1 = box(ax, mx, 0.70, mw, 0.13, "Presence (label shift)\n$\\pi_k\\propto n_k+c$", NEW, fs=5.8)
    M2 = box(ax, mx, 0.38, mw, 0.25, "Graph-pooled\ndisplacement field\n$D=(N_z+sQ)^{-1}N_zE$\nprior $Q=\\alpha I+\\lambda L_T$", NEW, fs=6.2, bold=True, lw=1.3)
    M3 = box(ax, mx, 0.16, mw, 0.13, "Shared covariance\n$\\sigma_j^2$ (inherited)", FROZEN, fs=6.0)
    edge(ax, bottom(E), top(St), text="$z$", tpos=(0.012, -0.02))
    edge(ax, (St[0] + St[2], St[1] + St[3] * 0.85), left(M1))
    edge(ax, (St[0] + St[2], St[1] + St[3] * 0.55), (M2[0], M2[1] + M2[3] * 0.35))
    edge(ax, (St[0] + St[2], St[1] + St[3] * 0.25), left(M3))
    edge(ax, (M2[0], M2[1] + M2[3] * 0.8), (E[0] + E[2], E[1] + E[3] * 0.25), color=NEW[1])
    ax.text(ex + ew - 0.002, 0.49, "$\\mu{=}T{+}D,\\ \\pi,\\ \\sigma^2$\nback to the E-step", ha="right", va="center", fontsize=5.4, color=NEW[1], linespacing=1.1)
    edge(ax, right(d1), (E[0], E[1] + E[3] * 0.85), text="$\\hat{y}$", tpos=(0, 0.01))
    edge(ax, right(d2), (E[0], E[1] + E[3] * 0.3), text="$W_s$", tpos=(0, 0.012))
    # L_T: below the statistics box into the field box
    xm = (St[0] + St[2] + M2[0]) / 2
    ax.plot([d3[0] + d3[2] * 0.5, d3[0] + d3[2] * 0.5, xm, xm], [d3[1], 0.10, 0.10, M2[1] + 0.02], color=NEW[1], lw=0.7)
    ax.add_patch(FancyArrowPatch((xm, M2[1] + 0.02), (M2[0], M2[1] + 0.03), arrowstyle="-|>", mutation_scale=7, color=NEW[1], lw=0.7, shrinkA=0, shrinkB=1))
    ax.text(0.47, 0.105, "$L_T$", ha="center", va="bottom", fontsize=6, color=NEW[1])
    o1 = box(ax, 0.895, 0.60, 0.10, 0.22, "Adapted\nposterior $z$\n(prediction)", DATA, fs=6.0)
    o2 = box(ax, 0.895, 0.20, 0.10, 0.22, "Adapted state\n$\\mu=T+D$, $\\pi$\n(reused online)", DATA, fs=5.8)
    edge(ax, (gx + gw, 0.71), left(o1)); edge(ax, (gx + gw, 0.31), left(o2))
    for i, (role, t) in enumerate([(DATA, "data"), (FROZEN, "frozen or inherited"), (NEW, "this work")]):
        ax.add_patch(FancyBboxPatch((0.01 + i * 0.16, 0.012), 0.018, 0.04, boxstyle="round,pad=0.001", fc=role[0], ec=role[1], lw=0.8))
        ax.text(0.034 + i * 0.16, 0.032, t, va="center", fontsize=6)
    save(fig, "fig1_framework.pdf")


def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(SINGLE, 1.55))
    rng = np.random.default_rng(3)
    centers = np.array([[0.25, 0.7], [0.72, 0.72], [0.5, 0.22]])
    pts = np.concatenate([c + 0.09 * rng.standard_normal((3, 2)) for c in centers])
    clusters = np.repeat(np.arange(3), 3)
    edges = [(i, j) for i in range(9) for j in range(i + 1, 9) if clusters[i] == clusters[j]] + [(0, 3), (4, 6), (2, 7)]
    true_field = np.array([[0.12, 0.05]] * 3 + [[-0.02, 0.12]] * 3 + [[0.08, -0.1]] * 3)
    counts = np.array([30, 1, 0, 25, 0, 2, 40, 0, 1])
    emp = true_field + 0.08 * rng.standard_normal((9, 2)) / np.sqrt(np.maximum(counts, 1))[:, None]
    W = np.zeros((9, 9))
    for i, j in edges:
        W[i, j] = W[j, i] = 1.0
    dinv = 1 / np.sqrt(W.sum(1)); L = np.eye(9) - dinv[:, None] * W * dinv[None, :]
    def solve(alpha, lam):
        N = np.diag(counts.astype(float)); return np.linalg.solve(N + alpha * np.eye(9) + lam * L, N @ np.where(counts[:, None] > 0, emp, 0))
    fields = [np.where(counts[:, None] > 0, emp, np.nan), solve(1.0, 0.0), solve(1.0, 6.0)]
    titles = ["Empirical $e_k=\\bar{x}_k-t_k$", "$\\lambda=0$ (anchor only)", "GRAFT ($\\lambda>0$)"]
    for ax, title, F, let in zip(axes, titles, fields, "abc"):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=6.3, pad=3, x=0.55); ax.text(-0.02, 1.02, let, transform=ax.transAxes, fontsize=8, weight="bold")
        for i, j in edges:
            ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], color="#d0d0d0", lw=0.8, zorder=1)
        for k in range(9):
            ax.scatter(pts[k, 0], pts[k, 1], s=8 + 1.1 * counts[k], color="white", edgecolor=NEW[1] if counts[k] == 0 else "#555555", lw=0.9, zorder=3)
            if counts[k] == 0:
                dy = -0.1 if pts[k, 1] > 0.75 else 0.06
                ax.text(pts[k, 0] + (0.0 if pts[k, 1] > 0.75 else 0.07), pts[k, 1] + dy, "absent", fontsize=4.8, ha="center" if pts[k, 1] > 0.75 else "left", color=NEW[1])
            v = F[k]
            if np.all(np.isfinite(v)):
                ax.annotate("", xy=(pts[k, 0] + 1.6 * v[0], pts[k, 1] + 1.6 * v[1]), xytext=(pts[k, 0], pts[k, 1]),
                            arrowprops=dict(arrowstyle="-|>", color=NEW[1] if ax is axes[2] else "#333333", lw=0.9, mutation_scale=6), zorder=4)
    axes[0].text(0.02, 0.02, "node size $\\propto n_k$", fontsize=5, transform=axes[0].transAxes)
    plt.subplots_adjust(wspace=0.05, left=0.01, right=0.99, top=0.84, bottom=0.02)
    save(fig, "fig2_module.pdf")


def fig3():
    fig, axes = plt.subplots(1, 3, figsize=(SINGLE, 1.5))
    rng = np.random.default_rng(1)
    tA, tB, tC = np.array([0.22, 0.75]), np.array([0.62, 0.72]), np.array([0.82, 0.62])
    shift = np.array([0.08, -0.32])
    xA = tA + shift + 0.05 * rng.standard_normal((25, 2)); xB = tB + shift + 0.05 * rng.standard_normal((25, 2))
    cA, cB, cC = OKABE_ITO["blue"], OKABE_ITO["green"], NEW[1]
    titles = ["Per-class fitting", "Anchor-only shrinkage", "GRAFT"]
    for ax, title, let in zip(axes, titles, "abc"):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=6.3, pad=3, x=0.55)
        ax.text(-0.02, 1.02, let, transform=ax.transAxes, fontsize=8, weight="bold")
        ax.scatter(xA[:, 0], xA[:, 1], s=4, color=cA, alpha=0.5, lw=0); ax.scatter(xB[:, 0], xB[:, 1], s=4, color=cB, alpha=0.5, lw=0)
        for t, c, lab in [(tA, cA, "A"), (tB, cB, "B"), (tC, cC, "C (absent)")]:
            ax.scatter(*t, marker="*", s=30, color=c, zorder=5); ax.text(t[0] + 0.03, t[1] + 0.04, lab, fontsize=5.5, color=c)
    a, b, c_ = axes
    muC = xB.mean(0) + np.array([0.05, 0.02])
    a.annotate("", xy=muC, xytext=tC, arrowprops=dict(arrowstyle="-|>", color=cC, lw=0.9, mutation_scale=6)); a.scatter(*muC, marker="X", s=22, color=cC, zorder=6)
    a.annotate("", xy=xA.mean(0), xytext=tA, arrowprops=dict(arrowstyle="-|>", color=cA, lw=0.8, mutation_scale=6))
    a.text(0.02, 0.04, "absent C captures\nB's samples", fontsize=5, color=cC)
    for t, x, c in [(tA, xA, cA), (tB, xB, cB)]:
        m = 0.6 * x.mean(0) + 0.4 * t
        b.annotate("", xy=m, xytext=t, arrowprops=dict(arrowstyle="-|>", color=c, lw=0.8, mutation_scale=6)); b.scatter(*m, marker="X", s=18, color=c, zorder=6)
    b.scatter(*tC, marker="X", s=18, color=cC, zorder=6); b.text(0.02, 0.04, "C stays at a\nbiased anchor", fontsize=5, color=cC)
    for t, x, c in [(tA, xA, cA), (tB, xB, cB)]:
        m = x.mean(0); c_.annotate("", xy=m, xytext=t, arrowprops=dict(arrowstyle="-|>", color=c, lw=0.8, mutation_scale=6)); c_.scatter(*m, marker="X", s=18, color=c, zorder=6)
    c_.plot([tB[0], tC[0]], [tB[1], tC[1]], color="#d0d0d0", lw=0.8, zorder=1)
    mC = tC + shift
    c_.annotate("", xy=mC, xytext=tC, arrowprops=dict(arrowstyle="-|>", color=cC, lw=0.9, mutation_scale=6, ls="--")); c_.scatter(*mC, marker="X", s=18, color=cC, zorder=6, alpha=0.5)
    c_.text(0.02, 0.04, "C borrows B's shift;\n$\\pi_C\\to 0$", fontsize=5, color=cC)
    plt.subplots_adjust(wspace=0.05, left=0.01, right=0.99, top=0.84, bottom=0.02)
    save(fig, "fig3_concept.pdf")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); print("figures written")
