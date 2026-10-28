"""Result figures for the manuscript in publication style (see pubstyle.py).  Reads results/raw/ when real runs
exist, otherwise results/placeholder/ (scripts/make_placeholder_results.py) and stamps a PLACEHOLDER watermark.
Usage: python visualization/plot_paper_figures.py
"""
import glob, json, os, sys, dataclasses
import numpy as np
import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pubstyle import apply, panel_label, watermark, METHOD_COLOR, METHOD_NAME, OKABE_ITO, SINGLE, DOUBLE
apply()
RAW, PH = os.path.join(ROOT, "results", "raw"), os.path.join(ROOT, "results", "placeholder")
OUT = [os.path.join(ROOT, "paper", "figures"), os.path.join(ROOT, "figures", "quantitative")]
SETTINGS = [("64", "1-4", "very low"), ("64", "2-10", "low"), ("64", "5-25", "medium"), ("1000", "5-25", "medium"), ("1000", "25-50", "high"), ("1000", "50-100", "very high"), ("-1", "all", "all")]


def src():
    return (RAW, False) if glob.glob(os.path.join(RAW, "avg11_*.json")) else (PH, True)


def save(fig, name):
    for o in OUT:
        os.makedirs(o, exist_ok=True); fig.savefig(os.path.join(o, name))
    plt.close(fig); print("wrote", name)


def jload(d, pat):
    fs = sorted(glob.glob(os.path.join(d, pat)))
    return [json.load(open(f)) for f in fs]


def keff_lines():
    """Accuracy versus effective classes: lines with markers (one axis, legend + direct end labels)."""
    d, ph = src()
    fig, ax = plt.subplots(figsize=(SINGLE, 2.2))
    xs = np.arange(len(SETTINGS))
    for m in ["zero_shot", "transclip", "stata", "moon", "count_shrinkage", "graft"]:
        ys, es = [], []
        for B, K, _ in SETTINGS:
            r = jload(d, f"avg11_B{B}_K{K}_{m}.json"); r = r[0] if r else {}
            ys.append(r.get("acc_mean", np.nan)); es.append(r.get("acc_std") or 0)
        ax.errorbar(xs, ys, yerr=es, color=METHOD_COLOR[m], marker="o", ms=3.5, lw=1.2 if m == "graft" else 0.9,
                    ls="-" if m in ("graft", "stata", "moon", "count_shrinkage") else "--", label=METHOD_NAME[m], capsize=1.5, elinewidth=0.6)
    ax.set_xticks(xs); ax.set_xticklabels([f"{n}\n$B$={'all' if B == '-1' else B}" for B, K, n in SETTINGS], fontsize=6)
    ax.set_ylabel("Top-1 accuracy (%)"); ax.set_xlabel("Effective classes per batch")
    ax.set_ylim(35, 75); ax.legend(ncol=2, loc="lower right", handlelength=1.6)
    ax.axhline(65.2, color=OKABE_ITO["grey"], lw=0.5, ls=":", zorder=0)
    watermark(fig, ph); save(fig, "plot_keff.pdf")


def sensitivity():
    d, ph = src()
    data = {}
    for r in jload(d, "sens_*.json"):
        data.setdefault(r["param"], {})[float(r["value"])] = (r["acc_mean"], r.get("acc_std", 0))
    if not data:
        return
    names = {"alpha": r"Anchor strength $\alpha$", "lam": r"Graph coupling $\lambda$", "knn": "Neighbours $m$"}
    fig, axes = plt.subplots(1, 3, figsize=(SINGLE, 1.45), sharey=True)
    for ax, p, let in zip(axes, ["alpha", "lam", "knn"], "abc"):
        v = sorted(data[p])
        ax.errorbar(range(len(v)), [data[p][x][0] for x in v], yerr=[data[p][x][1] for x in v], marker="o", ms=3.5, color=METHOD_COLOR["graft"], lw=1.0, capsize=1.5, elinewidth=0.6)
        ax.set_xticks(range(len(v))); ax.set_xticklabels([f"{x:g}" for x in v]); ax.set_xlabel(names[p], fontsize=7)
        panel_label(ax, let, x=-0.08 if let != "a" else -0.35)
    axes[0].set_ylabel("Accuracy (%)"); axes[0].set_ylim(68.5, 71.5)
    plt.subplots_adjust(wspace=0.12)
    watermark(fig, ph); save(fig, "plot_sensitivity.pdf")


def smoothness():
    d, ph = src()
    rs = sorted(jload(d, "smooth_*.json"), key=lambda r: r["ratio"])
    if not rs:
        return
    fig, ax = plt.subplots(figsize=(SINGLE, 1.5))
    names = [r["features"].split("__")[0].replace("_", " ").replace("oxford ", "").replace("stanford ", "") for r in rs]
    ax.bar(np.arange(len(rs)), [r["ratio"] for r in rs], yerr=[r.get("random_std", 0) for r in rs], width=0.6, color=METHOD_COLOR["graft"], error_kw=dict(lw=0.6, capsize=1.5))
    ax.axhline(1.0, color=OKABE_ITO["grey"], ls="--", lw=0.7); ax.text(len(rs) - 0.5, 1.02, "random graph", ha="right", va="bottom", fontsize=6, color=OKABE_ITO["grey"])
    ax.set_xticks(np.arange(len(rs))); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=6)
    ax.set_ylabel("Energy ratio"); ax.set_ylim(0, 1.15)
    watermark(fig, ph); save(fig, "plot_smoothness.pdf")


def embedding_view():
    import torch
    from tests.synthetic import make_problem
    from graft import GRAFTConfig, graft_batch
    present = [0, 1, 6, 7, 12, 13]
    X, Y, T, Dtrue = make_problem(K=24, d=48, n_per_class=25, present=present, seed=11)
    cfg = GRAFTConfig(sample_knn=0)
    _, _, st_g = graft_batch(X, T, cfg, return_state=True)
    _, _, st_a = graft_batch(X, T, dataclasses.replace(cfg, lam=0.0), return_state=True)
    Mu_true = torch.nn.functional.normalize(T + Dtrue, dim=-1)
    allpts = torch.cat([X, T, Mu_true]); mean = allpts.mean(0)
    U, S, V = torch.pca_lowrank(allpts - mean, q=2)
    P = lambda Z: ((Z - mean) @ V[:, :2]).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE, 1.75), sharex=True, sharey=True)
    absent = [k for k in range(24) if k not in present]
    for ax, st, title, let in zip(axes, [st_a, st_g], ["Anchor-only ($\\lambda=0$)", "GRAFT ($\\lambda>0$)"], "ab"):
        x = P(X); ax.scatter(x[:, 0], x[:, 1], s=3, color=OKABE_ITO["skyblue"], alpha=0.45, lw=0, label="test images")
        t = P(T); m = P(torch.nn.functional.normalize(st.Mu, dim=-1)); mt = P(Mu_true)
        ax.scatter(t[:, 0], t[:, 1], marker="*", s=22, color=OKABE_ITO["black"], label="text anchor", zorder=4)
        ax.scatter(mt[absent, 0], mt[absent, 1], marker="o", s=16, facecolors="none", edgecolors=METHOD_COLOR["graft"], lw=0.8, label="true centroid, absent class", zorder=4)
        for k in range(24):
            ax.annotate("", xy=m[k], xytext=t[k], arrowprops=dict(arrowstyle="-|>", lw=0.7, color=METHOD_COLOR["graft"] if k in absent else OKABE_ITO["blue"], mutation_scale=5))
        ax.set_title(title, fontsize=7.5); ax.set_xticks([]); ax.set_yticks([]); panel_label(ax, let, x=0.0)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(True)
    axes[0].legend(fontsize=5, loc="lower left", handlelength=1.0)
    plt.subplots_adjust(wspace=0.06)
    save(fig, "plot_embedding_synthetic.pdf")


def per_dataset():
    """Per-dataset accuracy in the low setting: zero-shot vs StatA vs GRAFT (grouped bars, single column)."""
    d, ph = src()
    per = sorted(jload(d, "perdataset_low_*.json"), key=lambda r: r["dataset"])
    if not per:
        return
    fig, a = plt.subplots(figsize=(SINGLE, 1.65))
    names = [r["dataset"].replace("_", " ").replace("oxford ", "").replace("stanford ", "") for r in per]
    xs = np.arange(len(per)); w = 0.27
    a.bar(xs - w, [r["zero_shot"] for r in per], w, color=METHOD_COLOR["zero_shot"], label="CLIP zero-shot")
    a.bar(xs, [r["stata"] for r in per], w, yerr=[r.get("stata_std", 0) for r in per], color=METHOD_COLOR["stata"], label="StatA", error_kw=dict(lw=0.5, capsize=1))
    a.bar(xs + w, [r["graft"] for r in per], w, yerr=[r.get("graft_std", 0) for r in per], color=METHOD_COLOR["graft"], label="GRAFT", error_kw=dict(lw=0.5, capsize=1))
    a.set_xticks(xs); a.set_xticklabels(names, rotation=35, ha="right", fontsize=6); a.set_ylabel("Top-1 accuracy (%)")
    a.set_ylim(0, 112); a.set_yticks([0, 25, 50, 75, 100]); a.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.03), handlelength=1.0)
    watermark(fig, ph); save(fig, "plot_perdataset.pdf")


def absent_mass():
    """Posterior mass assigned to absent classes versus the number of effective classes (ImageNet, B = 64)."""
    d, ph = src()
    mass = sorted(jload(d, "absentmass_keff*.json"), key=lambda r: r["keff"])
    if not mass:
        return
    fig, b = plt.subplots(figsize=(SINGLE, 1.25))
    ks = [r["keff"] for r in mass]
    for m in ["zero_shot", "stata", "graft"]:
        b.plot(ks, [r[m] for r in mass], marker="o", ms=3.5, color=METHOD_COLOR[m], label=METHOD_NAME[m])
    b.set_xscale("log"); b.set_xticks(ks); b.set_xticklabels([str(k) for k in ks]); b.minorticks_off()
    b.set_xlabel("Effective classes $K_{\\mathrm{eff}}$"); b.set_ylabel("Posterior mass on\nabsent classes (%)"); b.set_ylim(0, 35)
    b.legend(handlelength=1.4, ncol=3, loc="upper right")
    watermark(fig, ph); save(fig, "plot_absentmass.pdf")


def qualitative_grid(n_rows=2, n_cols=4):
    """Qualitative predictions: test images with the ground truth and the predictions of zero-shot, StatA and
    GRAFT.  Real inputs: results/raw/qualitative.json ({"items": [{image, label, zs, stata, graft}, ...]});
    otherwise results/placeholder/qualitative.json (sample photographs with illustrative placeholder predictions,
    watermarked).  Correct predictions are typeset in green, wrong ones in vermillion."""
    f = os.path.join(RAW, "qualitative.json"); ph = False
    if not os.path.exists(f):
        f = os.path.join(PH, "qualitative.json"); ph = True
    items = json.load(open(f))["items"] if os.path.exists(f) else []
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(SINGLE, 2.0))
    ok, bad = OKABE_ITO["green"], OKABE_ITO["vermillion"]
    for i, ax in enumerate(axes.ravel()):
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values(): s_.set_visible(True); s_.set_linewidth(0.5)
        if i < len(items):
            it = items[i]; im = plt.imread(os.path.join(ROOT, it["image"]))
            h, w = im.shape[:2]; c = min(h, w); y0, x0 = (h - c) // 2, (w - c) // 2
            ax.imshow(im[y0:y0 + c, x0:x0 + c]); ax.set_aspect("equal")
            gt, preds = it["label"], [("ZS", it["zs"]), ("StatA", it["stata"]), ("GRAFT", it["graft"])]
        else:
            ax.set_facecolor("#ececec"); gt, preds = "class", []
        ax.set_title(f"GT: {gt}", fontsize=5.5, pad=2)
        for j, (m, p) in enumerate(preds):
            ax.text(0.0, -0.1 - 0.2 * j, f"{m}: {p}", transform=ax.transAxes, fontsize=4.7, color=ok if p == gt else bad, va="top")
    plt.subplots_adjust(wspace=0.22, hspace=1.1, top=0.94, bottom=0.2)
    watermark(fig, ph); save(fig, "plot_qualitative.pdf")


if __name__ == "__main__":
    keff_lines(); sensitivity(); smoothness(); embedding_view(); per_dataset(); absent_mass(); qualitative_grid()
