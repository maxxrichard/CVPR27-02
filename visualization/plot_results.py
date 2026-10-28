"""Plotting scripts for the experiments in docs/experiment_plan.md.  They read the JSON files written by
scripts/run_batch.py / scripts/run_online.py / scripts/analyze_smoothness.py and refuse to draw anything when the
files are missing (no fabricated curves).  Expected layout:
  results/raw/<dataset>_B<batch>_K<kmin>-<kmax>_<method>.json   (batch protocol)
  results/raw/<dataset>_online_g<gamma>_<method>.json           (online protocol)
  results/raw/smooth_<dataset>.json                             (H1 analysis)
  results/raw/sens_<param>_<value>_<dataset>.json               (hyper-parameter sensitivity, batch protocol)
Usage: python visualization/plot_results.py --which keff|sensitivity|efficiency|smoothness|all
"""
import argparse, glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = os.path.join(os.path.dirname(__file__), "..", "results", "raw")
OUT = [os.path.join(os.path.dirname(__file__), "..", "paper", "figures"), os.path.join(os.path.dirname(__file__), "..", "figures", "quantitative")]
plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})
METHOD_STYLE = {"zero_shot": ("CLIP zero-shot", "#6b7280", ":"), "count_shrinkage": ("anchor-only shrinkage (StatA-like control)", "#2563eb", "--"),
                "global_shift": ("complete pooling (global shift)", "#7c3aed", "-."), "graft": ("GRAFT", "#c2610a", "-"), "graft_eb": ("GRAFT + EB", "#b45309", "-")}
SETTINGS = [("64", "1-4", "Very low"), ("64", "2-10", "Low"), ("64", "5-25", "Medium"), ("1000", "5-25", "Medium"), ("1000", "25-50", "High"), ("1000", "50-100", "Very high"), ("-1", "all", "All")]


def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RAW, pattern)))]


def save(fig, name):
    for o in OUT:
        os.makedirs(o, exist_ok=True); fig.savefig(os.path.join(o, name), bbox_inches="tight")
    plt.close(fig); print("wrote", name)


def plot_keff():
    files = glob.glob(os.path.join(RAW, "*_B*_K*_*.json"))
    if not files:
        print("[keff] no result files found; nothing drawn"); return
    rows = {}
    for f in files:
        m = re.match(r"(.+)_B(-?\d+)_K([\w-]+)_(\w+)\.json", os.path.basename(f))
        if not m: continue
        ds, B, K, meth = m.groups(); r = json.load(open(f))
        rows.setdefault(meth, {}).setdefault((B, K), []).append(r["acc_mean"])
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    xs = np.arange(len(SETTINGS))
    for meth, d in rows.items():
        ys = [np.mean(d.get((B, K), [np.nan])) for B, K, _ in SETTINGS]
        lab, col, ls = METHOD_STYLE.get(meth, (meth, None, "-"))
        ax.plot(xs, ys, marker="o", ms=3, color=col, ls=ls, label=lab)
    ax.set_xticks(xs); ax.set_xticklabels([f"{n}\nB={B}" for B, K, n in SETTINGS], fontsize=5.5)
    ax.set_ylabel("Top-1 accuracy (%, mean over datasets)"); ax.set_xlabel("number of effective classes per batch")
    ax.legend(fontsize=5.5, frameon=False); ax.grid(alpha=0.3)
    save(fig, "plot_keff.pdf")


def plot_sensitivity():
    files = glob.glob(os.path.join(RAW, "sens_*.json"))
    if not files:
        print("[sensitivity] no result files found; nothing drawn"); return
    data = {}
    for f in files:
        m = re.match(r"sens_(\w+)_([\d.e-]+)_(.+)\.json", os.path.basename(f))
        if not m: continue
        p, v, ds = m.groups(); r = json.load(open(f)); data.setdefault(p, {}).setdefault(float(v), []).append(r["acc_mean"])
    fig, axes = plt.subplots(1, len(data), figsize=(1.7 * len(data), 1.5), squeeze=False)
    for ax, (p, d) in zip(axes[0], sorted(data.items())):
        v = sorted(d); ax.plot(v, [np.mean(d[x]) for x in v], marker="o", ms=3, color="#c2610a")
        ax.set_xscale("symlog"); ax.set_xlabel(p); ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("accuracy (%)")
    save(fig, "plot_sensitivity.pdf")


def plot_efficiency():
    files = glob.glob(os.path.join(RAW, "*_B*_K*_*.json"))
    if not files:
        print("[efficiency] no result files found; nothing drawn"); return
    pts = {}
    for f in files:
        r = json.load(open(f)); pts.setdefault(r["method"], []).append((r["time_per_task_s"], r["acc_mean"]))
    fig, ax = plt.subplots(figsize=(2.2, 1.7))
    for m, p in pts.items():
        p = np.array(p); lab, col, _ = METHOD_STYLE.get(m, (m, None, "-")); ax.scatter(p[:, 0], p[:, 1], s=8, color=col, label=lab)
    ax.set_xscale("log"); ax.set_xlabel("seconds per task"); ax.set_ylabel("accuracy (%)"); ax.legend(fontsize=5, frameon=False)
    save(fig, "plot_efficiency.pdf")


def plot_smoothness():
    files = glob.glob(os.path.join(RAW, "smooth_*.json"))
    if not files:
        print("[smoothness] no result files found; nothing drawn"); return
    rs = [json.load(open(f)) for f in sorted(files)]
    fig, ax = plt.subplots(figsize=(3.4, 1.5))
    names = [r["features"].split("__")[0] for r in rs]
    ax.bar(np.arange(len(rs)), [r["ratio"] for r in rs], color="#c2610a")
    ax.axhline(1.0, color="#6b7280", ls="--", lw=0.8); ax.set_xticks(np.arange(len(rs))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=5.5)
    ax.set_ylabel("Dirichlet-energy ratio\n(graph / random)")
    save(fig, "plot_smoothness.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--which", default="all"); a = ap.parse_args()
    fns = {"keff": plot_keff, "sensitivity": plot_sensitivity, "efficiency": plot_efficiency, "smoothness": plot_smoothness}
    for k, fn in fns.items():
        if a.which in ("all", k): fn()
