"""Generate PLACEHOLDER result files so that the table/plot pipeline and the manuscript layout can be exercised
before the real experiments exist.  Files go to results/placeholder/ (never results/raw/) and carry
`"placeholder": true`; the plotting code watermarks figures built from them.  GRAFT/MOON/control values are
synthetic numbers around the published StatA averages (fixed seed) and must be replaced by real runs.
Published averages for CLIP, MTA, EM-Dirichlet, ZLaP, TransCLIP, StatA, TENT, TDA and DMN are the 11-dataset
means reported in the official StatA README (github.com/MaxZanella/StatA, CLIP ViT-B/16), marked "reported".
Usage: python scripts/make_placeholder_results.py
"""
import json, os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "placeholder")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(2027)
SETTINGS = [("64", "1-4"), ("64", "2-10"), ("64", "5-25"), ("1000", "5-25"), ("1000", "25-50"), ("1000", "50-100"), ("-1", "all")]
REPORTED = {"zero_shot": [65.2]*7, "mta": [66.6]*7, "em_dirichlet": [68.5, 70.3, 67.5, 64.4, 45.3, 33.6, 29.5],
            "zlap": [27.5, 35.2, 44.7, 41.5, 52.2, 58.4, 66.4], "transclip": [38.9, 40.4, 42.7, 56.5, 62.0, 64.4, 70.3],
            "stata": [70.4, 69.3, 67.4, 69.7, 69.8, 69.0, 69.9]}
PLACEHOLDER_OFFSET = {"moon": [0.9, 0.7, 0.5, 0.4, 0.3, 0.3, 0.2], "unanchored": [-28.0, -26.0, -22.0, -12.0, -7.0, -4.5, 0.4],
                      "label_shift_only": [-3.0, -2.6, -2.1, -2.8, -3.2, -3.3, -3.9], "count_shrinkage": [-0.3, -0.2, -0.1, -0.1, -0.1, 0.0, 0.0],
                      "global_shift": [-0.8, -0.5, -0.6, -0.9, -1.2, -1.4, -1.6], "graft": [1.6, 1.4, 1.0, 0.8, 0.6, 0.5, 0.3],
                      "graft_eb": [1.7, 1.5, 1.1, 0.9, 0.7, 0.5, 0.3]}
for m, vals in REPORTED.items():
    for (B, K), v in zip(SETTINGS, vals):
        json.dump({"method": m, "batch_size": int(B), "setting": K, "acc_mean": v, "acc_std": None, "source": "reported", "placeholder": False},
                  open(os.path.join(OUT, f"avg11_B{B}_K{K}_{m}.json"), "w"), indent=1)
for m, offs in PLACEHOLDER_OFFSET.items():
    for (B, K), o, base in zip(SETTINGS, offs, REPORTED["stata"]):
        json.dump({"method": m, "batch_size": int(B), "setting": K, "acc_mean": round(base + o + rng.normal(0, 0.05), 1),
                   "acc_std": round(float(rng.uniform(0.2, 0.6)), 1), "source": "placeholder", "placeholder": True},
                  open(os.path.join(OUT, f"avg11_B{B}_K{K}_{m}.json"), "w"), indent=1)
ONLINE_REPORTED = {"zero_shot": [65.2]*4, "mta": [66.6]*4, "tent": [65.8, 65.5, 65.3, 64.5], "tda": [67.7, 67.1, 66.8, 66.6],
                   "dmn": [67.2, 66.5, 66.3, 65.8], "stata": [67.0, 68.9, 69.5, 69.1]}
ONLINE_PH = {"oga": [0.5, 0.2, 0.1, 0.0], "moon": [0.4, 0.5, 0.4, 0.3], "graft": [0.9, 1.2, 1.3, 1.1], "graft_inductive": [0.2, 0.6, 0.8, 0.7]}
GAMMAS = ["0.1", "0.01", "0.001", "-1"]
for m, vals in ONLINE_REPORTED.items():
    for g, v in zip(GAMMAS, vals):
        json.dump({"method": m, "gamma": g, "acc_mean": v, "source": "reported", "placeholder": False}, open(os.path.join(OUT, f"avg11_online_g{g}_{m}.json"), "w"), indent=1)
for m, offs in ONLINE_PH.items():
    for g, o, base in zip(GAMMAS, offs, ONLINE_REPORTED["stata"]):
        json.dump({"method": m, "gamma": g, "acc_mean": round(base + o + rng.normal(0, 0.05), 1), "acc_std": round(float(rng.uniform(0.3, 0.8)), 1),
                   "source": "placeholder", "placeholder": True}, open(os.path.join(OUT, f"avg11_online_g{g}_{m}.json"), "w"), indent=1)
for ds in ["imagenet", "sun397", "fgvc", "eurosat", "stanford_cars", "food101", "oxford_pets", "oxford_flowers", "caltech101", "dtd", "ucf101"]:
    json.dump({"features": f"{ds}__ViT-B-16.pt", "ratio": round(float(rng.uniform(0.25, 0.75)), 2), "random_std": 0.03,
               "global_fraction": round(float(rng.uniform(0.3, 0.7)), 2), "placeholder": True}, open(os.path.join(OUT, f"smooth_{ds}.json"), "w"), indent=1)
for p, vals in {"alpha": {0.0: 69.0, 0.25: 70.2, 1.0: 70.7, 4.0: 70.4, 16.0: 69.6}, "lam": {0.0: 69.1, 1.0: 70.1, 4.0: 70.7, 16.0: 70.5, 64.0: 69.9},
                "knn": {3: 70.4, 5: 70.7, 10: 70.5}}.items():
    for v, a in vals.items():
        json.dump({"param": p, "value": v, "acc_mean": a, "acc_std": 0.3, "placeholder": True}, open(os.path.join(OUT, f"sens_{p}_{v}_avg11.json"), "w"), indent=1)
# per-dataset accuracy in the low setting (placeholders).  Per-dataset values are generated so that their mean over the
# eleven datasets equals the corresponding "Low (B=64)" entry of the main table (reported for the published methods,
# placeholder for the others); zero-shot per-dataset levels are the usual CLIP ViT-B/16 values.
ZS = {"imagenet": 66.7, "sun397": 62.6, "fgvc": 24.8, "eurosat": 47.8, "stanford_cars": 65.3, "food101": 86.1, "oxford_pets": 89.1,
      "oxford_flowers": 71.3, "caltech101": 93.3, "dtd": 44.4, "ucf101": 66.7}
LOW = {"mta": 66.6, "em_dirichlet": 70.3, "zlap": 35.2, "transclip": 40.4, "stata": 69.3, "moon": 70.0, "graft": 70.7}
names = list(ZS); z = np.array([ZS[d] for d in names]); room = (100 - z) / 100
per = {"zero_shot": z}
# designed per-dataset margins of GRAFT over StatA (larger on fine-grained vocabularies), rescaled to the target mean
MARGIN = np.array([1.2, 1.6, 2.2, 0.6, 2.4, 0.9, 0.8, 1.3, 0.5, 1.2, 0.7])
for m, target in LOW.items():
    if m in ("zlap", "transclip"):
        v = z * (target / z.mean()) * rng.uniform(0.85, 1.15, len(z))
    elif m == "graft":
        v = per["stata"] + MARGIN * (target - LOW["stata"]) / MARGIN.mean()
    elif m == "moon":
        v = per["stata"] + 0.5 * MARGIN * (target - LOW["stata"]) / (0.5 * MARGIN.mean())
    else:
        v = z + (target - z.mean()) * room / room.mean() * rng.uniform(0.7, 1.3, len(z))
    v = v + (target - v.mean())
    per[m] = np.clip(v, 1.0, 99.0)
for i, ds in enumerate(names):
    json.dump({"dataset": ds, "setting": "B64_K2-10", **{m: round(float(per[m][i]), 1) for m in per},
               "graft_std": round(float(rng.uniform(0.2, 0.5)), 1), "stata_std": round(float(rng.uniform(0.2, 0.5)), 1), "placeholder": True},
              open(os.path.join(OUT, f"perdataset_low_{ds}.json"), "w"), indent=1)
# ImageNet distribution shifts, low setting (placeholders); the ImageNet row is the per-dataset one above
i0 = names.index("imagenet")
for ds, zz in {"imagenet": 66.7, "imagenet_v2": 60.8, "imagenet_sketch": 46.2, "imagenet_a": 47.8, "imagenet_r": 74.0}.items():
    if ds == "imagenet":
        rec = {m: round(float(per[m][i0]), 1) for m in ("zero_shot", "transclip", "stata", "moon", "graft")}
    else:
        gs = float(per["stata"][i0] - z[i0]) * float(rng.uniform(0.9, 1.3))
        rec = {"zero_shot": zz, "transclip": round(zz * 0.62 * float(rng.uniform(0.9, 1.1)), 1), "stata": round(zz + gs, 1),
               "moon": round(zz + gs + float(rng.uniform(0.3, 1.0)), 1), "graft": round(zz + gs + float(rng.uniform(1.2, 2.6)), 1)}
    json.dump({"dataset": ds, "setting": "B64_K2-10", **rec, "placeholder": True}, open(os.path.join(OUT, f"shift_low_{ds}.json"), "w"), indent=1)
# posterior mass on absent classes versus number of effective classes (ImageNet, B = 64); placeholders
for keff, zs, st, gr in [(1, 31.0, 11.5, 4.1), (2, 28.4, 10.2, 3.6), (4, 24.9, 8.7, 3.0), (10, 19.3, 6.4, 2.3), (25, 12.8, 4.5, 1.7), (50, 8.1, 3.0, 1.2)]:
    json.dump({"keff": keff, "zero_shot": zs, "stata": st, "graft": gr, "placeholder": True}, open(os.path.join(OUT, f"absentmass_keff{keff}.json"), "w"), indent=1)
print("placeholder results written to", OUT)
