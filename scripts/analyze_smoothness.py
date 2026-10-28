"""Hypothesis H1: oracle cross-modal displacements (per-class image centroid minus text anchor, computed with
ground-truth labels) vary smoothly over the text kNN class graph.  Reports the Dirichlet-energy ratio against
degree-preserving random relabellings (ratio << 1 supports H1) and the fraction of displacement energy explained
by the global (complete-pooling) component.
  python scripts/analyze_smoothness.py --features features/imagenet__ViT-B-16.pt --out results/raw/smooth_imagenet.json
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from graft.class_graph import build_class_graph, normalized_laplacian, smoothness_ratio
from graft.data import load_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--knn", type=int, default=5)
    ap.add_argument("--rho", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = load_features(args.features)
    X = torch.nn.functional.normalize(d["features"].float(), dim=-1); Y = d["labels"]; T = torch.nn.functional.normalize(d["text"].float(), dim=-1)
    K = T.shape[0]
    cent = torch.zeros_like(T); n = torch.zeros(K)
    cent.index_add_(0, Y, X); n.index_add_(0, Y, torch.ones(len(Y)))
    present = n > 0
    cent[present] = cent[present] / n[present][:, None]
    D = torch.where(present[:, None], cent - T, torch.zeros_like(T))
    W = build_class_graph(T, args.knn, args.rho); L = normalized_laplacian(W)
    res = smoothness_ratio(D[present], L[present][:, present])
    g = D[present].mean(0, keepdim=True)
    res.update({"global_fraction": float((g.norm() ** 2 * present.sum()) / (D[present].norm() ** 2).clamp_min(1e-12)),
                "mean_displacement_norm": float(D[present].norm(dim=1).mean()), "n_classes_present": int(present.sum()),
                "knn": args.knn, "rho": args.rho, "features": os.path.basename(args.features)})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
