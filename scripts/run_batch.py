"""Batch (transductive) evaluation under the realistic effective-class protocol.

Example (StatA Table-1 style, "Low": 2-10 effective classes, batch 64, 1000 tasks):
  python scripts/run_batch.py --features features/imagenet__ViT-B-16.pt --method graft --batch_size 64 \
      --kmin 2 --kmax 10 --n_tasks 1000 --out results/raw/imagenet_B64_K2-10_graft.json
Methods: zero_shot | graft | graft_eb | count_shrinkage | global_shift | label_shift_only
Results are written as JSON with per-task accuracies (mean, std, worst-decile) and calibration.
"""
import argparse, json, os, sys, time, dataclasses
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from graft import GRAFTConfig, graft_batch, zero_shot, count_shrinkage, global_shift
from graft.baselines import label_shift_only
from graft.samplers import BatchSampler
from graft.metrics import accuracy, expected_calibration_error
from graft.data import load_features


def run_method(name, X, T, cfg):
    if name == "zero_shot":
        return zero_shot(X, T, cfg.logit_scale)
    if name == "graft":
        return graft_batch(X, T, cfg)
    if name == "graft_eb":
        return graft_batch(X, T, dataclasses.replace(cfg, empirical_bayes=True))
    if name == "count_shrinkage":
        return count_shrinkage(X, T, cfg)
    if name == "global_shift":
        return global_shift(X, T, cfg)
    if name == "label_shift_only":
        return label_shift_only(X, T, cfg)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--method", default="graft")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--kmin", type=int, default=None)
    ap.add_argument("--kmax", type=int, default=None)
    ap.add_argument("--n_tasks", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--config", default=None, help="JSON file overriding GRAFTConfig fields")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = GRAFTConfig(**json.load(open(args.config))) if args.config else GRAFTConfig()
    d = load_features(args.features, args.device)
    X_all, Y_all, T = d["features"], d["labels"], d["text"].to(args.device)
    sampler = BatchSampler(Y_all, args.batch_size, args.kmin, args.kmax, seed=args.seed)
    accs, zs_accs, eces, times = [], [], [], []
    for t in range(args.n_tasks):
        idx = torch.tensor(sampler.sample())
        X, Y = X_all[idx].to(args.device), Y_all[idx]
        t0 = time.time()
        y_hat, z = run_method(args.method, X, T, cfg)
        times.append(time.time() - t0)
        accs.append(accuracy(z, Y)); zs_accs.append(accuracy(y_hat, Y)); eces.append(expected_calibration_error(z, Y))
    accs = np.array(accs)
    res = {"method": args.method, "features": os.path.basename(args.features), "batch_size": args.batch_size,
           "kmin": args.kmin, "kmax": args.kmax, "n_tasks": args.n_tasks, "seed": args.seed, "config": dataclasses.asdict(cfg),
           "acc_mean": float(accs.mean()), "acc_std": float(accs.std()), "acc_worst_decile": float(np.sort(accs)[: max(1, len(accs) // 10)].mean()),
           "zero_shot_acc_mean": float(np.mean(zs_accs)), "ece_mean": float(np.mean(eces)), "time_per_task_s": float(np.mean(times)),
           "per_task_acc": accs.tolist()}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"{args.method}: acc {res['acc_mean']:.2f} +- {res['acc_std']:.2f} (zero-shot {res['zero_shot_acc_mean']:.2f}), ECE {res['ece_mean']:.2f}")


if __name__ == "__main__":
    main()
