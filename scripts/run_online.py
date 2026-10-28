"""Online evaluation on correlated streams (StatA Table-2 style protocol).

Example: python scripts/run_online.py --features features/imagenet__ViT-B-16.pt --batch_size 128 --gamma 0.01 \
             --n_runs 100 --out results/raw/imagenet_online_g0.01_graft.json
gamma in {0.1, 0.01, 0.001} controls stream correlation; gamma = -1 is the "Separate" stream.
"""
import argparse, json, os, sys, time, dataclasses
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from graft import GRAFTConfig, GRAFTOnline, zero_shot
from graft.samplers import OnlineSampler
from graft.metrics import accuracy, expected_calibration_error
from graft.data import load_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--method", default="graft", choices=["graft", "graft_eb", "zero_shot"])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--gamma", type=float, default=0.01)
    ap.add_argument("--n_runs", type=int, default=100)
    ap.add_argument("--predict_before_update", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = GRAFTConfig(**json.load(open(args.config))) if args.config else GRAFTConfig()
    if args.method == "graft_eb":
        cfg = dataclasses.replace(cfg, empirical_bayes=True)
    d = load_features(args.features, args.device)
    X_all, Y_all, T = d["features"], d["labels"], d["text"].to(args.device)
    run_accs, zs_accs, eces = [], [], []
    for r in range(args.n_runs):
        sampler = OnlineSampler(Y_all, args.batch_size, args.gamma, seed=args.seed * 1000 + r)
        model = GRAFTOnline(T, cfg, predict_before_update=args.predict_before_update)
        preds, zsp, ys = [], [], []
        for idx in sampler:
            idx = torch.tensor(idx); X = X_all[idx].to(args.device)
            if args.method == "zero_shot":
                y_hat, z = zero_shot(X, T, cfg.logit_scale)
            else:
                y_hat, z = model.step(X)
            preds.append(z.cpu()); zsp.append(y_hat.cpu()); ys.append(Y_all[idx])
        P, Z, Y = torch.cat(preds), torch.cat(zsp), torch.cat(ys)
        run_accs.append(accuracy(P, Y)); zs_accs.append(accuracy(Z, Y)); eces.append(expected_calibration_error(P, Y))
    a = np.array(run_accs)
    res = {"method": args.method, "features": os.path.basename(args.features), "batch_size": args.batch_size, "gamma": args.gamma,
           "n_runs": args.n_runs, "predict_before_update": args.predict_before_update, "config": dataclasses.asdict(cfg),
           "acc_mean": float(a.mean()), "acc_std": float(a.std()), "acc_worst_decile": float(np.sort(a)[: max(1, len(a) // 10)].mean()),
           "zero_shot_acc_mean": float(np.mean(zs_accs)), "ece_mean": float(np.mean(eces)), "per_run_acc": a.tolist()}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"{args.method} gamma={args.gamma}: acc {res['acc_mean']:.2f} +- {res['acc_std']:.2f} (zero-shot {res['zero_shot_acc_mean']:.2f})")


if __name__ == "__main__":
    main()
