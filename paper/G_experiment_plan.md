# Deliverable G: Experiment Execution Plan (pre-registered, nothing executed yet)

All experiments run on cached features. One GPU (any 8 GB card) suffices; feature extraction dominates cost.
Estimated wall-clock for the whole plan with CLIP ViT-B/16: feature extraction ~2 h (ImageNet included),
batch protocol ~3 h, online protocol ~1 h, ablations ~2 h.

## 1. Environment
```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git open_clip_torch
python -m pytest tests -q
```

## 2. Data
Datasets in the CoOp / StatA layout (follow DATASETS.md of github.com/MaxZanella/StatA): imagenet, caltech-101,
oxford_pets, stanford_cars, oxford_flowers, food-101, fgvc_aircraft, sun397, dtd, eurosat, ucf101, imagenetv2,
imagenet-sketch, imagenet-adversarial, imagenet-rendition. Standard CoOp split json files are required.

## 3. Feature extraction (once per dataset and backbone)
```bash
for ds in imagenet sun397 fgvc eurosat stanford_cars food101 oxford_pets oxford_flowers caltech101 dtd ucf101 imagenet_v2 imagenet_sketch imagenet_a imagenet_r; do
  python scripts/extract_features.py --dataset $ds --root $DATA --family clip --backbone ViT-B/16 --out features
done
# other backbones
python scripts/extract_features.py --dataset imagenet --root $DATA --family clip --backbone ViT-L/14 --out features
python scripts/extract_features.py --dataset imagenet --root $DATA --family open_clip --backbone ViT-B-16 --pretrained datacomp_xl_s13b_b90k --out features
python scripts/extract_features.py --dataset imagenet --root $DATA --family open_clip --backbone ViT-B-16-SigLIP --pretrained webli --out features
```

## 4. Experiments (research question -> command -> target table/figure)

| # | Question | Hypothesis | Datasets | Baselines | Metric | Config | Cost | Target |
|---|---|---|---|---|---|---|---|---|
| E1 | H1: is the oracle displacement field smooth on the class graph? | energy ratio << 1 | all 15 | none (label analysis) | Dirichlet-energy ratio, global fraction | `scripts/analyze_smoothness.py` | minutes | Sec. 6.5, plot_smoothness |
| E2 | Main realistic batch comparison | no collapse; > anchor-only at low K_eff | 11 | zero-shot, MTA, EM-Dirichlet, ZLaP, ECALP, TransCLIP, StatA, MOON + 4 controls | top-1 mean, worst decile | 7 settings, 1000 tasks | ~3 h | Tab. 1, plot_keff |
| E3 | Online streams | robust under correlation | 11 | TENT, TDA, DMN, OGA, StatA, MOON | top-1 | gamma in {0.1,0.01,0.001,-1}, B=128, 100 runs | ~1 h | Tab. 2 |
| E4 | Backbone transfer and shifts | gains persist on SigLIP/OpenCLIP | ImageNet + V2/S/A/R | zero-shot, StatA | top-1 | low setting | ~1 h | Tab. 3 |
| E5 | Calibration | presence improves ECE | 11 | TransCLIP, StatA, MOON, anchor-only | ECE, NLL, worst decile | low setting | included in E2 | Tab. 4 |
| E6 | Component ablation | each component contributes | 11 | controls | top-1 | configs/control_*.json, ablation_*.json | ~1 h | Tab. 5 |
| E7 | Prior / graph ablation incl. random graph | text graph > random graph | 11 | — | top-1 | configs/sens_*.json + random-graph flag | ~1 h | Tab. 6, plot_sensitivity |
| E8 | Likelihood / optimisation / cost | — | 11 | — | top-1, ms/task | configs/ablation_*.json | ~1 h | Tab. 7, Tab. 8, plot_efficiency |

### Commands
```bash
# E1
for f in features/*ViT-B-16.pt; do d=$(basename $f .pt | cut -d_ -f1); python scripts/analyze_smoothness.py --features $f --out results/raw/smooth_$d.json; done
# E2 (our implementation; for the published baselines use the official harness, see scripts/official_harness_patch.md)
for ds in imagenet sun397 fgvc eurosat stanford_cars food101 oxford_pets oxford_flowers caltech101 dtd ucf101; do
 for m in zero_shot graft graft_eb count_shrinkage global_shift label_shift_only; do
  for s in "64 1 4" "64 2 10" "64 5 25" "1000 5 25" "1000 25 50" "1000 50 100"; do set -- $s
   python scripts/run_batch.py --features features/${ds}__ViT-B-16.pt --method $m --batch_size $1 --kmin $2 --kmax $3 --n_tasks 1000 --out results/raw/${ds}_B$1_K$2-$3_$m.json
  done
  python scripts/run_batch.py --features features/${ds}__ViT-B-16.pt --method $m --batch_size -1 --n_tasks 1 --out results/raw/${ds}_B-1_Kall_$m.json
 done
done
# E3
for ds in ...; do for g in 0.1 0.01 0.001 -1; do python scripts/run_online.py --features features/${ds}__ViT-B-16.pt --batch_size 128 --gamma $g --n_runs 100 --out results/raw/${ds}_online_g${g}_graft.json; done; done
# E7 sensitivity (low setting)
for c in configs/sens_*.json; do n=$(basename $c .json); python scripts/run_batch.py --features features/imagenet__ViT-B-16.pt --method graft --config $c --batch_size 64 --kmin 2 --kmax 10 --n_tasks 1000 --out results/raw/${n#sens_}_imagenet.json; done
# tables and plots
python visualization/generate_tables.py && python visualization/plot_results.py --which all
```
Random-graph control for E7: pass a degree-preserving permuted `W_c` to `graft_batch` (helper in
`graft/class_graph.py::smoothness_ratio` shows the permutation); a CLI flag is a one-line addition in run_batch.py.

## 5. Pre-registered decision rules
* H1 supported on a dataset if energy ratio < 0.7 (mean over 20 permutations).
* Claim "no collapse" supported if GRAFT >= zero-shot in all 7 settings on the 11-dataset mean and in >= 9/11 datasets.
* Claim "cross-class pooling helps" supported if GRAFT > anchor-only control by > 1 point in very-low and low
  settings on the mean, with the random-graph control not matching the text graph.
* Any outcome is reported; the manuscript's discussion will be rewritten to the measured numbers.
