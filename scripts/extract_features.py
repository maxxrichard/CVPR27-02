"""Extract and cache image features and prompt-ensembled text prototypes for a dataset / backbone.

Supports OpenAI CLIP (`--family clip`, pip install git+https://github.com/openai/CLIP.git) and OpenCLIP /
SigLIP (`--family open_clip`, pip install open_clip_torch).  Datasets follow the CoOp / StatA layout and
split files (see docs/experiment_plan.md, Section 2).  Output: <out_root>/<dataset>__<backbone>.pt

Example:
  python scripts/extract_features.py --dataset dtd --root /data --family clip --backbone ViT-B/16 --out features/
"""
import argparse, json, os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets_registry import build_dataset, TEMPLATES   # local module in scripts/


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", required=True, help="dataset root (CoOp layout)")
    ap.add_argument("--family", default="clip", choices=["clip", "open_clip"])
    ap.add_argument("--backbone", default="ViT-B/16")
    ap.add_argument("--pretrained", default="openai", help="open_clip pretrained tag (e.g. webli for SigLIP)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="features")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.family == "clip":
        import clip
        model, preprocess = clip.load(args.backbone, device=args.device)
        tokenize = clip.tokenize
        encode_text = model.encode_text
        encode_image = model.encode_image
    else:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(args.backbone, pretrained=args.pretrained, device=args.device)
        tokenize = open_clip.get_tokenizer(args.backbone)
        encode_text = model.encode_text
        encode_image = model.encode_image
    model.eval()

    ds, classnames = build_dataset(args.dataset, args.root, args.split, preprocess)
    templates = TEMPLATES.get(args.dataset, TEMPLATES["default"])
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)

    feats, labels = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="images"):
            f = encode_image(x.to(args.device)).float()
            feats.append(torch.nn.functional.normalize(f, dim=-1).cpu()); labels.append(y)
        text = []
        for name in tqdm(classnames, desc="text"):
            toks = tokenize([t.format(name) for t in templates]).to(args.device)
            e = torch.nn.functional.normalize(encode_text(toks).float(), dim=-1).mean(0)
            text.append(torch.nn.functional.normalize(e, dim=-1).cpu())
    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.dataset}__{args.backbone.replace('/', '-')}" + ("" if args.family == "clip" else f"__{args.pretrained}")
    path = os.path.join(args.out, tag + ".pt")
    torch.save({"features": torch.cat(feats), "labels": torch.cat(labels), "text": torch.stack(text),
                "classnames": classnames, "meta": vars(args) | {"templates": templates}}, path)
    print("saved", path)


if __name__ == "__main__":
    main()
