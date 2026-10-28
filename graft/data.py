"""Cached-feature I/O.  A feature file is a torch checkpoint with keys
   'features' (N x d float), 'labels' (N long), 'text' (K x d float: class prototypes, prompt-ensembled),
   'classnames' (list[str]), and optional 'meta' (dict: backbone, dataset, templates)."""
import os
import torch


def load_features(path: str, device: str = "cpu"):
    d = torch.load(path, map_location=device)
    for k in ("features", "labels", "text"):
        assert k in d, f"missing key {k} in {path}"
    return d


def feature_path(root: str, dataset: str, backbone: str) -> str:
    return os.path.join(root, f"{dataset}__{backbone.replace('/', '-')}.pt")
