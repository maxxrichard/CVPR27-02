"""Minimal dataset registry (CoOp / StatA layout).  Each dataset directory is expected to contain the
images and a `split_<name>.json` file with the standard test split used by CoOp, Tip-Adapter, TransCLIP and
StatA (lists of [relative_path, label, classname]).  ImageNet and its variants use the standard folder layout
with the official class-name list in `classnames.txt`.  Only the pieces needed for feature extraction are
implemented; preparing the data follows docs/experiment_plan.md Section 2.
"""
import json, os
from PIL import Image
from torch.utils.data import Dataset

TEMPLATES = {
    "default": ["a photo of a {}."],
    "imagenet": ["itap of a {}.", "a bad photo of the {}.", "a origami {}.", "a photo of the large {}.",
                 "a {} in a video game.", "art of the {}.", "a photo of the small {}."],
    "caltech101": ["a photo of a {}."], "dtd": ["{} texture."], "eurosat": ["a centered satellite photo of {}."],
    "fgvc": ["a photo of a {}, a type of aircraft."], "food101": ["a photo of {}, a type of food."],
    "oxford_flowers": ["a photo of a {}, a type of flower."], "oxford_pets": ["a photo of a {}, a type of pet."],
    "stanford_cars": ["a photo of a {}."], "sun397": ["a photo of a {}."], "ucf101": ["a photo of a person doing {}."],
}
for v in ("imagenet_v2", "imagenet_sketch", "imagenet_a", "imagenet_r"):
    TEMPLATES[v] = TEMPLATES["imagenet"]

IMAGE_DIRS = {"caltech101": "101_ObjectCategories", "dtd": "images", "eurosat": "2750", "fgvc": "images",
              "food101": "images", "oxford_flowers": "jpg", "oxford_pets": "images", "stanford_cars": "",
              "sun397": "SUN397", "ucf101": "UCF-101-midframes"}
SPLIT_FILES = {"caltech101": "split_zhou_Caltech101.json", "dtd": "split_zhou_DescribableTextures.json",
               "eurosat": "split_zhou_EuroSAT.json", "fgvc": None, "food101": "split_zhou_Food101.json",
               "oxford_flowers": "split_zhou_OxfordFlowers.json", "oxford_pets": "split_zhou_OxfordPets.json",
               "stanford_cars": "split_zhou_StanfordCars.json", "sun397": "split_zhou_SUN397.json",
               "ucf101": "split_zhou_UCF101.json"}
DATASET_DIRS = {"caltech101": "caltech-101", "dtd": "dtd", "eurosat": "eurosat", "fgvc": "fgvc_aircraft",
                "food101": "food-101", "oxford_flowers": "oxford_flowers", "oxford_pets": "oxford_pets",
                "stanford_cars": "stanford_cars", "sun397": "sun397", "ucf101": "ucf101"}


class ListDataset(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        return self.transform(Image.open(p).convert("RGB")), y


def build_dataset(name, root, split, transform):
    if name.startswith("imagenet"):
        from torchvision.datasets import ImageFolder
        sub = {"imagenet": "imagenet/images/val", "imagenet_v2": "imagenetv2/imagenetv2-matched-frequency-format-val",
               "imagenet_sketch": "imagenet-sketch/images", "imagenet_a": "imagenet-adversarial/imagenet-a",
               "imagenet_r": "imagenet-rendition/imagenet-r"}[name]
        ds = ImageFolder(os.path.join(root, sub), transform=transform)
        cn_file = os.path.join(root, "imagenet", "classnames.txt")
        classnames = [l.strip().split(" ", 1)[1] if " " in l.strip() else l.strip() for l in open(cn_file)]
        if name in ("imagenet_a", "imagenet_r"):   # 200-class subsets: keep the folder order's class names
            folders = ds.classes
            wnids = [l.strip().split(" ", 1)[0] for l in open(cn_file)]
            m = dict(zip(wnids, classnames)); classnames = [m[f] for f in folders]
        return ds, classnames
    d = os.path.join(root, DATASET_DIRS[name])
    if name == "fgvc":
        names = [l.strip() for l in open(os.path.join(d, "variants.txt"))]
        items = []
        for l in open(os.path.join(d, f"images_variant_{split}.txt")):
            img, var = l.strip().split(" ", 1)
            items.append((os.path.join(d, "images", img + ".jpg"), names.index(var)))
        return ListDataset(items, transform), names
    sp = json.load(open(os.path.join(d, SPLIT_FILES[name])))[split]
    items = [(os.path.join(d, IMAGE_DIRS[name], p), int(y)) for p, y, _ in sp]
    K = max(y for _, y, _ in sp) + 1
    classnames = [None] * K
    for _, y, c in sp:
        classnames[int(y)] = c
    return ListDataset(items, transform), classnames
