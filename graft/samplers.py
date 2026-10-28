"""Evaluation samplers reproducing the realistic protocols of StatA (CVPR 2025), re-implemented from the
protocol description (no code copied).

BatchSampler  : each task draws K_eff ~ U{kmin..kmax} classes, then `batch_size` samples from their union
                (all of them if batch_size == -1 or the union is smaller).
OnlineSampler : a stream in which every class's samples are split over `slots` slots with Dirichlet(gamma)
                proportions; slots are concatenated (gamma small -> highly correlated stream).  gamma == -1
                gives the "Separate" stream: classes appear one after another in random order.
"""
from collections import defaultdict
from typing import List, Optional
import numpy as np
import torch


class BatchSampler:
    def __init__(self, labels: torch.Tensor, batch_size: int, kmin: Optional[int] = None, kmax: Optional[int] = None,
                 seed: int = 0):
        self.labels = labels.cpu().numpy()
        self.batch_size = batch_size
        self.by_class = defaultdict(list)
        for i, y in enumerate(self.labels):
            self.by_class[int(y)].append(i)
        self.classes = sorted(self.by_class)
        self.kmin, self.kmax = kmin, kmax
        self.rng = np.random.default_rng(seed)

    def sample(self) -> List[int]:
        K = len(self.classes)
        if self.kmin is None:
            chosen = self.classes
        else:
            k = int(self.rng.integers(self.kmin, self.kmax + 1))
            k = min(k, K)
            chosen = self.rng.choice(self.classes, size=k, replace=False).tolist()
        pool = np.concatenate([self.by_class[c] for c in chosen])
        if self.batch_size == -1 or len(pool) <= self.batch_size:
            idx = pool
        else:
            idx = self.rng.choice(pool, size=self.batch_size, replace=False)
        return self.rng.permutation(idx).tolist()


class OnlineSampler:
    def __init__(self, labels: torch.Tensor, batch_size: int, gamma: float, slots: Optional[int] = None, seed: int = 0):
        self.labels = labels.cpu().numpy()
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        by_class = defaultdict(list)
        for i, y in enumerate(self.labels):
            by_class[int(y)].append(i)
        classes = sorted(by_class)
        if gamma == -1:
            order = []
            for c in self.rng.permutation(classes):
                order.extend(self.rng.permutation(by_class[int(c)]).tolist())
        else:
            S = slots or len(classes)
            slot_lists = [[] for _ in range(S)]
            for c in classes:
                idx = self.rng.permutation(by_class[c])
                props = self.rng.dirichlet([gamma] * S)
                cuts = (np.cumsum(props)[:-1] * len(idx)).astype(int)
                for s, part in enumerate(np.split(idx, cuts)):
                    slot_lists[s].extend(part.tolist())
            order = []
            for sl in slot_lists:
                order.extend(self.rng.permutation(sl).tolist())
        self.order = order
        self.pos = 0

    def __iter__(self):
        return self

    def __next__(self) -> List[int]:
        if self.pos >= len(self.order):
            raise StopIteration
        b = self.order[self.pos:self.pos + self.batch_size]
        self.pos += self.batch_size
        return b
