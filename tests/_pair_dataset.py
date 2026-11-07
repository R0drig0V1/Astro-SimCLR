"""A tiny picklable dataset for the DataLoader reproducibility tests.

It lives in its own module, not in test_augmentations.py, because spawned DataLoader workers
(the macOS default) import the module that defines the dataset: importing only
``astro_simclr.augmentations`` keeps worker start-up fast, where the test module would pull
in Lightning.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from astro_simclr.augmentations import get_augmentation


class PairDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Pairs of augmented views of 16 fixed random 63x63 stamps, cropped to 27 px."""

    def __init__(self, name: str) -> None:
        self.stamps = np.random.default_rng(1).integers(0, 256, (16, 63, 63, 3), dtype=np.uint8)
        self.aug = get_augmentation(name, 27)

    def __len__(self) -> int:
        return len(self.stamps)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.aug.pair(self.stamps[idx])
