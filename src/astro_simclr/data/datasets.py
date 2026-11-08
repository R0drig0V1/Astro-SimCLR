"""Torch datasets over a :class:`SplitData`, on the :class:`Augmentation` protocol.

Ports of the legacy ``utils/dataset.py`` classes:

- :class:`StampDataset` (legacy ``Dataset_stamps_v2``): one view per stamp, with its
  features and label, for the classifiers and the linear evaluation;
- :class:`ContrastiveDataset` (legacy ``Dataset_simclr``): a pair of views per stamp, for
  SimCLR pretraining.

The legacy classes unpickled the file in ``__init__`` and applied ``img_float2int`` per
item; here the split is already uint8 (loaded once, usually memory-mapped from the
prepared layout), and the transform is any :class:`Augmentation` (task 1.5 provides the
legacy ones, including ``without_aug``, the legacy centre crop). Randomness comes from the
augmentation (global generators, seeded by ``seed_everything``); the datasets draw none.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from astro_simclr.augmentations import Augmentation, Stamp
from astro_simclr.data.schema import DISCARDED_FEATURES, LABEL_NAMES, DataError, SplitData

__all__ = ["ContrastiveDataset", "StampDataset"]


def _stamp(images: np.ndarray, index: int) -> Stamp:
    # A copy, so that augmentations get a writable, contiguous array and never touch the
    # memory map (also safe across DataLoader worker processes).
    return np.array(images[index], dtype=np.uint8, copy=True)


class StampDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Labelled stamps: item ``i`` is ``(augmentation.view(stamp), features, label)``.

    - ``features``: float32 tensor of the 26 stored features minus ``discarded_features``
      (legacy default: columns 13-15, leaving 23);
    - ``label``: int64 one-hot tensor of length 5 if ``one_hot`` (the legacy classifier
      default), otherwise an int64 scalar tensor (as the legacy SimCLR validation used).

    Raises:
        DataError: if ``split`` is unlabelled (the pretraining ``Train`` split).
        IndexError: if a discarded feature index is out of range.
    """

    def __init__(
        self,
        split: SplitData,
        augmentation: Augmentation,
        *,
        one_hot: bool = True,
        discarded_features: Sequence[int] = DISCARDED_FEATURES,
    ) -> None:
        if split.labels is None or split.features is None:
            raise DataError("StampDataset needs a labelled split (images, labels, features)")
        self._images = split.images
        self._augmentation = augmentation
        kept = np.delete(np.asarray(split.features), list(discarded_features), axis=1)
        # float64 -> float32 in one step, as the legacy np.array(..., dtype=np.float32).
        self._features = torch.from_numpy(kept.astype(np.float32))
        # A copy: the split may be a read-only memory map, which torch cannot wrap.
        labels = torch.from_numpy(np.array(split.labels, dtype=np.int64, copy=True))
        self._labels = (
            torch.nn.functional.one_hot(labels, num_classes=len(LABEL_NAMES)) if one_hot else labels
        )

    def __len__(self) -> int:
        return int(self._images.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self._augmentation.view(_stamp(self._images, index))
        return image, self._features[index], self._labels[index]


class ContrastiveDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Stamps only: item ``i`` is ``augmentation.pair(stamp)``, two independent views.

    Labels and features are ignored, so any split works (the pretraining ``Train`` split
    has none).
    """

    def __init__(self, split: SplitData, augmentation: Augmentation) -> None:
        self._images = split.images
        self._augmentation = augmentation

    def __len__(self) -> int:
        return int(self._images.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._augmentation.pair(_stamp(self._images, index))
