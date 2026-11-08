"""Tests of StampDataset and ContrastiveDataset (astro_simclr.data.datasets)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from astro_simclr.augmentations import AUGMENTATIONS, Augmentation, Stamp, get_augmentation
from astro_simclr.data import DataError, SplitData, load_prepared, prepare
from astro_simclr.data.datasets import ContrastiveDataset, StampDataset
from astro_simclr.data.synthetic import CLASSIFIER_FILES, PRETRAINING_FILE, make_synthetic


class _CentreCrop:
    """Deterministic stand-in for the legacy ``without_aug`` (centre crop + ToTensor)."""

    def __init__(self, size: int = 27) -> None:
        self.size = size
        self.seen: list[Stamp] = []

    def view(self, img: Stamp) -> torch.Tensor:
        self.seen.append(img)
        start = (img.shape[0] - self.size) // 2
        crop = img[start : start + self.size, start : start + self.size]
        return torch.from_numpy(crop).permute(2, 0, 1).float() / 255

    def pair(self, img: Stamp) -> tuple[torch.Tensor, torch.Tensor]:
        return self.view(img), self.view(img)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("datasets")
    make_synthetic(root / "raw", n_per_split=10, seed=2)
    prepare(root / "raw" / CLASSIFIER_FILES[0], root)
    prepare(root / "raw" / PRETRAINING_FILE, root)
    return root


def _split(prepared: Path, name: str, split: str) -> SplitData:
    return load_prepared(prepared / Path(name).stem).splits[split]


def test_stamp_dataset_item_matches_the_legacy_contract(prepared: Path) -> None:
    split = _split(prepared, CLASSIFIER_FILES[0], "Train")
    aug = _CentreCrop()
    assert isinstance(aug, Augmentation)
    dataset = StampDataset(split, aug)
    assert len(dataset) == 10
    image, features, label = dataset[3]
    assert image.shape == (3, 27, 27) and image.dtype == torch.float32
    # Legacy: np.array(features, dtype=np.float32) with columns 13-15 deleted.
    assert split.features is not None and split.labels is not None
    expected = np.delete(split.features[3].astype(np.float32), [13, 14, 15])
    assert features.dtype == torch.float32 and features.shape == (23,)
    np.testing.assert_array_equal(features.numpy(), expected)
    # Legacy: one-hot int64 labels for the classifiers.
    assert label.dtype == torch.int64 and label.shape == (5,)
    assert int(label.argmax()) == int(split.labels[3]) and int(label.sum()) == 1


def test_augmentation_gets_a_writable_copy_of_the_stamp(prepared: Path) -> None:
    split = _split(prepared, CLASSIFIER_FILES[0], "Train")
    aug = _CentreCrop()
    StampDataset(split, aug)[0]
    stamp = aug.seen[0]
    assert stamp.dtype == np.uint8 and stamp.shape == (63, 63, 3)
    assert stamp.flags.writeable and not isinstance(stamp, np.memmap)
    np.testing.assert_array_equal(stamp, split.images[0])


def test_integer_labels_and_other_discarded_features(prepared: Path) -> None:
    split = _split(prepared, CLASSIFIER_FILES[0], "Validation")
    dataset = StampDataset(split, _CentreCrop(), one_hot=False, discarded_features=())
    _, features, label = dataset[0]
    assert features.shape == (26,)
    assert label.shape == () and label.dtype == torch.int64
    assert split.labels is not None and int(label) == int(split.labels[0])


def test_stamp_dataset_needs_labels(prepared: Path) -> None:
    with pytest.raises(DataError, match=r"labelled split"):
        StampDataset(_split(prepared, PRETRAINING_FILE, "Train"), _CentreCrop())


def test_contrastive_dataset_returns_pairs(prepared: Path) -> None:
    split = _split(prepared, PRETRAINING_FILE, "Train")
    dataset = ContrastiveDataset(split, _CentreCrop(size=21))
    assert len(dataset) == 10
    first, second = dataset[9]
    assert first.shape == second.shape == (3, 21, 21)


def test_datasets_batch_through_a_dataloader(prepared: Path) -> None:
    stamps = StampDataset(_split(prepared, CLASSIFIER_FILES[0], "Test"), _CentreCrop())
    images, features, labels = next(iter(DataLoader(stamps, batch_size=4)))
    assert images.shape == (4, 3, 27, 27)
    assert features.shape == (4, 23) and labels.shape == (4, 5)
    pairs = ContrastiveDataset(_split(prepared, PRETRAINING_FILE, "Train"), _CentreCrop())
    x_i, x_j = next(iter(DataLoader(pairs, batch_size=5, drop_last=True)))
    assert x_i.shape == x_j.shape == (5, 3, 27, 27)


# The real registry of task 1.5: the datasets work with any of its augmentations.


def test_stamp_dataset_with_without_aug_is_the_legacy_resize_img(prepared: Path) -> None:
    # Legacy Dataset_stamps_v2 without augmentation: img_float2int, ToPILImage, then
    # Resize_img (CenterCrop + ToTensor). The stamps are already uint8 here.
    split = _split(prepared, CLASSIFIER_FILES[0], "Validation")
    dataset = StampDataset(split, get_augmentation("without_aug", 27))
    image, _, _ = dataset[4]
    expected = torch.from_numpy(np.array(split.images[4, 18:45, 18:45])).permute(2, 0, 1)
    torch.testing.assert_close(image, expected.float() / 255, rtol=0, atol=1e-6)


@pytest.mark.parametrize("name", sorted(AUGMENTATIONS))
def test_contrastive_dataset_with_every_registered_augmentation(name: str, prepared: Path) -> None:
    torch.manual_seed(0)
    dataset = ContrastiveDataset(
        _split(prepared, PRETRAINING_FILE, "Train"), get_augmentation(name, 27)
    )
    x_i, x_j = next(iter(DataLoader(dataset, batch_size=4, drop_last=True)))
    assert x_i.shape == x_j.shape == (4, 3, 27, 27)
    assert x_i.dtype == torch.float32
