"""Tests of ``training.device_data``: device-resident batches for the ``carrasco2021`` schedule."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from astro_simclr.config import ConfigError
from astro_simclr.data.schema import DataError
from astro_simclr.training.device_data import (
    DeviceTensorDataModule,
    TensorSplit,
    interim_uint8_crops,
)
from astro_simclr.training.tiling import PermutationStream, tile_indices


def _split(labels: list[int], offset: float = 0.0) -> TensorSplit:
    n = len(labels)
    # Image i is filled with offset + i, so a batch's images identify its indices.
    images = np.broadcast_to(
        (offset + np.arange(n, dtype=np.float32))[:, None, None, None], (n, 3, 21, 21)
    ).copy()
    features = np.tile(np.arange(n, dtype=np.float32)[:, None], (1, 23))
    return TensorSplit(images=images, features=features, labels=np.array(labels, dtype=np.int64))


_TRAIN_LABELS = [0, 0, 0, 0, 0, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4]  # tiled: 5 classes x 5 = 25


def _splits() -> dict[str, TensorSplit]:
    return {
        "Train": _split(_TRAIN_LABELS),
        "Validation": _split([0, 1, 2, 3, 4] * 3, offset=100.0),
        "Test": _split([4, 3, 2, 1, 0], offset=200.0),
    }


def _ready(**kwargs: object) -> DeviceTensorDataModule:
    datamodule = DeviceTensorDataModule(_splits(), **kwargs)  # type: ignore[arg-type]
    datamodule.setup("fit")
    return datamodule


def test_tensor_split_validates() -> None:
    good = _split([0, 1])
    with pytest.raises(ValueError, match="images"):
        TensorSplit(good.images.astype(np.float64), good.features, good.labels)
    with pytest.raises(ValueError, match="features"):
        TensorSplit(good.images, good.features[:, :3], good.labels)
    with pytest.raises(ValueError, match="empty"):
        TensorSplit(good.images[:0], good.features[:0], good.labels[:0])
    with pytest.raises(ValueError, match="labels must be in"):
        TensorSplit(good.images, good.features, np.array([0, 5], dtype=np.int64))


def test_train_epoch_is_one_pass_of_the_tiled_stream() -> None:
    datamodule = _ready(batch_size=4, seed=3)
    tiled = tile_indices(_TRAIN_LABELS)
    assert datamodule.tiled_indices is not None
    assert np.array_equal(datamodule.tiled_indices, tiled)
    loader = datamodule.train_dataloader()
    assert len(loader) == 25 // 4
    stream = PermutationStream(25, 4, 3)
    batches = list(loader)
    assert len(batches) == 6
    labels = np.array(_TRAIN_LABELS)
    for b, (images, features, one_hot) in enumerate(batches):
        expected = tiled[stream.batch(b)]
        assert images.shape == (4, 3, 21, 21)
        assert images.dtype == torch.float32
        assert np.array_equal(images[:, 0, 0, 0].numpy(), expected.astype(np.float32))
        assert np.array_equal(features[:, 0].numpy(), expected.astype(np.float32))
        assert np.array_equal(one_hot.argmax(dim=1).numpy(), labels[expected])
        assert one_hot.shape == (4, 5)


def test_train_loader_starts_at_the_global_step() -> None:
    datamodule = _ready(batch_size=4, seed=3)
    loader = datamodule.train_dataloader()
    loader._position = lambda: 8  # mid-epoch 1 of 6-batch epochs: batches 8..11
    stream = PermutationStream(25, 4, 3)
    tiled = tile_indices(_TRAIN_LABELS)
    starts = [int(images[0, 0, 0, 0]) for images, _, _ in loader]
    assert starts == [int(tiled[stream.batch(b)][0]) for b in (8, 9, 10, 11)]


def test_unbalanced_streams_the_split_as_it_is() -> None:
    datamodule = _ready(batch_size=5, balance=False)
    assert datamodule.tiled_indices is not None
    assert datamodule.tiled_indices.tolist() == list(range(15))
    assert len(datamodule.train_dataloader()) == 3


def test_eval_batches_are_in_file_order() -> None:
    datamodule = _ready(batch_size=4, eval_batch_size=4)
    loader = datamodule.val_dataloader()
    assert len(loader) == 4
    batches = list(loader)
    assert [len(images) for images, _, _ in batches] == [4, 4, 4, 3]
    values = torch.cat([images[:, 0, 0, 0] for images, _, _ in batches])
    assert values.tolist() == [100.0 + i for i in range(15)]
    test = list(datamodule.test_dataloader())
    assert torch.cat([one_hot for _, _, one_hot in test]).argmax(dim=1).tolist() == [4, 3, 2, 1, 0]


def test_stream_state_guards_resumes() -> None:
    loader = _ready(batch_size=4, seed=3).train_dataloader()
    state = loader.state_dict()
    assert state == {"n": 25, "batch_size": 4, "seed": 3}
    loader.load_state_dict(state)
    with pytest.raises(DataError, match="same data, batch_size and seed"):
        loader.load_state_dict({**state, "seed": 4})


def test_split_identity_is_in_the_stream_state() -> None:
    fold2 = {"splits_sha256": "a" * 64, "mode": "fold", "fold": 2}
    loader = _ready(batch_size=4, seed=3, run_identity=fold2).train_dataloader()
    state = loader.state_dict()
    assert state == {"n": 25, "batch_size": 4, "seed": 3, "split_run": fold2}
    loader.load_state_dict(state)
    with pytest.raises(ConfigError, match=r"from fold 3 of .* but this run is fold 2"):
        loader.load_state_dict({**state, "split_run": {**fold2, "fold": 3}})
    with pytest.raises(ConfigError, match="from a normal run"):
        loader.load_state_dict({"n": 25, "batch_size": 4, "seed": 3})
    normal = _ready(batch_size=4, seed=3).train_dataloader()
    with pytest.raises(ConfigError, match="this run is a normal run"):
        normal.load_state_dict(state)
    with pytest.raises(DataError, match="same data"):  # the stream is still checked
        loader.load_state_dict({**state, "seed": 4})


def test_source_callable_is_resolved_once_per_device() -> None:
    calls = []

    def source() -> dict[str, TensorSplit]:
        calls.append(1)
        return _splits()

    datamodule = DeviceTensorDataModule(source)  # batch 64 > 25: fine outside fit
    datamodule.setup("validate")
    datamodule.setup("test")
    assert len(calls) == 1
    with pytest.raises(DataError, match="fewer than batch_size"):
        datamodule.setup("fit")


def test_errors() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        DeviceTensorDataModule(_splits(), batch_size=0)
    with pytest.raises(ValueError, match="seed"):
        DeviceTensorDataModule(_splits(), seed=-1)
    unprepared = DeviceTensorDataModule(_splits())
    with pytest.raises(DataError, match="setup"):
        unprepared.train_dataloader()
    with pytest.raises(DataError, match="setup"):
        unprepared.val_dataloader()
    missing = DeviceTensorDataModule({"Train": _split([0, 1])})
    with pytest.raises(DataError, match="Validation"):
        missing.setup("fit")
    too_small = DeviceTensorDataModule({**_splits(), "Train": _split([0, 1])}, batch_size=4)
    with pytest.raises(DataError, match="fewer than batch_size"):
        too_small.setup("fit")


def test_interim_crops_of_the_prepared_layout(classifier_dir: Path, pretraining_dir: Path) -> None:
    splits = interim_uint8_crops(classifier_dir)
    train = splits["Train"]
    assert train.images.shape[1:] == (3, 21, 21)
    assert train.images.dtype == np.float32
    assert train.images.min() >= 0.0
    assert train.images.max() <= 1.0
    assert train.features.shape[1] == 23
    with pytest.raises(DataError, match="unlabelled"):
        interim_uint8_crops(pretraining_dir)
