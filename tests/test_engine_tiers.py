"""Tests of ``engine.tiers``: the three data tiers serve the same batches, and resume exactly.

The acceptance of task 4.3: the ``pinned`` tier (and the ``memmap`` one) give batches
``torch.equal`` to the ``device`` tier for the same seed and stream, on every accelerator
present; a resumed stream continues with exactly the batches of an uninterrupted one (on
any tier); the ``device`` tier gives today's ``DeviceTensorDataModule`` batches.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import torch
from _training_support import ACCELERATORS

from astro_simclr.config import ConfigError
from astro_simclr.data.schema import DataError
from astro_simclr.engine.data import BatchSource, DataTier, EvalSource, StorageDtype, TierChoice
from astro_simclr.engine.dataset import Dataset
from astro_simclr.engine.stream import tile_indices
from astro_simclr.engine.tiers import (
    StreamSpec,
    TrainSource,
    check_stream,
    open_eval,
    open_train,
)
from astro_simclr.training.device_data import DeviceTensorDataModule, TensorSplit

_N = 257  # not a multiple of the batch: the stream crosses pass boundaries mid-batch
_STEPS = 40  # > 4 passes of the 211-row selection at batch 16 (13 batches per pass)
_HOST_TIERS = (DataTier.PINNED, DataTier.MEMMAP)


def _images(n: int = _N, seed: int = 0) -> npt.NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    return rng.random((n, 3, 21, 21), dtype=np.float32)


def _dataset(
    tier: DataTier, tmp_path: Path, *, fits: bool = True, selection: bool = True
) -> Dataset:
    """A labelled dataset whose images are a file for the memmap tier, an array otherwise."""
    rng = np.random.default_rng(1)
    images = _images()
    if tier is DataTier.MEMMAP:
        path = tmp_path / "images.npy"
        if not path.exists():
            np.save(path, images)
        images = np.load(path, mmap_mode="r")
    dataset = Dataset(
        images=images,
        features=rng.random((_N, 23), dtype=np.float32),
        labels=(rng.permutation(_N) % 5).astype(np.int64),
        fits=rng.random((_N, 6), dtype=np.float32) if fits else None,
        n_classes=5,
    )
    # A split of the pool, in a scrambled order, as a fold or a candid split is.
    return dataset.select(np.random.default_rng(2).permutation(_N)[:211]) if selection else dataset


def _cpu(batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.cpu() for tensor in batch)


def _assert_same(
    left: list[tuple[torch.Tensor, ...]], right: list[tuple[torch.Tensor, ...]]
) -> None:
    assert len(left) == len(right)
    for a, b in zip(left, right, strict=True):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert x.dtype == y.dtype
            assert torch.equal(x, y)


def _train_batches(source: TrainSource, steps: range) -> list[tuple[torch.Tensor, ...]]:
    # Every batch is kept while later ones are fetched: a staging buffer reused too early
    # would change a kept batch.
    return [_cpu(source.batch(step)) for step in steps]


@pytest.mark.parametrize("accelerator", ACCELERATORS)
@pytest.mark.parametrize("tier", _HOST_TIERS)
@pytest.mark.parametrize("balance", [True, False])
def test_host_tiers_equal_the_device_tier(
    accelerator: str, tier: DataTier, balance: bool, tmp_path: Path
) -> None:
    spec = StreamSpec(batch_size=16, seed=7, balance=balance)
    reference = open_train(_dataset(DataTier.DEVICE, tmp_path), TierChoice(), spec, accelerator)
    source = open_train(_dataset(tier, tmp_path), TierChoice(tier), spec, accelerator)
    assert source.state() == reference.state()
    expected = _train_batches(reference, range(_STEPS))
    assert len(expected[0]) == 4  # images, features, one-hot labels, fits
    assert expected[0][2].dtype == torch.int64
    _assert_same(_train_batches(source, range(_STEPS)), expected)
    assert source.position == _STEPS
    if accelerator != "cpu":
        assert source.describe()["pinned_staging"] is True
        assert source.describe()["prefetch"] == 1


@pytest.mark.parametrize("accelerator", ACCELERATORS)
@pytest.mark.parametrize("tier", _HOST_TIERS)
def test_host_tiers_equal_the_device_tier_in_evaluation(
    accelerator: str, tier: DataTier, tmp_path: Path
) -> None:
    reference = open_eval(_dataset(DataTier.DEVICE, tmp_path), TierChoice(), accelerator)
    batches = open_eval(_dataset(tier, tmp_path), TierChoice(tier), accelerator, batch_size=64)
    assert len(batches) == len(reference) == 4  # 211 rows: 64, 64, 64, 19
    expected = [_cpu(batch) for batch in reference]
    assert expected[-1][0].shape[0] == 19
    _assert_same([_cpu(batch) for batch in batches], expected)
    _assert_same([_cpu(batch) for batch in batches], expected)  # a second pass is the same


@pytest.mark.parametrize("accelerator", ACCELERATORS)
def test_block_shuffle_is_served_identically_by_every_tier(
    accelerator: str, tmp_path: Path
) -> None:
    spec = StreamSpec(batch_size=16, seed=3, balance=False, shuffle="block", block_rows=8)
    per_tier = [
        _train_batches(
            open_train(_dataset(tier, tmp_path, fits=False), TierChoice(tier), spec, accelerator),
            range(_STEPS),
        )
        for tier in DataTier
    ]
    _assert_same(per_tier[1], per_tier[0])
    _assert_same(per_tier[2], per_tier[0])


def test_device_tier_equals_the_lightning_device_data_module() -> None:
    rng = np.random.default_rng(4)
    n = 90
    split = TensorSplit(
        images=_images(n),
        features=rng.random((n, 23), dtype=np.float32),
        labels=(rng.permutation(n) % 5).astype(np.int64),
        fits=rng.random((n, 6), dtype=np.float32),
    )
    module = DeviceTensorDataModule(
        {"Train": split, "Validation": split, "Test": split}, batch_size=8, seed=11
    )
    module.setup("fit")
    loader = module.train_dataloader()
    expected = [_cpu(loader._split.gather(loader.indices(step))) for step in range(40)]
    dataset = Dataset(
        images=split.images,
        features=split.features,
        labels=split.labels,
        fits=split.fits,
        n_classes=split.n_classes,
    )
    source = open_train(dataset, TierChoice(), StreamSpec(batch_size=8, seed=11), "cpu")
    assert source.batches_per_pass == len(loader)
    assert {k: v for k, v in source.state().items() if k in loader.state_dict()} == (
        loader.state_dict()
    )
    _assert_same(_train_batches(source, range(40)), expected)
    reference = [_cpu(batch) for batch in module.val_dataloader()]
    _assert_same([_cpu(batch) for batch in open_eval(dataset, TierChoice(), "cpu")], reference)


@pytest.mark.parametrize("accelerator", ACCELERATORS)
@pytest.mark.parametrize("tier", list(DataTier))
def test_resume_is_exact(accelerator: str, tier: DataTier, tmp_path: Path) -> None:
    spec = StreamSpec(batch_size=16, seed=5)
    dataset = _dataset(tier, tmp_path)
    uninterrupted = _train_batches(
        open_train(dataset, TierChoice(tier), spec, accelerator), range(_STEPS)
    )
    first = open_train(dataset, TierChoice(tier), spec, accelerator)
    _train_batches(first, range(17))  # stopped mid-pass, with step 17 prefetched
    saved, steps_done = first.state(), first.position
    # A fresh process: a new source, resumed from the checkpoint's stream and position,
    # on another tier too (the tiers serve the same values).
    other_tier = DataTier.PINNED if tier is DataTier.DEVICE else DataTier.DEVICE
    for resume_tier in (tier, other_tier):
        resumed = open_train(
            _dataset(resume_tier, tmp_path), TierChoice(resume_tier), spec, accelerator
        )
        resumed.resume(saved, steps_done)
        assert resumed.position == 17
        _assert_same(_train_batches(resumed, range(17, _STEPS)), uninterrupted[17:])


def test_out_of_order_steps_are_still_exact(tmp_path: Path) -> None:
    spec = StreamSpec(batch_size=16, seed=5)
    dataset = _dataset(DataTier.PINNED, tmp_path)
    expected = _train_batches(open_train(dataset, TierChoice(), spec, "cpu"), range(_STEPS))
    source = open_train(dataset, TierChoice(DataTier.PINNED), spec, "cpu")
    for step in (3, 4, 30, 2, 2, 39):
        _assert_same([_cpu(source.batch(step))], [expected[step]])


def test_resume_refuses_another_stream(tmp_path: Path) -> None:
    dataset = _dataset(DataTier.DEVICE, tmp_path)
    source = open_train(dataset, TierChoice(), StreamSpec(batch_size=16, seed=5), "cpu")
    for spec, choice in (
        (StreamSpec(batch_size=16, seed=6), TierChoice()),
        (StreamSpec(batch_size=8, seed=5), TierChoice()),
        (StreamSpec(batch_size=16, seed=5, balance=False), TierChoice()),
        (StreamSpec(batch_size=16, seed=5), TierChoice(DataTier.DEVICE, "float16")),
    ):
        other = open_train(dataset, choice, spec, "cpu")
        with pytest.raises(DataError, match="stream"):
            other.resume(source.state(), 10)
    with pytest.raises(ValueError, match="steps_done"):
        source.resume(source.state(), -1)
    check_stream(source.state(), dict(source.state()))


def test_sources_satisfy_the_frozen_protocols(tmp_path: Path) -> None:
    dataset = _dataset(DataTier.DEVICE, tmp_path)
    train: BatchSource = open_train(dataset, TierChoice(), StreamSpec(batch_size=16), "cpu")
    evaluation: EvalSource = open_eval(dataset, TierChoice(), "cpu")
    labels = dataset.labels_of_rows()
    assert labels is not None
    assert train.batches_per_pass == len(tile_indices(labels)) // 16
    assert "rows_sha256" in train.state()
    assert len(evaluation) == 4


@pytest.mark.parametrize("accelerator", ACCELERATORS)
@pytest.mark.parametrize("dtype", ["float16", "uint8"])
def test_quantised_storage_is_a_recorded_deviation(
    accelerator: str, dtype: StorageDtype, tmp_path: Path
) -> None:
    dataset = _dataset(DataTier.DEVICE, tmp_path)
    spec = StreamSpec(batch_size=16, seed=1)
    device = open_train(dataset, TierChoice(DataTier.DEVICE, dtype), spec, accelerator)
    pinned = open_train(dataset, TierChoice(DataTier.PINNED, dtype), spec, accelerator)
    exact = open_train(dataset, TierChoice(), spec, accelerator)
    assert device.describe()["deviation"] is True
    assert device.describe()["storage_dtype"] == dtype
    assert device.state()["storage_dtype"] == dtype
    assert exact.describe()["deviation"] is False
    quantised = _train_batches(device, range(10))
    _assert_same(_train_batches(pinned, range(10)), quantised)
    for got, want in zip(quantised, _train_batches(exact, range(10)), strict=True):
        assert got[0].dtype == torch.float32
        values = want[0].numpy()
        if dtype == "float16":
            dequantised = values.astype(np.float16).astype(np.float32)
        else:
            dequantised = np.rint(values * np.float32(255)).astype(np.uint8)
            dequantised = torch.from_numpy(dequantised).float().div(255.0).numpy()
        assert np.array_equal(got[0].numpy(), dequantised)
        assert torch.equal(got[1], want[1])  # only the images are quantised


def test_quantised_storage_refusals(tmp_path: Path) -> None:
    dataset = _dataset(DataTier.MEMMAP, tmp_path)
    with pytest.raises(ConfigError, match="memmap tier"):
        open_train(dataset, TierChoice(DataTier.MEMMAP, "uint8"), StreamSpec(), "cpu")
    with pytest.raises(ConfigError, match="memmap tier"):
        open_eval(dataset, TierChoice(DataTier.MEMMAP, "float16"), "cpu")
    raw = Dataset(images=_images() * 3)  # not in [0, 1]
    with pytest.raises(DataError, match="uint8"):
        open_train(raw, TierChoice(DataTier.PINNED, "uint8"), StreamSpec(balance=False), "cpu")


@pytest.mark.parametrize("tier", list(DataTier))
def test_unlabelled_datasets_serve_columns(tier: DataTier, tmp_path: Path) -> None:
    images = _dataset(tier, tmp_path, selection=False).images
    dataset = Dataset(images=images, kind="unlabelled")
    spec = StreamSpec(batch_size=16, seed=0, balance=False)
    source = open_train(dataset, TierChoice(tier), spec, "cpu")
    columns = source.columns(0)
    assert set(columns) == {"images"}
    assert torch.equal(
        columns["images"], torch.from_numpy(np.asarray(images)[source.order.rows(0)])
    )
    with pytest.raises(ConfigError, match="unlabelled"):
        source.batch(1)
    batches = open_eval(dataset, TierChoice(tier), "cpu", batch_size=100)
    passes = list(batches.iter_columns())
    assert [len(c["images"]) for c in passes] == [100, 100, 57]
    with pytest.raises(ConfigError, match="unlabelled"):
        next(iter(batches))


def test_spec_and_arguments_validate(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        StreamSpec(batch_size=0)
    with pytest.raises(ValueError, match="block_rows"):
        StreamSpec(block_rows=True)
    with pytest.raises(ValueError, match="seed"):
        StreamSpec(seed=-1)
    with pytest.raises(ValueError, match="shuffle"):
        StreamSpec(shuffle="pixel")  # type: ignore[arg-type]
    dataset = _dataset(DataTier.DEVICE, tmp_path)
    with pytest.raises(ValueError, match="batch_size"):
        open_eval(dataset, TierChoice(), "cpu", batch_size=0)
    with pytest.raises(ValueError, match="needs the labels"):
        open_train(Dataset(images=_images()), TierChoice(), StreamSpec(balance=True), "cpu")
    source = open_train(dataset, TierChoice(), StreamSpec(), "cpu")
    with pytest.raises(ValueError, match="step"):
        source.batch(-1)
