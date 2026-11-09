"""Tests of the host-planned augmentation (audit item 5; ADR 0006 and 0007, "Prefetch").

The plain engine's paper task (``tasks.paper.PaperTask``) plans each step's augmentation on
a host copy of the batch (``training.paper_augment.PaperAugment.plan``), uploads the plan
(``class_augment.upload_plan``) and applies it on the batch's device
(``PaperAugment.apply``). The contract: every draw is a function of the run's seed and the
step, so the same step gets the same augmentation wherever it runs.

1. ``upload_plan`` moves nested plans.
2. ``PaperTask.train_step`` trains on exactly the planned augmentation of its step, with
   every flag on, over several steps (and the plan changes with the step).
3. A run without augmentation plans nothing.

The bit-for-bit runs of every augmentation flag are ``test_engine_regression.py``'s goldens.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from astro_simclr.config import settings_from_mapping
from astro_simclr.presets import CARRASCO2021_PRESET, PRESETS
from astro_simclr.tasks.paper import build_paper_task, paper_augment
from astro_simclr.training.class_augment import stamp_fits, upload_plan
from astro_simclr.training.flux_rescale import fit_central_psf
from astro_simclr.training.paper_augment import AugmentBatch
from astro_simclr.training.tensor_splits import TensorSplit

#: Every training augmentation at once (R^2 0 so that the random synthetic stamps qualify).
_ALL: dict[str, Any] = {
    "paper_arch": "single",
    "rot90_augment": True,
    "aug_noise": 1.0,
    "aug_feature_jitter": 0.1,
    "synthetic_bogus": 0.25,
    "aug_min_r2": 0.0,
    "flux_rescale": 0.5,
}


def test_upload_plan_moves_nested_tensors() -> None:
    plan: dict[str, Any] = {
        "c5": {"keep": torch.ones(2, dtype=torch.bool)},
        "rot": torch.arange(2),
        "n": 3,
    }
    moved = upload_plan(plan, torch.device("cpu"))
    assert torch.equal(moved["c5"]["keep"], plan["c5"]["keep"])
    assert torch.equal(moved["rot"], plan["rot"])
    assert moved["n"] == 3
    assert moved is not plan


def _splits(n: int = 40, seed: int = 0) -> dict[str, TensorSplit]:
    """Train (with the 11-column fits), Validation and Test of random crops with a source."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:21, :21] - 10
    blob = np.exp(-(xx**2 + yy**2) / (2 * 1.1**2))

    def split(count: int, fits: bool) -> TensorSplit:
        images = rng.random((count, 3, 21, 21)).astype(np.float32)
        images[:, 0] += 3 * blob.astype(np.float32)
        images[:, 2] = images[:, 0] - 0.5 * images[:, 1]
        images -= images.min(axis=(2, 3), keepdims=True)
        images /= images.max(axis=(2, 3), keepdims=True)
        features = rng.normal(size=(count, 23)).astype(np.float32)
        labels = np.arange(count, dtype=np.int64) % 5
        table = None
        if fits:
            table = np.concatenate([stamp_fits(images), fit_central_psf(images)], axis=1)
        return TensorSplit(images, features, labels, fits=table)

    return {"Train": split(n, True), "Validation": split(8, False), "Test": split(8, False)}


def _train_batch(split: TensorSplit, rows: slice) -> AugmentBatch:
    """Rows ``rows`` of ``split`` as a training batch: crops, features, one-hot labels, fits."""
    assert split.fits is not None
    labels = torch.from_numpy(split.labels[rows])
    return (
        torch.from_numpy(split.images[rows]),
        torch.from_numpy(split.features[rows]),
        torch.nn.functional.one_hot(labels, 5).float(),
        torch.from_numpy(split.fits[rows]),
    )


def test_the_task_step_trains_on_its_planned_augmentation() -> None:
    torch.manual_seed(0)
    task = build_paper_task(settings_from_mapping({**PRESETS[CARRASCO2021_PRESET], **_ALL}), {})
    augment = task.augment
    assert augment.active and augment.needs_fits
    task.net.eval()  # no dropout: the loss is a function of the batch alone
    train = _splits()["Train"]
    previous = None
    for step in range(6):
        batch = _train_batch(train, slice(8 * (step % 5), 8 * (step % 5) + 8))
        host = (batch[0].clone(), batch[1].clone(), batch[2].clone(), batch[3].clone())
        plan = upload_plan(augment.plan(host, step), torch.device("cpu"))
        images, features, targets = augment.apply(batch, plan)
        expected = task.net.criterion(task.net(images, features), targets)
        loss = task.train_step(batch, step)
        assert torch.equal(loss, expected)
        # The batch is not modified, and the plan is a function of (seed, step) alone.
        assert all(torch.equal(a, b) for a, b in zip(batch, host, strict=True))
        again = augment.apply(batch, upload_plan(augment.plan(host, step), torch.device("cpu")))
        assert all(
            torch.equal(a, b) for a, b in zip(again, (images, features, targets), strict=True)
        )
        if step == 5:  # the rows of step 0, at another step: another augmentation
            assert previous is not None and not torch.equal(images, previous)
        if step == 0:
            previous = images


def test_a_run_without_augmentation_plans_nothing() -> None:
    augment = paper_augment(settings_from_mapping(dict(PRESETS[CARRASCO2021_PRESET])))
    assert not augment.active
    assert not augment.needs_fits
    batch = _train_batch(_splits()["Train"], slice(0, 8))
    assert augment.plan(batch, 3) == {}
