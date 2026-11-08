"""Tests of the augmentation prefetch (audit item 5; ADR 0006 and 0007, "Prefetch").

The training loader plans each step's augmentation on the host
(``PaperClassifierModule.augment_plan``) and uploads the plan with the batch; the step only
applies it (``apply_plan``). The contract: bit for bit the in-step path, on every device.

1. ``upload_plan`` moves nested plans.
2. The loader's planned batches equal the in-step augmentation.
3. A short augmented run gives the same ``metrics.json`` (timing aside) and the same
   predictions with the prefetch as without it, on every available accelerator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from _training_support import ACCELERATORS

from astro_simclr import experiments
from astro_simclr.config import settings_from_mapping
from astro_simclr.experiments import paper_module, run_command
from astro_simclr.presets import CARRASCO2021_PRESET, PRESETS
from astro_simclr.protocol.predictions import load_predictions
from astro_simclr.training.class_augment import stamp_fits, upload_plan
from astro_simclr.training.device_data import DeviceTensorDataModule, TensorSplit
from astro_simclr.training.flux_rescale import fit_central_psf

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


def test_the_loader_plans_each_step_exactly_as_the_step_would() -> None:
    torch.manual_seed(0)
    module = paper_module(settings_from_mapping({**PRESETS[CARRASCO2021_PRESET], **_ALL}))
    assert module.augmenting
    splits = _splits()
    planned = DeviceTensorDataModule(splits, batch_size=8, planner=module.augment_plan)
    planned.setup("fit")
    plain = DeviceTensorDataModule(splits, batch_size=8)
    plain.setup("fit")
    loader = planned.train_dataloader()
    batches = list(loader)
    assert len(batches) == len(plain.train_dataloader())
    for step, (with_plan, without) in enumerate(
        zip(batches, plain.train_dataloader(), strict=True)
    ):
        assert isinstance(with_plan[-1], dict)
        rows = with_plan[:-1]
        for got, want in zip(rows, without, strict=True):
            assert torch.equal(got, want)
        applied = module.apply_plan(rows, with_plan[-1])
        host = tuple(t.cpu() for t in without)
        expected = module.apply_plan(
            without, upload_plan(module.augment_plan(host, step), torch.device("cpu"))
        )
        for got, want in zip(applied, expected, strict=True):
            assert torch.equal(got, want)
    # Step 0 is the in-step path's (no Trainer: global step 0), bit for bit.
    in_step = module.augmented_batch(tuple(batches[0][:-1]))
    for got, want in zip(module.apply_plan(batches[0][:-1], batches[0][-1]), in_step, strict=True):
        assert torch.equal(got, want)
    # And the step accepts the planned batch.
    module.eval()
    loss = module.training_step(batches[0], 0)
    torch.testing.assert_close(loss, module.compute_loss(module(*in_step[:2]), in_step[2]))


def test_a_run_without_augmentation_gets_no_planner() -> None:
    module = paper_module(settings_from_mapping(dict(PRESETS[CARRASCO2021_PRESET])))
    assert not module.augmenting


def _comparable(metrics: dict[str, Any], out: Path) -> dict[str, Any]:
    """``metrics`` without the timing and with the output directory replaced."""
    text = json.dumps({k: v for k, v in metrics.items() if k != "timing"}, sort_keys=True)
    return dict(json.loads(text.replace(str(out), "OUT")))


@pytest.mark.parametrize("accelerator", ACCELERATORS)
def test_prefetch_gives_the_same_run_as_the_in_step_path(
    accelerator: str, classifier_f32_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = {
        **PRESETS[CARRASCO2021_PRESET],
        **_ALL,
        "accelerator": accelerator,
        "deterministic": True,
        "data_dir": classifier_f32_dir.parent,
        "dataset": classifier_f32_dir.name,
        "batch_size": 8,
        "max_steps": 6,
        "val_every_n_steps": 2,
        "horizon_increment": 0,
        "checkpoint_every_n_steps": 2,
    }
    runs = {}
    for prefetch in (True, False):
        monkeypatch.setattr(experiments, "_PREFETCH", prefetch)
        out = tmp_path / str(prefetch)
        config = settings_from_mapping({**settings, "output_dir": out})
        metrics = run_command("train-supervised", config, progress_bar=False)
        on_disk = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        assert on_disk["timing"]
        runs[prefetch] = (_comparable(on_disk, out), out, metrics)
    assert runs[True][0] == runs[False][0]
    for split in ("val", "test"):
        name = runs[True][2]["predictions"][split]["file"]
        first = load_predictions(runs[True][1] / name)
        second = load_predictions(runs[False][1] / name)
        np.testing.assert_array_equal(first.probabilities, second.probabilities)
