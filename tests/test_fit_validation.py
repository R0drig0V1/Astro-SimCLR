"""The fit's validations without the wasted work give the same run (audit item 2).

``ClassifierModule`` no longer builds the confusion matrix nor copies the labels and
probabilities to the CPU in the validations inside ``fit`` (only ``trainer.validate`` /
``trainer.test`` keep them), its metrics skip torchmetrics' value checks
(``validate_args=False``), and the progress bar redraws every
``progress_bar_refresh_rate`` batches. None of it may change a number: a short
``carrasco2021`` run on the CPU with the old behaviour patched back in must write the same
``metrics.json`` (except the timing and the settings that differ), the same per-stamp
predictions and the same best weights.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from astro_simclr.config import Settings
from astro_simclr.experiments import run_command
from astro_simclr.training import metrics as metrics_module
from astro_simclr.training import modules

_SCHEDULE: dict[str, Any] = {
    "schedule": "carrasco2021",
    "supervised_model": "paper_stamp_classifier",
    "image_size": 21,
    "eval_batch_size": 64,
    "early_stopping": False,
    "monitor": "loss_val",
    "optimizer": "Adam",
    "lr": 1e-3,
    "beta_loss": 0.5,
    "batch_size": 4,
    "max_steps": 10,
    "val_every_n_steps": 3,
    "horizon_increment": 10,
    "checkpoint_every_n_steps": 3,
}


def _run(data: Path, out: Path, **changes: Any) -> dict[str, Any]:
    settings = Settings(
        accelerator="cpu",
        deterministic=True,
        data_dir=data.parent,
        dataset=data.name,
        output_dir=out,
        **{**_SCHEDULE, **changes},
    )
    run_command("train-supervised", settings, progress_bar=True)
    result: dict[str, Any] = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    return result


def _comparable(result: dict[str, Any]) -> dict[str, Any]:
    """``metrics.json`` without what must differ: timing, output directory, refresh rate."""
    result = {key: value for key, value in result.items() if key != "timing"}
    config = dict(result["config"])
    for key in ("output_dir", "progress_bar_refresh_rate"):
        config.pop(key)
    result["config"] = config
    return result


def test_fit_without_wasted_validation_work_gives_the_same_run(
    classifier_f32_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"confusion_matrix": 0}

    def counted(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls["confusion_matrix"] += 1
        return metrics_module.confusion_matrix(*args, **kwargs)

    monkeypatch.setattr(modules, "confusion_matrix", counted)

    # Before: every validation keeps its outputs (no stage is ever "fitting"), torchmetrics
    # checks the values, and the bar redraws every batch (Lightning's default).
    with monkeypatch.context() as old:
        old.setattr(modules, "TrainerFn", types.SimpleNamespace(FITTING=object()))

        def checked(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs: Any) -> Any:
            return metrics_module.classification_metrics(
                y_pred, y_true, **{**kwargs, "validate_args": True}
            )

        old.setattr(modules, "classification_metrics", checked)
        before = _run(classifier_f32_dir, tmp_path / "before", progress_bar_refresh_rate=1)
    before_calls, calls["confusion_matrix"] = calls["confusion_matrix"], 0

    after = _run(classifier_f32_dir, tmp_path / "after")

    # The old path built a matrix at each of the fit's validations; the new one only at the
    # final validate and test.
    assert before_calls > 2
    assert calls["confusion_matrix"] == 2
    assert after["config"]["progress_bar_refresh_rate"] == 50
    assert _comparable(after) == _comparable(before)
    for stage in ("val", "test"):
        old_file = np.load(tmp_path / "before" / f"predictions_{stage}.npz")
        new_file = np.load(tmp_path / "after" / f"predictions_{stage}.npz")
        assert old_file.files == new_file.files
        for name in old_file.files:
            np.testing.assert_array_equal(new_file[name], old_file[name])
    old_best = torch.load(tmp_path / "before" / "checkpoints" / "best.ckpt", weights_only=True)
    new_best = torch.load(tmp_path / "after" / "checkpoints" / "best.ckpt", weights_only=True)
    assert old_best["global_step"] == new_best["global_step"]
    for name, tensor in old_best["state_dict"].items():
        assert torch.equal(new_best["state_dict"][name], tensor), name
