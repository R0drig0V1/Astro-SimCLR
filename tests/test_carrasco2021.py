"""The ``carrasco2021`` schedule through the plain runner, ``tasks.plain.run_plain`` (task 2.2a).

On the synthetic prepared data with float32 stamps (10 stamps per split, 2 per class; the
``classifier_f32_dir`` fixture, whose leaked Train stamp is dropped): batch 4 gives epochs of
2 steps, so step-based validation, the horizon and mid-epoch checkpoints all cross epoch
boundaries. What is checked here is the schedule's rule as ``metrics.json`` states it (the
horizon extension, the best and last steps and epochs, the checkpoints) and the settings'
validation; the exact numbers of every flag are the golden set's
(``test_engine_regression.py``), the resumes ``test_engine_resume.py``'s, the runner's
refusals ``test_tasks_plain.py``'s and the preset ``test_presets.py``'s.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _training_support import ACCELERATORS

from astro_simclr.config import ConfigError, Settings, settings_from_mapping
from astro_simclr.engine import load_best, load_checkpoint
from astro_simclr.tasks.plain import run_plain

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


def _settings(classifier_f32_dir: Path, out: Path, **changes: Any) -> Settings:
    base = Settings(
        accelerator="cpu",
        deterministic=True,
        data_dir=classifier_f32_dir.parent,
        dataset=classifier_f32_dir.name,
        output_dir=out,
        **_SCHEDULE,
    )
    return base.replace(**changes)


def test_run_plain_writes_the_schedule_s_metrics(classifier_f32_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    metrics = run_plain("train-supervised", _settings(classifier_f32_dir, out))
    on_disk = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert on_disk == metrics
    assert metrics["config"]["schedule"] == "carrasco2021"
    # The first validation (step 3) is always a best, so the horizon is at least 13.
    assert metrics["horizon"] >= 13
    assert metrics["last_step"] == metrics["horizon"]
    assert metrics["best_step"] % 3 == 0
    assert metrics["horizon"] == max(10, metrics["best_step"] + 10)
    # Epochs of 2 steps: a step's epoch is that of the batch it trained on.
    assert metrics["best_epoch"] == (metrics["best_step"] - 1) // 2
    assert metrics["last_epoch"] == (metrics["last_step"] - 1) // 2
    last = load_checkpoint(out / "ckpt" / "last.pt")
    assert last["steps_done"] == metrics["last_step"]  # written once more at the end
    assert last["policy"]["best_step"] == metrics["best_step"]
    assert last["policy"]["horizon"] == metrics["horizon"]
    best = load_best(out / "ckpt" / "best.pt")
    assert (best["steps_done"], best["score"]) == (metrics["best_step"], metrics["monitor_best"])
    for key in ("val_acc_best", "test_acc_at_best", "monitor_best"):
        assert isinstance(metrics[key], float)
    assert (out / "confusion_matrix_test.png").is_file()


@pytest.mark.parametrize("accelerator", ACCELERATORS)
def test_smoke_on_each_accelerator(
    accelerator: str, classifier_f32_dir: Path, tmp_path: Path
) -> None:
    settings = _settings(
        classifier_f32_dir,
        tmp_path,
        accelerator=accelerator,
        deterministic=False,
        max_steps=4,
        val_every_n_steps=2,
        horizon_increment=0,
        checkpoint_every_n_steps=2,
    )
    metrics = run_plain("train-supervised", settings)
    assert metrics["accelerator"] == accelerator
    assert metrics["last_step"] == 4
    assert metrics["best_step"] in (2, 4)


def test_settings_validate_the_schedule() -> None:
    def build(**changes: Any) -> Settings:
        return settings_from_mapping({**_SCHEDULE, **changes})

    assert build().schedule == "carrasco2021"
    with pytest.raises(ConfigError, match="schedule 'steps'"):
        build(schedule="steps")
    with pytest.raises(ConfigError, match="eval_batch_size=64"):
        build(eval_batch_size=100)
    with pytest.raises(ConfigError, match="early_stopping=false"):
        build(early_stopping=True)
    with pytest.raises(ConfigError, match="num_workers=0"):
        build(num_workers=2)
    with pytest.raises(ConfigError, match="max_steps"):
        build(max_steps=0)
    with pytest.raises(ConfigError, match="horizon_increment"):
        build(horizon_increment=-1)
    with pytest.raises(ConfigError, match="multiple of val_every_n_steps"):
        build(checkpoint_every_n_steps=4)
    # The epoch schedule does not pin them.
    assert build(schedule="epochs", eval_batch_size=100, early_stopping=True).eval_batch_size == 100
