"""The ``carrasco2021`` schedule through ``experiments.run_command`` (task 2.2a).

On the synthetic prepared data (10 stamps per split, 2 per class): batch 4 gives epochs of
2 steps, so step-based validation, the horizon and mid-epoch checkpoints all cross epoch
boundaries. The data path is the interim uint8 crops (2.2c wires the float path).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import lightning as L
import pytest
import torch
from _training_support import ACCELERATORS
from lightning.pytorch.callbacks import Callback, EarlyStopping

from astro_simclr.config import ConfigError, Settings, settings_from_mapping
from astro_simclr.experiments import _epochs_trainer, run_command
from astro_simclr.training.modules import ClassifierModule
from astro_simclr.training.schedule import HorizonCallback

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


def _settings(classifier_dir: Path, out: Path, **changes: Any) -> Settings:
    base = Settings(
        accelerator="cpu",
        deterministic=True,
        data_dir=classifier_dir.parent,
        dataset=classifier_dir.name,
        output_dir=out,
        **_SCHEDULE,
    )
    return base.replace(**changes)


def _best_state(out: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(out / "checkpoints" / "best.ckpt", weights_only=True)
    state: dict[str, torch.Tensor] = checkpoint["state_dict"]
    return state


def _horizon_state(checkpoint: dict[str, Any]) -> dict[str, Any]:
    (state,) = (v for k, v in checkpoint["callbacks"].items() if "HorizonCallback" in k)
    return dict(state)


class _CopyLast(Callback):
    """At step ``step``, copy ``last.ckpt`` (written at an earlier step) to ``dst``."""

    def __init__(self, step: int, src: Path, dst: Path) -> None:
        self.step, self.src, self.dst = step, src, dst

    def on_train_batch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule, *args: Any
    ) -> None:
        if trainer.global_step == self.step:
            self.dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.src, self.dst)


class _Probe(Callback):
    """Record the HorizonCallback's state when training (re)starts, and the validation steps."""

    seen: dict[str, Any]

    def __init__(self) -> None:
        self.val_steps: list[int] = []

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.val_steps.append(trainer.global_step)

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        (horizon,) = (c for c in trainer.callbacks if isinstance(c, HorizonCallback))  # type: ignore[attr-defined]
        self.seen = {
            "horizon": horizon.horizon,
            "best_step": horizon.best_step,
            "max_steps": trainer.fit_loop.epoch_loop.max_steps,
            "global_step": trainer.global_step,
        }


def test_run_command_writes_metrics(classifier_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    metrics = run_command("train-supervised", _settings(classifier_dir, out), progress_bar=False)
    on_disk = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert on_disk == metrics
    assert metrics["config"]["schedule"] == "carrasco2021"
    # The first validation (step 3) is always a best, so the horizon is at least 13.
    assert metrics["horizon"] >= 13
    assert metrics["last_step"] == metrics["horizon"]
    assert metrics["best_step"] % 3 == 0
    assert metrics["horizon"] == max(10, metrics["best_step"] + 10)
    assert metrics["best_epoch"] == (metrics["best_step"] - 1) // 2
    last = torch.load(out / "checkpoints" / "last.ckpt", weights_only=True)
    assert last["global_step"] == metrics["last_step"]  # written once more at the end
    for key in ("val_acc_best", "test_acc_at_best", "monitor_best"):
        assert isinstance(metrics[key], float)
    assert (out / "checkpoints" / "best.ckpt").is_file()
    assert (out / "checkpoints" / "last.ckpt").is_file()
    assert (out / "confusion_matrix_test.png").is_file()


def test_resume_from_mid_epoch_last_checkpoint(classifier_dir: Path, tmp_path: Path) -> None:
    # last.ckpt every 3 steps, on the validation grid, in epochs of 2 steps: the step-9 copy
    # is mid-epoch and was written after the step-9 validation, which the resume must not
    # lose (a ModelCheckpoint in on_train_batch_end would) nor repeat.
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    saved = resumed / "checkpoints" / "last.ckpt"
    copy = _CopyLast(10, whole / "checkpoints" / "last.ckpt", saved)
    whole_probe = _Probe()
    expected = run_command(
        "train-supervised",
        _settings(classifier_dir, whole),
        callbacks=[copy, whole_probe],
        progress_bar=False,
    )
    checkpoint = torch.load(saved, weights_only=True)
    assert checkpoint["global_step"] == 9
    state = _horizon_state(checkpoint)
    assert state["horizon"] > 10  # extended before the checkpoint, so its restore matters
    assert state["best_step"] is not None

    probe = _Probe()
    metrics = run_command(
        "train-supervised",
        _settings(classifier_dir, resumed),
        resume="auto",
        callbacks=[probe],
        progress_bar=False,
    )
    assert probe.seen == {
        "horizon": state["horizon"],
        "best_step": state["best_step"],
        "max_steps": state["horizon"],
        "global_step": 9,
    }
    assert whole_probe.val_steps[:3] == [3, 6, 9]
    assert probe.val_steps == [s for s in whole_probe.val_steps if s > 9]
    for key in ("best_step", "last_step", "horizon", "monitor_best", "val_acc_best"):
        assert metrics[key] == expected[key], key
    assert metrics["resumed_from"] == str(saved)
    best_whole, best_resumed = _best_state(whole), _best_state(resumed)
    assert all(torch.equal(best_whole[k], best_resumed[k]) for k in best_whole)


def test_resume_at_the_horizon_still_writes_best(classifier_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    expected = run_command("train-supervised", _settings(classifier_dir, out), progress_bar=False)
    (out / "checkpoints" / "best.ckpt").unlink()
    # The final last.ckpt is at the horizon: the fit loop is skipped, on_train_end never runs.
    metrics = run_command(
        "train-supervised", _settings(classifier_dir, out), resume="auto", progress_bar=False
    )
    assert metrics["best_step"] == expected["best_step"]
    assert metrics["monitor_best"] == expected["monitor_best"]
    assert metrics["val_acc_best"] == expected["val_acc_best"]


@pytest.mark.parametrize("accelerator", ACCELERATORS)
def test_smoke_on_each_accelerator(accelerator: str, classifier_dir: Path, tmp_path: Path) -> None:
    settings = _settings(
        classifier_dir,
        tmp_path,
        accelerator=accelerator,
        deterministic=False,
        max_steps=4,
        val_every_n_steps=2,
        horizon_increment=0,
        checkpoint_every_n_steps=2,
    )
    metrics = run_command("train-supervised", settings, progress_bar=False)
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


@pytest.mark.parametrize(
    ("command", "changes", "match"),
    [
        ("train-supervised", {"supervised_model": "stamp_classifier"}, "trains only"),
        ("pretrain", {}, "trains only"),
        ("train-supervised", {"schedule": "epochs"}, "needs schedule=carrasco2021"),
        ("train-supervised", {"image_size": 27}, "image_size=21"),
    ],
)
def test_run_command_rejects_mismatched_schedule(
    command: str, changes: dict[str, Any], match: str, classifier_dir: Path, tmp_path: Path
) -> None:
    settings = _settings(classifier_dir, tmp_path, **changes)
    with pytest.raises(ConfigError, match=match):
        run_command(command, settings, progress_bar=False)


def test_epochs_schedule_can_disable_early_stopping(tmp_path: Path) -> None:
    def callbacks(early_stopping: bool) -> list[Callback]:
        settings = Settings(output_dir=tmp_path, accelerator="cpu", early_stopping=early_stopping)
        _, trainer = _epochs_trainer(settings, ClassifierModule.MONITORABLE, [], False)
        return list(trainer.callbacks)  # type: ignore[attr-defined]

    assert any(isinstance(c, EarlyStopping) for c in callbacks(True))
    assert not any(isinstance(c, EarlyStopping) for c in callbacks(False))
