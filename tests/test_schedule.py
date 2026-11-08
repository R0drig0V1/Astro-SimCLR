"""Tests of ``training.schedule.HorizonCallback``, ALeRCE's horizon rule (task 2.2a).

A scripted module logs a predetermined ``loss_val`` at each validation and trains one
weight by SGD (so the weights differ at every step); the Trainer is configured as the
``carrasco2021`` schedule configures it (``val_check_interval``, no epoch limit).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lightning as L
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from astro_simclr.config import ConfigError
from astro_simclr.training.schedule import HorizonCallback

_STEPS_PER_EPOCH = 7  # not a multiple of the validation interval


class _Scripted(L.LightningModule):
    def __init__(self, scores: Sequence[float], metric: str = "loss_val") -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.scores = list(scores)
        self.metric = metric
        self.val_steps: list[int] = []
        self.weights_at: dict[int, torch.Tensor] = {}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self.w.sum()  # gradient 1: SGD moves w by -lr every step

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        return None

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return
        step = self.trainer.global_step
        index = len(self.val_steps)
        self.val_steps.append(step)
        self.weights_at[step] = self.w.detach().clone()
        score = self.scores[index] if index < len(self.scores) else 99.0
        self.log(self.metric, score)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def train_dataloader(self) -> DataLoader[torch.Tensor]:
        return DataLoader(torch.zeros(_STEPS_PER_EPOCH, 1), batch_size=1)  # type: ignore[arg-type]

    def val_dataloader(self) -> DataLoader[torch.Tensor]:
        return DataLoader(torch.zeros(1, 1), batch_size=1)  # type: ignore[arg-type]


def _fit(
    module: _Scripted,
    tmp_path: Path,
    *,
    max_steps: int,
    increment: int,
    mode: str = "min",
    monitor: str = "loss_val",
) -> tuple[L.Trainer, HorizonCallback]:
    horizon = HorizonCallback(
        dirpath=tmp_path,
        monitor=monitor,
        mode=mode,  # type: ignore[arg-type]
        initial_horizon=max_steps,
        increment=increment,
    )
    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=-1,
        max_steps=max_steps,
        val_check_interval=10,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
        callbacks=[horizon],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module)
    return trainer, horizon


def test_validates_every_10_steps_across_epochs(tmp_path: Path) -> None:
    module = _Scripted([3.0, 2.0, 1.0])
    trainer, _ = _fit(module, tmp_path, max_steps=30, increment=0)
    assert module.val_steps == [10, 20, 30]
    assert trainer.global_step == 30


def test_horizon_extends_and_best_is_restored(tmp_path: Path) -> None:
    # Steps 10..60: bests at 10 (horizon 35), 20 (45) and 40 (65); training ends at 65.
    module = _Scripted([5.0, 4.0, 6.0, 3.0, 7.0, 8.0])
    trainer, horizon = _fit(module, tmp_path, max_steps=30, increment=25)
    assert trainer.global_step == 65
    assert module.val_steps == [10, 20, 30, 40, 50, 60]
    assert horizon.horizon == 65
    assert horizon.best_step == 40
    assert horizon.best_model_score is not None
    assert float(horizon.best_model_score) == 3.0
    assert torch.equal(module.w.detach(), module.weights_at[40])  # restored in the module
    checkpoint = torch.load(horizon.best_model_path, weights_only=True)
    assert Path(horizon.best_model_path) == tmp_path / "best.ckpt"
    assert torch.equal(checkpoint["state_dict"]["w"], module.weights_at[40])
    assert checkpoint["global_step"] == 40
    assert checkpoint["epoch"] == 40 // _STEPS_PER_EPOCH


def test_mode_max_and_nan_never_improves(tmp_path: Path) -> None:
    module = _Scripted([math.nan, 0.5, 0.9, 0.7], metric="accuracy_val")
    _, horizon = _fit(
        module, tmp_path, max_steps=40, increment=0, mode="max", monitor="accuracy_val"
    )
    assert horizon.best_step == 30
    assert horizon.best_model_score is not None
    assert float(horizon.best_model_score) == pytest.approx(0.9)


def test_validate_after_fit_does_not_change_the_best(tmp_path: Path) -> None:
    module = _Scripted([2.0, 1.0])
    trainer, horizon = _fit(module, tmp_path, max_steps=20, increment=0)
    module.scores.append(-5.0)  # a better value, but outside fitting
    trainer.validate(module, verbose=False)
    assert horizon.best_step == 20


def test_without_validation_nothing_is_written(tmp_path: Path) -> None:
    _, horizon = _fit(_Scripted([]), tmp_path, max_steps=5, increment=0)
    assert horizon.best_step is None
    assert horizon.best_model_path == ""
    assert not (tmp_path / "best.ckpt").exists()


def test_missing_monitor_is_a_config_error(tmp_path: Path) -> None:
    module = _Scripted([1.0], metric="something_else")
    with pytest.raises(ConfigError, match="monitors 'loss_val'"):
        _fit(module, tmp_path, max_steps=10, increment=0)


def test_state_round_trip_and_monitor_check(tmp_path: Path) -> None:
    module = _Scripted([2.0, 1.0])
    _, horizon = _fit(module, tmp_path, max_steps=20, increment=15)
    state = horizon.state_dict()
    assert state["horizon"] == 35
    assert state["best_step"] == 20
    assert all(t.device.type == "cpu" for t in state["best_state"].values())
    restored = HorizonCallback(dirpath=tmp_path, initial_horizon=30, increment=15)
    restored.load_state_dict(state)
    assert (restored.horizon, restored.best_step) == (35, 20)
    assert restored.best_model_score is not None
    assert float(restored.best_model_score) == 1.0
    larger = HorizonCallback(dirpath=tmp_path, initial_horizon=50)
    larger.load_state_dict(state)
    assert larger.horizon == 50  # a larger max_steps in the resumed run wins
    other = HorizonCallback(dirpath=tmp_path, monitor="accuracy_val", mode="max")
    with pytest.raises(ConfigError, match="same monitor"):
        other.load_state_dict(state)


def test_constructor_checks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mode"):
        HorizonCallback(dirpath=tmp_path, mode="best")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="initial_horizon"):
        HorizonCallback(dirpath=tmp_path, initial_horizon=0)
    with pytest.raises(ValueError, match="initial_horizon"):
        HorizonCallback(dirpath=tmp_path, increment=-1)
    with pytest.raises(ValueError, match="last_every_n_steps"):
        HorizonCallback(dirpath=tmp_path, last_every_n_steps=0)
