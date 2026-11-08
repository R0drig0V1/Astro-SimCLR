"""End-to-end smoke test of the CLI (task 1.7): the whole chain, in-process, on the CPU.

``make-synthetic`` -> ``prepare-data`` -> ``pretrain`` -> ``linear-eval`` (on the pretrain
checkpoint) -> ``fine-tune`` -> ``train-supervised``, each through ``cli.main([...])`` on a
tiny synthetic dataset in ``tmp_path``, in the dry mode of the plan (``max_epochs=1
limit_train_batches=2 limit_val_batches=2``, small batches, the ``stamp1`` encoder), plus
the ``--resume`` equivalence. It runs inside ``scripts/check.sh`` with no skip marker; the
plan's budget is 60 s on CI.

``train-supervised`` is the one exception to the dry mode: the plan asks for a validation
accuracy >= 0.5 with seed 0, and the stamp classifier's BatchNorm statistics need a few
dozen steps before evaluation-mode accuracy means anything, so it trains 8 short epochs
(about 5 s; seed 0 reaches about 0.8, seeds 1-2 about 0.7).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from astro_simclr.cli import main
from astro_simclr.data.synthetic import CLASSIFIER_FILES, PRETRAINING_FILE

#: Stamps per split: enough for 8 batches of 5, small enough to stay fast.
_N = 40
_DRY = ["max_epochs=1", "limit_train_batches=2", "limit_val_batches=2"]


def _finite_numbers(value: object) -> bool:
    """True if every number in a JSON value is finite (``null`` is not)."""
    if value is None:
        return False
    if isinstance(value, bool | str):
        return True
    if isinstance(value, int | float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_numbers(item) for item in value)
    if isinstance(value, dict):
        return all(_finite_numbers(item) for item in value.values())
    return True


def _metrics(out: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((out / "metrics.json").read_text())
    return data


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """make-synthetic and prepare-data through the CLI; returns the data directory."""
    root = tmp_path_factory.mktemp("smoke")
    raw, data = root / "raw", root / "data"
    assert main(["make-synthetic", "--out", str(raw), "--n-per-split", str(_N)]) == 0
    for name in (*CLASSIFIER_FILES, PRETRAINING_FILE):
        assert main(["prepare-data", str(raw / name), "--out", str(data)]) == 0
    return data


def _train(command: str, data: Path, out: Path, *settings: str, extra: tuple[str, ...] = ()) -> int:
    return main(
        [
            command,
            "--data-dir",
            str(data),
            "--output-dir",
            str(out),
            "--accelerator",
            "cpu",
            "--seed",
            "0",
            *extra,
            "--set",
            "num_workers=0",
            *settings,
        ]
    )


def test_full_chain(prepared: Path, tmp_path: Path) -> None:
    pretrain = tmp_path / "pretrain"
    assert _train("pretrain", prepared, pretrain, "encoder_name=stamp1", "batch_size=8", *_DRY) == 0
    chained = {
        "linear-eval": tmp_path / "linear-eval",
        "fine-tune": tmp_path / "fine-tune",
    }
    for command, out in chained.items():
        code = _train(
            command, prepared, out, "batch_size=5", *_DRY, extra=("--pretrained", str(pretrain))
        )
        assert code == 0
    supervised = tmp_path / "supervised"
    assert _train("train-supervised", prepared, supervised, "batch_size=5", "max_epochs=8") == 0

    for out in (pretrain, *chained.values(), supervised):
        assert (out / "checkpoints" / "best.ckpt").is_file()
        assert (out / "checkpoints" / "last.ckpt").is_file()
        metrics = _metrics(out)
        assert metrics["seed"] == 0
        assert metrics["config"]["seed"] == 0
        assert _finite_numbers(metrics["monitor_best"])
        assert metrics["best_epoch"] is not None
    assert _metrics(pretrain)["monitor"] == "loss_train_enc"
    assert _metrics(pretrain)["config"]["encoder_name"] == "stamp1"

    for out in (*chained.values(), supervised):
        metrics = _metrics(out)
        for key in ("val_acc_best", "test_acc_at_best", "val_macro_f1", "test_macro_f1"):
            assert _finite_numbers(metrics[key]), key
        assert _finite_numbers(metrics["evaluation"])
        assert len(metrics["evaluation"]["test"]["confusion_matrix"]) == 5
        assert (out / "confusion_matrix_val.png").stat().st_size > 0
        assert (out / "confusion_matrix_test.png").stat().st_size > 0
    # linear-eval and fine-tune really used the pretrain checkpoint's encoder.
    for out in chained.values():
        assert _metrics(out)["config"]["pretrained"] == str(pretrain)
    assert _metrics(supervised)["val_acc_best"] >= 0.5


def _last_state(out: Path) -> dict[str, Any]:
    checkpoint = torch.load(
        out / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=True
    )
    (early_stopping,) = (
        state for key, state in checkpoint["callbacks"].items() if key.startswith("EarlyStopping")
    )
    return {
        "epoch": checkpoint["epoch"],
        "global_step": checkpoint["global_step"],
        "lr": checkpoint["optimizer_states"][0]["param_groups"][0]["lr"],
        "scheduler_epoch": checkpoint["lr_schedulers"][0]["last_epoch"],
        "wait_count": early_stopping["wait_count"],
    }


def test_resume_matches_an_uninterrupted_run(prepared: Path, tmp_path: Path) -> None:
    # SimCLR, because its learning rate changes every epoch (cosine warmup). A huge
    # min_delta makes every epoch after the first a non-improvement, so wait_count grows by
    # one per epoch whatever the losses are: an EarlyStopping restarted from scratch would
    # end at 1 instead of 3. (Resume does not restore the random generators, so the losses
    # themselves are not compared.)
    common = (
        "encoder_name=stamp1",
        "batch_size=8",
        "limit_train_batches=2",
        "min_delta=1000",
        "patience=10",
    )
    resumed, straight = tmp_path / "resumed", tmp_path / "straight"
    assert _train("pretrain", prepared, resumed, *common, "max_epochs=2") == 0
    assert _last_state(resumed)["epoch"] == 1
    assert (
        _train("pretrain", prepared, resumed, *common, "max_epochs=4", extra=("--resume", "auto"))
        == 0
    )
    assert _train("pretrain", prepared, straight, *common, "max_epochs=4") == 0

    state = _last_state(resumed)
    assert state == _last_state(straight)
    assert state["epoch"] == 3
    assert state["wait_count"] == 3
    assert _metrics(resumed)["resumed_from"] == str(resumed / "checkpoints" / "last.ckpt")
    assert _metrics(resumed)["last_epoch"] == 3
