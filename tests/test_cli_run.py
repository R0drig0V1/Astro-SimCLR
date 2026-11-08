"""Tests of ``astro-simclr run CONFIG`` (task 4.2): resolution, ``--show``, the lowering to
``run_command`` and its equality with the flat subcommand it aliases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from astro_simclr import experiments
from astro_simclr.cli import main
from astro_simclr.config import Settings
from astro_simclr.config_tree import config_sha256, preset, resolve_mapping
from astro_simclr.protocol.split_runs import SplitRequest

_DRY = ["max_epochs=1", "limit_train_batches=1", "limit_val_batches=1"]


def _write(path: Path, data: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_show_prints_the_resolved_config_and_its_hash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _write(tmp_path / "c.yaml", {"preset": "carrasco2021"})
    assert main(["run", str(config), "--show", "--set", "model.arch=p4"]) == 0
    out = capsys.readouterr().out
    expected = resolve_mapping({"preset": "carrasco2021"}, ["model.arch=p4"])
    assert out.startswith("# command: train-supervised\n")
    assert f"# config hash: {config_sha256(expected)}\n" in out
    shown = yaml.safe_load(out)
    assert shown["model"]["arch"] == "p4"
    assert list(shown) == [
        "run",
        "runtime",
        "data",
        "model",
        "optim",
        "schedule",
        "augment",
        "eval",
    ]


def test_errors_exit_with_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _write(tmp_path / "c.yaml", {"preset": "carrasco2021"})
    assert main(["run", str(config), "--show", "--set", "model.depth=3"]) == 2
    assert "unknown key(s) model.depth" in capsys.readouterr().err
    flat = _write(tmp_path / "flat.yaml", {"seed": 3})
    assert main(["run", str(flat), "--show"]) == 2
    assert "needs the training command" in capsys.readouterr().err


def test_run_lowers_to_run_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Settings, dict[str, Any]]] = []

    def fake(command: str, settings: Settings, **kwargs: Any) -> dict[str, Any]:
        calls.append((command, settings, kwargs))
        return {"best_step": 10, "timing": {"seconds_per_step": 0.5}}

    monkeypatch.setattr(experiments, "run_command", fake)
    config = _write(tmp_path / "c.yaml", {"preset": "carrasco2021"})
    splits = tmp_path / "splits.json"
    argv = ["run", str(config), "--resume", "auto", "--splits", str(splits), "--fold", "2"]
    assert main([*argv, "--set", "run.seed=3"]) == 0
    (command, settings, kwargs) = calls[0]
    assert command == "train-supervised"
    assert settings == preset("carrasco2021").to_settings().replace(seed=3)
    assert kwargs == {"resume": "auto", "split": SplitRequest(splits, 2)}


def test_flat_config_with_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, Settings]] = []

    def fake(command: str, settings: Settings, **_: Any) -> dict[str, Any]:
        seen.append((command, settings))
        return {}

    monkeypatch.setattr(experiments, "run_command", fake)
    flat = _write(tmp_path / "flat.yaml", {"encoder_name": "stamp1"})
    assert main(["run", str(flat), "--command", "pretrain", "--set", "batch_size=4"]) == 0
    command, settings = seen[0]
    assert command == "pretrain"
    assert (settings.encoder_name, settings.batch_size, settings.max_epochs) == ("stamp1", 4, 400)


@pytest.fixture(scope="module")
def data_dir(synthetic_prepared: dict[str, Path]) -> Path:
    """The directory that holds the prepared synthetic datasets."""
    return next(iter(synthetic_prepared.values())).parent


def test_run_trains_like_the_flat_subcommand(data_dir: Path, tmp_path: Path) -> None:
    """The same dry run through ``run`` and through ``train-supervised`` (its alias)."""
    nested = _write(
        tmp_path / "nested.yaml",
        {
            "run": {"output_dir": str(tmp_path / "nested")},
            "runtime": {"accelerator": "cpu"},
            "data": {"dir": str(data_dir), "batch_size": 5},
        },
    )
    assert main(["run", str(nested), "--set", *_DRY]) == 0
    flat_out = tmp_path / "flat"
    argv = ["train-supervised", "--data-dir", str(data_dir), "--output-dir", str(flat_out)]
    assert main([*argv, "--accelerator", "cpu", "--set", "batch_size=5", *_DRY]) == 0

    def results(out: Path) -> dict[str, Any]:
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        keys = ("command", "best_epoch", "monitor_best", "val_acc_best", "test_acc_at_best")
        return {key: metrics.get(key) for key in keys}

    assert results(tmp_path / "nested") == results(flat_out)
    assert results(flat_out)["monitor_best"] is not None
