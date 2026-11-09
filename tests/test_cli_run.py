"""Tests of ``astro-simclr run CONFIG`` (task 4.2): resolution, ``--show``, the lowering to
the plain engine's ``run_plain``, its equality with the flat subcommand it aliases, and the
refusal of what task 4.8 removed (``runtime.engine lightning``, the legacy commands)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from astro_simclr.cli import main
from astro_simclr.config import Settings
from astro_simclr.config_tree import config_sha256, preset, resolve_mapping
from astro_simclr.protocol.split_runs import SplitRequest
from astro_simclr.tasks import plain

#: A short carrasco2021 run: 3 steps, one validation, no horizon extension.
_SHORT = ["max_steps=3", "val_every_n_steps=3", "horizon_increment=0", "checkpoint_every_n_steps=3"]
#: The same, as nested paths.
_SHORT_NESTED = [f"schedule.{item}" for item in _SHORT]


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


def test_run_lowers_to_run_plain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Settings, dict[str, Any]]] = []

    def fake(command: str, settings: Settings, **kwargs: Any) -> dict[str, Any]:
        calls.append((command, settings, kwargs))
        return {"best_step": 10, "timing": {"seconds_per_step": 0.5}}

    monkeypatch.setattr(plain, "run_plain", fake)
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

    monkeypatch.setattr(plain, "run_plain", fake)
    flat = _write(tmp_path / "flat.yaml", {"seed": 3})
    argv = ["run", str(flat), "--command", "train-supervised", "--set", "batch_size=4"]
    assert main(argv) == 0
    command, settings = seen[0]
    assert command == "train-supervised"
    assert (settings.seed, settings.batch_size) == (3, 4)


def test_lightning_configs_are_refused_naming_the_tag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Task 4.8: runtime.engine lightning, and the legacy commands, which only it trained.
    config = _write(tmp_path / "c.yaml", {"preset": "carrasco2021"})
    assert main(["run", str(config), "--set", "runtime.engine=lightning"]) == 2
    assert "lightning-final" in capsys.readouterr().err
    flat = _write(tmp_path / "flat.yaml", {"encoder_name": "stamp1"})
    out = ["--set", f"output_dir={tmp_path / 'out'}"]
    assert main(["run", str(flat), "--command", "pretrain", *out]) == 2
    assert "lightning-final" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_run_trains_like_the_flat_subcommand(classifier_f32_dir: Path, tmp_path: Path) -> None:
    """The same short run through ``run`` and through ``train-supervised`` (its alias)."""
    data_dir = classifier_f32_dir.parent
    nested = _write(
        tmp_path / "nested.yaml",
        {
            "preset": "carrasco2021",
            "run": {"output_dir": str(tmp_path / "nested")},
            "runtime": {"accelerator": "cpu"},
            "data": {"dir": str(data_dir), "batch_size": 4},
        },
    )
    assert main(["run", str(nested), "--set", *_SHORT_NESTED]) == 0
    flat_out = tmp_path / "flat"
    argv = ["train-supervised", "--preset", "carrasco2021", "--data-dir", str(data_dir)]
    argv += ["--output-dir", str(flat_out), "--accelerator", "cpu"]
    assert main([*argv, "--set", "batch_size=4", *_SHORT]) == 0

    def results(out: Path) -> dict[str, Any]:
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        keys = ("command", "best_step", "monitor_best", "val_acc_best", "test_acc_at_best")
        return {key: metrics.get(key) for key in keys} | {"run": metrics["run"]}

    assert results(tmp_path / "nested") == results(flat_out)
    assert results(flat_out)["monitor_best"] is not None
