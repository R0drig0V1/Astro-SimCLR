"""Tests of the RJ23 regime on Lightning (astro_simclr.training.rj23, task 3.4b), on the CPU.

Tiny runs on a synthetic 5-class ``raw-alerts`` layout (``_rj23_support``): an end-to-end
run with the real evaluation, and runs whose monitored values are scripted (the callback's
``_evaluate`` replaced) to pin the stop at patience, the kept weights, the cap and resume.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, cast, get_args

import lightning as L
import numpy as np
import pytest
import torch
from _rj23_support import DATASET, rj23_layout
from lightning.pytorch.callbacks import Callback

from astro_simclr.cli import main
from astro_simclr.config import (
    RJ23_BN_POSITIONS,
    RJ23_KEEPS,
    RJ23_LARGE_VALUES,
    RJ23_SCENARIOS,
    ConfigError,
    Settings,
    settings_from_mapping,
)
from astro_simclr.data.raw_alerts import load_raw_alerts
from astro_simclr.data.rj23_input import VIEW_NAMES, LargeValues
from astro_simclr.data.schema import DataError
from astro_simclr.experiments import run_command
from astro_simclr.presets import PRESETS, RJ23_LEARNING_RATES, preset_for, preset_report
from astro_simclr.protocol.alerts import alert_table_from_raw, write_alert_table
from astro_simclr.protocol.bench_report import benchmark_report
from astro_simclr.protocol.predictions import load_predictions
from astro_simclr.protocol.split_runs import file_sha256
from astro_simclr.rj23_model import RJ23_TABLE2, BNPosition, build_rj23_classifier
from astro_simclr.rj23_regime import KEEPS, RJ23Schedule, predict_probabilities
from astro_simclr.runs.manifest import read_manifest
from astro_simclr.tasks.plain import run_plain
from astro_simclr.training.rj23 import (
    RJ23DataModule,
    RJ23Module,
    RJ23Request,
    RJ23ScheduleCallback,
    run_rj23,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning", "ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def layout(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return rj23_layout(tmp_path_factory.mktemp("rj23-training"))


def _settings(data_dir: Path, out: Path, **changes: object) -> Settings:
    values: dict[str, object] = {
        **PRESETS["rj23-multiscale"],
        "dataset": DATASET,
        "data_dir": data_dir,
        "output_dir": out,
        "accelerator": "cpu",
        "val_every_n_steps": 2,
        "max_steps": 40,
        "patience": 2,
        "batch_size": 16,
        "eval_batch_size": 8,
        "checkpoint_every_n_steps": 2,
        "progress_bar_refresh_rate": 0,
    }
    values.update(changes)
    return settings_from_mapping(values)


def _script(monkeypatch: pytest.MonkeyPatch, values: list[float]) -> None:
    """Make validation ``k`` of any run return ``values[k]`` (resumes included)."""

    def evaluate(
        self: RJ23ScheduleCallback, trainer: L.Trainer, module: L.LightningModule
    ) -> float:
        return values[len(self.history)]

    monkeypatch.setattr(RJ23ScheduleCallback, "_evaluate", evaluate)


class _Recorder(Callback):
    """The batches seen (by step) and the weights after chosen steps."""

    def __init__(self, keep_at: tuple[int, ...] = (), fail_at: int | None = None) -> None:
        self.batches: dict[int, tuple[float, list[int]]] = {}
        self.weights: dict[int, dict[str, torch.Tensor]] = {}
        self.keep_at = keep_at
        self.fail_at = fail_at

    def on_train_batch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule, batch: Any, batch_idx: int
    ) -> None:
        images, _, labels = batch
        self.batches[trainer.global_step] = (float(images.sum()), labels.tolist())

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        step = trainer.global_step
        if step in self.keep_at:
            model = cast(RJ23Module, pl_module).model
            self.weights[step] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if step == self.fail_at:
            raise _Crash


class _Crash(RuntimeError):
    """A crash in the middle of a run."""


def _weights(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state: dict[str, torch.Tensor] = checkpoint["state_dict"]
    return state


def _equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left)


# --- An end-to-end tiny run ----------------------------------------------------------------


def test_tiny_run_end_to_end(layout: tuple[Path, Path], tmp_path: Path) -> None:
    data_dir, splits = layout
    out = tmp_path / "run"
    settings = _settings(data_dir, out)
    metrics = run_rj23(settings, RJ23Request(splits))
    assert json.loads((out / "metrics.json").read_text()) == metrics

    # The schedule's record agrees with the policy on its own history.
    record = metrics["schedule"]
    steps = [step for step, _ in record["history"]]
    assert steps == list(range(0, record["stop_step"] + 1, 2))
    decision = RJ23Schedule(eval_every=2, patience=2, max_steps=40).decide(
        record["stop_step"], [value for _, value in record["history"]]
    )
    assert decision.stop and decision.reason == record["stopped_by"]
    assert decision.best_step == record["best_step"] == metrics["best_step"]
    assert metrics["last_step"] == metrics["stop_step"] == record["stop_step"]

    # The bench-report contract: seed, machine, splits SHA-256, per-alert predictions.
    assert metrics["benchmark"]["splits_sha256"] == file_sha256(splits)
    assert metrics["machine"]["device"] == "cpu"
    assert set(metrics["predictions"]) == {"val", "test"}
    data = load_raw_alerts(data_dir / DATASET)
    table = alert_table_from_raw(data)
    alerts = tmp_path / "alerts.csv"
    write_alert_table(table, alerts)
    lines = benchmark_report([out], splits, alerts, n_boot=20)
    assert any("macro-F1" in line for line in lines)

    # Both keeps are evaluated and written; the primary (stop) owns the plain names.
    keep = metrics["keep"]
    assert keep["primary"] == "stop"
    for name in KEEPS:
        assert (out / keep[name]["checkpoint"]).is_file()
        for stage in ("val", "test"):
            assert (out / keep[name]["predictions"][stage]["file"]).is_file()
    assert keep["best"]["predictions"]["val"]["file"] == "predictions_val_best.npz"
    assert metrics["evaluation"] == keep["stop"]["evaluation"]

    parts = metrics["benchmark"]["parts"]
    assert set(parts) == {"Train", "Validation", "Test"}
    assert keep["stop"]["predictions"]["val"]["n"] == parts["Validation"]["n"]

    # The preset report names the scenario and the deviations of this tiny run.
    preset = metrics["preset"]
    assert preset["name"] == "rj23-multiscale" and not preset["faithful"]
    assert set(preset["deviations"]) == {"val_every_n_steps", "max_steps", "patience", "batch_size"}
    assert metrics["timing"]["steps"] == metrics["last_step"]

    # The run's manifest (ADR 0015): finished, the object splits hashed as a whole split.
    manifest = read_manifest(out)
    assert manifest.status == "finished" and manifest.run_id == metrics["run"]["run_id"]
    assert (manifest.identity.split, manifest.identity.splits_sha256) == (
        "whole",
        file_sha256(splits),
    )


def test_stop_weights_reproduce_the_predictions(layout: tuple[Path, Path], tmp_path: Path) -> None:
    data_dir, splits = layout
    out = tmp_path / "run"
    run_rj23(_settings(data_dir, out, max_steps=4, patience=5), RJ23Request(splits))
    from astro_simclr.protocol.object_splits import read_object_splits
    from astro_simclr.rj23_data import rj23_parts

    data = load_raw_alerts(data_dir / DATASET)
    part = rj23_parts(data, read_object_splits(splits), "multiscale")["Validation"]
    model = build_rj23_classifier("multiscale", n_classes=5)
    model.load_state_dict(_weights(out / "checkpoints/stop.ckpt"))
    probabilities = predict_probabilities(
        model, torch.from_numpy(part.images), torch.from_numpy(part.positions), 8
    )
    stored = load_predictions(out / "predictions_val.npz", n_classes=5)
    np.testing.assert_array_equal(stored.index, part.candids)
    np.testing.assert_allclose(stored.probabilities, probabilities.numpy(), atol=1e-6)


# --- Patience, keep, cap (scripted validations) -------------------------------------------


def test_stop_at_patience_keeps_the_stop_and_the_best(
    layout: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Validations at steps 0, 2, 4, 6, 8: best at 2, then 3 without a strict improvement.
    _script(monkeypatch, [0.1, 0.3, 0.3, 0.2, 0.25])
    data_dir, splits = layout
    out = tmp_path / "run"
    recorder = _Recorder(keep_at=(2, 8))
    settings = _settings(data_dir, out, patience=3)
    metrics = run_rj23(settings, RJ23Request(splits), callbacks=[recorder])
    record = metrics["schedule"]
    assert record["history"] == [[0, 0.1], [2, 0.3], [4, 0.3], [6, 0.2], [8, 0.25]]
    assert (record["stopped_by"], record["capped"]) == ("patience", False)
    assert (metrics["best_step"], metrics["stop_step"], metrics["last_step"]) == (2, 8, 8)
    assert metrics["monitor_best"] == 0.3
    best, stop = _weights(out / "checkpoints/best.ckpt"), _weights(out / "checkpoints/stop.ckpt")
    assert _equal(best, recorder.weights[2])
    assert _equal(stop, recorder.weights[8])
    assert not _equal(best, stop)
    assert max(recorder.batches) == 7  # no step after the stop

    # A run stopped by patience, resumed, trains nothing more.
    again = _Recorder()
    resumed = run_rj23(settings, RJ23Request(splits), resume="auto", callbacks=[again])
    assert again.batches == {}
    assert (resumed["stop_step"], resumed["timing"]["steps"]) == (8, 0)
    assert _equal(_weights(out / "checkpoints/stop.ckpt"), stop)

    # keep: best makes the best weights' predictions primary.
    out_best = tmp_path / "best"
    metrics_best = run_rj23(
        dataclasses.replace(settings, output_dir=out_best, rj23_keep="best"), RJ23Request(splits)
    )
    assert metrics_best["evaluation"] == metrics_best["keep"]["best"]["evaluation"]
    assert metrics_best["predictions"]["val"]["file"] == "predictions_val.npz"
    assert metrics_best["keep"]["stop"]["predictions"]["val"]["file"] == "predictions_val_stop.npz"


def test_the_cap_stops_and_is_reported(
    layout: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _script(monkeypatch, [0.1, 0.2, 0.3, 0.4])  # improving: patience never runs out
    data_dir, splits = layout
    metrics = run_rj23(
        _settings(data_dir, tmp_path / "run", max_steps=6, patience=5), RJ23Request(splits)
    )
    record = metrics["schedule"]
    assert (record["stopped_by"], record["capped"]) == ("cap", True)
    assert (metrics["best_step"], metrics["stop_step"], metrics["last_step"]) == (6, 6, 6)


def test_decision_mode_stops_on_the_selection_half(
    layout: tuple[Path, Path], tmp_path: Path
) -> None:
    data_dir, splits = layout
    metrics = run_rj23(
        _settings(data_dir, tmp_path / "run", max_steps=4), RJ23Request(splits, "decision")
    )
    assert metrics["schedule"]["selection"] == "selection"
    assert metrics["benchmark"]["mode"] == "decision"
    assert set(metrics["predictions"]) == {"selection", "decision"}
    assert set(metrics["benchmark"]["parts"]) == {"Train", "selection", "decision"}
    assert read_manifest(tmp_path / "run").identity.split == "screen"


def test_a_csv_alert_table(layout: tuple[Path, Path], tmp_path: Path) -> None:
    data_dir, splits = layout
    alerts = tmp_path / "alerts.csv"
    write_alert_table(alert_table_from_raw(load_raw_alerts(data_dir / DATASET)), alerts)
    metrics = run_rj23(
        _settings(data_dir, tmp_path / "run", max_steps=2), RJ23Request(splits, alerts=alerts)
    )
    assert metrics["benchmark"]["alerts"] == str(alerts)


# --- Resume ----------------------------------------------------------------------------------


def test_resume_continues_the_stream_and_the_history(
    layout: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _script(monkeypatch, [0.1, 0.2, 0.3, 0.4, 0.5])
    data_dir, splits = layout
    request = RJ23Request(splits)
    whole = _Recorder()
    full = run_rj23(
        _settings(data_dir, tmp_path / "whole", max_steps=8, checkpoint_every_n_steps=4),
        request,
        callbacks=[whole],
    )
    settings = _settings(data_dir, tmp_path / "parts", max_steps=8, checkpoint_every_n_steps=4)
    with pytest.raises(_Crash):
        run_rj23(settings, request, callbacks=[_Recorder(fail_at=5)])
    assert (settings.output_dir / "checkpoints/last.ckpt").is_file()  # written at step 4
    second = _Recorder()
    resumed = run_rj23(settings, request, resume="auto", callbacks=[second])
    assert sorted(second.batches) == [4, 5, 6, 7]
    for step, batch in second.batches.items():
        assert batch == whole.batches[step]  # the stream is a function of the step
    assert resumed["schedule"]["history"] == full["schedule"]["history"]
    assert resumed["resumed_from"] == str(settings.output_dir / "checkpoints/last.ckpt")
    assert resumed["timing"]["steps"] == 4

    # A finished run resumed trains nothing and reports the same stop.
    again = _Recorder()
    finished = run_rj23(settings, request, resume="auto", callbacks=[again])
    assert again.batches == {}
    assert finished["stop_step"] == 8 and finished["timing"]["steps"] == 0

    # Another seed is another stream: refused.
    with pytest.raises(DataError, match="stream"):
        run_rj23(dataclasses.replace(settings, seed=1), request, resume="auto")
    # Another protocol is another run: refused.
    with pytest.raises(ConfigError, match=r"the same run|the checkpoint ran"):
        run_rj23(settings, RJ23Request(splits, "decision"), resume="auto")


def test_a_fit_that_ends_early_says_to_resume(layout: tuple[Path, Path], tmp_path: Path) -> None:
    class StopAt3(Callback):
        def on_train_batch_end(self, trainer: L.Trainer, *args: Any) -> None:
            if trainer.global_step == 3:
                trainer.should_stop = True

    data_dir, splits = layout
    with pytest.raises(ConfigError, match="--resume auto"):
        run_rj23(
            _settings(data_dir, tmp_path / "run", patience=50),
            RJ23Request(splits),
            callbacks=[StopAt3()],
        )


# --- Adapter pieces ------------------------------------------------------------------------


def test_data_module_and_callback_refusals(layout: tuple[Path, Path], tmp_path: Path) -> None:
    from astro_simclr.protocol.object_splits import read_object_splits
    from astro_simclr.rj23_data import rj23_parts

    data_dir, splits = layout
    parts = rj23_parts(load_raw_alerts(data_dir / DATASET), read_object_splits(splits), "lowres")
    with pytest.raises(ValueError, match="no part"):
        RJ23DataModule(parts, train="Pool", batch_size=4, seed=0, n_classes=5, identity={})
    module = RJ23DataModule(
        parts, train="Train", batch_size=4, seed=0, n_classes=5, identity={"mode": "x"}
    )
    with pytest.raises(DataError, match="has setup"):
        module.part("Train")
    module.setup("fit")
    loader = module.train_dataloader()
    state = loader.state_dict()
    with pytest.raises(ConfigError, match="the same run"):
        loader.load_state_dict({**state, "rj23_run": {"mode": "y"}})
    with pytest.raises(DataError, match="stream"):
        loader.load_state_dict({**state, "seed": 9})
    loader.load_state_dict(state)
    images, positions, labels = next(iter(loader))
    assert images.shape == (4, 6, 16, 16) and positions.shape == (4, 3) and labels.shape == (4,)

    schedule = RJ23Schedule(eval_every=2, patience=1, max_steps=4)
    with pytest.raises(ValueError, match="multiple"):
        RJ23ScheduleCallback(
            schedule,
            module,
            selection="Validation",
            n_classes=5,
            eval_batch_size=4,
            last_checkpoint=tmp_path / "last.ckpt",
            last_every_n_steps=3,
        )
    callback = RJ23ScheduleCallback(
        schedule,
        module,
        selection="Validation",
        n_classes=5,
        eval_batch_size=4,
        last_checkpoint=tmp_path / "last.ckpt",
    )
    saved = callback.state_dict()
    with pytest.raises(ConfigError, match="schedule"):
        callback.load_state_dict({**saved, "selection": "selection"})
    callback.load_state_dict(saved)
    assert callback.decision is None and not callback.stopped
    with pytest.raises(ConfigError, match="mode"):
        RJ23Request(splits, "cv")  # type: ignore[arg-type]


def test_run_rj23_needs_its_schedule_and_raw_alerts(
    layout: tuple[Path, Path], tmp_path: Path, classifier_dir: Path
) -> None:
    data_dir, splits = layout
    with pytest.raises(ConfigError, match="run_rj23 needs"):
        run_rj23(Settings(), RJ23Request(splits))
    with pytest.raises(ConfigError, match=r"training\.rj23\.run_rj23"):
        run_command("train-supervised", _settings(data_dir, tmp_path / "x"))
    # The plain engine's runner trains it with its request only.
    with pytest.raises(ConfigError, match="needs its object splits"):
        run_plain("train-supervised", _settings(data_dir, tmp_path / "p"))
    phase2 = _settings(classifier_dir.parent, tmp_path / "y", dataset=classifier_dir.name)
    with pytest.raises(DataError, match="raw-alerts"):
        run_rj23(phase2, RJ23Request(splits))


# --- Settings, presets and the CLI ---------------------------------------------------------


def test_presets_match_the_model_and_input_constants() -> None:
    assert RJ23_SCENARIOS == VIEW_NAMES
    assert get_args(BNPosition) == RJ23_BN_POSITIONS
    assert get_args(LargeValues) == RJ23_LARGE_VALUES
    assert RJ23_KEEPS == KEEPS
    for scenario in RJ23_SCENARIOS:
        assert RJ23_LEARNING_RATES[scenario] == RJ23_TABLE2[scenario].learning_rate
        preset = PRESETS[f"rj23-{scenario}"]
        assert preset["lr"] == RJ23_TABLE2[scenario].learning_rate
        assert preset["rj23_scenario"] == scenario
        settings = settings_from_mapping(dict(preset))
        assert preset_for(settings) == f"rj23-{scenario}"
        report = preset_report(settings)
        assert report is not None and report["faithful"] and report["diagnostics"] == []


def test_the_rj23_preset_values() -> None:
    settings = settings_from_mapping(dict(PRESETS["rj23-full"]))
    schedule = RJ23Schedule(
        eval_every=settings.val_every_n_steps,
        patience=settings.patience,
        max_steps=settings.max_steps,
    )
    assert schedule == RJ23Schedule()
    assert (settings.batch_size, settings.optimizer, settings.rj23_keep) == (256, "Adam", "stop")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"supervised_model": "stamp_classifier"}, "supervised_model=rj23"),
        ({"with_features": True}, "with_features=false"),
        ({"monitor": "loss_val"}, "monitor=macro_f1_val"),
        ({"early_stopping": True}, "early_stopping=false"),
        ({"num_workers": 2}, "num_workers=0"),
        ({"patience": 0}, "patience >= 1"),
        ({"max_steps": 1250}, "max_steps a multiple"),
        ({"checkpoint_every_n_steps": 750}, "checkpoint_every_n_steps a multiple"),
        ({"rj23_scenario": "huge"}, "rj23_scenario 'huge'"),
        ({"rj23_keep": "last"}, "rj23_keep 'last'"),
    ],
)
def test_rj23_settings_are_checked(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        settings_from_mapping({**PRESETS["rj23-full"], **changes})


def test_rj23_fields_belong_to_the_schedule() -> None:
    with pytest.raises(ConfigError, match="apply only to schedule=rj23"):
        Settings(rj23_scenario="lowres")
    with pytest.raises(ConfigError, match="needs schedule=rj23"):
        Settings(supervised_model="rj23")
    assert preset_for(Settings()) is None


def test_cli_trains_an_rj23_scenario(
    layout: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir, splits = layout
    out = tmp_path / "cli"
    base = ["train-supervised", "--preset", "rj23-lowres", "--data-dir", str(data_dir)]
    base += ["--output-dir", str(out), "--accelerator", "cpu"]
    sets = ["--set", f"dataset={DATASET}", "val_every_n_steps=2", "max_steps=2"]
    sets += ["batch_size=8", "checkpoint_every_n_steps=2", "progress_bar_refresh_rate=0"]
    assert main([*base, "--splits", str(splits), *sets]) == 0
    printed = capsys.readouterr().out
    assert "val macro-F1" in printed and "test macro-F1" in printed
    assert json.loads((out / "metrics.json").read_text())["preset"]["name"] == "rj23-lowres"

    assert main([*base, *sets]) == 2
    assert "needs --splits" in capsys.readouterr().err
    assert main([*base, "--splits", str(splits), "--fold", "0", *sets]) == 2
    assert "no CV folds" in capsys.readouterr().err
    other = ["train-supervised", "--output-dir", str(tmp_path / "o"), "--alerts", "a.csv"]
    assert main(other) == 2
    assert "--alerts belongs to the rj23 schedule" in capsys.readouterr().err


def test_run_trains_a_nested_rj23_config_on_both_engines(
    layout: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir, splits = layout
    config = tmp_path / "rj23.yaml"
    nested = {
        "preset": "rj23-lowres",
        "run": {"output_dir": str(tmp_path / "run")},
        "runtime": {"accelerator": "cpu"},
        "data": {"dir": str(data_dir), "dataset": DATASET, "batch_size": 8},
        "schedule": {"max_steps": 2, "val_every_n_steps": 2, "checkpoint_every_n_steps": 2},
    }
    config.write_text(json.dumps(nested), encoding="utf-8")
    argv = ["run", str(config), "--splits", str(splits)]
    assert main([*argv, "--set", "runtime.progress_bar_refresh_rate=0"]) == 0
    assert "val macro-F1" in capsys.readouterr().out
    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert metrics["preset"]["name"] == "rj23-lowres"
    plain = ["--set", "runtime.engine=plain", "run.output_dir=" + str(tmp_path / "plain")]
    assert main([*argv, *plain]) == 0
    assert "val macro-F1" in capsys.readouterr().out
    on_plain = json.loads((tmp_path / "plain" / "metrics.json").read_text())
    assert on_plain["machine"]["engine"] == "plain"
    assert on_plain["evaluation"] == metrics["evaluation"]
