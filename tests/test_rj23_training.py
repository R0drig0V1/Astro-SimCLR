"""Tests of the RJ23 regime's runs (``schedule: rj23``, task 3.4b) on the plain engine, on the CPU.

Tiny runs of ``tasks.plain.run_plain`` on a synthetic 5-class ``raw-alerts`` layout
(``_rj23_support``): an end-to-end run with the real evaluation, and runs whose monitored
values are scripted (``tasks.rj23.macro_f1_score`` replaced by a probe that also records
the training batches) to pin the stop at patience, the kept weights and resume. Then the
settings, the presets and the CLI. ``test_tasks_rj23.py`` holds the runner's refusals, the
sources and the cap; the golden set (``test_engine_regression.py``) the numbers.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, get_args

import numpy as np
import pytest
import torch
from _rj23_support import DATASET, rj23_layout

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
from astro_simclr.engine import (
    EvalOutput,
    Task,
    TrainBatch,
    Validation,
    load_best,
    load_checkpoint,
)
from astro_simclr.presets import PRESETS, RJ23_LEARNING_RATES, preset_for, preset_report
from astro_simclr.protocol.alerts import alert_table_from_raw, write_alert_table
from astro_simclr.protocol.bench_report import benchmark_report
from astro_simclr.protocol.object_splits import read_object_splits
from astro_simclr.protocol.predictions import load_predictions
from astro_simclr.protocol.split_runs import file_sha256
from astro_simclr.rj23_data import rj23_parts
from astro_simclr.rj23_model import RJ23_TABLE2, BNPosition, build_rj23_classifier
from astro_simclr.rj23_regime import KEEPS, RJ23Schedule, predict_probabilities
from astro_simclr.runs.layout import BEST_CHECKPOINT, LAST_CHECKPOINT
from astro_simclr.runs.manifest import read_manifest
from astro_simclr.tasks import rj23 as rj23_tasks
from astro_simclr.tasks.plain import run_plain
from astro_simclr.tasks.rj23 import RJ23Request, RJ23TrainSource

#: The unpatched training stream (each probe wraps this one, not an earlier probe).
_BATCH = RJ23TrainSource.batch


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


def _run(
    settings: Settings,
    request: RJ23Request,
    probe: _Probe | None = None,
    *,
    resume: str | None = None,
) -> dict[str, Any]:
    """``run_plain`` of an RJ23 run (``resume`` as its), with ``probe``'s hooks if any."""
    return run_plain(
        "train-supervised",
        settings,
        rj23=request,
        resume=resume,
        on_step=None if probe is None else probe.on_step,
        on_validation=None if probe is None else probe.on_validation,
    )


class _Crash(RuntimeError):
    """A crash in the middle of a run."""


class _Probe:
    """Record the training batches of the runs that follow; script their validations.

    Patches ``RJ23TrainSource.batch`` (records batch ``s`` as its image sum and labels) and,
    with ``values``, ``tasks.rj23.macro_f1_score``: the validation after ``s`` steps returns
    ``values[s // 2]`` (validations every 2 steps, resumes included). Its ``on_validation``
    keeps the weights at the steps of ``keep_at``; its ``on_step`` raises :class:`_Crash`
    after ``fail_at`` steps.
    """

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        values: Sequence[float] | None = None,
        *,
        keep_at: tuple[int, ...] = (),
        fail_at: int | None = None,
    ) -> None:
        self.batches: dict[int, tuple[float, list[int]]] = {}
        self.weights: dict[int, dict[str, torch.Tensor]] = {}
        self.keep_at = keep_at
        self.fail_at = fail_at
        self.steps_done = 0
        probe = self

        def batch(source: RJ23TrainSource, step: int) -> TrainBatch:
            drawn = _BATCH(source, step)
            probe.batches[step] = (float(drawn[0].sum()), drawn[2].tolist())
            probe.steps_done = step + 1
            return drawn

        monkeypatch.setattr(RJ23TrainSource, "batch", batch)
        if values is not None:
            scripted = list(values)

            def score(n_classes: int) -> Callable[[Sequence[EvalOutput]], float]:
                return lambda outputs: scripted[probe.steps_done // 2]

            monkeypatch.setattr(rj23_tasks, "macro_f1_score", score)

    def on_step(self, steps_done: int) -> None:
        """Crash after ``fail_at`` steps."""
        if steps_done == self.fail_at:
            raise _Crash

    def on_validation(self, record: Validation, task: Task) -> None:
        """Keep the weights of the validations at ``keep_at``."""
        if record.steps_done in self.keep_at:
            state = task.module.state_dict()
            self.weights[record.steps_done] = {k: v.detach().clone() for k, v in state.items()}


def _stop(out: Path) -> dict[str, torch.Tensor]:
    """The ``stop`` weights of a plain run: ``ckpt/last.pt`` at the stop."""
    return load_checkpoint(out / LAST_CHECKPOINT)["model"]


def _best(out: Path) -> dict[str, torch.Tensor]:
    return load_best(out / BEST_CHECKPOINT)["model"]


def _equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left)


# --- An end-to-end tiny run ----------------------------------------------------------------


def test_tiny_run_end_to_end(layout: tuple[Path, Path], tmp_path: Path) -> None:
    data_dir, splits = layout
    out = tmp_path / "run"
    settings = _settings(data_dir, out)
    metrics = _run(settings, RJ23Request(splits))
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
    assert metrics["machine"]["engine"] == "plain"
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
    _run(_settings(data_dir, out, max_steps=4, patience=5), RJ23Request(splits))
    data = load_raw_alerts(data_dir / DATASET)
    part = rj23_parts(data, read_object_splits(splits), "multiscale")["Validation"]
    model = build_rj23_classifier("multiscale", n_classes=5)
    model.load_state_dict(_stop(out))
    probabilities = predict_probabilities(
        model, torch.from_numpy(part.images), torch.from_numpy(part.positions), 8
    )
    stored = load_predictions(out / "predictions_val.npz", n_classes=5)
    np.testing.assert_array_equal(stored.index, part.candids)
    np.testing.assert_allclose(stored.probabilities, probabilities.numpy(), atol=1e-6)


# --- Patience and keep (scripted validations) ----------------------------------------------


def test_stop_at_patience_keeps_the_stop_and_the_best(
    layout: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Validations at steps 0, 2, 4, 6, 8: best at 2, then 3 without a strict improvement.
    values = [0.1, 0.3, 0.3, 0.2, 0.25]
    data_dir, splits = layout
    out = tmp_path / "run"
    probe = _Probe(monkeypatch, values, keep_at=(2, 8))
    settings = _settings(data_dir, out, patience=3)
    metrics = _run(settings, RJ23Request(splits), probe)
    record = metrics["schedule"]
    assert record["history"] == [[0, 0.1], [2, 0.3], [4, 0.3], [6, 0.2], [8, 0.25]]
    assert (record["stopped_by"], record["capped"]) == ("patience", False)
    assert (metrics["best_step"], metrics["stop_step"], metrics["last_step"]) == (2, 8, 8)
    assert metrics["monitor_best"] == 0.3
    best, stop = _best(out), _stop(out)
    assert _equal(best, probe.weights[2])
    assert _equal(stop, probe.weights[8])
    assert not _equal(best, stop)
    assert max(probe.batches) == 7  # no step after the stop

    # A run stopped by patience, resumed, trains nothing more.
    again = _Probe(monkeypatch, values)
    resumed = _run(settings, RJ23Request(splits), again, resume="auto")
    assert again.batches == {}
    assert (resumed["stop_step"], resumed["timing"]["steps"]) == (8, 0)
    assert _equal(_stop(out), stop)

    # keep: best makes the best weights' predictions primary.
    out_best = tmp_path / "best"
    metrics_best = _run(
        settings.replace(output_dir=out_best, rj23_keep="best"), RJ23Request(splits)
    )
    assert metrics_best["evaluation"] == metrics_best["keep"]["best"]["evaluation"]
    assert metrics_best["predictions"]["val"]["file"] == "predictions_val.npz"
    assert metrics_best["keep"]["stop"]["predictions"]["val"]["file"] == "predictions_val_stop.npz"


def test_decision_mode_stops_on_the_selection_half(
    layout: tuple[Path, Path], tmp_path: Path
) -> None:
    data_dir, splits = layout
    metrics = _run(
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
    settings = _settings(data_dir, tmp_path / "run", max_steps=2)
    metrics = _run(settings, RJ23Request(splits, alerts=alerts))
    assert metrics["benchmark"]["alerts"] == str(alerts)


# --- Resume ----------------------------------------------------------------------------------


def test_resume_continues_the_stream_and_the_history(
    layout: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    data_dir, splits = layout
    request = RJ23Request(splits)
    whole = _Probe(monkeypatch, values)
    full = _run(
        _settings(data_dir, tmp_path / "whole", max_steps=8, checkpoint_every_n_steps=4),
        request,
        whole,
    )
    settings = _settings(data_dir, tmp_path / "parts", max_steps=8, checkpoint_every_n_steps=4)
    with pytest.raises(_Crash):
        _run(settings, request, _Probe(monkeypatch, values, fail_at=5))
    assert read_manifest(settings.output_dir).status == "failed"
    assert load_checkpoint(settings.output_dir / LAST_CHECKPOINT)["steps_done"] == 4
    second = _Probe(monkeypatch, values)
    resumed = _run(settings, request, second, resume="auto")
    assert sorted(second.batches) == [4, 5, 6, 7]
    for step, batch in second.batches.items():
        assert batch == whole.batches[step]  # the stream is a function of the step
    assert resumed["schedule"]["history"] == full["schedule"]["history"]
    assert resumed["resumed_from"] == str(settings.output_dir / LAST_CHECKPOINT)
    assert resumed["timing"]["steps"] == 4
    assert _equal(_stop(settings.output_dir), _stop(tmp_path / "whole"))  # an exact resume
    assert read_manifest(settings.output_dir).status == "finished"

    # A finished run resumed trains nothing and reports the same stop.
    again = _Probe(monkeypatch, values)
    finished = _run(settings, request, again, resume="auto")
    assert again.batches == {}
    assert finished["stop_step"] == 8 and finished["timing"]["steps"] == 0

    # Another seed or another protocol is another run (its id): refused.
    with pytest.raises(ConfigError, match="resume the same run"):
        _run(settings.replace(seed=1), request, resume="auto")
    with pytest.raises(ConfigError, match="resume the same run"):
        _run(settings, RJ23Request(splits, "decision"), resume="auto")


def test_request_refuses_an_unknown_mode(layout: tuple[Path, Path]) -> None:
    _, splits = layout
    with pytest.raises(ConfigError, match="mode"):
        RJ23Request(splits, "cv")  # type: ignore[arg-type]


def test_an_rj23_run_needs_its_request_and_raw_alerts(
    layout: tuple[Path, Path], tmp_path: Path, classifier_dir: Path
) -> None:
    data_dir, splits = layout
    with pytest.raises(ConfigError, match="needs its object splits"):
        run_plain("train-supervised", _settings(data_dir, tmp_path / "p"))
    phase2 = _settings(classifier_dir.parent, tmp_path / "y", dataset=classifier_dir.name)
    with pytest.raises(DataError, match="raw-alerts"):
        _run(phase2, RJ23Request(splits))


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
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["preset"]["name"] == "rj23-lowres"
    assert metrics["machine"]["engine"] == "plain"

    assert main([*base, *sets]) == 2
    assert "needs --splits" in capsys.readouterr().err
    assert main([*base, "--splits", str(splits), "--fold", "0", *sets]) == 2
    assert "no CV folds" in capsys.readouterr().err
    other = ["train-supervised", "--output-dir", str(tmp_path / "o"), "--alerts", "a.csv"]
    assert main(other) == 2
    assert "--alerts belongs to the rj23 schedule" in capsys.readouterr().err


def test_run_trains_a_nested_rj23_config(
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
    assert metrics["machine"]["engine"] == "plain"  # the only engine since task 4.8
