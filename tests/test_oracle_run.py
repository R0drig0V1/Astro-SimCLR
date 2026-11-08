"""Tests of the M1 oracle run, its report and the ``fetch-m1`` / ``m1-oracle`` CLI (task 2.2b).

The checks run on synthetic prepared data and random M1-shaped weights; the real numbers
come from ``astro-simclr m1-oracle`` on ``red`` (``docs/architecture.md``, "oracle").
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _tf_writer import event_file, random_m1, synthetic_m1_dir

from astro_simclr.cli import main
from astro_simclr.data import DataError, prepare
from astro_simclr.data.synthetic import CLASSIFIER_FILES, make_synthetic
from astro_simclr.oracle import m1
from astro_simclr.oracle.run import (
    CurvePoint,
    read_val_curve,
    run_m1_oracle,
    selection_loss,
    write_curve,
    write_report,
)
from astro_simclr.reference import read_oracle_report


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    raw = tmp_path_factory.mktemp("oracle_raw")
    make_synthetic(raw, n_per_split=10, seed=3)
    out = tmp_path_factory.mktemp("oracle_prepared")
    return {
        "f32": prepare(raw / CLASSIFIER_FILES[0], out / "f32", images="float32"),
        "u8": prepare(raw / CLASSIFIER_FILES[0], out / "u8"),
    }


def _manual_loss(logits: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> float:
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_p = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    ce = -log_p[np.arange(len(labels)), labels].mean()
    entropy = -(np.exp(log_p) * log_p).sum(axis=1).mean()
    return float(ce - 0.5 * entropy)


def test_selection_loss_is_the_mean_of_batch_means() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(130, 5))
    labels = rng.integers(0, 5, size=130)
    batches = [slice(0, 64), slice(64, 128), slice(128, 130)]  # the last batch has 2 stamps
    expected = np.mean([_manual_loss(logits[b], labels[b]) for b in batches])
    assert selection_loss(logits, labels) == pytest.approx(expected, abs=1e-12)
    assert selection_loss(logits, labels) != pytest.approx(_manual_loss(logits, labels))
    with pytest.raises(ValueError, match=r"\(N, 5\)"):
        selection_loss(logits[:, :4], labels)
    with pytest.raises(ValueError, match=r"\(N, 5\)"):
        selection_loss(logits[:0], labels[:0])


def test_curve_round_trip(tmp_path: Path) -> None:
    events = tmp_path / "val.events"
    events.write_bytes(
        event_file(
            [(0, {"loss": 0.75, "accuracy": 0.2}), (10, {"loss": -0.0616, "accuracy": 0.95})]
        )
    )
    points = read_val_curve(events)
    assert [p.step for p in points] == [0, 10]
    path = tmp_path / "sub" / "curve.csv"
    write_curve(points, path)
    rows = list(csv.reader(path.open()))
    assert rows == [
        ["step", "val_loss", "val_accuracy"],
        ["0", "0.75", "0.2"],
        ["10", "-0.0616", "0.95"],
    ]


@pytest.mark.parametrize(
    ("events", "match"),
    [
        ([(0, {"loss": 1.0})], "no 'loss' and 'accuracy'"),
        ([(0, {"loss": 1.0}), (10, {"accuracy": 0.5})], "steps differ"),
    ],
)
def test_curve_errors(
    tmp_path: Path, events: list[tuple[int, dict[str, float]]], match: str
) -> None:
    path = tmp_path / "val.events"
    path.write_bytes(event_file(events))
    with pytest.raises(ValueError, match=match):
        read_val_curve(path)


def test_run_on_synthetic_data(prepared: dict[str, Path], tmp_path: Path) -> None:
    curve = [CurvePoint(0, 0.5, 0.4), CurvePoint(10, -0.06, 0.95), CurvePoint(20, -0.01, 0.96)]
    report = run_m1_oracle(prepared["f32"], random_m1(11), curve=curve)
    port = report["checks"]["port_equivalence"]
    assert port["passed"] is True
    assert port["max_abs_logit_diff"] <= 1e-4
    assert port["argmax_agreement"] == 1.0
    assert port["n"] == 10
    data = report["checks"]["data_path"]
    assert data["passed"] is False  # 10 random-weight stamps cannot be 475 +- 2 correct
    assert data["correct_passed"] is False
    assert report["passed"] is False
    approx = report["nan_approximation"]
    assert approx["approximated"]["n"] + approx["others"]["n"] == 10
    assert set(report["diagnostics_validation"]) == {
        "no_nan_approximation",
        "uint8_21_padded",
        "uint8_27",
    }
    assert report["test"]["n"] == 10
    assert set(report["test"]["recall"]) == {"AGN", "SN", "VS", "Asteroid", "Bogus"}
    summary = report["m1"]["val_curve"]
    assert (summary["best_step"], summary["peak_val_accuracy_step"]) == (10, 20)
    assert summary["last_step"] == 20
    path = tmp_path / "report.json"
    write_report(report, path)
    assert json.loads(path.read_text()) == report
    assert read_oracle_report(path) is False


def test_report_passes_when_both_checks_pass(
    prepared: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from astro_simclr.oracle import run

    weights = random_m1(12)
    first = run_m1_oracle(prepared["f32"], weights)
    data = first["checks"]["data_path"]
    monkeypatch.setattr(run, "M1_VAL_CORRECT", data["correct"])
    monkeypatch.setattr(run, "M1_VAL_LOSS", data["loss"])
    report = run_m1_oracle(prepared["f32"], weights)
    assert report["checks"]["data_path"]["exact"] is True
    assert report["passed"] is True
    assert "val_curve" not in report["m1"]


def test_run_needs_the_float32_stamps(prepared: dict[str, Path]) -> None:
    with pytest.raises(DataError, match="images_f32"):
        run_m1_oracle(prepared["u8"], random_m1(13))


def test_cli_fetch_and_oracle(
    prepared: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    served, files = synthetic_m1_dir(tmp_path)
    monkeypatch.setattr(m1, "M1_FILES", files)
    cache = tmp_path / "cache"
    source = ["--cache-dir", str(cache), "--source-dir", str(served)]
    assert main(["fetch-m1", *source]) == 0
    assert "m1_weights.npz" in capsys.readouterr().out
    out = tmp_path / "out"
    code = main(["m1-oracle", "--data-dir", str(prepared["f32"]), "--out", str(out), *source])
    printed = capsys.readouterr().out
    assert code == 1  # random weights: the data-path check fails
    assert "port equivalence" in printed
    assert "passed" in printed
    assert printed.rstrip().endswith("FAIL")
    assert read_oracle_report(out / "m1_oracle.json") is False
    rows = list(csv.reader((out / "m1_val_curve.csv").open()))
    assert rows[0] == ["step", "val_loss", "val_accuracy"]
    assert len(rows) == 3


def test_cli_fetch_failure_is_a_data_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["fetch-m1", "--cache-dir", str(tmp_path / "c"), "--source-dir", str(tmp_path)])
    assert code == 1
    assert "cannot fetch M1" in capsys.readouterr().err
