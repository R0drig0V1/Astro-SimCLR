"""Tests of ``make-object-splits`` and ``bench-report`` end to end, on synthetic data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from _bench_support import MACHINE, as_predictions, bench_run, noisy_predictions, synthetic_table

from astro_simclr.cli import main
from astro_simclr.config import ConfigError
from astro_simclr.data.schema import DataError
from astro_simclr.protocol.alerts import RJ23_LABEL_NAMES, AlertTable, write_alert_table
from astro_simclr.protocol.bench_report import benchmark_report
from astro_simclr.protocol.object_splits import read_object_splits
from astro_simclr.protocol.predictions import Predictions
from astro_simclr.protocol.split_runs import file_sha256

#: The CLI flag of RJ23's 6 classes: the setup also checks that nothing assumes 5 classes.
LABELS = ["--labels", ",".join(RJ23_LABEL_NAMES)]


class Setup:
    """A 6-class alert table, its object splits and a writer of benchmark runs on Test."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.table = synthetic_table(900, seed=3, label_names=RJ23_LABEL_NAMES)
        self.alerts = root / "alerts.csv"
        write_alert_table(self.table, self.alerts)
        self.splits = root / "splits.json"
        args = ["make-object-splits", str(self.alerts), "--out", str(self.splits), *LABELS]
        assert main(args) == 0
        self.sha = file_sha256(self.splits)
        loaded = read_object_splits(self.splits)
        self.rows = loaded.rows(self.table, "Test")

    def predictions(self, accuracy: float, seed: int) -> Predictions:
        labels = self.table.label[self.rows]
        predicted = noisy_predictions(labels, accuracy, 6, seed)
        return as_predictions(labels, predicted, self.table.candid[self.rows], 6)

    def runs(
        self, name: str, accuracy: float, seeds: range = range(5), *, sha: str | None = None
    ) -> list[Path]:
        return [
            bench_run(
                self.root,
                f"{name}/seed{s}",
                self.predictions(accuracy, int(100 * accuracy) + s),
                splits_sha256=self.sha if sha is None else sha,
                seed=s,
            )
            for s in seeds
        ]


@pytest.fixture
def setup(tmp_path: Path) -> Setup:
    return Setup(tmp_path)


def test_make_object_splits_cli(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    again = ["make-object-splits", str(setup.alerts), "--out", str(setup.splits), *LABELS]
    assert main([*again, "--overwrite"]) == 0
    out = capsys.readouterr().out
    assert "Train:" in out and "decision:" in out and f"wrote {setup.splits}" in out
    assert main(again) == 2
    assert "already exists" in capsys.readouterr().err
    assert main([*again, "--overwrite", "--seed", "1", "--fractions", "0.7", "0.15", "0.15"]) == 0
    assert read_object_splits(setup.splits).fractions == (0.7, 0.15, 0.15)


def test_make_object_splits_conflicts(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    table = setup.table
    labels = table.label.copy()
    labels[0] = (labels[0] + 1) % 6
    other = np.flatnonzero(table.oid == table.oid[0])
    bad = setup.root / "bad.csv"
    rows = AlertTable.from_rows(
        table.oid.tolist(),
        table.candid.tolist(),
        table.jd.tolist(),
        labels.tolist(),
        table.label_names,
    )
    write_alert_table(rows, bad)
    out = setup.root / "bad.json"
    args = ["make-object-splits", str(bad), "--out", str(out), *LABELS]
    if other.size > 1:
        assert main(args) == 1
        assert "different labels" in capsys.readouterr().err
        assert main([*args, "--drop-conflicts"]) == 0
        assert f"dropped {table.oid[0]}" in capsys.readouterr().out
    assert main([*args[:4], "--overwrite", "--labels", "A,B"]) == 1  # unknown label names
    assert "unknown label" in capsys.readouterr().err


def test_report_on_runs(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    runs = setup.runs("model", 0.8)
    code = main(
        [
            "bench-report",
            *map(str, runs),
            "--splits",
            str(setup.splits),
            "--alerts",
            str(setup.alerts),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "view all alerts:" in out and "view first stamp:" in out
    assert "macro-F1" in out and "(%, mean ± std over 5 seeds)" in out
    assert "prior-reweighted to [9774, 3615, 10211, 9180, 15950, 580]" in out
    assert "Satellite" in out and "MDE" in out
    assert out.rstrip().endswith("bench-report: REPORT (Test is never a decision input)")
    assert "WARN" not in out


def test_report_comparison_identical_and_better(setup: Setup) -> None:
    base = setup.runs("base", 0.7)
    same = benchmark_report(
        base, setup.splits, setup.alerts, baseline_dirs=base[::-1][:5], n_boot=300
    )
    assert sum("+0.00 points, 95 % CI [+0.00, +0.00]" in line for line in same) == 2
    better = benchmark_report(
        setup.runs("better", 0.95),
        setup.splits,
        setup.alerts,
        baseline_dirs=base,
        n_boot=300,
        reference_counts=[1] * 6,
    )
    text = "\n".join(better)
    assert "MDE paired" in text and "prior-reweighted to [1, 1, 1, 1, 1, 1]" in text
    diffs = [line for line in better if "seed-ensemble macro-F1:" in line]
    assert len(diffs) == 2 and all("+" in d.split("points")[0] for d in diffs)


def test_report_first_stamp_view_counts(setup: Setup) -> None:
    lines = benchmark_report(setup.runs("m", 0.8, range(2)), setup.splits, setup.alerts)
    views = [line for line in lines if line.startswith("view ")]
    n_objects = np.unique(setup.table.oid[setup.rows]).size
    assert views == [
        f"view all alerts: {setup.rows.size} alerts of {n_objects} objects",
        f"view first stamp: {n_objects} alerts of {n_objects} objects",
    ]
    assert "WARN: 2 runs run(s)" in "\n".join(lines)


def test_report_refusals(setup: Setup) -> None:
    good = setup.runs("good", 0.8, range(1))

    def report(
        dirs: list[Path],
        *,
        stage: str = "test",
        reference_counts: list[float] | None = None,
        allow_mixed_machines: bool = False,
    ) -> list[str]:
        return benchmark_report(
            dirs,
            setup.splits,
            setup.alerts,
            stage=stage,
            reference_counts=reference_counts,
            allow_mixed_machines=allow_mixed_machines,
        )

    with pytest.raises(ConfigError, match="another object-splits file"):
        report(setup.runs("other", 0.8, range(1), sha="0" * 64))
    with pytest.raises(ConfigError, match="another object-splits file"):
        report([bench_run(setup.root, "nosha", setup.predictions(0.8, 0), splits_sha256=None)])
    with pytest.raises(ConfigError, match="no per-alert val"):
        report(good, stage="val")
    with pytest.raises(ConfigError, match="unknown stage"):
        report(good, stage="train")
    with pytest.raises(ConfigError, match="at least one"):
        report([])
    with pytest.raises(ConfigError, match="needs 6 values"):
        report(good, reference_counts=[1, 2])
    with pytest.raises(ConfigError, match="repeat a seed"):
        report([good[0], good[0]])
    # Predictions of other alerts, or of other labels.
    pred = setup.predictions(0.8, 0)
    short = Predictions(pred.probabilities[1:], pred.labels[1:], pred.index[1:], 6)
    with pytest.raises(ConfigError, match="not exactly"):
        report([bench_run(setup.root, "short", short, splits_sha256=setup.sha)])
    relabelled = Predictions(pred.probabilities, (pred.labels + 1) % 6, pred.index, 6)
    with pytest.raises(ConfigError, match="labels differ"):
        report([bench_run(setup.root, "relabel", relabelled, splits_sha256=setup.sha)])
    # Mixed machines: refused unless allowed.
    gpu = bench_run(
        setup.root,
        "gpu",
        pred,
        splits_sha256=setup.sha,
        seed=9,
        machine={**MACHINE, "device": "cuda"},
    )
    with pytest.raises(ConfigError, match="different machines"):
        report([*good, gpu])
    assert "WARN: mixed machines" in "\n".join(report([*good, gpu], allow_mixed_machines=True))
    nomachine = setup.root / "nomachine"
    bench_run(setup.root, "nomachine", pred, splits_sha256=setup.sha, seed=8)
    metrics = json.loads((nomachine / "metrics.json").read_text())
    metrics.pop("machine")
    (nomachine / "metrics.json").write_text(json.dumps(metrics))
    assert "do not record their machine" in "\n".join(report([nomachine]))


def test_report_data_errors(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    # A 5-class predictions file is not a 6-class benchmark run.
    pred = setup.predictions(0.8, 0)
    five = Predictions(pred.probabilities[:, :5], np.minimum(pred.labels, 4), pred.index)
    run = bench_run(setup.root, "five", five, splits_sha256=setup.sha)
    args = ["bench-report", str(run), "--splits", str(setup.splits), "--alerts", str(setup.alerts)]
    assert main(args) == 1
    assert "probabilities (N, 6)" in capsys.readouterr().err
    # Splits of another table.
    other = setup.root / "other.csv"
    write_alert_table(synthetic_table(900, seed=4), other)
    with pytest.raises(DataError, match="another alert table"):
        benchmark_report([run], setup.splits, other)


def test_report_without_benchmark_label_names(tmp_path: Path) -> None:
    names = ("A", "B", "C")
    table = synthetic_table(300, seed=2, label_names=names)
    alerts, splits = tmp_path / "a.csv", tmp_path / "s.json"
    write_alert_table(table, alerts)
    assert main(["make-object-splits", str(alerts), "--out", str(splits), "--labels", "A,B,C"]) == 0
    rows = read_object_splits(splits).rows(table, "Test")
    labels = table.label[rows]
    pred = as_predictions(labels, noisy_predictions(labels, 0.9, 3, 0), table.candid[rows], 3)
    run = bench_run(tmp_path, "r", pred, splits_sha256=file_sha256(splits))
    lines = benchmark_report([run], splits, alerts)
    assert not any("prior-reweighted" in line for line in lines)


def test_report_on_r5_defaults(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """R5's five classes are the default, reweighted to RJ23's 5-class counts."""
    table = synthetic_table(400, seed=6)
    alerts, splits = tmp_path / "a.csv", tmp_path / "s.json"
    write_alert_table(table, alerts)
    assert main(["make-object-splits", str(alerts), "--out", str(splits)]) == 0
    rows = read_object_splits(splits).rows(table, "Test")
    labels = table.label[rows]
    pred = as_predictions(labels, noisy_predictions(labels, 0.9, 5, 0), table.candid[rows], 5)
    run = bench_run(tmp_path, "r", pred, splits_sha256=file_sha256(splits))
    capsys.readouterr()
    assert main(["bench-report", str(run), "--splits", str(splits), "--alerts", str(alerts)]) == 0
    out = capsys.readouterr().out
    assert "prior-reweighted to [9774, 3615, 10211, 9180, 15950]" in out
    assert "Satellite" not in out
