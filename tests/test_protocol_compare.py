"""Tests of ``compare-variants`` and the decision helpers (screen, cross-validation)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _protocol_support import MACHINE, fake_run, make_layout, predictions

from astro_simclr.cli import main
from astro_simclr.config import ConfigError
from astro_simclr.protocol.compare import (
    compare_variants,
    cv_comparison,
    load_run_predictions,
    run_ece,
    screen_comparison,
)
from astro_simclr.protocol.predictions import Predictions
from astro_simclr.protocol.splits import ProtocolSplits, make_splits

LABELS = [i // 20 for i in range(100)]


def _wrong(labels: list[int], positions: list[int]) -> list[int]:
    """``labels`` with the stamps at ``positions`` predicted as the next class."""
    return [(y + 1) % 5 if i in positions else y for i, y in enumerate(labels)]


def _side(root: Path, name: str, errors: list[list[int]], **extra: Any) -> list[Path]:
    return [
        fake_run(
            root,
            f"{name}{seed}",
            {
                "test": predictions(LABELS, _wrong(LABELS, wrong)),
                "val": predictions(LABELS, LABELS),
            },
            seed=seed,
            **extra,
        )
        for seed, wrong in enumerate(errors)
    ]


def test_compare_variants_reports(tmp_path: Path) -> None:
    base = _side(tmp_path, "base", [list(range(0, 100, 5)), list(range(1, 100, 5))])
    variant = _side(tmp_path, "var", [[0, 1], [1, 2]])
    result = compare_variants(base, variant, n_boot=500)
    text = "\n".join(result.lines)
    # Baseline: 80 % each seed; the ensemble errs where both seeds err (none) or ties.
    assert "baseline  accuracy 0.8000 ± 0.0000 (n=2)" in text
    assert "variant   accuracy 0.9800 ± 0.0000 (n=2)" in text
    assert result.bootstrap.observed == pytest.approx(0.18)
    assert result.bootstrap.low > 0
    assert result.mcnemar.only_variant > result.mcnemar.only_baseline
    assert "test is never a decision input" in text
    assert "WARN: 2 baseline run(s); the protocol uses 5 seeds" in text
    assert result.lines[-1] == "compare-variants: REPORT"
    eces = [run_ece(load_run_predictions(path, "test").predictions) for path in base]
    assert f"ECE (15 bins, top label; secondary, per run): baseline {eces[0]:.4f}" in text
    val = compare_variants(base, variant, stage="val", n_boot=10)
    assert "chose each run's checkpoint" in "\n".join(val.lines)
    assert val.bootstrap.observed == 0.0


def test_compare_variants_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = _side(tmp_path, "base", [[0]])
    variant = _side(tmp_path, "var", [[1]])
    argv = ["compare-variants", "--baseline", *map(str, base), "--variant", *map(str, variant)]
    assert main([*argv, "--n-boot", "50"]) == 0
    assert "McNemar (seed ensembles): baseline only 1, variant only 1" in capsys.readouterr().out


def test_mixed_machines_are_refused(tmp_path: Path) -> None:
    base = _side(tmp_path, "base", [[0]])
    gpu = {**MACHINE, "device": "cuda", "gpu": "NVIDIA GeForce GTX 1050", "cuda": "12.6"}
    variant = _side(tmp_path, "var", [[1]], machine=gpu)
    with pytest.raises(ConfigError, match="different machines"):
        compare_variants(base, variant, n_boot=10)
    allowed = compare_variants(base, variant, n_boot=10, allow_mixed_machines=True)
    assert any("WARN: mixed machines" in line for line in allowed.lines)
    unknown = _side(tmp_path, "old", [[1]], machine="?")
    lines = compare_variants(base, unknown, n_boot=10).lines
    assert any("do not record their machine" in line for line in lines)


def test_invalid_sets_are_refused(tmp_path: Path) -> None:
    base = _side(tmp_path, "base", [[0], [1]])
    variant = _side(tmp_path, "var", [[1]])
    with pytest.raises(ConfigError, match="at least one baseline"):
        compare_variants([], variant)
    with pytest.raises(ConfigError, match="repeat a seed"):
        compare_variants([base[0], base[0]], variant)
    other = fake_run(tmp_path, "other", {"test": predictions(LABELS[:50], LABELS[:50])})
    with pytest.raises(ConfigError, match="not paired"):
        compare_variants(base, [other])
    no_preds = tmp_path / "no_preds"
    no_preds.mkdir()
    (no_preds / "metrics.json").write_text(json.dumps({"command": "train-supervised"}))
    with pytest.raises(ConfigError, match="re-run"):
        compare_variants(base, [no_preds])
    # Two sets evaluated on different splits.
    relabelled = _side(tmp_path, "relabel", [[1]])
    metrics = json.loads((relabelled[0] / "metrics.json").read_text())
    metrics["predictions"]["test"]["split"] = "fold 0"
    (relabelled[0] / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(ConfigError, match="different sets"):
        compare_variants(base, relabelled)


def test_runs_on_other_data_are_refused(tmp_path: Path) -> None:
    base = _side(tmp_path, "base", [[0]], data={"source": {"sha256": "a" * 64}})
    variant = _side(tmp_path, "var", [[1]], data={"source": {"sha256": "b" * 64}})
    with pytest.raises(ConfigError, match="different data"):
        compare_variants(base, variant, n_boot=10)
    same = _side(tmp_path, "same", [[1]], data={"source": {"sha256": "a" * 64}})
    assert compare_variants(base, same, n_boot=10).lines[-1] == "compare-variants: REPORT"


def test_load_run_predictions_accepts_the_metrics_file(tmp_path: Path) -> None:
    run = _side(tmp_path, "base", [[0]])[0]
    loaded = load_run_predictions(run / "metrics.json", "test")
    assert loaded.split == "Test" and loaded.seed == 0
    assert loaded.predictions.labels.size == 100


# --- Decision helpers ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def splits(tmp_path_factory: pytest.TempPathFactory) -> ProtocolSplits:
    path = make_layout(tmp_path_factory.mktemp("cmp") / "layout", train=30, validation=10, test=2)
    return make_splits(path, selection_per_class=3, screen_per_class=5)


def _on(index: np.ndarray, labels: np.ndarray, wrong: int) -> Predictions:
    predicted = labels.copy()
    predicted[:wrong] = (predicted[:wrong] + 1) % 5
    return predictions(labels.tolist(), predicted.tolist(), index=index.tolist())


def _pool_labels(splits: ProtocolSplits, index: np.ndarray) -> np.ndarray:
    # make_layout labels: class = position // per_class (30 in Train, 10 in Validation).
    train, val = splits.locate(index)
    return np.concatenate([train // 30, val // 10]).astype(np.int64)


def test_screen_comparison(splits: ProtocolSplits) -> None:
    decision = splits.screen.decision
    labels = (decision // 10).astype(np.int64)
    base = [_on(decision, labels, 0), _on(decision, labels, 1)]
    worse = [_on(decision, labels, 3), _on(decision, labels, 2)]  # 0.88 and 0.92 of 25
    result = screen_comparison(base, worse, splits)
    assert result.baseline == pytest.approx(0.98) and result.variant == pytest.approx(0.90)
    assert result.decision == "drop"
    assert screen_comparison(base, base, splits).decision == "keep"
    full = predictions(LABELS[:50], LABELS[:50])  # all of Validation, not the decision half
    with pytest.raises(ValueError, match="decision half"):
        screen_comparison([full], worse, splits)
    with pytest.raises(ValueError, match="no predictions"):
        screen_comparison([], worse, splits)
    relabelled = [_on(decision, (labels + 1) % 5, 0)]
    with pytest.raises(ValueError, match="labels"):
        screen_comparison(base, relabelled, splits)


def test_cv_comparison(splits: ProtocolSplits) -> None:
    base: dict[int, list[Predictions]] = {}
    better: dict[int, list[Predictions]] = {}
    for fold in splits.folds:
        labels = _pool_labels(splits, fold.evaluation)
        base[fold.fold] = [_on(fold.evaluation, labels, 20), _on(fold.evaluation, labels, 18)]
        better[fold.fold] = [_on(fold.evaluation, labels, 0), _on(fold.evaluation, labels, 1)]
    result = cv_comparison(base, better, splits, n_boot=300)
    assert result.decision == "adopt"
    assert result.interval.metric == "balanced_accuracy"
    assert all(v > b for v, b in zip(result.variant_folds, result.baseline_folds, strict=True))
    assert cv_comparison(better, base, splits, n_boot=300).decision == "worse"
    assert cv_comparison(base, base, splits, n_boot=50).decision == "inconclusive"
    with pytest.raises(ValueError, match="fold 4"):
        cv_comparison({k: v for k, v in base.items() if k != 4}, better, splits)
    shifted = dict(base)
    shifted[0] = [
        _on(splits.folds[1].evaluation, _pool_labels(splits, splits.folds[1].evaluation), 0)
    ]
    with pytest.raises(ValueError, match="fold 0's evaluation part"):
        cv_comparison(shifted, better, splits)
