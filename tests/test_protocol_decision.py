"""Tests of ``compare-variants --splits FILE``: the CV and screen decisions on split runs.

The runs are hand-made (``_protocol_support.fake_split_run``): predictions on each fold's
evaluation part (pool indices) or on the screen's decision half (Validation indices) of a
small stored splits file, with a chosen number of errors. Training split runs for real is
tested in ``test_split_runs.py``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _protocol_support import fake_split_run, make_layout, predictions

from astro_simclr.cli import main
from astro_simclr.config import ConfigError
from astro_simclr.protocol.compare import compare_decision, compare_variants
from astro_simclr.protocol.predictions import Predictions, save_predictions, seed_mean_correct
from astro_simclr.protocol.split_runs import file_sha256
from astro_simclr.protocol.splits import ProtocolSplits, make_splits, write_splits
from astro_simclr.protocol.stats import paired_bootstrap

#: make_layout labels: class = position // per_class (30 per class in Train, 10 in Validation).
TRAIN_PER_CLASS, VAL_PER_CLASS = 30, 10
#: Errors per fold of each seed: (baseline, variant).
Errors = Callable[[int, int], int]


@pytest.fixture(scope="module")
def stored(tmp_path_factory: pytest.TempPathFactory) -> tuple[ProtocolSplits, Path]:
    root = tmp_path_factory.mktemp("decision")
    layout = make_layout(root / "layout", train=TRAIN_PER_CLASS, validation=VAL_PER_CLASS, test=2)
    splits = make_splits(layout, selection_per_class=3, screen_per_class=5)
    path = root / "splits.json"
    write_splits(splits, path)
    return splits, path


def _labels(splits: ProtocolSplits, pool: np.ndarray) -> np.ndarray:
    train, val = splits.locate(pool)
    return np.concatenate([train // TRAIN_PER_CLASS, val // VAL_PER_CLASS]).astype(np.int64)


def _wrong_at(index: np.ndarray, labels: np.ndarray, wrong: np.ndarray) -> Predictions:
    predicted = labels.copy()
    predicted[wrong] = (predicted[wrong] + 1) % 5
    return predictions(labels.tolist(), predicted.tolist(), index=index.tolist())


def _on(index: np.ndarray, labels: np.ndarray, wrong: int, offset: int = 0) -> Predictions:
    """Predictions with ``wrong`` errors, at positions ``offset, offset + 1, ...``."""
    return _wrong_at(index, labels, (np.arange(wrong) + offset) % index.size)


def _cv_side(
    root: Path,
    name: str,
    stored: tuple[ProtocolSplits, Path],
    errors: Callable[[int, int], int],
    *,
    seeds: tuple[int, ...] = (0, 1),
    folds: tuple[int, ...] | None = None,
    machine: dict[str, Any] | None = None,
    source: str = "0" * 64,
) -> list[Path]:
    splits, path = stored
    sha = file_sha256(path)
    runs = []
    for fold in splits.folds:
        if folds is not None and fold.fold not in folds:
            continue
        labels = _labels(splits, fold.evaluation)
        for seed in seeds:
            pred = _on(fold.evaluation, labels, errors(fold.fold, seed), offset=seed)
            runs.append(
                fake_split_run(
                    root,
                    f"{name}/fold{fold.fold}-seed{seed}",
                    pred,
                    sha256=sha,
                    seed=seed,
                    fold=fold.fold,
                    machine=machine,
                    source=source,
                )
            )
    return runs


def _screen_side(
    root: Path, name: str, stored: tuple[ProtocolSplits, Path], wrong: list[int]
) -> list[Path]:
    splits, path = stored
    decision = splits.screen.decision
    labels = (decision // VAL_PER_CLASS).astype(np.int64)
    return [
        fake_split_run(
            root,
            f"{name}/seed{seed}",
            _on(decision, labels, n),
            sha256=file_sha256(path),
            seed=seed,
        )
        for seed, n in enumerate(wrong)
    ]


def _argv(base: list[Path], variant: list[Path], path: Path, *extra: str) -> list[str]:
    return [
        "compare-variants",
        "--splits",
        str(path),
        "--baseline",
        *map(str, base),
        "--variant",
        *map(str, variant),
        "--n-boot",
        "300",
        *extra,
    ]


def test_cv_superiority_adopts_and_reports_per_fold(
    tmp_path: Path, stored: tuple[ProtocolSplits, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    splits, path = stored
    base = _cv_side(tmp_path, "base", stored, lambda fold, seed: 12)
    better = _cv_side(tmp_path, "var", stored, lambda fold, seed: seed)
    result = compare_decision(base, better, path, n_boot=300)
    assert result.mode == "fold" and result.verdict == "ADOPT" and result.passed
    assert result.cv is not None and result.cv.rule == "superiority"
    assert len(result.cv.baseline_folds) == len(splits.folds)
    assert all(
        v > b for v, b in zip(result.cv.variant_folds, result.cv.baseline_folds, strict=True)
    )
    text = "\n".join(result.lines)
    for fold in splits.folds:
        assert f"fold {fold.fold}: baseline" in text
    assert "paired by pool index" in text and "McNemar" in text
    assert main(_argv(base, better, path)) == 0
    assert "compare-variants: ADOPT" in capsys.readouterr().out
    # The reverse is worse: not adopted, exit 1.
    assert main(_argv(better, base, path)) == 1
    assert "compare-variants: NOT ADOPTED (worse)" in capsys.readouterr().out


def test_both_rules_on_an_equal_variant(
    tmp_path: Path, stored: tuple[ProtocolSplits, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    _, path = stored
    base = _cv_side(tmp_path, "base", stored, lambda fold, seed: 3 + seed)
    same = _cv_side(tmp_path, "same", stored, lambda fold, seed: 3 + seed)
    # Equal is not better: superiority cannot adopt it ...
    superiority = compare_decision(base, same, path, n_boot=300)
    assert superiority.cv is not None and superiority.cv.decision == "inconclusive"
    assert not superiority.passed
    # ... non-inferiority can (the efficiency rows).
    ni = compare_decision(base, same, path, rule="non-inferiority", n_boot=300)
    assert ni.cv is not None and ni.cv.decision == "adopt" and ni.cv.margin == 0.005
    assert ni.verdict == "ADOPT"
    assert main(_argv(base, same, path, "--rule", "non-inferiority")) == 0
    out = capsys.readouterr().out
    assert "non-inferiority, margin 0.0050 (0.50 points" in out
    # A clearly worse variant is "worse" under non-inferiority too.
    worse = _cv_side(tmp_path, "worse", stored, lambda fold, seed: 20)
    ni_worse = compare_decision(base, worse, path, rule="non-inferiority", n_boot=300)
    assert ni_worse.cv is not None and ni_worse.cv.decision == "worse"
    assert main(_argv(base, worse, path, "--rule", "non-inferiority", "--margin", "0.01")) == 1


def test_bootstrap_is_paired_by_pool_index(
    tmp_path: Path, stored: tuple[ProtocolSplits, Path]
) -> None:
    splits, path = stored
    base = _cv_side(tmp_path, "base", stored, lambda fold, seed: 5 + 2 * seed)
    same = _cv_side(tmp_path, "same", stored, lambda fold, seed: 5 + 2 * seed)
    # Paired: an identical variant differs by exactly 0 in every resample, although the
    # per-stamp scores vary (an unpaired resample would give a non-zero width).
    result = compare_decision(base, same, path, n_boot=300)
    assert result.cv is not None
    interval = result.cv.interval
    assert (interval.low, interval.observed, interval.high) == (0.0, 0.0, 0.0)
    # The interval is the bootstrap of the folds' seed-mean scores concatenated in fold
    # order, the same stamp at the same position on both sides.
    better = _cv_side(tmp_path, "better", stored, lambda fold, seed: seed)
    got = compare_decision(base, better, path, n_boot=300, seed=3).cv
    assert got is not None

    def scores(errors: Errors) -> np.ndarray:
        parts = []
        for fold in splits.folds:
            labels = _labels(splits, fold.evaluation)
            runs = [_on(fold.evaluation, labels, errors(fold.fold, s), offset=s) for s in (0, 1)]
            parts.append(seed_mean_correct(runs))
        return np.concatenate(parts)

    labels = np.concatenate([_labels(splits, fold.evaluation) for fold in splits.folds])
    expected = paired_bootstrap(
        scores(lambda f, s: 5 + 2 * s),
        scores(lambda f, s: s),
        labels,
        metric="balanced_accuracy",
        n_boot=300,
        seed=3,
    )
    assert got.interval == expected
    # A run whose stamps are in another order cannot be paired by position: refused.
    fold = splits.folds[0]
    labels0 = _labels(splits, fold.evaluation)
    order = np.random.default_rng(0).permutation(fold.evaluation.size)
    shuffled = _wrong_at(fold.evaluation[order], labels0[order], np.arange(2))
    save_predictions(shuffled, base[0] / "predictions_evaluation.npz")
    with pytest.raises(ConfigError, match="fold 0's evaluation part"):
        compare_decision(base, better, path, n_boot=50)


def test_sides_must_cover_the_same_folds_and_seeds(
    tmp_path: Path, stored: tuple[ProtocolSplits, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    _, path = stored
    base = _cv_side(tmp_path, "base", stored, lambda fold, seed: 4)
    variant = _cv_side(tmp_path, "var", stored, lambda fold, seed: 2)
    missing_fold = _cv_side(tmp_path, "nofold", stored, lambda f, s: 2, folds=(0, 1, 2, 3))
    with pytest.raises(ConfigError, match=r"miss fold\(s\) \[4\]"):
        compare_decision(base, missing_fold, path)
    one_seed = _cv_side(tmp_path, "oneseed", stored, lambda f, s: 2, seeds=(0,))
    with pytest.raises(
        ConfigError, match="in the baseline runs but not in the variant runs: fold 0 seed 1"
    ):
        compare_decision(base, one_seed, path)
    other_seeds = _cv_side(tmp_path, "seeds23", stored, lambda f, s: 2, seeds=(2, 3))
    with pytest.raises(ConfigError, match="same folds and seeds"):
        compare_decision(base, other_seeds, path)
    with pytest.raises(ConfigError, match="repeat fold 0 seed 0"):
        compare_decision([*base, base[0]], variant, path)
    assert main(_argv(base, missing_fold, path)) == 2
    assert "miss fold(s) [4]" in capsys.readouterr().err
    # Every fold needs the same seeds: fold 0 with one seed, the others with two.
    fewer = [run for run in base if run.name != "fold0-seed1"]
    fewer_var = [run for run in variant if run.name != "fold0-seed1"]
    with pytest.raises(ConfigError, match="different numbers of seeds"):
        compare_decision(fewer, fewer_var, path)
    # One seed per fold on both sides is allowed, with a warning (the plan uses 2).
    single = _cv_side(tmp_path, "single", stored, lambda f, s: 4, seeds=(0,))
    result = compare_decision(single, one_seed, path, n_boot=50)
    assert "WARN: fold 0 has 1 seed(s); the plan uses 2" in "\n".join(result.lines)


def test_runs_must_belong_to_the_splits_file(
    tmp_path: Path, stored: tuple[ProtocolSplits, Path]
) -> None:
    splits, path = stored
    base = _cv_side(tmp_path, "base", stored, lambda fold, seed: 4)
    variant = _cv_side(tmp_path, "var", stored, lambda fold, seed: 2)
    edited = tmp_path / "edited.json"
    edited.write_text(path.read_text() + "\n")  # same splits, another file: another SHA-256
    with pytest.raises(ConfigError, match="trained on another splits file"):
        compare_decision(base, variant, edited)
    other_data = _cv_side(tmp_path, "data", stored, lambda f, s: 2, source="f" * 64)
    with pytest.raises(ConfigError, match="different data"):
        compare_decision(base, other_data, path)
    gpu = {"device": "cuda", "gpu": "GTX 1050", "torch": "2.14.0", "cuda": "12.6"}
    other_machine = _cv_side(tmp_path, "gpu", stored, lambda f, s: 2, machine=gpu)
    with pytest.raises(ConfigError, match="different machines"):
        compare_decision(base, other_machine, path)
    mixed = compare_decision(base, other_machine, path, allow_mixed_machines=True, n_boot=50)
    assert any("WARN: mixed machines" in line for line in mixed.lines)
    screen = _screen_side(tmp_path, "screen", stored, [0, 1])
    with pytest.raises(ConfigError, match="mix CV folds and the screen"):
        compare_decision(base, screen, path)
    normal = tmp_path / "normal"
    normal.mkdir()
    (normal / "metrics.json").write_text(json.dumps({"seed": 0, "predictions": {}}))
    with pytest.raises(ConfigError, match="not a --splits run"):
        compare_decision([normal], variant, path)
    # A split run in the report mode points to --splits.
    with pytest.raises(ConfigError, match="compare-variants --splits FILE"):
        compare_variants(base, variant)
    assert splits.folds  # the fixture is a CV file


def test_screen_only_drops(
    tmp_path: Path, stored: tuple[ProtocolSplits, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    _, path = stored
    base = _screen_side(tmp_path, "base", stored, [0, 1])
    much_worse = _screen_side(tmp_path, "worse", stored, [3, 2])  # 0.88 and 0.92 of 25
    equal = _screen_side(tmp_path, "equal", stored, [1, 0])
    dropped = compare_decision(base, much_worse, path)
    assert dropped.mode == "screen" and dropped.verdict == "DROP" and not dropped.passed
    assert dropped.screen is not None and dropped.screen.variant == pytest.approx(0.90)
    kept = compare_decision(base, equal, path)
    assert kept.verdict == "KEEP" and kept.passed
    assert "the screen never adopts" in "\n".join(kept.lines)
    assert main(_argv(base, much_worse, path)) == 1
    assert "compare-variants: DROP" in capsys.readouterr().out
    assert main(_argv(base, equal, path)) == 0
    with pytest.raises(ConfigError, match="drop-only"):
        compare_decision(base, equal, path, rule="non-inferiority")
    with pytest.raises(
        ConfigError, match="in the baseline runs but not in the variant runs: seed 1"
    ):
        compare_decision(base, equal[:1], path)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--split", "val"], "--split is for normal runs"),
        (["--margin", "0.01"], "--margin needs --rule non-inferiority"),
        (["--rule", "non-inferiority", "--margin", "0.5"], "fraction of balanced accuracy"),
    ],
)
def test_decision_flag_errors(
    tmp_path: Path,
    stored: tuple[ProtocolSplits, Path],
    capsys: pytest.CaptureFixture[str],
    extra: list[str],
    message: str,
) -> None:
    _, path = stored
    base = _cv_side(tmp_path, "base", stored, lambda fold, seed: 4)
    assert main(_argv(base, base, path, *extra)) == 2
    assert message in capsys.readouterr().err


def test_rule_flags_need_splits(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = str(tmp_path)
    for extra in (["--rule", "superiority"], ["--margin", "0.01"]):
        argv = ["compare-variants", "--baseline", run, "--variant", run, *extra]
        assert main(argv) == 2
        assert "needs --splits FILE" in capsys.readouterr().err


def test_malformed_split_runs_are_refused(
    tmp_path: Path, stored: tuple[ProtocolSplits, Path]
) -> None:
    splits, path = stored
    base = _screen_side(tmp_path, "base", stored, [0, 1])
    # A screen run evaluated on the selection half instead of the decision half.
    selection = splits.screen.selection
    wrong_half = fake_split_run(
        tmp_path,
        "selection/seed0",
        _on(selection, (selection // VAL_PER_CLASS).astype(np.int64), 0),
        sha256=file_sha256(path),
        seed=0,
    )
    other = _screen_side(tmp_path, "other", stored, [0, 0])
    with pytest.raises(ConfigError, match=r"cannot be compared.*decision half"):
        compare_decision(base, [wrong_half, other[1]], path)
    metrics = json.loads((other[0] / "metrics.json").read_text())
    metrics["protocol"]["mode"] = "bootstrap"
    (other[0] / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(ConfigError, match="unknown split-run mode 'bootstrap'"):
        compare_decision(base, other, path)
    with pytest.raises(ConfigError, match="at least one variant run"):
        compare_decision(base, [], path)
