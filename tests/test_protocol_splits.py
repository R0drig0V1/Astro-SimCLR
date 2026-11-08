"""Tests of the leakage helper, the stored decision splits and ``make-splits``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _protocol_support import N_CLASSES, make_layout

from astro_simclr.cli import main
from astro_simclr.data.prepared import load_prepared
from astro_simclr.protocol.leakage import find_duplicates, train_exclusions, train_keep_indices
from astro_simclr.protocol.splits import (
    SplitsError,
    make_splits,
    read_splits,
    verify_splits,
    write_splits,
)

ROOT = Path(__file__).resolve().parents[1]
#: Sizes small enough for fast tests (the defaults are 100 and 50).
_SMALL: dict[str, Any] = {"selection_per_class": 3, "screen_per_class": 5}


@pytest.fixture(scope="module")
def layout(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """30/10/10 stamps per class; Train 7 duplicates Validation 12 (as in ``red``)."""
    root = tmp_path_factory.mktemp("protocol")
    return make_layout(
        root / "small", train=30, validation=10, test=10, duplicate=(7, "Validation", 12)
    )


# --- Leakage -------------------------------------------------------------------------------


def test_find_duplicates(layout: Path) -> None:
    duplicates = find_duplicates(load_prepared(layout))
    assert [d.as_dict() for d in duplicates] == [
        {
            "split_a": "Train",
            "index_a": 7,
            "split_b": "Validation",
            "index_b": 12,
            "label_a": "AGN",
            "label_b": "SN",
            "features_equal": True,
        }
    ]
    assert train_exclusions(duplicates) == [7]


def test_validation_test_duplicates_are_reported_not_excluded(tmp_path: Path) -> None:
    path = make_layout(tmp_path / "vt", train=2, validation=2, test=2)
    data = load_prepared(path)
    # Rebuild with a Validation stamp copied into Test, then also into Train.
    images = {name: np.array(split.images) for name, split in data.splits.items()}
    images["Test"][3] = images["Validation"][1]
    images["Train"][0] = images["Validation"][1]
    from dataclasses import replace

    data = replace(
        data,
        splits={name: replace(split, images=images[name]) for name, split in data.splits.items()},
    )
    found = {(d.split_a, d.index_a, d.split_b, d.index_b) for d in find_duplicates(data)}
    assert found == {
        ("Train", 0, "Validation", 1),
        ("Train", 0, "Test", 3),
        ("Validation", 1, "Test", 3),
    }
    assert train_exclusions(find_duplicates(data)) == [0]


def test_train_keep_indices() -> None:
    assert train_keep_indices(5, [3, 1, 3]).tolist() == [0, 2, 4]
    assert train_keep_indices(3, []).tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="range"):
        train_keep_indices(3, [3])


# --- Splits --------------------------------------------------------------------------------


def test_make_splits_partitions(layout: Path) -> None:
    splits = make_splits(layout, **_SMALL)
    assert splits.excluded_train == (7,)
    n_pool = splits.n_train + splits.n_validation
    assert (splits.n_train, splits.n_validation) == (150, 50)
    data = load_prepared(layout)
    train_labels, val_labels = data.splits["Train"].labels, data.splits["Validation"].labels
    assert train_labels is not None and val_labels is not None
    labels = np.concatenate([train_labels, val_labels])

    screen = splits.screen
    assert np.intersect1d(screen.selection, screen.decision).size == 0
    for half in (screen.selection, screen.decision):
        assert np.bincount(val_labels[half], minlength=N_CLASSES).tolist() == [5] * N_CLASSES

    evaluation = np.concatenate([fold.evaluation for fold in splits.folds])
    assert sorted(evaluation.tolist()) == [i for i in range(n_pool) if i != 7]
    for fold in splits.folds:
        parts = (fold.evaluation, fold.selection, fold.training)
        assert sum(p.size for p in parts) == n_pool - 1
        assert np.unique(np.concatenate(parts)).size == n_pool - 1  # disjoint
        assert 7 not in np.concatenate(parts)
        assert np.bincount(labels[fold.selection], minlength=N_CLASSES).tolist() == [3] * 5
        # Stratified: every class is split as evenly as possible across folds.
        counts = np.bincount(labels[fold.evaluation], minlength=N_CLASSES)
        assert counts.min() >= 7 and counts.max() <= 8
    train_part, val_part = splits.locate(splits.folds[0].evaluation)
    assert train_part.max() < 150 and val_part.max() < 50
    assert train_part.size + val_part.size == splits.folds[0].evaluation.size
    with pytest.raises(ValueError, match="pool"):
        splits.locate(np.array([n_pool]))


def test_round_trip_and_determinism(layout: Path, tmp_path: Path) -> None:
    splits = make_splits(layout, **_SMALL)
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    write_splits(splits, first)
    write_splits(make_splits(layout, **_SMALL), second)
    assert first.read_bytes() == second.read_bytes()
    loaded = read_splits(first)
    verify_splits(loaded, layout)
    assert loaded.excluded_train == splits.excluded_train
    assert loaded.duplicates == splits.duplicates
    for a, b in zip(loaded.folds, splits.folds, strict=True):
        for part in ("evaluation", "selection", "training"):
            np.testing.assert_array_equal(getattr(a, part), getattr(b, part))
    np.testing.assert_array_equal(loaded.screen.decision, splits.screen.decision)
    # One list of integers per line.
    assert max(len(line) for line in first.read_text().splitlines()) > 100
    assert len(first.read_text().splitlines()) < 75


def test_another_seed_draws_other_splits(layout: Path) -> None:
    a, b = make_splits(layout, **_SMALL), make_splits(layout, seed=1, **_SMALL)
    assert not np.array_equal(a.screen.selection, b.screen.selection)
    assert not np.array_equal(a.folds[0].evaluation, b.folds[0].evaluation)


def test_too_small_classes(layout: Path) -> None:
    with pytest.raises(SplitsError, match="fewer than"):
        make_splits(layout, selection_per_class=3, screen_per_class=6)
    with pytest.raises(SplitsError, match="too few"):
        make_splits(layout, selection_per_class=35, screen_per_class=5)
    with pytest.raises(SplitsError, match="n_folds"):
        make_splits(layout, n_folds=1, **_SMALL)
    with pytest.raises(SplitsError, match="n_folds"):
        make_splits(layout, n_folds=11, **_SMALL)  # fold_of has one digit per stamp
    with pytest.raises(SplitsError, match=">= 1"):
        make_splits(layout, selection_per_class=0, screen_per_class=5)


def test_classes_smaller_than_the_folds(tmp_path: Path) -> None:
    tiny = make_layout(tmp_path / "tiny", train=1, validation=2, test=1)
    with pytest.raises(SplitsError, match="fewer than 5 folds"):
        make_splits(tiny, selection_per_class=1, screen_per_class=1)


def test_verify_rejects_another_dataset(layout: Path, tmp_path: Path) -> None:
    other = make_layout(tmp_path / "other", train=30, validation=10, test=10, seed=1)
    with pytest.raises(SplitsError, match="regenerate"):
        verify_splits(make_splits(layout, **_SMALL), other)


def test_verify_accepts_a_renamed_copy(layout: Path, tmp_path: Path) -> None:
    import shutil

    copy = tmp_path / "moved-elsewhere"
    shutil.copytree(layout, copy)
    verify_splits(make_splits(layout, **_SMALL), copy)


def test_verify_rejects_unstratified_folds(layout: Path, tmp_path: Path) -> None:
    raw = _plain(layout, tmp_path)
    fold_of = list(raw["cv"]["fold_of"])
    selections = raw["cv"]["selection"]
    # Move two class-0 Train stamps of fold 0 into fold 1 (outside every selection set).
    moved = [i for i in range(30) if fold_of[i] == "0" and all(i not in sel for sel in selections)][
        :2
    ]
    for i in moved:
        fold_of[i] = "1"
    raw["cv"]["fold_of"] = "".join(fold_of)
    path = tmp_path / "unstratified.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(SplitsError, match="not stratified"):
        verify_splits(read_splits(path), layout)


def test_verify_rechecks_the_excluded_stamps(layout: Path) -> None:
    from dataclasses import replace

    splits = make_splits(layout, **_SMALL)
    with pytest.raises(SplitsError, match="leakage check"):
        verify_splits(replace(splits, excluded_train=()), layout)


def test_verify_rejects_unbalanced_splits(layout: Path, tmp_path: Path) -> None:
    raw = _plain(layout, tmp_path)
    # Swap one selection stamp for another class's stamp outside the fold.
    selection = raw["cv"]["selection"][0]
    fold_of = raw["cv"]["fold_of"]
    swap = next(
        i
        for i in range(150)
        if fold_of[i] not in ("0", "-") and i not in selection and i // 30 != selection[0] // 30
    )
    raw["cv"]["selection"][0] = sorted([*selection[1:], swap])
    path = tmp_path / "unbalanced.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(SplitsError, match="not balanced"):
        verify_splits(read_splits(path), layout)


def _plain(layout: Path, tmp_path: Path) -> dict[str, Any]:
    path = tmp_path / "plain.json"
    write_splits(make_splits(layout, **_SMALL), path)
    raw: dict[str, Any] = json.loads(path.read_text())
    return raw


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda r: r.update(format="other"), "not a astro-simclr-protocol-splits"),
        (lambda r: r.update(format_version=9), "format_version"),
        (lambda r: r.pop("leakage"), "malformed"),
        (lambda r: r["screen"].update(per_class="5"), "per_class"),
        (lambda r: r["screen"].update(per_class=0, selection=[], decision=[]), "positive"),
        (lambda r: r["screen"].update(per_class=True), "positive"),
        (lambda r: r["cv"].update(n_folds=1, selection=r["cv"]["selection"][:1]), "2-10"),
        (lambda r: r["screen"].update(selection=[1, 0]), "sorted"),
        (lambda r: r["screen"].update(selection=[True]), "integers"),
        (lambda r: r["screen"].update(decision=r["screen"]["selection"]), "overlap"),
        (lambda r: r["screen"].update(selection=r["screen"]["selection"][:-1]), "per_class"),
        (lambda r: r["leakage"].update(excluded_train=[500]), "Train indices"),
        (lambda r: r["cv"].update(n_folds=4), "cv needs"),
        (lambda r: r["cv"].update(fold_of=r["cv"]["fold_of"][:-1]), "fold_of"),
        (lambda r: r["cv"].update(fold_of="9" + r["cv"]["fold_of"][1:]), "fold_of"),
        (lambda r: r["leakage"].update(excluded_train=[]), "excluded"),
        (
            lambda r: r["cv"]["selection"].__setitem__(0, r["cv"]["selection"][0][:-1]),
            "selection\\[0\\]",
        ),
    ],
)
def test_read_rejects_malformed_files(layout: Path, tmp_path: Path, edit: Any, match: str) -> None:
    raw = _plain(layout, tmp_path)
    edit(raw)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(SplitsError, match=match):
        read_splits(path)


def test_read_rejects_unreadable_files(tmp_path: Path) -> None:
    with pytest.raises(SplitsError, match="cannot read"):
        read_splits(tmp_path / "missing.json")
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(SplitsError, match="cannot read"):
        read_splits(tmp_path / "bad.json")


def test_make_splits_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # The CLI uses the protocol's sizes: 100 per class to select, 50 + 50 to screen.
    path = make_layout(
        tmp_path / "big", train=130, validation=100, test=2, duplicate=(0, "Test", 1)
    )
    out = tmp_path / "splits.json"
    assert main(["make-splits", str(path), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "Train 0 (AGN) = Test 1 (AGN)" in printed
    assert "excluded from Train: [0]" in printed
    verify_splits(read_splits(out), path)
    assert main(["make-splits", str(path), "--out", str(out)]) == 2  # exists
    assert main(["make-splits", str(path), "--out", str(out), "--overwrite"]) == 0


def test_stored_red_splits_are_well_formed() -> None:
    """The committed splits of ``red`` (generated on the real data) read back cleanly."""
    splits = read_splits(ROOT / "configs" / "splits" / "td_ztf_stamp_17_06_20_red-002.json")
    assert splits.excluded_train == (2995,)
    assert splits.duplicates[0]["label_a"] == "Bogus" and splits.duplicates[0]["label_b"] == "SN"
    assert (splits.n_train, splits.n_validation) == (50582, 500)
    assert splits.screen.selection.size == splits.screen.decision.size == 250
    assert [fold.selection.size for fold in splits.folds] == [500] * 5
    assert sum(fold.evaluation.size for fold in splits.folds) == 50582 + 500 - 1
