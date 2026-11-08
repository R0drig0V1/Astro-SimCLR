"""Tests of the phase-5 label budgets: the draw, its guarantees, the file and its verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _bench_support import synthetic_table
from _protocol_support import N_CLASSES, make_layout

from astro_simclr.cli import main
from astro_simclr.data.prepared import load_prepared
from astro_simclr.data.schema import LABEL_NAMES
from astro_simclr.protocol.alerts import write_alert_table
from astro_simclr.protocol.label_budgets import (
    BudgetError,
    BudgetPart,
    LabelBudgets,
    cv_parts,
    cv_parts_from_files,
    make_label_budgets,
    object_split_parts,
    object_split_parts_from_files,
    read_label_budgets,
    resolve,
    verify_label_budgets,
    write_label_budgets,
)
from astro_simclr.protocol.object_splits import make_object_splits, write_object_splits
from astro_simclr.protocol.splits import make_splits, read_splits, write_splits

ROOT = Path(__file__).resolve().parents[1]
ARCSEC = 1 / 3600
NAMES = ("A", "B", "C")
SOURCE: dict[str, Any] = {
    "kind": "cv-splits",
    "file": "parent.json",
    "sha256": "0" * 64,
    "fingerprint": {"dataset": {"n": 1}},
}


def _part(name: str = "fold0", *, seed: int = 0, per_class: int = 40) -> BudgetPart:
    """A part with multi-alert objects and one object holding alerts of two classes."""
    rng = np.random.default_rng(seed)
    objects, labels = [], []
    for c in range(len(NAMES)):
        for k in range(per_class):
            alerts = 1 + int(rng.integers(0, 3))
            objects += [1000 * c + k] * alerts
            labels += [c] * alerts
    objects += [7, 7]  # object 7 (class A's) also has alerts of class C
    labels += [2, 2]
    order = rng.permutation(len(objects))
    rows = np.sort(rng.choice(10 * len(objects), size=len(objects), replace=False))
    return BudgetPart(
        name,
        rows.astype(np.int64),
        np.asarray(objects)[order],
        np.asarray(labels, dtype=np.int64)[order],
        rows[:5].astype(np.int64),
    )


def _make(parts: list[BudgetPart], **kwargs: Any) -> LabelBudgets:
    settings: dict[str, Any] = {"budgets": (2, 5, 20), "n_repeats": 4, **kwargs}
    return make_label_budgets(
        parts, source=SOURCE, label_names=NAMES, object_id="test ids", **settings
    )


def _class_objects(part: BudgetPart, rows: np.ndarray, c: int) -> set[int]:
    member = np.isin(part.rows, rows) & (part.labels == c)
    return set(part.objects[member].tolist())


def test_subsets_hold_exact_nested_budgets() -> None:
    parts = [_part("fold0"), _part("fold1", seed=1)]
    budgets = _make(parts)
    assert budgets.selection_per_class == {2: 0, 5: 1, 20: 4}
    for part in parts:
        subsets = resolve(budgets, part)
        assert len(subsets) == 3 * 4 + 1
        by_key = {(s.budget, s.repeat): s for s in subsets}
        for repeat in range(4):
            previous: list[set[int]] = [set()] * len(NAMES)
            for budget in (2, 5, 20):
                subset = by_key[(budget, repeat)]
                for c in range(len(NAMES)):
                    held = _class_objects(part, subset.labelled, c)
                    assert len(held) == budget
                    assert previous[c] <= held  # nested
                    previous[c] = held
                    # Every alert of class c of a held object is labelled, none of another class.
                    member = np.isin(part.objects, list(held)) & (part.labels == c)
                    assert set(part.rows[member]) <= set(subset.labelled)
                # Selection: the stored size per class, no object of the subset.
                subset_objects = set(part.objects[np.isin(part.rows, subset.labelled)])
                for c in range(len(NAMES)):
                    chosen = _class_objects(part, subset.selection, c)
                    assert len(chosen) == budgets.selection_per_class[budget]
                    assert not chosen & subset_objects
        full = by_key[(None, 0)]
        assert np.array_equal(full.labelled, part.rows)
        assert np.array_equal(full.selection, part.selection)
    # The repeats and the parts differ.
    first = budgets.objects[0]
    assert first[0] != first[1]
    assert budgets.objects[0][0] != budgets.objects[1][0]


def test_two_class_object_counts_once_in_each_class() -> None:
    part = _part(per_class=8)
    budgets = _make([part], budgets=(8,), selection_fraction=0.0)
    labelled = resolve(budgets, part)[0].labelled
    # Every object is taken at the full class size, so object 7 is labelled in A and in C.
    assert {7} <= _class_objects(part, labelled, 0)
    assert 7 in _class_objects(part, labelled, 2)
    assert len(_class_objects(part, labelled, 2)) == 8


def test_selection_size_is_capped_by_the_smallest_remainder() -> None:
    part = _part(per_class=22)
    budgets = _make([part], budgets=(20,), selection_fraction=0.2)
    # floor(0.2 x 20) = 4, but some class has 22 - 20 = 2 objects left (or 1 for class C,
    # which holds object 7 when A's subset holds it too).
    assert budgets.selection_per_class[20] in (1, 2)
    left = min(len({o for o in part.objects[part.labels == c]}) - 20 for c in range(len(NAMES)))
    assert budgets.selection_per_class[20] <= left


def test_deterministic_and_seeded() -> None:
    parts = [_part()]
    assert _make(parts) == _make(parts)
    assert _make(parts, seed=1).objects != _make(parts).objects


def test_round_trip_and_verify(tmp_path: Path) -> None:
    parts = [_part("fold0"), _part("fold1", seed=1)]
    budgets = _make(parts)
    path = tmp_path / "budgets.json"
    write_label_budgets(budgets, path)
    again = read_label_budgets(path)
    assert again == budgets
    verify_label_budgets(again, parts, source=SOURCE)
    # Each id list is on one line.
    assert len(path.read_text().splitlines()) < 400


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"budgets": (5, 2)}, "increasing"),
        ({"budgets": ()}, "increasing"),
        ({"budgets": (0, 5)}, "increasing"),
        ({"n_repeats": 0}, "n_repeats"),
        ({"selection_fraction": 0.3}, "at most 20"),
        ({"selection_fraction": -0.1}, "at most 20"),
        ({"budgets": (50,)}, "fewer than the largest budget 50"),
        ({"budgets": (40,)}, "no object left"),
    ],
)
def test_draw_refuses_bad_settings(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(BudgetError, match=match):
        _make([_part()], **kwargs)


def test_draw_refuses_bad_parts() -> None:
    part = _part()
    with pytest.raises(BudgetError, match="no training part"):
        _make([])
    with pytest.raises(BudgetError, match="names repeat"):
        _make([part, part])
    short = BudgetPart("x", part.rows, part.objects[:-1], part.labels, part.selection)
    with pytest.raises(BudgetError, match="differ in length"):
        _make([short])
    wrong = BudgetPart("x", part.rows, part.objects, part.labels + 5, part.selection)
    with pytest.raises(BudgetError, match="labels must be in range"):
        _make([wrong])
    rows = part.rows.copy()
    rows[1] = rows[0]
    with pytest.raises(BudgetError, match="rows repeat"):
        _make([BudgetPart("x", rows, part.objects, part.labels, part.selection)])


def _plain(tmp_path: Path, parts: list[BudgetPart]) -> dict[str, Any]:
    path = tmp_path / "b.json"
    write_label_budgets(_make(parts), path)
    raw: dict[str, Any] = json.loads(path.read_text())
    return raw


def _write(tmp_path: Path, raw: dict[str, Any]) -> Path:
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(raw))
    return path


def _a(raw: dict[str, Any]) -> dict[str, Any]:
    entry: dict[str, Any] = raw["parts"][0]["repeats"][0]["A"]
    return entry


def _swap_in_a_class_b_object(raw: dict[str, Any]) -> None:
    _a(raw)["objects"][0] = raw["parts"][0]["repeats"][0]["B"]["objects"][-1]


def _selection_in_subset(raw: dict[str, Any]) -> None:
    _a(raw)["selection"]["5"][0] = _a(raw)["objects"][0]


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (_swap_in_a_class_b_object, "not in the part"),
        (lambda raw: _a(raw)["objects"].pop(), "does not hold 20 objects"),
        (lambda raw: _a(raw)["objects"].__setitem__(1, _a(raw)["objects"][0]), "20 objects"),
        (_selection_in_subset, "is in the subset"),
        (lambda raw: _a(raw)["selection"]["20"].pop(), "does not hold 4 objects"),
        (lambda raw: raw["source"].__setitem__("sha256", "1" * 64), "not from this parent"),
        (lambda raw: raw["source"].__setitem__("fingerprint", {}), "not from this parent"),
        (lambda raw: raw["parts"][0].__setitem__("name", "other"), "parts"),
    ],
)
def test_verify_refuses_other_or_broken_draws(tmp_path: Path, change: Any, match: str) -> None:
    parts = [_part("fold0"), _part("fold1", seed=1)]
    raw = _plain(tmp_path, parts)
    change(raw)
    with pytest.raises(BudgetError, match=match):
        verify_label_budgets(read_label_budgets(_write(tmp_path, raw)), parts, source=SOURCE)


def _drop(key: str) -> Any:
    return lambda raw: raw.pop(key)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda raw: raw.__setitem__("format", "x"), "not a astro-simclr-label-budgets"),
        (lambda raw: raw.__setitem__("format_version", 9), "unsupported format_version"),
        (lambda raw: raw["source"].__setitem__("kind", "x"), "source must be"),
        (lambda raw: raw.__setitem__("label_names", ["A"]), "label_names"),
        (lambda raw: raw.__setitem__("label_names", ["A", "A", "B"]), "label_names"),
        (lambda raw: raw.__setitem__("budgets", "x"), "budgets must be a list"),
        (lambda raw: raw.__setitem__("budgets", [True]), "a budget must be"),
        (lambda raw: raw.__setitem__("n_repeats", 0), "n_repeats must be"),
        (lambda raw: raw.__setitem__("selection_fraction", "x"), "must be a number"),
        (lambda raw: raw.__setitem__("selection_per_class", {}), "one size per budget"),
        (lambda raw: raw["selection_per_class"].__setitem__("5", -1), "non-negative"),
        (lambda raw: raw["selection_per_class"].__setitem__("5", 3), "exceeds the fraction"),
        (lambda raw: raw.__setitem__("seed", "x"), "seed must be"),
        (lambda raw: raw.__setitem__("object_id", 3), "object_id must be"),
        (lambda raw: raw.__setitem__("parts", []), "parts must be"),
        (lambda raw: raw["parts"][0].pop("name"), "needs a name"),
        (lambda raw: raw["parts"][0]["repeats"].pop(), "must hold n_repeats"),
        (lambda raw: raw["parts"][0]["repeats"][0].pop("A"), "one entry per class"),
        (lambda raw: _a(raw).pop("selection"), "needs objects and selection"),
        (lambda raw: _a(raw)["selection"].pop("5"), "one list per budget"),
        (lambda raw: _a(raw).__setitem__("objects", [1.5]), "list of object ids"),
        (lambda raw: raw["parts"][1].__setitem__("name", "fold0"), "names repeat"),
    ],
)
def test_read_refuses_malformed_files(tmp_path: Path, change: Any, match: str) -> None:
    raw = _plain(tmp_path, [_part("fold0"), _part("fold1", seed=1)])
    change(raw)
    with pytest.raises(BudgetError, match=match):
        read_label_budgets(_write(tmp_path, raw))


def test_read_unreadable(tmp_path: Path) -> None:
    with pytest.raises(BudgetError, match="cannot read"):
        read_label_budgets(tmp_path / "missing.json")
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(BudgetError, match="cannot read"):
        read_label_budgets(tmp_path / "bad.json")


def test_part_index_names_the_missing_part() -> None:
    budgets = _make([_part()])
    with pytest.raises(KeyError, match="no part 'x'"):
        budgets.part_index("x")


# --- The CD21 parts (stored CV splits) -----------------------------------------------------


@pytest.fixture(scope="module")
def layout(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("budgets")
    # Train stamps 0-2 (class 0) are one object, 3 and 4 (class 0) another.
    positions = {"Train": {0: (40.0, 10.0), 1: (40.0, 10.0 + 0.5 * ARCSEC), 2: (40.0, 10.0)}}
    positions["Train"].update({3: (90.0, 5.0), 4: (90.0, 5.0)})
    return make_layout(root / "data", train=40, validation=10, test=5, positions=positions)


def _pool_labels(path: Path) -> np.ndarray:
    data = load_prepared(path)
    train, val = data.splits["Train"].labels, data.splits["Validation"].labels
    assert train is not None and val is not None
    return np.concatenate([train, val]).astype(np.int64)


def test_cv_parts_follow_the_folds_and_the_groups(layout: Path) -> None:
    splits = make_splits(layout, selection_per_class=3, screen_per_class=2)
    labels = _pool_labels(layout)
    parts = cv_parts(splits, labels)
    assert [p.name for p in parts] == [f"fold{k}" for k in range(5)]
    for part, fold in zip(parts, splits.folds, strict=True):
        assert np.array_equal(part.rows, fold.training)
        assert np.array_equal(part.selection, fold.selection)
        assert np.array_equal(part.labels, labels[fold.training])
    # A group's stamps share their smallest pool index in the training part as object id.
    grouping = splits.grouping
    assert grouping is not None
    for part in parts:
        for row, obj in zip(part.rows.tolist(), part.objects.tolist(), strict=True):
            members = np.flatnonzero(grouping.group_of == grouping.group_of[row])
            assert obj == int(np.intersect1d(members, part.rows)[0])
    assert any(np.unique(p.objects).size < p.rows.size for p in parts)  # a multi-stamp group
    with pytest.raises(BudgetError, match="pool labels"):
        cv_parts(splits, labels[:-1])


def test_cv_parts_of_a_version_1_file_are_stamps(layout: Path, tmp_path: Path) -> None:
    splits = make_splits(layout, selection_per_class=3, screen_per_class=2)
    ungrouped = type(splits)(**{**_fields(splits), "grouping": None})
    parts = cv_parts(ungrouped, _pool_labels(layout))
    assert np.array_equal(parts[0].objects, parts[0].rows)


def _fields(obj: Any) -> dict[str, Any]:
    return {name: getattr(obj, name) for name in obj.__slots__}


def test_cli_make_and_verify_on_cv_splits(
    layout: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    splits_path = tmp_path / "splits.json"
    write_splits(make_splits(layout, selection_per_class=3, screen_per_class=2), splits_path)
    out = tmp_path / "budgets.json"
    args = ["--splits", str(splits_path), "--data", str(layout)]
    make = ["make-label-budgets", "--out", str(out), *args, "--budgets", "1", "4"]
    assert main([*make, "--repeats", "3"]) == 0
    assert "selection objects per class (1: 0, 4: 0)" in capsys.readouterr().out
    assert main(["verify-label-budgets", str(out), *args]) == 0
    assert "OK (5 parts)" in capsys.readouterr().out
    # The file names the parent's identity and is refused on another parent.
    budgets = read_label_budgets(out)
    assert budgets.source["fingerprint"]["dataset"] == {
        key: read_splits(splits_path).dataset[key]
        for key in ("source_sha256", "labels_sha256", "n")
    }
    assert budgets.object_id == (
        "the smallest pool index of the object's group in the training part"
    )
    found = cv_parts_from_files(splits_path, layout)
    verify_label_budgets(budgets, found.parts, source=found.source)
    splits_path.write_text(splits_path.read_text().replace('"seed": 0', '"seed": 0 '))
    assert main(["verify-label-budgets", str(out), *args]) == 1
    assert "not from this parent" in capsys.readouterr().err
    # Existing output, and incomplete parent options.
    assert main(make) == 2
    assert main(["make-label-budgets", "--out", str(tmp_path / "x.json")]) == 2
    assert main(["verify-label-budgets", str(out), "--splits", str(splits_path)]) == 2


def test_infeasible_budget_names_the_class(layout: Path, tmp_path: Path) -> None:
    splits = make_splits(layout, selection_per_class=3, screen_per_class=2)
    parts = cv_parts(splits, _pool_labels(layout))
    with pytest.raises(BudgetError, match=f"class {LABEL_NAMES[0]} has"):
        make_label_budgets(
            parts, source=SOURCE, label_names=LABEL_NAMES, object_id="x", budgets=(100,)
        )


# --- The R5 part (stored object splits) ----------------------------------------------------


def test_object_split_parts_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    table = synthetic_table(600, seed=3)
    splits = make_object_splits(table, seed=0)
    parts = object_split_parts(splits, table)
    assert [p.name for p in parts] == ["Train"]
    (part,) = parts
    assert np.array_equal(part.rows, splits.rows(table, "Train"))
    assert np.array_equal(part.selection, splits.rows(table, "selection"))
    assert set(part.objects.tolist()) == set(splits.groups["Train"])
    budgets = make_label_budgets(
        parts,
        source=SOURCE,
        label_names=table.label_names,
        object_id="oid",
        budgets=(2, 5),
        n_repeats=3,
    )
    for subset in resolve(budgets, part):
        if subset.budget is None:
            continue
        held = table.oid[subset.labelled]
        for c in range(len(table.label_names)):
            assert np.unique(held[table.label[subset.labelled] == c]).size == subset.budget

    alerts, splits_path = tmp_path / "alerts.csv", tmp_path / "osplits.json"
    write_alert_table(table, alerts)
    write_object_splits(splits, splits_path)
    out = tmp_path / "r5-budgets.json"
    args = ["--object-splits", str(splits_path), "--alerts", str(alerts)]
    assert main(["make-label-budgets", "--out", str(out), *args, "--budgets", "3"]) == 0
    assert "parts: Train" in capsys.readouterr().out
    assert main(["verify-label-budgets", str(out), *args]) == 0
    found = object_split_parts_from_files(splits_path, alerts, table.label_names)
    assert found.object_id == "the alert table's oid"
    assert read_label_budgets(out).source["fingerprint"] == {"table_sha256": table.sha256}
    assert main(["verify-label-budgets", str(out), *args, "--data", str(tmp_path)]) == 2


# --- The stored CD21 file ------------------------------------------------------------------


def test_stored_cd21_file_is_well_formed() -> None:
    path = ROOT / "configs/label-budgets/td_ztf_stamp_17_06_20_red-002-grouped.json"
    budgets = read_label_budgets(path)
    splits_path = ROOT / "configs/splits/td_ztf_stamp_17_06_20_red-002-grouped.json"
    assert (
        budgets.source["sha256"]
        == __import__("hashlib").sha256(splits_path.read_bytes()).hexdigest()
    )
    assert budgets.part_names == tuple(f"fold{k}" for k in range(5))
    assert budgets.budgets == (10, 100, 1000)
    assert budgets.n_repeats == 20
    assert budgets.selection_per_class == {10: 2, 100: 20, 1000: 83}
    assert budgets.label_names == LABEL_NAMES
    assert len(budgets.label_names) == N_CLASSES
    # Without the data (not versioned), check what the parent splits file alone fixes: every
    # object is a stamp of the fold's training part, and no object is in two roles.
    splits = read_splits(splits_path)
    for p, fold in enumerate(splits.folds):
        training = set(fold.training.tolist())
        for repeat in range(budgets.n_repeats):
            subset = {o for ids in budgets.objects[p][repeat] for o in ids}
            assert subset <= training
            for c in range(N_CLASSES):
                chosen = set(budgets.selection[p][repeat][c][1000])
                assert chosen <= training
                assert not chosen & subset


def test_without_full_budget() -> None:
    part = _part()
    budgets = _make([part], include_full=False)
    assert all(s.budget is not None for s in resolve(budgets, part))
