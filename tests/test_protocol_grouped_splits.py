"""Tests of the grouped (format version 2) decision splits and of ``leakage-audit``.

The layout: 30/10/10 stamps per class (Train rows ``30 c .. 30 c + 29`` are class ``c``,
Validation and Test rows ``10 c .. 10 c + 9``), with planted objects:

- Train 0-5 (AGN): a chain of 0.5" steps, one group;
- Train 31, 32 and Validation 12 (SN): one object across Train and Validation;
- Validation 40, 41 (Bogus): one object inside Validation;
- Train 60 (VS) and Train 125 (Bogus): one position, two classes;
- Train 7 (AGN), Validation 33 (Asteroid) and Test 3 (AGN): one object; Validation 33 is
  also a byte copy of Train 7's stamp (the leaked stamp, excluded from Train);
- Train 90, 91 (Asteroid): 0.7" apart across the RA wrap at 0/360.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _protocol_support import N_CLASSES, make_layout

from astro_simclr.cli import main
from astro_simclr.data.prepared import load_prepared
from astro_simclr.data.schema import DataError
from astro_simclr.protocol.audit import format_audit, leakage_audit
from astro_simclr.protocol.groups import grouping_for
from astro_simclr.protocol.split_runs import SplitRun
from astro_simclr.protocol.splits import (
    ProtocolSplits,
    SplitsError,
    check_group_integrity,
    make_splits,
    read_splits,
    verify_splits,
    write_splits,
)

ROOT = Path(__file__).resolve().parents[1]
ARCSEC = 1 / 3600
_SMALL: dict[str, Any] = {"selection_per_class": 3, "screen_per_class": 5}
N_TRAIN, N_VALIDATION = 150, 50

POSITIONS: dict[str, dict[int, tuple[float, float]]] = {
    "Train": {
        **{i: (40.0, 10.0 + 0.5 * i * ARCSEC) for i in range(6)},
        31: (120.0, -20.0),
        32: (120.0, -20.0 + 1.0 * ARCSEC),
        60: (250.0, 70.0),
        125: (250.0, 70.0),
        7: (300.0, 45.0),
        90: (359.9999, 5.0),
        91: (0.0001, 5.0),
    },
    "Validation": {
        12: (120.0 + 0.5 * ARCSEC, -20.0),
        33: (300.0, 45.0),
        40: (15.0, 0.0),
        41: (15.0, 0.0),
    },
    "Test": {3: (300.0, 45.0 + 0.3 * ARCSEC)},
}


@pytest.fixture(scope="module")
def layout(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("grouped")
    return make_layout(
        root / "grouped",
        train=30,
        validation=10,
        test=10,
        duplicate=(7, "Validation", 33),
        positions=POSITIONS,
    )


def _labels(path: Path) -> np.ndarray:
    data = load_prepared(path)
    train, val = data.splits["Train"].labels, data.splits["Validation"].labels
    assert train is not None and val is not None
    return np.concatenate([train, val])


def _assert_integral(splits: ProtocolSplits, labels: np.ndarray) -> None:
    """Every rule of the grouped splits, checked from scratch."""
    assert splits.grouping is not None
    group_of = splits.grouping.group_of
    n_pool = splits.n_train + splits.n_validation
    excluded = set(splits.excluded_train)
    for members in splits.grouping.multi():
        folds = {
            f.fold for f in splits.folds for m in members if m not in excluded and m in f.evaluation
        }
        assert len(folds) <= 1, members
    for fold in splits.folds:
        parts = [fold.evaluation, fold.selection, fold.training, fold.unused]
        joined = np.concatenate(parts)
        assert joined.size == np.unique(joined).size == n_pool - len(excluded)
        assert not set(joined.tolist()) & excluded
        assert not set(group_of[fold.selection].tolist()) & set(group_of[fold.training].tolist())
        assert not set(group_of[fold.evaluation].tolist()) & set(
            group_of[np.concatenate([fold.training, fold.selection])].tolist()
        )
        counts = np.bincount(labels[fold.selection], minlength=N_CLASSES)
        assert counts.tolist() == [splits.selection_per_class] * N_CLASSES
    n_train = splits.n_train
    selection = set(group_of[n_train + splits.screen.selection].tolist())
    decision = set(group_of[n_train + splits.screen.decision].tolist())
    assert not selection & decision
    validation_groups = set(group_of[n_train:].tolist())
    expected = [i for i in range(n_train) if group_of[i] in validation_groups]
    assert list(splits.screen_excluded_train) == expected
    for half in (splits.screen.selection, splits.screen.decision):
        counts = np.bincount(labels[n_train + half], minlength=N_CLASSES)
        assert counts.tolist() == [splits.screen_per_class] * N_CLASSES


@pytest.mark.parametrize("seed", range(4))
def test_grouped_splits_keep_every_group_whole(layout: Path, seed: int) -> None:
    splits = make_splits(layout, seed=seed, **_SMALL)
    assert splits.format_version == 2
    assert splits.grouping is not None and splits.grouping.radius_arcsec == 1.5
    multi = splits.grouping.multi()
    assert [0, 1, 2, 3, 4, 5] in multi
    assert [31, 32, N_TRAIN + 12] in multi
    assert [60, 125] in multi
    assert [90, 91] in multi
    assert [N_TRAIN + 40, N_TRAIN + 41] in multi
    # Train 7 is the byte duplicate of Validation 33 (excluded); Test 3 is not in the pool.
    assert [7, N_TRAIN + 33] in multi
    assert splits.excluded_train == (7,)
    assert splits.screen_excluded_train == (7, 31, 32)
    assert splits.screen_fallback is None
    _assert_integral(splits, _labels(layout))
    verify_splits(splits, layout)


def test_property_random_clusters(tmp_path: Path) -> None:
    """Property: on random clustered positions, every seed gives integral, balanced splits."""
    for seed in range(3):
        rng = np.random.default_rng(seed)
        positions: dict[str, dict[int, tuple[float, float]]] = {"Train": {}, "Validation": {}}
        for _ in range(25):
            ra, dec = rng.uniform(0, 360), rng.uniform(-30, 80)
            for name, size in (("Train", 150), ("Validation", 50)):
                for row in rng.choice(size, size=int(rng.integers(0, 3)), replace=False):
                    positions[name][int(row)] = (ra, dec + rng.uniform(0, 1) * ARCSEC)
        path = make_layout(
            tmp_path / f"random{seed}", train=30, validation=10, test=2, positions=positions
        )
        splits = make_splits(path, seed=seed, **_SMALL)
        _assert_integral(splits, _labels(path))
        verify_splits(splits, path)


def test_determinism_and_round_trip(layout: Path, tmp_path: Path) -> None:
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    write_splits(make_splits(layout, **_SMALL), first)
    write_splits(make_splits(layout, **_SMALL), second)
    assert first.read_bytes() == second.read_bytes()
    raw = json.loads(first.read_text())
    assert raw["format_version"] == 2
    assert raw["grouping"]["method"] == "position"
    assert raw["grouping"]["radius_arcsec"] == 1.5
    assert raw["grouping"]["n_groups"] == N_TRAIN + N_VALIDATION - 11
    loaded = read_splits(first)
    original = make_splits(layout, **_SMALL)
    assert loaded.grouping is not None and original.grouping is not None
    np.testing.assert_array_equal(loaded.grouping.group_of, original.grouping.group_of)
    for a, b in zip(loaded.folds, original.folds, strict=True):
        for part in ("evaluation", "selection", "training", "unused"):
            np.testing.assert_array_equal(getattr(a, part), getattr(b, part))
    assert loaded.screen_excluded_train == original.screen_excluded_train
    verify_splits(loaded, layout)


def test_a_wider_radius_groups_more(layout: Path) -> None:
    narrow = make_splits(layout, radius_arcsec=0.4, **_SMALL)
    assert narrow.grouping is not None
    # 0.5" steps no longer chain Train 0-5, and Train 90/91 (0.72") split up.
    assert [0, 1, 2, 3, 4, 5] not in narrow.grouping.multi()
    assert [90, 91] not in narrow.grouping.multi()


def test_selection_groups_leave_the_training_part(tmp_path: Path) -> None:
    # Five-stamp AGN objects in Train: a selection set of 1 per class takes one member of
    # whichever group it draws, and the other members become unused in that fold.
    positions = {"Train": {i: (10.0 * (i // 5), 0.0) for i in range(30)}}
    path = make_layout(tmp_path / "chunks", train=30, validation=4, test=1, positions=positions)
    splits = make_splits(path, selection_per_class=1, screen_per_class=2)
    labels = _labels(path)
    _assert_integral(splits, labels)
    assert splits.grouping is not None
    group_of = splits.grouping.group_of
    for fold in splits.folds:
        agn = int(fold.selection[labels[fold.selection] == 0][0])
        mates = set(np.flatnonzero(group_of == group_of[agn]).tolist()) - {agn}
        assert mates <= set(fold.unused.tolist())
    assert any(fold.unused.size >= 4 for fold in splits.folds)
    # The groups of 5 AGN: per class, folds differ by more than 1, within one group's size.
    verify_splits(splits, path)


def test_screen_fallback_trims_both_halves(tmp_path: Path) -> None:
    # Validation Bogus 40-45 are one object: 6 stamps never fit a half of 5, so only the 4
    # other Bogus are left, 2 per half, and every class is trimmed to 2 per half.
    positions = {"Validation": {i: (77.0, 7.0) for i in range(40, 46)}}
    path = make_layout(tmp_path / "fallback", train=30, validation=10, test=1, positions=positions)
    splits = make_splits(path, **_SMALL)
    assert splits.screen_per_class == 2
    assert splits.screen_fallback is not None and "requested 5" in splits.screen_fallback
    assert not set(range(40, 46)) & set(
        np.concatenate([splits.screen.selection, splits.screen.decision]).tolist()
    )
    _assert_integral(splits, _labels(path))
    out = tmp_path / "fallback.json"
    write_splits(splits, out)
    loaded = read_splits(out)
    assert loaded.screen_fallback == splits.screen_fallback
    verify_splits(loaded, path)
    # An object bigger than both halves together leaves nothing to screen.
    positions = {"Validation": {i: (77.0, 7.0) for i in range(40, 50)}}
    path = make_layout(tmp_path / "empty", train=30, validation=10, test=1, positions=positions)
    with pytest.raises(SplitsError, match="no class"):
        make_splits(path, **_SMALL)


def test_too_few_stamps_outside_the_selection_groups(tmp_path: Path) -> None:
    # The SN are 2 objects of 20 (15 Train and 5 Validation alerts each), one per fold: the
    # selection group of each fold is all its SN, so its training part has none.
    positions = {
        "Train": {i: (5.0 * (i // 15), 0.0) for i in range(30, 60)},
        "Validation": {i: (10.0 if i < 15 else 15.0, 0.0) for i in range(10, 20)},
    }
    path = make_layout(tmp_path / "sn", train=30, validation=10, test=1, positions=positions)
    with pytest.raises(SplitsError, match="SN"):
        make_splits(path, selection_per_class=3, screen_per_class=5, n_folds=2)


def _plain(layout: Path, tmp_path: Path) -> dict[str, Any]:
    path = tmp_path / "plain.json"
    write_splits(make_splits(layout, **_SMALL), path)
    raw: dict[str, Any] = json.loads(path.read_text())
    return raw


def _write(raw: dict[str, Any], tmp_path: Path) -> Path:
    path = tmp_path / "edited.json"
    path.write_text(json.dumps(raw))
    return path


def _move_group_member(raw: dict[str, Any]) -> None:
    """Move Train 1 (of the 0-5 group) to another fold's evaluation part."""
    fold_of = list(raw["cv"]["fold_of"])
    selections = raw["cv"]["selection"]
    # A fold whose selection set does not hold Train 1 (selection sets stay outside folds).
    target = next(
        k for k in range(raw["cv"]["n_folds"]) if str(k) != fold_of[0] and 1 not in selections[k]
    )
    fold_of[1] = str(target)
    raw["cv"]["fold_of"] = "".join(fold_of)


def _split_screen_group(raw: dict[str, Any]) -> None:
    """Swap Validation 41 with a Bogus of the other half, so 40 and 41 are in both halves."""
    screen = raw["screen"]
    half, other = (
        ("selection", "decision") if 40 in screen["selection"] else ("decision", "selection")
    )
    assert 41 in screen[half]
    swap = next(i for i in screen[other] if 40 <= i < 50)
    screen[half] = sorted([*(i for i in screen[half] if i != 41), swap])
    screen[other] = sorted([*(i for i in screen[other] if i != swap), 41])


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (_move_group_member, "two folds"),
        (_split_screen_group, "both screen halves"),
        (lambda r: r["grouping"].update(screen_excluded_train=[7, 31]), "screen_excluded_train"),
        (lambda r: r.pop("grouping"), "grouping block"),
        (lambda r: r["grouping"].update(method="sky"), "method"),
        (lambda r: r["grouping"].update(radius_arcsec=0), "radius_arcsec"),
        (lambda r: r["grouping"].update(radius_arcsec=True), "radius_arcsec"),
        (lambda r: r["grouping"].update(method="external"), "no radius_arcsec"),
        (lambda r: r["grouping"].update(groups={}), "list of index lists"),
        (lambda r: r["grouping"]["groups"].append([3]), "two or more"),
        (lambda r: r["grouping"]["groups"].append([0, 199]), "shares a stamp"),
        (lambda r: r["grouping"].update(n_groups=1), "n_groups"),
        (lambda r: r["grouping"].update(source="ids.npz"), "source"),
        (lambda r: r["grouping"].update(screen_fallback=3), "screen_fallback"),
    ],
)
def test_read_refuses_a_broken_grouping(
    layout: Path, tmp_path: Path, edit: Any, match: str
) -> None:
    raw = _plain(layout, tmp_path)
    edit(raw)
    with pytest.raises(SplitsError, match=match):
        read_splits(_write(raw, tmp_path))


def test_verify_refuses_a_broken_grouping(layout: Path, tmp_path: Path) -> None:
    splits = make_splits(layout, **_SMALL)
    # Forget the Train 0-5 object: the file is consistent, but not what the data say.
    raw = _plain(layout, tmp_path)
    raw["grouping"]["groups"] = [g for g in raw["grouping"]["groups"] if g[0] != 0]
    raw["grouping"]["n_groups"] += 5
    with pytest.raises(SplitsError, match="fresh position grouping"):
        verify_splits(read_splits(_write(raw, tmp_path)), layout)
    # Another radius than the data were grouped with.
    raw = _plain(layout, tmp_path)
    raw["grouping"]["radius_arcsec"] = 0.4
    with pytest.raises(SplitsError, match="fresh position grouping"):
        verify_splits(read_splits(_write(raw, tmp_path)), layout)
    # An in-memory object with a group across two parts (not through a file): a fold-0
    # selection stamp joins a fold-0 training stamp that the same fold evaluates.
    assert splits.grouping is not None
    fold = splits.folds[0]
    chosen = int(fold.selection[0])
    home = next(f for f in splits.folds if chosen in f.evaluation)
    mate = int(np.intersect1d(fold.training, home.evaluation)[0])
    group_of = splits.grouping.group_of.copy()
    group_of[chosen] = group_of[mate]
    broken = replace(splits, grouping=replace(splits.grouping, group_of=group_of))
    with pytest.raises(SplitsError, match="selection and training"):
        verify_splits(broken, layout)
    short = replace(splits, grouping=replace(splits.grouping, group_of=group_of[:-1]))
    with pytest.raises(SplitsError, match="one group per pool stamp"):
        check_group_integrity(short)


def test_external_group_ids(layout: Path, tmp_path: Path) -> None:
    ids = tmp_path / "oid.npz"
    train = np.array([f"ZTF{i:04d}" for i in range(N_TRAIN)], dtype="<U12")
    validation = np.array([f"ZTF{i + 1000:04d}" for i in range(N_VALIDATION)])
    train[10:13] = "ZTFshared"  # an AGN object of three alerts
    validation[0] = train[20]  # Validation 0 (AGN) is Train 20's object
    np.savez(ids, Train=train, Validation=validation)
    splits = make_splits(layout, group_ids_file=ids, **_SMALL)
    assert splits.grouping is not None and splits.grouping.method == "external"
    assert splits.grouping.radius_arcsec is None
    assert splits.grouping_source is not None and splits.grouping_source["name"] == "oid.npz"
    assert splits.grouping.multi() == [[10, 11, 12], [20, N_TRAIN]]
    assert splits.screen_excluded_train == (20,)
    _assert_integral(splits, _labels(layout))
    out = tmp_path / "external.json"
    write_splits(splits, out)
    loaded = read_splits(out)
    verify_splits(loaded, layout)  # integrity only: the ids are not given
    verify_splits(loaded, layout, group_ids_file=ids)
    other = tmp_path / "other.npz"
    np.savez(other, Train=np.arange(N_TRAIN), Validation=np.arange(N_VALIDATION) + N_TRAIN)
    with pytest.raises(SplitsError, match="group ids of"):
        verify_splits(loaded, layout, group_ids_file=other)


def test_screen_run_leaves_the_grouped_train_alerts(layout: Path, tmp_path: Path) -> None:
    splits = make_splits(layout, **_SMALL)
    screen = SplitRun(tmp_path / "s.json", "0" * 64, splits, None).parts()
    assert not {7, 31, 32} & set(screen.training.tolist())
    assert screen.training.size == N_TRAIN - 3
    fold = SplitRun(tmp_path / "s.json", "0" * 64, splits, 0)
    parts = fold.parts()
    np.testing.assert_array_equal(parts.training, splits.folds[0].training)
    assert splits.grouping is not None
    assert fold.describe()["grouping"] == {
        "method": "position",
        "radius_arcsec": 1.5,
        "n_groups": splits.grouping.n_groups,
        "n_screen_excluded_train": 3,
    }


# --- Version 1 -----------------------------------------------------------------------------


def test_version_1_files_are_still_read(layout: Path, tmp_path: Path) -> None:
    """A version-1 object (no grouping) writes and reads back as version 1, unchanged."""
    grouped = make_splits(layout, **_SMALL)
    v1 = replace(grouped, grouping=None, screen_excluded_train=(), screen_fallback=None)
    # Version 1 has no unused stamps: rebuild the folds as version 1 would read them.
    path = tmp_path / "v1.json"
    write_splits(v1, path)
    raw = json.loads(path.read_text())
    assert raw["format_version"] == 1 and "grouping" not in raw
    loaded = read_splits(path)
    assert loaded.format_version == 1 and loaded.grouping is None
    assert loaded.screen_training_excluded() == (7,)
    for fold in loaded.folds:
        assert fold.unused.size == 0
    assert SplitRun(path, "0" * 64, loaded, None).describe()["grouping"] is None


def test_stored_v1_red_file_reads_and_its_screen_trains_as_before() -> None:
    splits = read_splits(ROOT / "configs" / "splits" / "td_ztf_stamp_17_06_20_red-002.json")
    assert splits.format_version == 1
    screen = SplitRun(Path("v1.json"), "0" * 64, splits, None).parts()
    assert screen.training.size == 50582 - 1


def test_stored_grouped_red_file_is_well_formed() -> None:
    """The committed grouped splits of ``red`` (made on the real data) read back cleanly."""
    path = ROOT / "configs" / "splits" / "td_ztf_stamp_17_06_20_red-002-grouped.json"
    splits = read_splits(path)
    assert splits.format_version == 2
    assert splits.grouping is not None and splits.grouping.radius_arcsec == 1.5
    assert splits.excluded_train == (2995,)
    assert len(splits.screen_excluded_train) == 107
    assert splits.screen_fallback is None
    assert splits.screen.selection.size == splits.screen.decision.size == 250
    assert [fold.selection.size for fold in splits.folds] == [500] * 5
    assert sum(fold.evaluation.size for fold in splits.folds) == 50582 + 500 - 1
    # The same dataset as the version-1 file.
    v1 = read_splits(ROOT / "configs" / "splits" / "td_ztf_stamp_17_06_20_red-002.json")
    assert splits.dataset == v1.dataset


# --- leakage-audit -------------------------------------------------------------------------


def test_leakage_audit(layout: Path) -> None:
    report = leakage_audit(layout)
    assert report["grouping"] == {"method": "position", "radius_arcsec": 1.5, "source": None}
    assert report["rows"] == {"Train": 150, "Validation": 50, "Test": 50}
    # The pool's groups, with Test 3 in Train 7's group (with Validation 33).
    assert report["groups"]["with_two_or_more_rows"] == 6
    assert report["groups"]["largest"] == 6
    assert report["groups"]["mixed_class"] == 2  # Train 60/125, and Train 7/Validation 33
    pairs = {(p["a"], p["b"]): p for p in report["pairs"]}
    tv = pairs[("Train", "Validation")]
    assert (tv["groups"], tv["rows_a"], tv["rows_b"]) == (2, 3, 2)
    assert tv["rows_b_per_class"]["SN"] == 1
    tt = pairs[("Train", "Test")]
    assert (tt["groups"], tt["rows_b"]) == (1, 1)
    assert pairs[("Validation", "Test")]["groups"] == 1  # through Train 7
    near = report["near_train"]
    assert near["Validation"]["indices"] == [12, 33]
    assert near["Test"]["indices"] == [3]
    assert near["Validation"]["rows_per_class"]["SN"] == 1
    assert report["within"]["Validation"]["groups"] == 1
    assert report["within"]["Train"]["rows"] == 12
    assert len(report["byte_duplicates"]) == 1
    text = "\n".join(format_audit(report))
    assert "Validation rows sharing a position with Train: 2 of 50" in text
    assert 'groups by position within 1.5"' in text
    assert "byte duplicate: Train 7 (AGN) = Validation 33 (Asteroid)" in text


def test_leakage_audit_external(layout: Path, tmp_path: Path) -> None:
    ids = tmp_path / "oid.npz"
    np.savez(
        ids,
        Train=np.arange(150),
        Validation=np.array([0, *range(1001, 1050)]),
        Test=np.arange(2000, 2050),
    )
    report = leakage_audit(layout, group_ids_file=ids)
    assert report["grouping"]["method"] == "external"
    assert report["near_train"]["Validation"]["indices"] == [0]
    assert report["near_train"]["Test"]["indices"] == []
    assert "by external ids (oid.npz)" in "\n".join(format_audit(report))
    np.savez(ids, Train=np.arange(150), Validation=np.arange(50))
    with pytest.raises(DataError, match="no Test split"):
        leakage_audit(layout, group_ids_file=ids)


def test_audit_of_an_unlabelled_split(layout: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from astro_simclr.protocol import audit

    data = load_prepared(layout)
    test = replace(data.splits["Test"], labels=None)
    unlabelled = replace(data, splits={**data.splits, "Test": test})
    monkeypatch.setattr(audit, "load_prepared", lambda path: unlabelled)
    report = leakage_audit(layout)
    assert report["near_train"]["Test"]["rows_per_class"] is None
    assert report["groups"]["mixed_class"] == 0  # not computed without every label
    assert "unlabelled" in "\n".join(format_audit(report))


def test_cli(layout: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["leakage-audit", str(layout)]) == 0
    assert "Test rows sharing a position with Train: 1 of 50" in capsys.readouterr().out
    assert main(["leakage-audit", str(layout), "--json", "--radius-arcsec", "0.4"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["grouping"]["radius_arcsec"] == 0.4
    for bad in ("0", "-1", "inf", "x"):
        with pytest.raises(SystemExit):
            main(["leakage-audit", str(layout), "--radius-arcsec", bad])
    capsys.readouterr()
    big = make_layout(
        tmp_path / "big",
        train=130,
        validation=100,
        test=2,
        positions={"Validation": {0: (1.0, 1.0), 1: (1.0, 1.0)}},
    )
    out = tmp_path / "splits.json"
    assert main(["make-splits", str(big), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert 'grouping: position within 1.5"' in printed
    assert "0 Train stamps grouped with Validation" in printed
    assert "/0\n" in printed or "/0," in printed  # unused counts
    verify_splits(read_splits(out), big)
    ids = tmp_path / "oid.npz"
    np.savez(ids, Train=np.arange(650), Validation=np.arange(500) + 650)
    out2 = tmp_path / "ext.json"
    assert main(["make-splits", str(big), "--out", str(out2), "--group-ids", str(ids)]) == 0
    assert f"external ids ({ids})" in capsys.readouterr().out


def test_grouping_for_the_pool_matches_the_file(layout: Path) -> None:
    data = load_prepared(layout)
    grouping = grouping_for(data, ("Train", "Validation"))
    splits = make_splits(layout, **_SMALL)
    assert splits.grouping is not None
    np.testing.assert_array_equal(grouping.group_of, splits.grouping.group_of)
