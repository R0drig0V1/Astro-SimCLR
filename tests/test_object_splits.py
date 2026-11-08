"""Tests of the stored object splits: drawing, the file, and the verification against data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _bench_support import synthetic_table

from astro_simclr.protocol.alerts import AlertTable
from astro_simclr.protocol.object_splits import (
    HALF_NAMES,
    SPLIT_NAMES,
    make_object_splits,
    read_object_splits,
    verify_object_splits,
    write_object_splits,
)
from astro_simclr.protocol.splits import SplitsError


def _conflicted(table: AlertTable) -> AlertTable:
    """The table with the first alert of its first multi-alert object relabelled."""
    oids, counts = np.unique(table.oid, return_counts=True)
    target = oids[counts > 1][0]
    labels = table.label.copy()
    row = int(np.flatnonzero(table.oid == target)[0])
    labels[row] = (labels[row] + 1) % len(table.label_names)
    return AlertTable.from_rows(
        table.oid.tolist(),
        table.candid.tolist(),
        table.jd.tolist(),
        labels.tolist(),
        table.label_names,
    )


@pytest.fixture(scope="module")
def table() -> AlertTable:
    return synthetic_table(3000, seed=0)


def test_draw_round_trip_and_verify(table: AlertTable, tmp_path: Path) -> None:
    splits = make_object_splits(table, seed=0, name="syn.csv")
    path = tmp_path / "splits.json"
    write_object_splits(splits, path)
    loaded = read_object_splits(path)
    assert loaded == splits
    verify_object_splits(loaded, table)
    # Every alert is in one split; the splits of an object agree (no leakage).
    split_of = splits.split_of(table)
    assert (split_of >= 0).all()
    assert sum(splits.rows(table, s).size for s in SPLIT_NAMES) == len(table)
    for name in SPLIT_NAMES:
        assert (
            splits.counts[name]["alerts"]
            == np.bincount(
                table.label[split_of == SPLIT_NAMES.index(name)], minlength=len(table.label_names)
            ).tolist()
        )
    # The halves partition Validation, by object.
    halves = [splits.rows(table, h) for h in HALF_NAMES]
    np.testing.assert_array_equal(np.union1d(*halves), splits.rows(table, "Validation"))
    assert not set(table.oid[halves[0]]) & set(table.oid[halves[1]])
    assert splits.table["n_groups"] == np.unique(table.oid).size
    # One line per list, so the file stays readable.
    assert len(path.read_text().splitlines()) < 80
    with pytest.raises(ValueError, match="unknown part"):
        splits.rows(table, "Dev")


def test_deterministic_per_seed(table: AlertTable, tmp_path: Path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    write_object_splits(make_object_splits(table, seed=4), a)
    write_object_splits(make_object_splits(table, seed=4), b)
    assert a.read_bytes() == b.read_bytes()
    assert make_object_splits(table, seed=5).groups != make_object_splits(table, seed=4).groups


def test_fractions_and_stratification(table: AlertTable) -> None:
    splits = make_object_splits(table, fractions=(0.8, 0.1, 0.1))
    total = np.bincount(table.label, minlength=len(table.label_names))
    train = np.asarray(splits.counts["Train"]["alerts"])
    # Within 2 percentage points per class, even for the smallest class (SN) here.
    np.testing.assert_allclose(train / total, 0.8, atol=0.02)
    with pytest.raises(SplitsError, match="fractions"):
        make_object_splits(table, fractions=(0.5, 0.5))
    with pytest.raises(SplitsError, match="fractions"):
        make_object_splits(table, fractions=(0.6, 0.3, 0.2))
    with pytest.raises(SplitsError, match="on_conflict"):
        make_object_splits(table, on_conflict="ignore")  # type: ignore[arg-type]


def test_conflicting_labels_refused_or_dropped(table: AlertTable, tmp_path: Path) -> None:
    bad = _conflicted(table)
    with pytest.raises(SplitsError, match="1 groups have rows with different labels"):
        make_object_splits(bad)
    splits = make_object_splits(bad, on_conflict="drop")
    assert len(splits.conflicts) == 1
    dropped = splits.conflicts[0].group
    assert all(dropped not in splits.groups[s] for s in SPLIT_NAMES)
    path = tmp_path / "s.json"
    write_object_splits(splits, path)
    verify_object_splits(read_object_splits(path), bad)


def test_custom_group_key(table: AlertTable) -> None:
    # Two objects per position group: the group, not the oid, is kept whole.
    oids = np.unique(table.oid)
    position = {oid: f"pos{i // 2}" for i, oid in enumerate(oids)}
    # Make each position group single-class by taking the label of its first object.
    first_label: dict[str, int] = {}
    for oid, label in zip(table.oid.tolist(), table.label.tolist(), strict=True):
        first_label.setdefault(position[oid], label)
    groups = [position[o] for o in table.oid.tolist()]
    labels = [first_label[g] for g in groups]
    grouped = AlertTable.from_rows(
        table.oid.tolist(),
        table.candid.tolist(),
        table.jd.tolist(),
        labels,
        table.label_names,
        group=groups,
    )
    splits = make_object_splits(grouped)
    assert splits.group_key == "group"
    verify_object_splits(splits, grouped)
    split_of = splits.split_of(grouped)
    for g in set(groups[:200]):
        assert np.unique(split_of[np.asarray(groups) == g]).size == 1
    # The same alerts grouped by oid are another table (the group is in the fingerprint).
    with pytest.raises(SplitsError, match="another alert table"):
        verify_object_splits(splits, table)


def test_too_small_class() -> None:
    tiny = AlertTable.from_rows(
        [f"o{i}" for i in range(8)], list(range(1, 9)), [1.0] * 8, [0] * 6 + [1] * 2, ("A", "B")
    )
    with pytest.raises(SplitsError, match="B has 2 groups"):
        make_object_splits(tiny)


def _tamper(path: Path, change: Any) -> Path:
    raw = json.loads(path.read_text())
    change(raw)
    out = path.with_name("tampered.json")
    out.write_text(json.dumps(raw))
    return out


def test_verify_refuses_other_data(table: AlertTable, tmp_path: Path) -> None:
    splits = make_object_splits(table)
    other = synthetic_table(3000, seed=1)
    with pytest.raises(SplitsError, match="another alert table"):
        verify_object_splits(splits, other)
    path = tmp_path / "s.json"
    write_object_splits(splits, path)

    def move(raw: dict[str, Any]) -> None:
        oid = raw["groups"]["Train"].pop(0)
        raw["groups"]["Test"] = sorted([*raw["groups"]["Test"], oid])

    with pytest.raises(SplitsError, match="counts"):
        verify_object_splits(read_object_splits(_tamper(path, move)), table)

    def drop(raw: dict[str, Any]) -> None:
        raw["groups"]["Train"].pop(0)

    with pytest.raises(SplitsError, match="1 missing"):
        verify_object_splits(read_object_splits(_tamper(path, drop)), table)

    def drop_half(raw: dict[str, Any]) -> None:
        raw["halves"]["selection"].pop(0)

    with pytest.raises(SplitsError, match="halves do not cover"):
        verify_object_splits(read_object_splits(_tamper(path, drop_half)), table)

    def lopsided(raw: dict[str, Any]) -> None:
        val = raw["groups"]["Validation"]
        raw["halves"] = {"selection": val[:-1], "decision": val[-1:]}

    with pytest.raises(SplitsError, match="outside the bound"):
        verify_object_splits(read_object_splits(_tamper(path, lopsided)), table)

    def names(raw: dict[str, Any]) -> None:
        raw["label_names"][0] = "QSO"

    with pytest.raises(SplitsError, match="label names differ"):
        verify_object_splits(read_object_splits(_tamper(path, names)), table)

    def conflicts(raw: dict[str, Any]) -> None:
        raw["conflicts"] = [{"group": "ZTF99", "labels": ["AGN", "SN"]}]

    with pytest.raises(SplitsError, match="conflicts"):
        verify_object_splits(read_object_splits(_tamper(path, conflicts)), table)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda raw: raw.update(format="other"), "not a astro-simclr-object-splits"),
        (lambda raw: raw.update(format_version=2), "unsupported format_version"),
        (lambda raw: raw.pop("halves"), "malformed"),
        (lambda raw: raw.update(seed="0"), "seed must be an integer"),
        (lambda raw: raw.update(group_key="ra"), "group_key"),
        (lambda raw: raw["table"].pop("sha256"), "sha256"),
        (lambda raw: raw["groups"]["Test"].append("ZTF 1"), "valid group ids"),
        (lambda raw: raw["groups"]["Test"].reverse(), "sorted"),
        (lambda raw: raw["fractions"].update(Test=0.5), "fractions"),
        (
            lambda raw: raw["groups"].update(
                Test=sorted([*raw["groups"]["Test"], raw["groups"]["Train"][0]])
            ),
            "two splits",
        ),
        (
            lambda raw: raw["halves"].update(
                decision=sorted([*raw["halves"]["decision"], raw["groups"]["Train"][0]])
            ),
            "Validation groups only",
        ),
        (lambda raw: raw.update(conflicts={}), "conflicts must be a list"),
        (
            lambda raw: raw.update(conflicts=[{"group": "x", "labels": ["AGN"]}]),
            "2\\+ known labels",
        ),
        (
            lambda raw: raw.update(
                conflicts=[{"group": raw["groups"]["Train"][0], "labels": ["AGN", "SN"]}]
            ),
            "conflicting group is also in a split",
        ),
    ],
)
def test_read_refuses_malformed_files(
    table: AlertTable, tmp_path: Path, change: Any, message: str
) -> None:
    path = tmp_path / "s.json"
    write_object_splits(make_object_splits(table), path)
    with pytest.raises(SplitsError, match=message):
        read_object_splits(_tamper(path, change))


def test_read_unreadable(tmp_path: Path) -> None:
    with pytest.raises(SplitsError, match="cannot read"):
        read_object_splits(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    with pytest.raises(SplitsError, match="cannot read"):
        read_object_splits(bad)
