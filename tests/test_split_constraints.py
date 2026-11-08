"""Tests of constrained grouped splits (task 3.5's cross-benchmark rule, ADR 0018).

Without constraints the draw must be unchanged number for number; with them every
constrained group lands in an allowed part, and the stored file round-trips and verifies.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from _bench_support import synthetic_table

from astro_simclr.protocol.alerts import AlertTable
from astro_simclr.protocol.grouping import assign_groups, grouped_split
from astro_simclr.protocol.object_splits import (
    make_object_splits,
    read_object_splits,
    verify_object_splits,
    write_object_splits,
)
from astro_simclr.protocol.splits import SplitsError


@pytest.fixture(scope="module")
def table() -> AlertTable:
    return synthetic_table(3000, seed=0)


def test_an_all_true_mask_is_the_unconstrained_draw() -> None:
    rng = np.random.default_rng(1)
    sizes, labels = rng.integers(1, 4, 300), rng.integers(0, 3, 300)
    free = assign_groups(sizes, labels, (0.6, 0.2, 0.2), seed=3)
    masked = assign_groups(sizes, labels, (0.6, 0.2, 0.2), seed=3, allowed=np.ones((300, 3), bool))
    np.testing.assert_array_equal(free, masked)


def test_constrained_groups_land_in_allowed_parts() -> None:
    rng = np.random.default_rng(2)
    sizes, labels = np.ones(200, dtype=np.int64), rng.integers(0, 2, 200)
    mask = np.ones((200, 3), dtype=bool)
    mask[:60] = [True, False, False]  # pinned to part 0
    mask[60:80] = [False, True, True]  # never part 0
    part = assign_groups(sizes, labels, (0.6, 0.2, 0.2), seed=0, allowed=mask)
    assert (part[:60] == 0).all() and (part[60:80] != 0).all()
    # The free groups make up for the pinned ones: each class stays near its targets.
    for c in (0, 1):
        counts = np.bincount(part[labels == c], minlength=3) / (labels == c).sum()
        assert counts == pytest.approx([0.6, 0.2, 0.2], abs=0.05)


def test_mask_errors() -> None:
    with pytest.raises(ValueError, match="allowed must be"):
        assign_groups([1, 1], [0, 0], (0.5, 0.5), seed=0, allowed=np.ones((2, 3), bool))
    with pytest.raises(SplitsError, match="no allowed part"):
        assign_groups([1, 1], [0, 0], (0.5, 0.5), seed=0, allowed=[[True, True], [False, False]])
    with pytest.raises(SplitsError, match=r"no group is allowed in part\(s\) \[0\]"):
        assign_groups([1] * 4, [0] * 4, (0.5, 0.5), seed=0, allowed=[[False, True]] * 4)
    with pytest.raises(ValueError, match="out of range"):
        grouped_split(["a", "b"], [0, 0], (0.5, 0.5), seed=0, allowed={"a": [2]})


def test_object_splits_with_constraints_round_trip(table: AlertTable, tmp_path: Path) -> None:
    oids = np.unique(table.oid).tolist()
    allowed: dict[str, tuple[str, ...]] = {o: ("Train",) for o in oids[:300]}
    allowed |= {o: ("Validation", "Test") for o in oids[300:400]}
    allowed["ZTF99notthere"] = ("Train",)  # not a group of the table: ignored
    allowed[oids[400]] = ("Train", "Validation", "Test")  # no constraint at all
    splits = make_object_splits(table, seed=0, allowed=allowed)
    assert len(splits.constraints) == 400
    for o in oids[:300]:
        assert o in splits.groups["Train"]
    for o in oids[300:400]:
        assert o not in splits.groups["Train"]
    path = tmp_path / "s.json"
    write_object_splits(splits, path)
    raw = json.loads(path.read_text())
    assert raw["format_version"] == 2
    assert [e["splits"] for e in raw["constraints"]] == [["Train"], ["Validation", "Test"]]
    loaded = read_object_splits(path)
    assert loaded == splits
    verify_object_splits(loaded, table)
    # Unconstrained splits are unchanged by the new field.
    free = make_object_splits(table, seed=0)
    assert (
        free.constraints == {}
        and free.groups == make_object_splits(table, seed=0, allowed={}).groups
    )


def test_constraint_errors_and_tampering(table: AlertTable, tmp_path: Path) -> None:
    oids = np.unique(table.oid).tolist()
    with pytest.raises(SplitsError, match="invalid"):
        make_object_splits(table, allowed={oids[0]: ("Holdout",)})
    with pytest.raises(SplitsError, match="invalid"):
        make_object_splits(table, allowed={oids[0]: ()})
    splits = make_object_splits(table, seed=0, allowed={oids[0]: ("Train",)})
    path = tmp_path / "s.json"
    write_object_splits(splits, path)
    raw = json.loads(path.read_text())
    raw["constraints"] = [{"splits": ["Test"], "groups": [oids[0]]}]
    path.write_text(json.dumps(raw))
    with pytest.raises(SplitsError, match="outside its allowed splits"):
        read_object_splits(path)
    for bad in (
        [{"splits": ["Train", "Validation", "Test"], "groups": [oids[0]]}],
        [{"splits": ["Test", "Train"], "groups": [oids[0]]}],
        [{"splits": ["Train"], "groups": [oids[0]]}, {"splits": ["Train"], "groups": [oids[0]]}],
        {"Train": [oids[0]]},
    ):
        raw["constraints"] = bad
        path.write_text(json.dumps(raw))
        with pytest.raises(SplitsError):
            read_object_splits(path)
    # A version-1 file (no constraints) is still read.
    raw = json.loads(json.dumps({**json.loads(path.read_text()), "format_version": 1}))
    del raw["constraints"]
    path.write_text(json.dumps(raw))
    assert read_object_splits(path).constraints == {}


def test_verify_checks_the_constraints_against_the_table(table: AlertTable) -> None:
    oids = np.unique(table.oid).tolist()
    splits = make_object_splits(table, seed=0, allowed={oids[0]: ("Train",)})
    moved = {**splits.constraints, oids[0]: ("Test",)}
    from dataclasses import replace

    with pytest.raises(SplitsError, match="outside its allowed"):
        verify_object_splits(replace(splits, constraints=moved), table)
    with pytest.raises(SplitsError, match="not a group of the table"):
        verify_object_splits(replace(splits, constraints={"ZTF99zzzzzzz": ("Train",)}), table)
