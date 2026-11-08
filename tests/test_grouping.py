"""Tests of the grouped, class-stratified assignment and of the cross-benchmark rule.

The property tests run on many random tables (seeded, so they are reproducible): whatever
the group sizes and class mix, no group may straddle two parts, and every class must be
within the draw's bound of its targets.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from astro_simclr.protocol.grouping import (
    CROSS_BENCHMARK_ALLOWED,
    assert_no_shared_groups,
    assign_groups,
    check_fractions,
    check_stratified,
    group_rows,
    grouped_halves,
    grouped_split,
    shared_groups,
    stratification_bound,
)
from astro_simclr.protocol.splits import SplitsError


def _random_rows(seed: int) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Random groups (integer ids, shuffled rows) with heavy-tailed sizes and 2-6 classes."""
    rng = np.random.default_rng(seed)
    n_classes = int(rng.integers(2, 7))
    n_groups = int(rng.integers(10 * n_classes, 400))
    labels = np.concatenate([np.arange(n_classes).repeat(4), rng.integers(0, n_classes, n_groups)])
    sizes = np.minimum(rng.zipf(2.2, size=labels.size), 60)
    ids = rng.permutation(10**6)[: labels.size]
    order = rng.permutation(int(sizes.sum()))
    return np.repeat(ids, sizes)[order], np.repeat(labels, sizes)[order]


@pytest.mark.parametrize("seed", range(60))
def test_property_no_group_in_two_parts(seed: int) -> None:
    groups, labels = _random_rows(seed)
    fractions = (0.6, 0.2, 0.2) if seed % 2 else (0.7, 0.15, 0.15)
    part = grouped_split(groups, labels, fractions, seed=seed)
    # Leakage is impossible: every group's rows are in a single part.
    for g in np.unique(groups):
        assert np.unique(part[groups == g]).size == 1
    members = [set(groups[part == k].tolist()) for k in range(3)]
    assert not (members[0] & members[1] or members[0] & members[2] or members[1] & members[2])
    assert (part >= 0).all()
    # Within the bound, for every class and part.
    check_stratified(groups, labels, part, fractions)
    for c in np.unique(labels):
        sizes = group_rows(groups[labels == c], labels[labels == c]).size
        total = sizes.sum()
        for k, share in enumerate(fractions):
            deviation = np.count_nonzero((part == k) & (labels == c)) - share * total
            assert -stratification_bound(sizes) - 1e-9 <= deviation <= sizes.max() + 1e-9


def test_stratification_tolerance_with_small_groups() -> None:
    """With small groups (1-3 rows) every class is within 1 point of its fractions."""
    rng = np.random.default_rng(5)
    weights = np.array([9774, 3615, 10211, 9180, 15950, 580], dtype=np.float64)
    labels_g = rng.choice(6, size=8000, p=weights / weights.sum())
    sizes = rng.integers(1, 4, size=labels_g.size)
    groups = np.repeat(np.arange(labels_g.size), sizes)
    labels = np.repeat(labels_g, sizes)
    part = grouped_split(groups, labels, (0.6, 0.2, 0.2), seed=0)
    for c in range(6):
        share = np.bincount(part[labels == c], minlength=3) / np.count_nonzero(labels == c)
        np.testing.assert_allclose(share, [0.6, 0.2, 0.2], atol=0.01)
    # Random order (not largest-first): the parts' rows per group stay alike.
    per_group = [np.count_nonzero(part == k) / np.unique(groups[part == k]).size for k in range(3)]
    assert max(per_group) - min(per_group) < 0.1


def test_deterministic_per_seed() -> None:
    groups, labels = _random_rows(1)
    first = grouped_split(groups, labels, (0.6, 0.2, 0.2), seed=3)
    np.testing.assert_array_equal(first, grouped_split(groups, labels, (0.6, 0.2, 0.2), seed=3))
    assert not np.array_equal(first, grouped_split(groups, labels, (0.6, 0.2, 0.2), seed=4))
    # Row order does not matter: the draw works on sorted group ids.
    order = np.random.default_rng(0).permutation(groups.size)
    shuffled = grouped_split(groups[order], labels[order], (0.6, 0.2, 0.2), seed=3)
    np.testing.assert_array_equal(shuffled, first[order])


def test_string_group_ids_and_conflicts() -> None:
    groups = np.array(["a", "a", "b", "c", "c", "d", "e", "f"])
    labels = np.array([0, 1, 0, 0, 0, 1, 1, 1])
    found = group_rows(groups, labels)
    assert found.ids.tolist() == ["a", "b", "c", "d", "e", "f"]
    assert found.size.tolist() == [2, 1, 2, 1, 1, 1]
    assert found.conflicts == {"a": (0, 1)}
    with pytest.raises(SplitsError, match=r"1 groups .*a: A, B"):
        grouped_split(groups, labels, (0.5, 0.5), seed=0, names=("A", "B"))
    part = grouped_split(groups, labels, (0.5, 0.5), seed=0, drop_conflicts=True)
    assert part[:2].tolist() == [-1, -1] and (part[2:] >= 0).all()


def test_halves_are_grouped_and_stratified() -> None:
    groups, labels = _random_rows(2)
    selection, decision = grouped_halves(groups, labels, seed=0)
    assert np.intersect1d(groups[selection], groups[decision]).size == 0
    assert np.union1d(selection, decision).size == groups.size
    part = np.zeros(groups.size, dtype=np.int64)
    part[decision] = 1
    check_stratified(groups, labels, part, (0.5, 0.5))
    # Its own stream: not the first two parts of a 3-way split.
    again = grouped_halves(groups, labels, seed=0)
    np.testing.assert_array_equal(again[0], selection)


def test_too_few_groups_and_bad_inputs() -> None:
    with pytest.raises(SplitsError, match="B has 2 groups"):
        assign_groups([1] * 7, [0] * 5 + [1] * 2, (0.6, 0.2, 0.2), seed=0, names=("A", "B"))
    # Three singletons cannot fill 60/20/20: the greedy draw gives Train two of them.
    with pytest.raises(SplitsError, match="A has 3 groups"):
        assign_groups([1, 1, 1], [0, 0, 0], (0.6, 0.2, 0.2), seed=0, names=("A",))
    with pytest.raises(SplitsError, match="class 1 has 1 groups"):
        assign_groups([1, 1, 1], [0, 0, 1], (0.5, 0.5), seed=0)
    for bad in ((1.0,), (0.5, 0.6), (0.5, -0.5, 1.0), (float("nan"), 1.0)):
        with pytest.raises(SplitsError, match="fractions"):
            check_fractions(bad)
    with pytest.raises(ValueError, match="1-D"):
        group_rows([1, 2], [0])
    with pytest.raises(ValueError, match="non-negative"):
        group_rows([1, 2], [0, -1])
    assert stratification_bound([]) == 0.0
    assert stratification_bound([3, 1], n_parts=2) == 3.0


def test_check_stratified_catches_violations() -> None:
    groups = np.arange(20)
    labels = np.zeros(20, dtype=np.int64)
    good = grouped_split(groups, labels, (0.5, 0.5), seed=0)
    check_stratified(groups, labels, good, (0.5, 0.5))
    lopsided = np.zeros(20, dtype=np.int64)
    lopsided[:2] = 1
    with pytest.raises(SplitsError, match="selection holds 18 X rows"):
        check_stratified(
            groups, labels, lopsided, (0.5, 0.5), names=("X",), part_names=("selection", "decision")
        )
    straddle = good.copy()
    pairs = np.repeat(np.arange(10), 2)  # groups of two rows, one in each part
    straddle[::2], straddle[1::2] = 0, 1
    with pytest.raises(SplitsError, match="two parts"):
        check_stratified(pairs, labels, straddle, (0.5, 0.5))


def test_cross_benchmark_rule() -> None:
    # Benchmark A: groups 1-6 in Train, Validation, Test; B reuses 2 (Train) and 6 (Test).
    groups_a = np.array([1, 1, 2, 3, 4, 5, 6])
    split_a = np.array([0, 0, 0, 1, 1, 2, 2])
    groups_b = np.array([2, 7, 8, 6, 6, 9])
    split_b = np.array([2, 0, 1, 0, 0, -1])
    table = shared_groups(groups_a, split_a, groups_b, split_b)
    assert table[0][2].tolist() == [2]  # A Train / B Test: the leak that matters
    assert table[2][0].tolist() == [6]  # A Test / B Train
    assert sum(cell.size for row in table for cell in row) == 2
    # The plan's matrix (the default) forbids both: Train with the other's Test.
    with pytest.raises(SplitsError, match=r"A Train / B Test: 1 \(\[2\]\).*A Test / B Train"):
        assert_no_shared_groups(groups_a, split_a, groups_b, split_b)
    with pytest.raises(SplitsError, match="A Train / B Test"):
        assert_no_shared_groups(groups_a, split_a, groups_b, split_b, allowed=())
    counts = assert_no_shared_groups(
        groups_a, split_a, groups_b, split_b, allowed={("Train", "Test"), ("Test", "Train")}
    )
    assert counts.tolist() == [[0, 0, 1], [0, 0, 0], [1, 0, 0]]
    # Disjoint benchmarks pass with all nine pairs refused; rows in no split never count.
    clean = assert_no_shared_groups(
        groups_a, split_a, np.array([9, 10]), np.array([0, -1]), allowed=()
    )
    assert clean.sum() == 0
    assert assert_no_shared_groups([1], [0], [1], [-1], allowed=()).sum() == 0
    with pytest.raises(ValueError, match="unknown split"):
        assert_no_shared_groups(groups_a, split_a, groups_b, split_b, allowed={("Train", "Dev")})
    with pytest.raises(ValueError, match="1-D"):
        shared_groups(groups_a, split_a[:2], groups_b, split_b)


def test_cross_benchmark_rule_with_oids() -> None:
    a = np.array(["ZTF18aaa", "ZTF18bbb"])
    b = np.array(["ZTF18bbb", "ZTF19ccc"])
    # Validation/Validation and Train/Train are allowed by the plan's matrix ...
    counts = assert_no_shared_groups(a, [0, 1], b, [1, 1])
    assert counts[1, 1] == 1
    assert assert_no_shared_groups(a, [0, 0], b, [0, 0])[0, 0] == 1
    # ... but not when every pair is refused, and Validation/Train never is.
    with pytest.raises(SplitsError, match="A Validation / B Validation"):
        assert_no_shared_groups(a, [0, 1], b, [1, 1], allowed=())
    with pytest.raises(SplitsError, match="A Validation / B Train"):
        assert_no_shared_groups(a, [0, 1], b, [0, 0])
    assert len(CROSS_BENCHMARK_ALLOWED) == 5
