"""Tests of the object groups (``protocol.groups``): the exact neighbour search and union-find."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from _protocol_support import make_layout

from astro_simclr.data.prepared import load_prepared
from astro_simclr.data.schema import DataError
from astro_simclr.protocol.groups import (
    EXTERNAL,
    POSITION,
    canonical_groups,
    external_groups,
    group_members,
    grouping_for,
    load_group_ids,
    neighbour_pairs,
    position_groups,
    union_groups,
)

ARCSEC = 1 / 3600


def _brute_pairs(
    ra: npt.NDArray[np.float64], dec: npt.NDArray[np.float64], radius: float
) -> set[tuple[int, int]]:
    """Every pair within ``radius`` arcseconds, by the vector dot product (another formula)."""
    r, d = np.radians(ra), np.radians(dec)
    xyz = np.stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)], axis=1)
    # The chord length avoids arccos' poor precision near 1.
    chord = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    separation = np.degrees(2 * np.arcsin(np.clip(chord / 2, 0, 1)))
    i, j = np.nonzero(np.triu(separation <= radius * ARCSEC, k=1))
    return set(zip(i.tolist(), j.tolist(), strict=True))


def _clustered(
    rng: np.random.Generator, n_centres: int, centres_dec: tuple[float, float]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Clusters of 1-6 points within a few arcseconds of random centres (some at the RA wrap)."""
    ras, decs = [], []
    for k in range(n_centres):
        ra0 = 0.0 if k % 4 == 0 else rng.uniform(0, 360)
        dec0 = rng.uniform(*centres_dec)
        for _ in range(int(rng.integers(1, 7))):
            dec = np.clip(dec0 + rng.normal(0, 1.2) * ARCSEC, -90, 90)
            # Spread the RA on the sky, not in RA degrees (they shrink by cos dec).
            scale = 1 / max(np.cos(np.radians(dec)), 1e-9)
            ras.append((ra0 + rng.normal(0, 1.2) * ARCSEC * scale) % 360)
            decs.append(dec)
    return np.asarray(ras), np.asarray(decs)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("band", [(-30.0, 85.0), (89.99, 90.0), (-90.0, -89.99)])
def test_neighbour_pairs_match_brute_force(seed: int, band: tuple[float, float]) -> None:
    """Property: on clustered positions (equator to poles, RA wrap), the search is exact."""
    rng = np.random.default_rng(seed)
    ra, dec = _clustered(rng, 40, band)
    for radius in (0.7, 1.5, 4.0):
        first, second = neighbour_pairs(ra, dec, radius)
        assert set(zip(first.tolist(), second.tolist(), strict=True)) == _brute_pairs(
            ra, dec, radius
        )
        assert np.all(first < second)
        assert np.array_equal(np.lexsort((second, first)), np.arange(first.size))


def test_radius_boundary_is_inclusive() -> None:
    eps = 1e-4  # arcseconds: far above the float error, far below any real separation
    for ra0, dec0 in ((10.0, 0.0), (200.0, 60.0)):
        dec = np.array([dec0, dec0 + (1.5 - eps) * ARCSEC, dec0 + (1.5 + eps) * ARCSEC])
        ra = np.full(3, ra0)
        first, second = neighbour_pairs(ra[[0, 1]], dec[[0, 1]], 1.5)
        assert (first.tolist(), second.tolist()) == ([0], [1])
        first, _ = neighbour_pairs(ra[[0, 2]], dec[[0, 2]], 1.5)
        assert first.size == 0
    # Along RA on the equator, the same boundary.
    first, _ = neighbour_pairs([0.0, (1.5 - eps) * ARCSEC], [0.0, 0.0], 1.5)
    assert first.size == 1
    first, _ = neighbour_pairs([0.0, (1.5 + eps) * ARCSEC], [0.0, 0.0], 1.5)
    assert first.size == 0


def test_ra_wrap_around() -> None:
    # 0.36" either side of RA 0 (and the same point written as -0.0001 and 359.9999).
    ra = np.array([359.9999, 0.0001, -0.0001, 180.0])
    dec = np.array([5.0, 5.0, 5.0, 5.0])
    first, second = neighbour_pairs(ra, dec, 1.5)
    assert set(zip(first.tolist(), second.tolist(), strict=True)) == {(0, 1), (0, 2), (1, 2)}
    assert position_groups(ra, dec, 1.5).group_of.tolist() == [0, 0, 0, 1]


def test_poles() -> None:
    # At the pole, every RA is the same point; 1" from it, RAs 180 degrees apart are 2" apart.
    ra = np.array([0.0, 90.0, 270.0, 0.0, 180.0])
    dec = np.array([90.0, 90.0, 90.0, 90 - 1 * ARCSEC, 90 - 1 * ARCSEC])
    first, second = neighbour_pairs(ra, dec, 1.5)
    found = set(zip(first.tolist(), second.tolist(), strict=True))
    assert {(0, 1), (0, 2), (1, 2), (0, 3), (0, 4)} <= found
    assert (3, 4) not in found  # 2" apart through the pole
    assert found == _brute_pairs(ra, dec, 1.5)
    # The south pole as well.
    first, _ = neighbour_pairs([10.0, 190.0], [-90.0, -90.0], 0.1)
    assert first.size == 1


def test_groups_are_transitive_and_canonical() -> None:
    # A chain of 1" steps: each link is within 1.5", the ends are 3" apart.
    dec = np.array([30.0, 50.0, 30 + 1 * ARCSEC, 30 + 2 * ARCSEC, 50 + 10 * ARCSEC])
    ra = np.full(5, 100.0)
    grouping = position_groups(ra, dec, 1.5)
    assert grouping.method == POSITION and grouping.radius_arcsec == 1.5
    assert grouping.group_of.tolist() == [0, 1, 0, 0, 2]
    assert grouping.n_groups == 3
    assert grouping.multi() == [[0, 2, 3]]
    assert group_members(grouping.group_of) == [[0, 2, 3], [1], [4]]


def test_union_groups_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    pairs = rng.integers(0, 50, size=(40, 2))
    expected = union_groups(50, pairs[:, 0], pairs[:, 1])
    for _ in range(5):
        shuffled = rng.permutation(pairs)
        flipped = shuffled[:, ::-1] if rng.integers(2) else shuffled
        np.testing.assert_array_equal(union_groups(50, flipped[:, 0], flipped[:, 1]), expected)
    # Canonical: groups numbered in the order of their smallest row.
    first_rows = [np.flatnonzero(expected == g)[0] for g in range(expected.max() + 1)]
    assert first_rows == sorted(first_rows)
    assert union_groups(3, np.empty(0, np.int64), np.empty(0, np.int64)).tolist() == [0, 1, 2]
    assert group_members(np.empty(0, dtype=np.int64)) == []


def test_canonical_and_external_groups() -> None:
    assert canonical_groups(["b", "a", "b", "c"]).tolist() == [0, 1, 0, 2]
    assert canonical_groups([7, 7, 3]).tolist() == [0, 0, 1]
    grouping = external_groups(np.array(["ZTF18a", "ZTF19b", "ZTF18a"]))
    assert grouping.method == EXTERNAL and grouping.radius_arcsec is None
    assert grouping.group_of.tolist() == [0, 1, 0]
    with pytest.raises(ValueError, match="one-dimensional"):
        external_groups(np.zeros((2, 2)))


def test_errors() -> None:
    with pytest.raises(ValueError, match="positive"):
        neighbour_pairs([0.0], [0.0], 0)
    with pytest.raises(ValueError, match="1-d"):
        neighbour_pairs([0.0, 1.0], [0.0], 1.5)
    with pytest.raises(DataError, match="non-finite"):
        neighbour_pairs([np.nan], [0.0], 1.5)
    with pytest.raises(DataError, match="outside"):
        neighbour_pairs([0.0], [90.5], 1.5)
    with pytest.raises(ValueError, match="same length"):
        union_groups(3, np.array([0]), np.array([1, 2]))
    with pytest.raises(ValueError, match="range"):
        union_groups(3, np.array([0]), np.array([3]))
    assert neighbour_pairs([], [], 1.5)[0].size == 0


def test_load_group_ids(tmp_path: Path) -> None:
    good = tmp_path / "ids.npz"
    np.savez(good, Train=np.array(["a", "b"]), Validation=np.array([1, 2]))
    ids, source = load_group_ids(good)
    assert sorted(ids) == ["Train", "Validation"]
    assert source["name"] == "ids.npz" and len(source["sha256"]) == 64
    pickled = tmp_path / "pickled.npz"
    np.savez(pickled, Train=np.array([{"a": 1}], dtype=object))
    with pytest.raises(DataError, match="cannot read"):
        load_group_ids(pickled)
    floats = tmp_path / "floats.npz"
    np.savez(floats, Train=np.array([1.5]))
    with pytest.raises(DataError, match="integers or strings"):
        load_group_ids(floats)
    with pytest.raises(DataError, match="cannot read"):
        load_group_ids(tmp_path / "missing.npz")


def test_grouping_for_external_ids_checks_the_splits(tmp_path: Path) -> None:
    data = load_prepared(make_layout(tmp_path / "ids", train=2, validation=1, test=1))
    ids = {"Train": np.arange(10), "Validation": np.array([0, 1, 2, 3, 4])}
    grouping = grouping_for(data, ("Train", "Validation"), group_ids=ids)
    assert grouping.group_of[:10].tolist() == list(range(10))
    assert grouping.group_of[10:].tolist() == [0, 1, 2, 3, 4]  # same ids as Train 0-4
    with pytest.raises(DataError, match="no Test split"):
        grouping_for(data, ("Train", "Test"), group_ids=ids)
    with pytest.raises(DataError, match="shape"):
        grouping_for(data, ("Train",), group_ids={"Train": np.arange(3)})


def test_grouping_needs_features(tmp_path: Path) -> None:
    from dataclasses import replace

    data = load_prepared(make_layout(tmp_path / "nofeat", train=1, validation=1, test=1))
    train = replace(data.splits["Train"], features=None)
    data = replace(data, splits={**data.splits, "Train": train})
    with pytest.raises(DataError, match="no features"):
        grouping_for(data, ("Train",))
