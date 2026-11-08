"""Object groups of a prepared dataset: alerts at the same sky position, or with the same id.

``red`` removed exact duplicates only, so one object (or one artefact) can have several
distinct alerts, in one split or across splits. On ``td_ztf_stamp_17_06_20_red-002``, 24 of
the 500 Validation stamps and 27 of the 500 Test stamps have a Train alert within 1.5"
(mostly Bogus), and Train holds groups of up to 16 alerts. A split that is stratified but
not grouped can then train on one alert of an object and evaluate on another, which favours
whatever memorises more (``docs/evaluation-protocol.md``, "Leakage").

A *grouping* gives every row a group id. Two sources:

- **position** (the default): de-standardise ``ra`` and ``dec`` with M1's statistics
  (:func:`~astro_simclr.data.m1_features.destandardise`), link every pair of rows whose
  great-circle separation is at most ``radius_arcsec`` (1.5" by default, inclusive), and
  take the connected components (union-find). The grouping is therefore transitive: a chain
  of alerts each within the radius of the next is one group, even when its ends are farther
  apart.
- **external**: one id per row (an ``oid`` column, from task 3.1's ``fetch-alerts``); rows
  with equal ids are one group. A row without a known object must get an id of its own.
  The ids come in a ``.npz`` file with one array per split (``Train``, ``Validation``,
  optionally ``Test``), read by :func:`load_group_ids` (integers or strings, no pickles).

The neighbour search is exact and needs numpy only: the rows are sorted by ``dec`` and each
row is compared with the following rows while their ``dec`` difference is at most the
radius. That window is sufficient anywhere on the sphere, poles included, because the
separation of two points is never smaller than their ``dec`` difference. The separation is
the haversine formula, which is accurate at arcsecond scales and handles the RA wrap at
0/360 (it only uses ``sin`` of the RA difference). The cost is ``O(N w)`` for a window of
``w`` rows (16 on ``red``); a dataset where many rows share a declination band within the
radius degrades towards ``O(N^2)`` but stays correct.

**Cross-matching two sets** (:func:`match_any_within`, the corpus index of task 5.1): which
rows of one set lie within the radius of *any* row of a reference set. The same search with
the reference sorted by ``dec``: each row is compared with the reference rows whose ``dec``
is within the radius of its own (two ``searchsorted`` bounds), so the cost is ``O(N w)`` for
a window of ``w`` reference rows, and the RA wrap and the poles are handled as above.

Group ids are canonical: groups are numbered 0, 1, ... in the order of their smallest row
index, so two groupings of the same rows are equal exactly when their arrays are equal.
"""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt

from astro_simclr.data.m1_features import destandardise
from astro_simclr.data.schema import DataError, StampData

__all__ = [
    "DEFAULT_RADIUS_ARCSEC",
    "EXTERNAL",
    "POSITION",
    "Grouping",
    "GroupingMethod",
    "canonical_groups",
    "external_groups",
    "group_members",
    "grouping_for",
    "load_group_ids",
    "match_any_within",
    "neighbour_pairs",
    "position_groups",
    "split_positions",
    "union_groups",
]

GroupingMethod = Literal["position", "external"]
#: :attr:`Grouping.method` of a position grouping.
POSITION: Final = "position"
#: :attr:`Grouping.method` of a grouping by external ids (an ``oid`` column).
EXTERNAL: Final = "external"
#: The default matching radius (arcseconds): about one ZTF pixel (1.01") plus astrometric
#: scatter; the integrator's audit of ``red`` used it.
DEFAULT_RADIUS_ARCSEC: Final = 1.5
#: Tolerance on ``|dec| <= 90`` (degrees): de-standardisation loses about 1e-13.
_DEC_TOLERANCE: Final = 1e-6

Indices = npt.NDArray[np.int64]
Floats = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Grouping:
    """A canonical group id per row (module docstring), and how it was made.

    ``radius_arcsec`` is set for a position grouping and ``None`` for an external one.
    """

    method: GroupingMethod
    radius_arcsec: float | None
    group_of: Indices

    @property
    def n_groups(self) -> int:
        """The number of groups (singletons included)."""
        return int(self.group_of.max()) + 1 if self.group_of.size else 0

    def multi(self) -> list[list[int]]:
        """The groups with two or more rows, each as sorted row indices, by smallest row."""
        return [members for members in group_members(self.group_of) if len(members) > 1]


def canonical_groups(labels: npt.ArrayLike) -> Indices:
    """Renumber arbitrary group labels as 0, 1, ... in the order of first appearance.

    Equal labels stay in one group; the result does not depend on the label values.
    """
    _, first, inverse = np.unique(np.asarray(labels), return_index=True, return_inverse=True)
    # Rank each distinct label by the position of its first row.
    rank = np.empty(first.size, dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(first.size, dtype=np.int64)
    return rank[inverse.reshape(-1)]


def group_members(group_of: Indices) -> list[list[int]]:
    """The rows of each group, as sorted lists, for the canonical ids ``0..n_groups-1``."""
    order = np.argsort(group_of, kind="stable")
    bounds = np.flatnonzero(np.diff(group_of[order])) + 1
    return [chunk.tolist() for chunk in np.split(order, bounds)] if group_of.size else []


def union_groups(n: int, first: Indices, second: Indices) -> Indices:
    """Canonical group ids of ``n`` rows linked by the pairs ``(first[k], second[k])``.

    Union-find with path halving; the root of each set is its smallest row, so the result
    is canonical without a renumbering pass beyond :func:`canonical_groups`.

    Raises:
        ValueError: if the pair arrays differ in length or hold an index outside ``range(n)``.
    """
    first, second = np.asarray(first, dtype=np.int64), np.asarray(second, dtype=np.int64)
    if first.shape != second.shape:
        raise ValueError("the pair arrays must have the same length")
    if first.size and (min(first.min(), second.min()) < 0 or max(first.max(), second.max()) >= n):
        raise ValueError(f"pair indices must be in range({n})")
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(first.tolist(), second.tolist(), strict=True):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)
    return canonical_groups([find(x) for x in range(n)])


def _separation_deg(ra1: Floats, dec1: Floats, ra2: Floats, dec2: Floats) -> Floats:
    """Great-circle separation (degrees) by the haversine formula."""
    r1, d1, r2, d2 = (np.radians(v) for v in (ra1, dec1, ra2, dec2))
    h = np.sin((d2 - d1) / 2) ** 2 + np.cos(d1) * np.cos(d2) * np.sin((r2 - r1) / 2) ** 2
    return np.asarray(np.degrees(2 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))), dtype=np.float64)


def _check_positions(ra: Floats, dec: Floats) -> None:
    if ra.shape != dec.shape or ra.ndim != 1:
        raise ValueError(f"ra and dec must be 1-d of one length, got {ra.shape} and {dec.shape}")
    if not (np.isfinite(ra).all() and np.isfinite(dec).all()):
        raise DataError("ra/dec hold non-finite values: cannot group by position")
    if dec.size and np.abs(dec).max() > 90 + _DEC_TOLERANCE:
        msg = (
            f"dec reaches {float(np.abs(dec).max()):.6g} degrees, outside [-90, 90]: were the "
            "features standardised with M1's statistics? (astro-simclr check-features)"
        )
        raise DataError(msg)


def neighbour_pairs(
    ra: npt.ArrayLike, dec: npt.ArrayLike, radius_arcsec: float
) -> tuple[Indices, Indices]:
    """Every pair of rows within ``radius_arcsec`` of each other (inclusive), exactly.

    ``ra`` and ``dec`` are in degrees (any RA range: the separation only uses its sine).
    Returns ``(first, second)`` with ``first < second``, sorted by ``first`` then
    ``second``. The dec-sorted window search of the module docstring.

    Raises:
        ValueError: if ``radius_arcsec`` is not positive, or the shapes differ.
        DataError: if a coordinate is not finite or ``|dec|`` exceeds 90.
    """
    if not radius_arcsec > 0:
        raise ValueError(f"radius_arcsec must be positive, got {radius_arcsec}")
    ra_, dec_ = np.asarray(ra, dtype=np.float64), np.asarray(dec, dtype=np.float64)
    _check_positions(ra_, dec_)
    radius = radius_arcsec / 3600.0
    order = np.argsort(dec_, kind="stable")
    sorted_dec = dec_[order]
    # For each sorted row, one past the last later row whose dec difference fits the window.
    end = np.searchsorted(sorted_dec, sorted_dec + radius, side="right")
    found_a: list[Indices] = []
    found_b: list[Indices] = []
    width = int((end - np.arange(order.size)).max()) if order.size else 0
    for offset in range(1, width):
        rows = np.flatnonzero(end > np.arange(order.size) + offset)
        a, b = order[rows], order[rows + offset]
        close = _separation_deg(ra_[a], dec_[a], ra_[b], dec_[b]) <= radius
        found_a.append(np.minimum(a[close], b[close]))
        found_b.append(np.maximum(a[close], b[close]))
    if not found_a:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    first, second = np.concatenate(found_a), np.concatenate(found_b)
    keep = np.lexsort((second, first))
    return first[keep].astype(np.int64), second[keep].astype(np.int64)


def match_any_within(
    ra: npt.ArrayLike,
    dec: npt.ArrayLike,
    ref_ra: npt.ArrayLike,
    ref_dec: npt.ArrayLike,
    radius_arcsec: float,
) -> tuple[npt.NDArray[np.bool_], Indices]:
    """Which rows lie within ``radius_arcsec`` (inclusive) of any reference row, exactly.

    All coordinates are in degrees (any RA range). Returns ``(matched, nearest)``:
    ``matched[i]`` is whether row ``i`` has a reference row within the radius, and
    ``nearest[i]`` the closest such reference row (the smallest index among equally close
    ones), or -1. The dec-window search of the module docstring, over the reference rows.

    Raises:
        ValueError: if ``radius_arcsec`` is not positive, or the shapes differ.
        DataError: if a coordinate is not finite or ``|dec|`` exceeds 90.
    """
    if not radius_arcsec > 0:
        raise ValueError(f"radius_arcsec must be positive, got {radius_arcsec}")
    ra_, dec_ = np.asarray(ra, dtype=np.float64), np.asarray(dec, dtype=np.float64)
    ref_ra_, ref_dec_ = np.asarray(ref_ra, dtype=np.float64), np.asarray(ref_dec, dtype=np.float64)
    _check_positions(ra_, dec_)
    _check_positions(ref_ra_, ref_dec_)
    radius = radius_arcsec / 3600.0
    order = np.argsort(ref_dec_, kind="stable")
    sorted_dec = ref_dec_[order]
    # The reference rows of row i are sorted positions lo[i] ... hi[i] - 1.
    lo = np.searchsorted(sorted_dec, dec_ - radius, side="left")
    hi = np.searchsorted(sorted_dec, dec_ + radius, side="right")
    best = np.full(ra_.size, np.inf)
    nearest = np.full(ra_.size, -1, dtype=np.int64)
    width = int((hi - lo).max()) if ra_.size else 0
    for offset in range(width):
        rows = np.flatnonzero(lo + offset < hi)
        ref = order[lo[rows] + offset]
        separation = _separation_deg(ra_[rows], dec_[rows], ref_ra_[ref], ref_dec_[ref])
        closer = (separation <= radius) & (
            (separation < best[rows]) | ((separation == best[rows]) & (ref < nearest[rows]))
        )
        best[rows[closer]] = separation[closer]
        nearest[rows[closer]] = ref[closer]
    return nearest >= 0, nearest


def position_groups(ra: npt.ArrayLike, dec: npt.ArrayLike, radius_arcsec: float) -> Grouping:
    """Group rows whose positions are linked by chains of pairs within ``radius_arcsec``.

    Raises:
        ValueError: if ``radius_arcsec`` is not positive, or the shapes differ.
        DataError: if a coordinate is not finite or ``|dec|`` exceeds 90.
    """
    first, second = neighbour_pairs(ra, dec, radius_arcsec)
    n = int(np.asarray(ra).size)
    return Grouping(POSITION, float(radius_arcsec), union_groups(n, first, second))


def external_groups(ids: Sequence[Hashable] | npt.ArrayLike) -> Grouping:
    """Group rows with equal ids (an ``oid`` column). The ids must be one sortable type.

    Raises:
        ValueError: if ``ids`` is not one-dimensional.
    """
    array = np.asarray(ids)
    if array.ndim != 1:
        raise ValueError(f"group ids must be one-dimensional, got shape {array.shape}")
    return Grouping(EXTERNAL, None, canonical_groups(array))


def split_positions(data: StampData, names: Sequence[str]) -> tuple[Floats, Floats]:
    """``(ra, dec)`` in degrees of the splits ``names`` of ``data``, concatenated in order.

    The stored features are standardised; :func:`destandardise` undoes it with M1's
    statistics (``check-features`` tells whether a file's features match them).

    Raises:
        DataError: if a split has no features.
    """
    ras, decs = [], []
    for name in names:
        features = data.splits[name].features
        if features is None:
            raise DataError(f"split {name} has no features: cannot group by position")
        values = destandardise(np.asarray(features))
        ras.append(values["ra"])
        decs.append(values["dec"])
    return np.concatenate(ras), np.concatenate(decs)


def grouping_for(
    data: StampData,
    names: Sequence[str],
    *,
    radius_arcsec: float = DEFAULT_RADIUS_ARCSEC,
    group_ids: Mapping[str, Any] | None = None,
) -> Grouping:
    """The grouping of the rows of splits ``names`` (concatenated in order).

    With ``group_ids`` (split name -> one id per row), an external grouping; otherwise a
    position grouping at ``radius_arcsec``.

    Raises:
        DataError: if ``group_ids`` lacks a split or has the wrong length for one, or (by
            position) a split has no features or impossible coordinates.
    """
    if group_ids is None:
        ra, dec = split_positions(data, names)
        return position_groups(ra, dec, radius_arcsec)
    columns = []
    for name in names:
        if name not in group_ids:
            raise DataError(f"the group ids have no {name} split")
        column = np.asarray(group_ids[name])
        if column.shape != (len(data.splits[name]),):
            msg = f"the {name} group ids have shape {column.shape}, not ({len(data.splits[name])},)"
            raise DataError(msg)
        columns.append(column)
    return external_groups(np.concatenate(columns))


def load_group_ids(path: Path) -> tuple[dict[str, npt.NDArray[Any]], dict[str, str]]:
    """Read external group ids: a ``.npz`` with one 1-d array per split (module docstring).

    Returns the arrays by split name and the file's ``{"name", "sha256"}`` (recorded in the
    splits file, so a later check can tell whether the same ids were used).

    Raises:
        DataError: if the file cannot be read, needs pickles (object arrays), or holds an
            array that is not one-dimensional integers or strings.
    """
    try:
        raw = path.read_bytes()
        with np.load(path, allow_pickle=False) as archive:
            ids = {name: archive[name] for name in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile) as err:
        raise DataError(f"{path}: cannot read the group ids ({err})") from None
    for name, column in ids.items():
        if column.ndim != 1 or column.dtype.kind not in "iuU":
            msg = f"{path}: group ids {name!r} must be a 1-d array of integers or strings"
            raise DataError(msg)
    return ids, {"name": path.name, "sha256": hashlib.sha256(raw).hexdigest()}
