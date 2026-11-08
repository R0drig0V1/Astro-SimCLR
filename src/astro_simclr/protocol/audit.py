"""``leakage-audit``: how many objects a prepared dataset shares across its splits.

The byte-duplicate check (:mod:`astro_simclr.protocol.leakage`) finds the same stamp in two
splits; this audit finds the same *object*: rows of different splits in one group
(:mod:`astro_simclr.protocol.groups`; by position within ``radius_arcsec`` of each other,
or by external ids). The groups are built over all three splits together, so a Validation
alert and a Test alert of one object are linked too, and so are rows linked only through a
row of a third split.

The report (:func:`leakage_audit`, a JSON-ready dict) holds:

- ``groups``: their number, how many have two or more rows, the largest, and how many mix
  classes;
- ``within``: per split, its groups of two or more of its own rows, and those rows per
  class;
- ``pairs``: per split pair (Train-Validation, Train-Test, Validation-Test), the groups
  with rows in both, and those rows of each split per class;
- ``near_train``: per evaluation split (Validation, Test), its rows that share a
  position with a Train row: a Train row within the radius (position), or the same id
  (external). This is the number the leakage note quotes (``docs/evaluation-protocol.md``);
- ``byte_duplicates``: :func:`~astro_simclr.protocol.leakage.find_duplicates`.

It reads the features and labels only (plus every stamp once, for the byte check).
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from astro_simclr.data.prepared import load_prepared
from astro_simclr.data.schema import LABEL_NAMES, SPLITS, StampData
from astro_simclr.protocol.groups import (
    DEFAULT_RADIUS_ARCSEC,
    POSITION,
    Grouping,
    grouping_for,
    load_group_ids,
    neighbour_pairs,
    split_positions,
)
from astro_simclr.protocol.leakage import find_duplicates

__all__ = ["format_audit", "leakage_audit"]

Indices = npt.NDArray[np.int64]


def _per_class(labels: Indices | None, rows: Indices) -> dict[str, int] | None:
    """Rows per class name (``None`` for an unlabelled split)."""
    if labels is None:
        return None
    counts = np.bincount(labels[rows], minlength=len(LABEL_NAMES))
    return {name: int(counts[k]) for k, name in enumerate(LABEL_NAMES)}


def _near_train(
    data: StampData, grouping: Grouping, split_of: Indices, offsets: dict[str, int]
) -> dict[str, Indices]:
    """Per non-Train split, its local rows with a Train row in reach (module docstring)."""
    train = SPLITS.index("Train")
    if grouping.method == POSITION:
        assert grouping.radius_arcsec is not None  # noqa: S101 - a position grouping's radius
        ra, dec = split_positions(data, SPLITS)
        first, second = neighbour_pairs(ra, dec, grouping.radius_arcsec)
        # first < second and Train comes first, so a Train neighbour is always ``first``.
        linked = second[split_of[first] == train]
    else:
        train_groups = np.unique(grouping.group_of[split_of == train])
        linked = np.flatnonzero(np.isin(grouping.group_of, train_groups))
    out = {}
    for name in SPLITS:
        if name == "Train":
            continue
        rows = np.unique(linked[split_of[linked] == SPLITS.index(name)])
        out[name] = (rows - offsets[name]).astype(np.int64)
    return out


def leakage_audit(
    path: Path,
    *,
    radius_arcsec: float = DEFAULT_RADIUS_ARCSEC,
    group_ids_file: Path | None = None,
) -> dict[str, Any]:
    """The audit of the prepared dataset at ``path`` (module docstring), as a dict.

    Groups by position within ``radius_arcsec``, or by the ids of ``group_ids_file`` (which
    must then cover Train, Validation and Test).

    Raises:
        DataError: if ``path`` is not a valid prepared layout, a split has no features, a
            coordinate is impossible, or the group ids cannot be read or do not fit.
        ValueError: if ``radius_arcsec`` is not positive.
    """
    data = load_prepared(path)
    source = None
    if group_ids_file is None:
        grouping = grouping_for(data, SPLITS, radius_arcsec=radius_arcsec)
    else:
        ids, source = load_group_ids(group_ids_file)
        grouping = grouping_for(data, SPLITS, group_ids=ids)
    sizes = [len(data.splits[name]) for name in SPLITS]
    offsets = dict(zip(SPLITS, np.cumsum([0, *sizes[:-1]]).tolist(), strict=True))
    split_of = np.repeat(np.arange(len(SPLITS), dtype=np.int64), sizes)
    labels = {
        name: None if data.splits[name].labels is None else np.asarray(data.splits[name].labels)
        for name in SPLITS
    }
    group_of = grouping.group_of
    size = np.bincount(group_of, minlength=grouping.n_groups)

    def local_rows(name: str, groups: Indices) -> Indices:
        rows = np.flatnonzero((split_of == SPLITS.index(name)) & np.isin(group_of, groups))
        return np.asarray(rows - offsets[name], dtype=np.int64)

    within = {}
    for name in SPLITS:
        own = group_of[split_of == SPLITS.index(name)]
        repeated = np.flatnonzero(np.bincount(own, minlength=grouping.n_groups) > 1)
        rows = local_rows(name, repeated)
        within[name] = {
            "groups": int(repeated.size),
            "rows": int(rows.size),
            "rows_per_class": _per_class(labels[name], rows),
        }
    pairs = []
    for name_a, name_b in combinations(SPLITS, 2):
        shared = np.intersect1d(
            group_of[split_of == SPLITS.index(name_a)], group_of[split_of == SPLITS.index(name_b)]
        )
        rows_a, rows_b = local_rows(name_a, shared), local_rows(name_b, shared)
        pairs.append(
            {
                "a": name_a,
                "b": name_b,
                "groups": int(shared.size),
                "rows_a": int(rows_a.size),
                "rows_b": int(rows_b.size),
                "rows_a_per_class": _per_class(labels[name_a], rows_a),
                "rows_b_per_class": _per_class(labels[name_b], rows_b),
            }
        )
    mixed = 0
    all_labels = [labels[name] for name in SPLITS]
    if all(column is not None for column in all_labels):
        flat = np.concatenate([np.asarray(column) for column in all_labels])
        for members in grouping.multi():
            mixed += int(np.unique(flat[members]).size > 1)
    near = _near_train(data, grouping, split_of, offsets)
    return {
        "dataset": path.name,
        "grouping": {
            "method": grouping.method,
            "radius_arcsec": grouping.radius_arcsec,
            "source": source,
        },
        "rows": dict(zip(SPLITS, sizes, strict=True)),
        "groups": {
            "n": grouping.n_groups,
            "with_two_or_more_rows": int(np.count_nonzero(size > 1)),
            "largest": int(size.max(initial=0)),
            "mixed_class": mixed,
        },
        "within": within,
        "pairs": pairs,
        "near_train": {
            name: {
                "rows": int(rows.size),
                "of": len(data.splits[name]),
                "rows_per_class": _per_class(labels[name], rows),
                "class_sizes": _per_class(
                    labels[name], np.arange(len(data.splits[name]), dtype=np.int64)
                ),
                "indices": rows.tolist(),
            }
            for name, rows in near.items()
        },
        "byte_duplicates": [dup.as_dict() for dup in find_duplicates(data)],
    }


def _classes(counts: dict[str, int] | None, sizes: dict[str, int] | None = None) -> str:
    if counts is None:
        return "unlabelled"
    if sizes is None:
        return ", ".join(f"{name} {n}" for name, n in counts.items())
    return ", ".join(f"{name} {n}/{sizes[name]}" for name, n in counts.items())


def format_audit(report: dict[str, Any]) -> Sequence[str]:
    """The report of :func:`leakage_audit` as lines of text (the CLI's output)."""
    grouping = report["grouping"]
    how = (
        f'by position within {grouping["radius_arcsec"]}"'
        if grouping["method"] == POSITION
        else f"by external ids ({grouping['source']['name']})"
    )
    groups = report["groups"]
    lines = [
        f"leakage-audit: {report['dataset']}, groups {how}",
        f"groups: {groups['n']} over {sum(report['rows'].values())} rows; "
        f"{groups['with_two_or_more_rows']} with 2+ rows (largest {groups['largest']}), "
        f"{groups['mixed_class']} mixing classes",
    ]
    for name, entry in report["within"].items():
        lines.append(
            f"within {name}: {entry['groups']} groups of 2+ rows, {entry['rows']} rows "
            f"({_classes(entry['rows_per_class'])})"
        )
    for pair in report["pairs"]:
        lines.append(
            f"{pair['a']}-{pair['b']}: {pair['groups']} shared groups; {pair['a']} rows "
            f"{pair['rows_a']} ({_classes(pair['rows_a_per_class'])}); {pair['b']} rows "
            f"{pair['rows_b']} ({_classes(pair['rows_b_per_class'])})"
        )
    for name, entry in report["near_train"].items():
        lines.append(
            f"{name} rows sharing a position with Train: {entry['rows']} of {entry['of']} "
            f"({_classes(entry['rows_per_class'], entry['class_sizes'])})"
        )
    for dup in report["byte_duplicates"]:
        lines.append(
            f"byte duplicate: {dup['split_a']} {dup['index_a']} ({dup['label_a']}) = "
            f"{dup['split_b']} {dup['index_b']} ({dup['label_b']})"
        )
    return lines
