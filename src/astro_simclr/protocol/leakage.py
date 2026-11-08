"""Stamps duplicated across the splits of a prepared dataset, and the Train stamps to drop.

The deduplication of ``red`` (``docs/data.md``, "Provenance") removed duplicates *within*
each split only, so one Train stamp of ``td_ztf_stamp_17_06_20_red-002`` is also a
Validation stamp (Train 2995, labelled Bogus, is Validation 115, labelled SN, with the same
features). Phase 2 removes such stamps **from Train only** (plan, task 2.3, "Leakage"):
Validation and Test keep their 500 stamps, as in the paper, so the reference numbers stay
comparable.

Two stamps are duplicates when their prepared uint8 arrays (63, 63, 3) are byte-identical.
Identical float stamps always give identical uint8 stamps; the converse can merge stamps
that differ by less than one quantisation step in every pixel, which for a 11,907-value
stamp is a duplicate for every practical purpose. Duplicates within one split are not
looked for (``red``'s Train stamps are all distinct). Only whole-stamp equality is
detected, not near-duplicates (shifted or re-normalised copies).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from astro_simclr.data.schema import LABEL_NAMES, SPLITS, SplitData, StampData

__all__ = ["Duplicate", "find_duplicates", "train_exclusions", "train_keep_indices"]

#: Digest size of the per-stamp hash (bytes): 128 bits, collisions are not a concern.
_DIGEST_SIZE: Final = 16


@dataclass(frozen=True, slots=True)
class Duplicate:
    """One stamp found in two splits: ``split_a[index_a]`` equals ``split_b[index_b]``.

    ``split_a`` comes before ``split_b`` in :data:`~astro_simclr.data.schema.SPLITS`
    order. The labels are class names (``None`` for an unlabelled split), and
    ``features_equal`` is whether the 26 stored features are identical too (``None`` if a
    split has no features).
    """

    split_a: str
    index_a: int
    split_b: str
    index_b: int
    label_a: str | None
    label_b: str | None
    features_equal: bool | None

    def as_dict(self) -> dict[str, object]:
        """The duplicate as a JSON-ready dict (for the stored splits and the CLI)."""
        return {
            "split_a": self.split_a,
            "index_a": self.index_a,
            "split_b": self.split_b,
            "index_b": self.index_b,
            "label_a": self.label_a,
            "label_b": self.label_b,
            "features_equal": self.features_equal,
        }


def _digests(split: SplitData) -> list[bytes]:
    """A 128-bit BLAKE2b digest of every stamp's bytes, in file order."""
    images = split.images
    return [
        hashlib.blake2b(
            np.ascontiguousarray(images[i]).tobytes(), digest_size=_DIGEST_SIZE
        ).digest()
        for i in range(len(split))
    ]


def _label(split: SplitData, index: int) -> str | None:
    return None if split.labels is None else LABEL_NAMES[int(split.labels[index])]


def _features_equal(a: SplitData, i: int, b: SplitData, j: int) -> bool | None:
    if a.features is None or b.features is None:
        return None
    return bool(np.array_equal(a.features[i], b.features[j]))


def find_duplicates(data: StampData) -> list[Duplicate]:
    """Every stamp that appears in two different splits of ``data``.

    Compares each pair of splits (Train-Validation, Train-Test, Validation-Test). The
    result is sorted by split pair, then by ``index_a``. A stamp repeated *within* a split
    is reported once per copy in the other split.

    The whole split is read once (about 600 MB for ``red``'s Train, paged in from the memory
    map; a few seconds).
    """
    digests = {name: _digests(data.splits[name]) for name in SPLITS}
    found: list[Duplicate] = []
    for position, name_a in enumerate(SPLITS):
        for name_b in SPLITS[position + 1 :]:
            where_b: dict[bytes, list[int]] = {}
            for j, digest in enumerate(digests[name_b]):
                where_b.setdefault(digest, []).append(j)
            split_a, split_b = data.splits[name_a], data.splits[name_b]
            for i, digest in enumerate(digests[name_a]):
                for j in where_b.get(digest, ()):
                    found.append(
                        Duplicate(
                            name_a,
                            i,
                            name_b,
                            j,
                            _label(split_a, i),
                            _label(split_b, j),
                            _features_equal(split_a, i, split_b, j),
                        )
                    )
    return found


def train_exclusions(duplicates: Iterable[Duplicate]) -> list[int]:
    """The sorted, distinct Train indices that duplicate a Validation or Test stamp.

    These are removed from Train only; a Validation-Test duplicate is not removed (both
    evaluation sets keep their 500 stamps) and must be reported instead.
    """
    return sorted({dup.index_a for dup in duplicates if dup.split_a == "Train"})


def train_keep_indices(n_train: int, excluded: Iterable[int]) -> npt.NDArray[np.int64]:
    """``arange(n_train)`` without ``excluded``: the Train stamps a run may use, in order.

    Raises:
        ValueError: if an excluded index is outside ``range(n_train)``.
    """
    drop = np.asarray(sorted(set(excluded)), dtype=np.int64)
    if drop.size and (drop[0] < 0 or drop[-1] >= n_train):
        msg = f"excluded Train indices must be in range({n_train}), got {drop.tolist()}"
        raise ValueError(msg)
    keep = np.ones(n_train, dtype=bool)
    keep[drop] = False
    return np.flatnonzero(keep).astype(np.int64)
