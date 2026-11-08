"""The near-duplicate check of the corpus index against the evaluation stamps (task 5.1).

The corpus index drops every archive row within 1.5" of an evaluation alert and every
listed ``oid`` and ``candid`` (:mod:`.corpus_index`). This check is the backstop for what
those keys miss: an evaluation stamp whose cutout is in the corpus under a position or an
id that does not match (a wrong or missing coordinate, an object renamed). It compares
pixels, not metadata, in two stages:

1. **Candidates, by cosine on a downsampled view** (cheap, every pair). Each normalised
   stamp (the stored-stamp normalisation, :func:`~.reader.normalise_stamps`) is
   average-pooled by 7 to 9x9 per channel; each channel's 81 values are centred on their
   mean, and the 243 values are scaled to unit norm (:func:`descriptors`; a stamp with no
   structure gets the zero vector, which matches nothing). For each corpus stamp, the
   ``candidates`` reference stamps of highest cosine are kept if their cosine is at least
   ``cosine``. All three channels are used, not the difference alone: on the difference
   channel, every isolated point source pooled to 9x9 looks alike, while the science and
   reference channels carry the field around the source.
2. **Confirmation at full resolution.** A candidate is a near-duplicate when the two
   normalised 63x63x3 stamps differ by at most ``max_abs_diff`` at every pixel. The same
   cutout normalised twice agrees to about 1e-6 (``fetch-alerts`` measures that on the
   stored stamps); two observations of the sky never do (their noise differs), so the
   default 1e-3 separates "the same cutout" from "a similar-looking source". The stage-1
   candidates that fail it are counted, not dropped: a pooled cosine of 0.99 is common
   between unrelated point sources on an empty field.

The cosines are float32 matrix products (numpy), in blocks; the confirmation reads the
reference stamps in place (memory-mapped).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from astro_simclr.data.schema import STAMP_SHAPE

__all__ = [
    "POOL",
    "NearDuplicateRules",
    "NearDuplicates",
    "ReferenceStamps",
    "descriptors",
    "find_near_duplicates",
]

#: Pooling factor of the descriptor: 63 = 7 x 9, so 9x9 per channel.
POOL: Final = 7
_SIDE: Final = STAMP_SHAPE[0] // POOL
#: Query rows per block of the cosine product: 1024 x 51,582 CD21 stamps is ~210 MB.
_BLOCK: Final = 1024


@dataclass(frozen=True, slots=True)
class NearDuplicateRules:
    """The thresholds of the check (module docstring).

    Raises:
        ValueError: unless ``0 < cosine < 1``, ``max_abs_diff >= 0`` and
            ``candidates >= 1``.
    """

    cosine: float = 0.99
    max_abs_diff: float = 1e-3
    candidates: int = 4

    def __post_init__(self) -> None:
        if not 0 < self.cosine < 1 or self.max_abs_diff < 0 or self.candidates < 1:
            raise ValueError(
                "need 0 < cosine < 1 (a float32 cosine of a copy is just below 1), "
                "max_abs_diff >= 0 and candidates >= 1, got "
                f"{self.cosine}, {self.max_abs_diff}, {self.candidates}"
            )

    def to_json(self) -> dict[str, Any]:
        """The record of the index's rules."""
        return {
            "descriptor": f"mean-pool {POOL} to {_SIDE}x{_SIDE}x3, centred per channel, unit norm",
            "cosine": self.cosine,
            "max_abs_diff": self.max_abs_diff,
            "candidates": self.candidates,
        }


def descriptors(stamps: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float32]:
    """The ``(N, 243)`` float32 descriptors of normalised ``(N, 63, 63, 3)`` stamps.

    Mean-pooled by :data:`POOL` (in float64), centred per channel, scaled to unit norm; a
    stamp whose pooled channels are all constant gets zeros.

    Raises:
        ValueError: if ``stamps`` is not ``(N, 63, 63, 3)``.
    """
    if stamps.ndim != 4 or stamps.shape[1:] != STAMP_SHAPE:
        raise ValueError(f"stamps must be (N, 63, 63, 3), got {stamps.shape}")
    n = stamps.shape[0]
    blocks = np.asarray(stamps, dtype=np.float64).reshape(n, _SIDE, POOL, _SIDE, POOL, 3)
    pooled = blocks.mean(axis=(2, 4))
    pooled -= pooled.mean(axis=(1, 2), keepdims=True)
    flat = pooled.reshape(n, -1)
    norm = np.linalg.norm(flat, axis=1, keepdims=True)
    scaled = np.divide(flat, norm, out=np.zeros_like(flat), where=norm > 0)
    return scaled.astype(np.float32)


@dataclass(frozen=True, slots=True)
class ReferenceStamps:
    """The evaluation stamps checked against: named sets of normalised stamps.

    Attributes:
        names: one name per set (``Train``, ``Validation``, ``Test``).
        images: the sets' float32 ``(N_k, 63, 63, 3)`` stamps (memory maps are fine).
    """

    names: tuple[str, ...]
    images: tuple[npt.NDArray[np.float32], ...]
    _starts: npt.NDArray[np.int64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.names) != len(self.images):
            raise ValueError("one name per image set")
        sizes = [0] + [int(i.shape[0]) for i in self.images]
        object.__setattr__(self, "_starts", np.cumsum(sizes).astype(np.int64))

    def __len__(self) -> int:
        return int(self._starts[-1])

    def locate(self, row: int) -> tuple[str, int]:
        """``(set name, row in that set)`` of global reference row ``row``."""
        which = int(np.searchsorted(self._starts, row, side="right")) - 1
        return self.names[which], row - int(self._starts[which])

    def stamp(self, row: int) -> npt.NDArray[np.float32]:
        """The ``(63, 63, 3)`` stamp of global reference row ``row``."""
        which = int(np.searchsorted(self._starts, row, side="right")) - 1
        stamp: npt.NDArray[np.float32] = self.images[which][row - int(self._starts[which])]
        return stamp

    def descriptors(self, chunk_rows: int = 4096) -> npt.NDArray[np.float32]:
        """The descriptors of every reference stamp, in global row order."""
        parts = [np.empty((0, 3 * _SIDE * _SIDE), dtype=np.float32)]
        for images in self.images:
            for start in range(0, images.shape[0], chunk_rows):
                parts.append(descriptors(images[start : start + chunk_rows]))
        return np.concatenate(parts)


@dataclass
class NearDuplicates:
    """The check's outcome over the corpus positions it was given.

    Attributes:
        matched: the corpus positions with a confirmed near-duplicate, sorted.
        matches: one record per confirmed pair: the corpus ``position``, the reference
            ``set`` and ``row``, the ``cosine`` and the ``max_abs_diff``.
        candidates: stage-1 pairs (cosine at least the threshold), confirmed or not.
        checked: corpus stamps compared.
    """

    matched: list[int] = field(default_factory=list)
    matches: list[dict[str, Any]] = field(default_factory=list)
    candidates: int = 0
    checked: int = 0


def _candidates(
    query: npt.NDArray[np.float32], bank: npt.NDArray[np.float32], rules: NearDuplicateRules
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float32]]:
    """Stage 1: ``(query rows, bank rows, cosines)`` of the pairs at or above the threshold.

    At most ``rules.candidates`` per query row, the highest cosines first (ties by the
    lower bank row, so the result is deterministic), sorted by query row.
    """
    found: list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float32]]]
    found = []
    for start in range(0, query.shape[0], _BLOCK):
        sims = query[start : start + _BLOCK] @ bank.T
        qi, bj = np.nonzero(sims >= rules.cosine)
        values = sims[qi, bj]
        order = np.lexsort((bj, -values, qi))
        qi, bj, values = qi[order], bj[order], values[order]
        rank = np.arange(qi.size) - np.searchsorted(qi, qi, side="left")
        keep = rank < rules.candidates
        found.append((qi[keep].astype(np.int64) + start, bj[keep].astype(np.int64), values[keep]))
    if not found:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
    return (
        np.concatenate([f[0] for f in found]),
        np.concatenate([f[1] for f in found]),
        np.concatenate([f[2] for f in found]),
    )


def find_near_duplicates(
    chunks: Iterable[tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]],
    reference: ReferenceStamps,
    rules: NearDuplicateRules,
    *,
    bank: npt.NDArray[np.float32] | None = None,
) -> NearDuplicates:
    """Check corpus stamps against ``reference`` (module docstring).

    ``chunks`` yields ``(positions, normalised stamps (B, 63, 63, 3))``
    (:meth:`~.reader.CorpusStamps.chunks`). ``bank`` is ``reference.descriptors()`` when
    already computed.
    """
    result = NearDuplicates()
    if len(reference) == 0:
        for positions, _ in chunks:
            result.checked += int(positions.size)
        return result
    bank = reference.descriptors() if bank is None else bank
    matched: set[int] = set()
    for positions, stamps in chunks:
        result.checked += int(positions.size)
        qi, bj, cosines = _candidates(descriptors(stamps), bank, rules)
        result.candidates += int(qi.size)
        for i, ref, cosine in zip(qi.tolist(), bj.tolist(), cosines.tolist(), strict=True):
            diff = float(np.abs(stamps[i] - reference.stamp(ref)).max())
            if diff <= rules.max_abs_diff:
                position = int(positions[i])
                matched.add(position)
                name, local = reference.locate(ref)
                result.matches.append(
                    {
                        "position": position,
                        "set": name,
                        "row": local,
                        "cosine": cosine,
                        "max_abs_diff": diff,
                    }
                )
    result.matched = sorted(matched)
    return result
