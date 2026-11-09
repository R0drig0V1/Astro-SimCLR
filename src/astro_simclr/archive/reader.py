"""Read the stamps of a corpus index from the archive's night shards (phase 5, task 5.1).

The archive corpus (:mod:`.layout`) stores **raw** cutouts (float32, NaN and inf kept),
split into ``cutouts-NNNNN.npy`` files of at most 4096 rows per night. Pretraining reads
the rows that a corpus index (:mod:`.corpus_index`) keeps, normalised **exactly like the
stored CD21 stamps** that the 5.0b pilot already trains on (ADR 0025, the 63-px input):

- per stamp and channel, over the full 63x63, ``x -= nanmin(x)``, ``x /= nanmax(x)`` with
  no epsilon, then NaN -> 0 (a constant or all-NaN channel becomes 0, an inf pixel 0):
  :func:`~astro_simclr.alerts.fetch.alerce_normalise`, in float64, then cast to float32.
  It is the normalisation that ``fetch-alerts`` checks against the stored stamps (at most
  1e-6 apart on the rows it verifies), so a corpus stamp and a CD21 stamp of the same alert
  agree to that precision.

:class:`CorpusStamps` memory-maps every night that the index uses (``layout.load_night``:
the files are checked against their ``meta.json``, read-only, nothing unpickled) and maps
index position ``i`` to its cutout file and row. :meth:`CorpusStamps.read` serves any set
of positions **sorted by file offset** (one forward sweep through each file the batch
touches; the data tier 3 of ADR 0018, over the night shards instead of one file) and puts
each stamp at its requested position. Nothing is cached here: the OS page cache is the
cache.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from astro_simclr.alerts.fetch import alerce_normalise
from astro_simclr.archive.layout import CORPUS_INDEX_DTYPE, load_night, night_dir
from astro_simclr.data.schema import STAMP_SHAPE, DataError

__all__ = ["STAMP_BYTES", "CorpusStamps", "normalise_stamps"]

#: Bytes of one float32 (63, 63, 3) stamp.
STAMP_BYTES: Final = int(np.prod(STAMP_SHAPE)) * 4


def normalise_stamps(raw: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
    """The stored-stamp normalisation of raw ``(..., 63, 63, 3)`` cutouts, float32 (module doc).

    The result is finite and in [0, 1]; the input is not modified.
    """
    return alerce_normalise(raw).astype(np.float32)


class CorpusStamps:
    """The cutouts of a corpus index's rows, memory-mapped from the archive (module docstring).

    Args:
        archive: the corpus directory (``ingest-archive --out``).
        rows: the index rows (:data:`~astro_simclr.archive.layout.CORPUS_INDEX_DTYPE`):
            position ``i`` is the stamp of night ``rows["night"][i]``, row ``rows["row"][i]``.

    Raises:
        DataError: if ``rows`` is not a corpus index table, a night is missing or invalid
            (``load_night``), or a row is outside its night, or its ``oid`` or ``candid``
            is not the night's (the index was built from another archive).
    """

    def __init__(self, archive: Path, rows: npt.NDArray[np.void]) -> None:
        if rows.dtype != CORPUS_INDEX_DTYPE or rows.ndim != 1:
            raise DataError(f"the corpus index rows must be {CORPUS_INDEX_DTYPE.descr}")
        self.archive = archive
        self._files: list[npt.NDArray[np.float32]] = []
        n = len(rows)
        self._file = np.empty(n, dtype=np.int64)
        self._offset = np.empty(n, dtype=np.int64)
        for night in np.unique(rows["night"]).tolist():
            loaded = load_night(night_dir(archive, int(night)))
            at = np.flatnonzero(rows["night"] == night)
            local = rows["row"][at]
            if local.size and (local.min() < 0 or local.max() >= len(loaded)):
                raise DataError(f"night {night}: an index row is outside 0..{len(loaded) - 1}")
            stored = loaded.index[local]
            if not (
                np.array_equal(stored["oid"], rows["oid"][at])
                and np.array_equal(stored["candid"], rows["candid"][at])
            ):
                raise DataError(
                    f"night {night}: the index's oid/candid differ from the archive's rows; "
                    "the index was built from another corpus"
                )
            # Row r of the night is row r - start of the file holding it.
            starts = np.cumsum([0] + [len(chunk) for chunk in loaded.cutouts])
            which = np.searchsorted(starts, local, side="right") - 1
            self._file[at] = which + len(self._files)
            self._offset[at] = local - starts[which]
            self._files.extend(loaded.cutouts)
        # The storage order of each position: by file, then by row in the file.
        self._key = self._file * (int(self._offset.max()) + 1 if n else 1) + self._offset

    def __len__(self) -> int:
        return int(self._file.shape[0])

    @property
    def nbytes(self) -> int:
        """Bytes of every indexed stamp as float32 (what a host copy would hold)."""
        return len(self) * STAMP_BYTES

    def read(
        self, positions: npt.ArrayLike, out: npt.NDArray[np.float32] | None = None
    ) -> npt.NDArray[np.float32]:
        """The raw cutouts ``(B, 63, 63, 3)`` of index positions ``positions``, in that order.

        The positions are read sorted by file offset (module docstring). ``out`` (a
        float32 ``(B, 63, 63, 3)`` array) receives them when given.

        Raises:
            IndexError: if a position is outside ``range(len(self))``.
        """
        index = np.asarray(positions, dtype=np.int64).reshape(-1)
        if index.size and (index.min() < 0 or index.max() >= len(self)):
            raise IndexError(f"positions must be in range({len(self)})")
        target = np.empty((index.size, *STAMP_SHAPE), dtype=np.float32) if out is None else out
        order = np.argsort(self._key[index], kind="stable")
        files = self._file[index[order]]
        bounds = np.flatnonzero(np.diff(files)) + 1
        for part in np.split(order, bounds) if order.size else []:
            source = self._files[int(self._file[index[part[0]]])]
            target[part] = source[self._offset[index[part]]]
        return target

    def normalised(self, positions: npt.ArrayLike) -> npt.NDArray[np.float32]:
        """The normalised stamps ``(B, 63, 63, 3)`` of ``positions`` (:func:`normalise_stamps`)."""
        return normalise_stamps(self.read(positions))

    def chunks(
        self, positions: npt.NDArray[np.int64], chunk_rows: int = 4096
    ) -> Iterator[tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]]:
        """``(positions, normalised stamps)`` over ``positions``, ``chunk_rows`` at a time.

        ``positions`` should be in storage order (an index's rows are) so that the reads
        sweep each file forward once.

        Raises:
            ValueError: if ``chunk_rows < 1`` (on the first iteration).
        """
        if chunk_rows < 1:
            raise ValueError(f"chunk_rows must be >= 1, got {chunk_rows}")
        for start in range(0, positions.size, chunk_rows):
            part = positions[start : start + chunk_rows]
            yield part, self.normalised(part)
