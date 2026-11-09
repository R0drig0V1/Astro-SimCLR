"""The stamp fits of a corpus index: the sidecar that lets a corpus run use the noise view.

The C5 noise view (:mod:`astro_simclr.simclr.views`, ADR 0006) moves the difference of a
stamp through its fit ``diff ~ a*sci + b*tmpl + c`` and scales the noise by the stamp's
background sigmas: :func:`~astro_simclr.stamp_relation.stamp_fits`, (N, 6) float32
(:data:`~astro_simclr.stamp_relation.FIT_COLUMNS`). A layout run computes them once from
its Train stamps when it starts; a corpus run cannot (the stamps are read per batch from
the archive, and 145k-390k stamps take minutes to fit), so they are computed **once per
index** by ``build-corpus-fits`` and stored beside it (ADR 0019, amendment of 2026-11-09):

- **The values:** row ``i`` is ``stamp_fits`` of index position ``i``'s stamp as the pilot
  reads it (:meth:`~astro_simclr.archive.reader.CorpusStamps.normalised`: the stored-stamp
  normalisation, float32), channels first and contiguous: exactly the input that a layout
  run gives ``stamp_fits``, so a stamp has the same fits through either path (a test checks
  it bit for bit).
- **The binding:** the sidecar records the SHA-256 of the index file it was computed from
  (and the index's row count and ``exclusion_sha256``); :func:`verify_corpus_fits` refuses
  fits of any other index, so a rebuilt index at the same path cannot be paired with stale
  fits. The archive is checked against the index before the stamps are read
  (:func:`~astro_simclr.archive.corpus_index.check_archive`), as the pilot does.

**Why a sidecar, not a field of the index** (the alternative was an index format v2): the
fits are a pure function of the index and the archive, not part of the exclusion, so an
index already built (and its ``exclusion_sha256``, the value a manifest records) stays
valid and is not rebuilt for them; a run without the noise view needs nothing new.

**The file** (``build-corpus-fits INDEX_FILE --out FITS_FILE``; not committed: it names
local paths) is an ``.npz`` without pickles, written atomically
(:func:`~astro_simclr.archive.corpus_index.write_npz_atomically`): ``fits``, float32
``(n, 6)`` in index order, and ``meta``, a JSON record (:data:`FITS_FORMAT` version
:data:`FITS_VERSION`) as a 0-d string. About 24 bytes per row (3.5 MB for 145k rows).
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from astro_simclr.archive.corpus_index import (
    check_archive,
    read_corpus_index,
    write_npz_atomically,
)
from astro_simclr.archive.reader import CorpusStamps
from astro_simclr.data.files import sha256_file
from astro_simclr.data.schema import DataError
from astro_simclr.stamp_relation import FIT_COLUMNS, stamp_fits

__all__ = [
    "FITS_FORMAT",
    "FITS_VERSION",
    "CorpusFits",
    "compute_corpus_fits",
    "read_corpus_fits",
    "verify_corpus_fits",
    "write_corpus_fits",
]

#: ``format`` of the sidecar's ``meta``; its version is :data:`FITS_VERSION`.
FITS_FORMAT: Final = "astro-simclr-corpus-fits"
FITS_VERSION: Final = 1
#: Rows read, normalised and fitted at a time (``stamp_fits`` also works in 4096-row chunks).
DEFAULT_CHUNK_ROWS: Final = 4096
_PROGRESS = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class CorpusFits:
    """The stamp fits of a corpus index (module docstring).

    Attributes:
        fits: float32 ``(n, 6)``, row ``i`` the fit of index position ``i``.
        meta: the record: the index (path, SHA-256, ``n``, ``exclusion_sha256``), the
            archive, the columns and how the values were computed.
    """

    fits: npt.NDArray[np.float32]
    meta: dict[str, Any]

    def __len__(self) -> int:
        return int(self.fits.shape[0])

    @property
    def index_sha256(self) -> str:
        """SHA-256 of the index file the fits were computed from."""
        return str(self.meta["index"]["sha256"])


def compute_corpus_fits(
    index_path: Path,
    *,
    archive: Path | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    progress: _PROGRESS | None = None,
) -> CorpusFits:
    """``stamp_fits`` of every stamp of the index at ``index_path`` (module docstring).

    ``archive`` defaults to the one the index records; it must hold the indexed nights
    unchanged. The stamps are read in storage order, ``chunk_rows`` at a time;
    ``progress`` receives one line per chunk.

    Raises:
        ValueError: if ``chunk_rows < 1``.
        DataError: if the index is unreadable, or the archive is not the one it indexed.
        OSError: if a file cannot be read.
    """
    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be >= 1, got {chunk_rows}")
    say = progress or (lambda _line: None)
    index_sha256 = sha256_file(index_path)
    index = read_corpus_index(index_path)
    source = Path(archive or index.meta["archive"]["path"])
    check_archive(index, source)
    stamps = CorpusStamps(source, index.rows)
    fits = np.empty((len(stamps), len(FIT_COLUMNS)), dtype=np.float32)
    for positions, normalised in stamps.chunks(np.arange(len(stamps)), chunk_rows):
        # The layout path's input to stamp_fits: float32 (B, 3, 63, 63), C-contiguous.
        fits[positions] = stamp_fits(np.ascontiguousarray(normalised.transpose(0, 3, 1, 2)))
        say(f"fits: {int(positions[-1]) + 1}/{len(stamps)} stamps")
    meta: dict[str, Any] = {
        "format": FITS_FORMAT,
        "format_version": FITS_VERSION,
        "created_by": f"astro-simclr {version('astro-simclr')}",
        "index": {
            "path": str(index_path),
            "sha256": index_sha256,
            "n": len(index),
            "exclusion_sha256": index.meta["exclusion_sha256"],
        },
        "archive": {"path": str(source), "corpus_sha256": index.meta["archive"]["corpus_sha256"]},
        "columns": list(FIT_COLUMNS),
        "input": "CorpusStamps.normalised (normalise_stamps: the stored-stamp min-max, "
        "NaN -> 0), float32, channels first",
        "function": "astro_simclr.stamp_relation.stamp_fits (float64, stored as float32)",
        "chunk_rows": chunk_rows,
        "n": len(index),
    }
    return CorpusFits(fits, meta)


def write_corpus_fits(fits: CorpusFits, path: Path, *, overwrite: bool = False) -> None:
    """Write ``fits`` to the ``.npz`` ``path`` (module docstring), atomically.

    Raises:
        DataError: if ``path`` exists and ``overwrite`` is False.
        OSError: if the file cannot be written.
    """
    write_npz_atomically(path, {"fits": fits.fits}, fits.meta, overwrite=overwrite)


def read_corpus_fits(path: Path) -> CorpusFits:
    """Read a sidecar written by :func:`write_corpus_fits` (no pickles).

    Raises:
        DataError: if the file is unreadable, of another format or version, or ``fits`` is
            not a finite float32 ``(meta["n"], 6)`` array.
    """
    try:
        with np.load(path, allow_pickle=False) as archive:
            fits = archive["fits"]
            meta = json.loads(str(archive["meta"]))
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as err:
        raise DataError(f"{path}: not a corpus fits file ({err})") from None
    if (
        not isinstance(meta, dict)
        or meta.get("format") != FITS_FORMAT
        or meta.get("format_version") != FITS_VERSION
    ):
        raise DataError(f"{path}: not an {FITS_FORMAT} (version {FITS_VERSION})")
    shape = (meta.get("n"), len(FIT_COLUMNS))
    if fits.dtype != np.float32 or fits.shape != shape or not np.isfinite(fits).all():
        raise DataError(f"{path}: the fits are not finite float32 of shape {shape}")
    return CorpusFits(fits, meta)


def verify_corpus_fits(fits: CorpusFits, *, index_sha256: str, n: int, source: Path) -> None:
    """Check that ``fits`` (read from ``source``) were computed from the index ``index_sha256``.

    Raises:
        DataError: if the fits name another index file, or hold another number of rows.
    """
    if fits.index_sha256 != index_sha256 or len(fits) != n:
        raise DataError(
            f"{source} holds the fits of another index (SHA-256 {fits.index_sha256[:12]}..., "
            f"{len(fits)} rows; this index is {index_sha256[:12]}..., {n} rows): rebuild them "
            "with `astro-simclr build-corpus-fits INDEX_FILE --out FITS_FILE`"
        )
