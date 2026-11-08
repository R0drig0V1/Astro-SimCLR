"""File helpers shared by the loaders and ``inspect-data``: hashing and atomic writes."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

__all__ = ["atomic_write_bytes", "sha256_file"]

#: Read size for hashing: large enough to be I/O bound on multi-GB pickles.
_CHUNK = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of the file at ``path``, read in chunks.

    Raises:
        OSError: if the file cannot be read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` through a temporary file in the same directory.

    A reader never sees a half-written file: an interrupted write leaves only the
    ``.tmp`` file, which the next write replaces.

    Raises:
        OSError: if the directory is not writable.
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
