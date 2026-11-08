"""Stream the avro alerts out of one nightly ZTF archive tarball, never extracting it.

The ZTF public alert archive (``https://ztf.uw.edu/alerts/public/``) holds one
``ztf_public_YYYYMMDD.tar.gz`` per night, each a gzip-compressed tar of one avro file per
alert. :class:`TarballStream` reads such a file in one sequential pass:

- the raw bytes go through a SHA-256 (the night's provenance) and a byte counter;
- gzip decompresses them, and the decompressed stream is capped
  (:data:`MAX_DECOMPRESSED_BYTES`), so a gzip bomb stops at the cap instead of filling the
  memory or the time budget;
- :mod:`tarfile` reads the tar in stream mode (``r|``), member by member; only regular
  files whose name ends in ``.avro`` and whose size is at most
  :data:`MAX_MEMBER_BYTES` are read into memory and yielded; every other member is skipped
  and counted (directories, links, devices, other files, oversized alerts);
- the number of members is capped (:data:`MAX_MEMBERS`);
- tar's own metadata members (pax headers, GNU long names and links), which :mod:`tarfile`
  reads whole into memory, are refused above :data:`MAX_HEADER_BYTES`, and GNU sparse
  members are refused (``_BoundedTarInfo``; ZTF's tarballs use neither);
- after the last member the gzip stream is read to its end, so its CRC-32 and length
  trailer is checked: a truncated download or a corrupt tail raises instead of being
  committed.

Nothing is written to disk and the member names are never used as paths, so a hostile tar
(absolute names, ``..``, links, devices) cannot place a file anywhere. The tarball is
untrusted: a corrupt or truncated gzip or tar stream raises :class:`TarballError` (a
:class:`~astro_simclr.data.schema.DataError`), and the caller must not record the night as
complete. After the last member the rest of the file is read through the hash, so
:attr:`TarballStream.sha256` is the SHA-256 of the whole file.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import re
import tarfile
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import BinaryIO, Final, Self

from astro_simclr.alerts.alerce import MAX_AVRO_BYTES
from astro_simclr.data.schema import DataError

__all__ = [
    "MAX_DECOMPRESSED_BYTES",
    "MAX_HEADER_BYTES",
    "MAX_MEMBERS",
    "MAX_MEMBER_BYTES",
    "TARBALL_PATTERN",
    "Member",
    "StreamStats",
    "TarballError",
    "TarballStream",
    "night_of",
    "tarball_name",
]

#: The archive's file name for one night (the night is the UT date ``YYYYMMDD``).
TARBALL_PATTERN: Final = re.compile(r"ztf_public_(\d{8})\.tar\.gz")
#: Largest avro member read: the cap of the ALeRCE download (a ZTF alert with 30 days of
#: previous candidates is about 60-100 KB).
MAX_MEMBER_BYTES: Final = MAX_AVRO_BYTES
#: Largest decompressed tar stream: about 20x a 2020 night (about 2.4 GB compressed, and
#: the gzipped FITS cutouts barely compress again).
MAX_DECOMPRESSED_BYTES: Final = 64 * 1024**3
#: Most tar members accepted in one tarball: about 10x the alerts of a busy night.
MAX_MEMBERS: Final = 5_000_000
#: Largest tar metadata member (pax header, GNU long name or link): real ones are tiny.
MAX_HEADER_BYTES: Final = 1024 * 1024
#: The tar member types whose body tarfile reads into memory as metadata.
_HEADER_TYPES: Final = frozenset(
    {tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.SOLARIS_XHDTYPE}
    | {tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK}
)
#: Read size of the raw file.
_CHUNK: Final = 1024 * 1024


class TarballError(DataError):
    """The tarball is not a readable gzip-compressed tar, or it exceeds a limit."""


def night_of(path: Path) -> int:
    """The night ``YYYYMMDD`` of an archive tarball, from its file name.

    Raises:
        TarballError: if the name is not ``ztf_public_YYYYMMDD.tar.gz`` with a valid date.
    """
    match = TARBALL_PATTERN.fullmatch(path.name)
    if match is None:
        raise TarballError(f"{path.name}: not an archive tarball name (ztf_public_YYYYMMDD.tar.gz)")
    text = match.group(1)
    try:
        date(int(text[:4]), int(text[4:6]), int(text[6:]))
    except ValueError:
        raise TarballError(f"{path.name}: {text} is not a valid date") from None
    return int(text)


def tarball_name(night: int) -> str:
    """The archive's file name of ``night`` (``YYYYMMDD``)."""
    return f"ztf_public_{night:08d}.tar.gz"


@dataclass(frozen=True, slots=True)
class Member:
    """One avro member of the tarball.

    Attributes:
        position: its position among all the tarball's members (0-based), for tracing.
        name: its name in the tar (untrusted text, never used as a path).
        data: its bytes.
    """

    position: int
    name: str
    data: bytes


@dataclass(slots=True)
class StreamStats:
    """What the stream saw; complete once the iteration ends.

    Attributes:
        members: tar members seen (every kind).
        avro: avro members yielded.
        skipped_not_file: directories, links, devices and other non-regular members.
        skipped_not_avro: regular files whose name does not end in ``.avro``.
        skipped_oversize: avro members larger than the member cap.
        compressed_bytes: bytes of the file.
        decompressed_bytes: bytes of the decompressed tar stream read.
        sha256: hex SHA-256 of the whole file (empty until the stream is exhausted).
    """

    members: int = 0
    avro: int = 0
    skipped_not_file: int = 0
    skipped_not_avro: int = 0
    skipped_oversize: int = 0
    compressed_bytes: int = 0
    decompressed_bytes: int = 0
    sha256: str = ""

    def to_json(self) -> dict[str, int | str]:
        """The stats as a JSON object (``meta.json["source"]`` of a night)."""
        return {
            "members": self.members,
            "avro": self.avro,
            "skipped_not_file": self.skipped_not_file,
            "skipped_not_avro": self.skipped_not_avro,
            "skipped_oversize": self.skipped_oversize,
            "compressed_bytes": self.compressed_bytes,
            "decompressed_bytes": self.decompressed_bytes,
            "sha256": self.sha256,
        }


class _BoundedTarInfo(tarfile.TarInfo):
    """A tar header that refuses oversized metadata members and sparse files.

    :mod:`tarfile` reads the body of a pax or GNU long-name header in one piece, before
    the member is handed to the caller, so the member cap cannot bound it; a hostile
    header declaring gigabytes (which compress about 1000:1) would fill the memory. The
    check runs on each 512-byte header as it is parsed, before its body is read.
    """

    # ``_proc_member`` dispatches every parsed header to its reader (pax, GNU long name,
    # sparse, regular) before the body is read; it is the one hook every supported
    # Python calls (3.13 no longer routes through ``frombuf``). The tests pin it.
    def _proc_member(self, archive: tarfile.TarFile) -> Self:
        if self.type in _HEADER_TYPES and self.size > MAX_HEADER_BYTES:
            raise TarballError(
                f"a tar metadata member of {self.size} bytes (more than {MAX_HEADER_BYTES})"
            )
        if self.type == tarfile.GNUTYPE_SPARSE:
            raise TarballError("a GNU sparse tar member (not accepted)")
        member: Self = super()._proc_member(archive)  # type: ignore[misc]
        return member


@dataclass
class _HashingReader:
    """The raw file, read through a SHA-256 and a byte counter."""

    handle: BinaryIO
    stats: StreamStats
    digest: hashlib._Hash = field(default_factory=hashlib.sha256)

    def read(self, size: int = -1) -> bytes:
        chunk = self.handle.read(size if size is not None and size >= 0 else _CHUNK)
        self.digest.update(chunk)
        self.stats.compressed_bytes += len(chunk)
        return chunk

    def seek(self, _offset: int, /) -> object:
        raise io.UnsupportedOperation("the tarball is read as a stream")

    def drain(self) -> None:
        while self.read(_CHUNK):
            pass


@dataclass
class _LimitedReader:
    """The decompressed stream, refused past ``limit`` bytes."""

    source: gzip.GzipFile
    stats: StreamStats
    limit: int

    def read(self, size: int = -1) -> bytes:
        room = self.limit - self.stats.decompressed_bytes
        want = _CHUNK if size is None or size < 0 else size
        chunk = self.source.read(min(want, room + 1))
        self.stats.decompressed_bytes += len(chunk)
        if self.stats.decompressed_bytes > self.limit:
            raise TarballError(f"the tarball decompresses to more than {self.limit} bytes")
        return chunk

    # The rest of tarfile's file protocol: a stream is never written, sought or closed here.
    def write(self, _data: bytes, /) -> object:
        raise io.UnsupportedOperation("the tarball is read only")

    def seek(self, _offset: int, /) -> object:
        raise io.UnsupportedOperation("the tarball is read as a stream")

    def tell(self) -> int:
        return self.stats.decompressed_bytes

    def close(self) -> None:
        """The underlying gzip stream is closed by its own ``with`` block."""


class TarballStream:
    """Iterate the avro members of one archive tarball (module docstring).

    ``source`` is the tarball's path, or an open binary stream (``ingest-archive -``: a
    download piped in, so no tarball is ever stored); a stream is read to its end but not
    closed. ``name`` names it in messages (default: the path's file name). Iterate it
    once; :attr:`stats` is complete (with the SHA-256) when the iteration ends without an
    exception.

    Raises (while iterating):
        TarballError: if the file is not a gzip-compressed tar, is truncated or corrupt,
            decompresses past ``max_decompressed_bytes`` or has more than ``max_members``.
        OSError: if the file cannot be read.
    """

    def __init__(
        self,
        source: Path | BinaryIO,
        *,
        name: str | None = None,
        max_member_bytes: int = MAX_MEMBER_BYTES,
        max_decompressed_bytes: int = MAX_DECOMPRESSED_BYTES,
        max_members: int = MAX_MEMBERS,
    ) -> None:
        self.source = source
        self.name = name or (source.name if isinstance(source, Path) else "<stream>")
        self.max_member_bytes = max_member_bytes
        self.max_decompressed_bytes = max_decompressed_bytes
        self.max_members = max_members
        self.stats = StreamStats()

    def __iter__(self) -> Iterator[Member]:
        if isinstance(self.source, Path):
            with self.source.open("rb") as handle:
                yield from self._read(handle)
        else:
            yield from self._read(self.source)

    def _read(self, handle: BinaryIO) -> Iterator[Member]:
        raw = _HashingReader(handle, self.stats)
        try:
            with gzip.GzipFile(fileobj=raw, mode="rb") as unzipped:
                limited = _LimitedReader(unzipped, self.stats, self.max_decompressed_bytes)
                with tarfile.open(fileobj=limited, mode="r|", tarinfo=_BoundedTarInfo) as tar:
                    yield from self._members(tar)
                # tarfile stops at the end-of-archive blocks; reading on to the gzip
                # end checks its CRC-32 and length trailer (BadGzipFile, EOFError).
                while limited.read(_CHUNK):
                    pass
        # gzip and tarfile raise these on corrupt or truncated input (a real I/O
        # error is another OSError and propagates).
        except (gzip.BadGzipFile, EOFError, zlib.error, tarfile.TarError) as err:
            raise TarballError(
                f"{self.name}: not a readable .tar.gz ({type(err).__name__}: {err})"
            ) from None
        raw.drain()
        self.stats.sha256 = raw.digest.hexdigest()

    def _members(self, tar: tarfile.TarFile) -> Iterator[Member]:
        for info in tar:
            position = self.stats.members
            self.stats.members += 1
            if self.stats.members > self.max_members:
                raise TarballError(f"{self.name}: more than {self.max_members} members")
            if not info.isreg():
                self.stats.skipped_not_file += 1
                continue
            if not info.name.endswith(".avro"):
                self.stats.skipped_not_avro += 1
                continue
            if info.size > self.max_member_bytes:
                self.stats.skipped_oversize += 1
                continue
            extracted = tar.extractfile(info)
            if extracted is None:  # pragma: no cover - regular files always extract
                self.stats.skipped_not_file += 1
                continue
            data = extracted.read(self.max_member_bytes + 1)
            if len(data) != info.size:  # pragma: no cover - tarfile raises on truncation
                raise TarballError(f"{self.name}: member {position} is truncated")
            self.stats.avro += 1
            yield Member(position, info.name, data)
