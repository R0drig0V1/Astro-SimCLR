"""The label sources of R5: dated snapshots of two public tables, and their parsers (task 3.5).

R5's AGN, VS and part of its SN come from the **ALeRCE light-curve classifier's labelled
set** (Sánchez-Sáez et al. 2021), Zenodo record 10.5281/zenodo.4279623, CC BY 4.0. The CSV
is inside a 1.12 GB archive: :func:`download_zenodo` streams the archive to disk, checks its
size and Zenodo's MD5 (pinned here, read from the record on 2026-11-08) and its SHA-256,
reads the one member it needs **without extracting anything** (no path from the archive is
ever used), writes the CSV and deletes the archive (``keep_archive`` keeps it).

The rest of the SN come from the **ZTF Bright Transient Survey** (BTS; Fremling et al. 2020,
ApJ 895, 32; Perley et al. 2020, ApJ 904, 35): its public sample explorer, queried for the
spectroscopically classified SN with the quality and purity cuts (:data:`BTS_URL`, the query
of the task-3.0 probe, 6,758 rows on 2026-11-08). It needs no login. On 2026-11-08 the BTS
pages state no terms of use or licence; they ask to cite the BTS papers (``docs/data.md``,
"R5", records the check). :func:`download_bts` fetches it once through task 3.1's polite
:class:`~astro_simclr.alerts.transport.Fetcher`.

Each source is written as ``SOURCES/<name>.csv`` with ``SOURCES/<name>.json``: the URL, the
retrieval time (UTC), size and SHA-256, the licence or terms, the citation. The snapshots
are never committed (ADR 0012); the R5 build records their SHA-256 and refuses a file that
does not match its record (:func:`load_source`).

Both files are **untrusted**: they are parsed with the ``csv`` module under a fixed schema
(:func:`read_zenodo`, :func:`read_bts`), sizes are capped, every ``oid`` must look like a
ZTF object id, and a malformed row is an error naming its line.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tarfile
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Final

from astro_simclr.alerts.alerce import is_oid
from astro_simclr.alerts.snapshot import utc_now
from astro_simclr.alerts.transport import USER_AGENT, https_opener
from astro_simclr.data.files import sha256_file
from astro_simclr.data.schema import DataError

__all__ = [
    "BTS",
    "BTS_URL",
    "MAX_CSV_BYTES",
    "ZENODO",
    "ZENODO_ARCHIVE_MD5",
    "ZENODO_ARCHIVE_SIZE",
    "ZENODO_ARCHIVE_URL",
    "ZENODO_MEMBER",
    "SourceRow",
    "download_bts",
    "download_zenodo",
    "load_source",
    "read_bts",
    "read_zenodo",
    "urllib_stream",
]

#: Source names (file stems under the sources directory).
ZENODO: Final = "zenodo-4279623"
BTS: Final = "bts"

ZENODO_RECORD_URL: Final = "https://zenodo.org/api/records/4279623"
ZENODO_ARCHIVE: Final = "files_for_lc_classifier_SanchezSaez2020.tar.gz"
ZENODO_ARCHIVE_URL: Final = f"{ZENODO_RECORD_URL}/files/{ZENODO_ARCHIVE}/content"
#: Zenodo's size and MD5 of the archive (``/api/records/4279623``, read 2026-11-08).
ZENODO_ARCHIVE_SIZE: Final = 1_121_636_335
ZENODO_ARCHIVE_MD5: Final = "f0c8c47fb530613718a3cf0c4678a3e4"
#: The labelled set inside the archive (10.2 MB, 123,496 rows on 2026-11-08).
ZENODO_MEMBER: Final = "labeled_set_lc_classifier_SanchezSaez_2020.csv"
ZENODO_LICENCE: Final = "CC-BY-4.0"
ZENODO_CITATION: Final = (
    "Sánchez-Sáez et al. 2021, AJ 161, 141 (ALeRCE light curve classifier); "
    "doi:10.5281/zenodo.4279623"
)

#: The BTS explorer query: spectroscopic SN (``subsample=sn``), quality and purity cuts, CSV.
BTS_URL: Final = (
    "https://sites.astro.caltech.edu/ztf/bts/explorer.php?f=s&subsample=sn&classstring="
    "&classexclude=&quality=y&purity=y&ztflink=lasair&lastdet=&startsavedate="
    "&startpeakdate=&startlastdate=&startra=&startdec=&startz=&startdur=&startrise="
    "&startfade=&startpeakmag=&startabsmag=&starthostabs=&starthostcol=&startb=&startav="
    "&endsavedate=&endpeakdate=&endlastdate=&endra=&enddec=&endz=&enddur=&endrise="
    "&endfade=&endpeakmag=&endabsmag=&endhostabs=&endhostcol=&endb=&endav=&sort=peakmag"
    "&format=csv"
)
BTS_TERMS: Final = (
    "No terms of use or licence stated on the BTS pages (explorer.php, bts.php, "
    "explorer_info.html; checked 2026-11-08); the explorer asks to cite the BTS papers."
)
BTS_CITATION: Final = (
    "Fremling et al. 2020, ApJ 895, 32 (BTS I); Perley et al. 2020, ApJ 904, 35 (BTS II)"
)

#: Largest accepted CSV (both are about 1-10 MB).
MAX_CSV_BYTES: Final = 64 * 1024 * 1024
#: Longest accepted Zenodo download, in seconds (1.12 GB at a slow 1 MB/s is 19 minutes).
ZENODO_DEADLINE: Final = 6 * 3600.0
_CHUNK: Final = 1024 * 1024

#: ``url -> a readable binary stream`` (closed by the caller's ``with``).
type Stream = Callable[[str], AbstractContextManager[IO[bytes]]]


def urllib_stream(url: str, timeout: float = 120.0) -> AbstractContextManager[IO[bytes]]:
    """GET ``url`` with the package's ``User-Agent``, following https redirects only.

    Raises:
        ValueError: if ``url`` is not ``https://``.
        OSError, urllib.error.HTTPError: if the request fails.
    """
    import urllib.request

    if not url.startswith("https://"):
        raise ValueError(f"refusing to request {url!r}: only https:// URLs")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    response: AbstractContextManager[IO[bytes]] = https_opener().open(request, timeout=timeout)
    return response


@dataclass(frozen=True, slots=True)
class SourceRow:
    """One labelled object of a source table.

    Attributes:
        row: the data row's position in the file (0 for the first data row).
        oid: the ZTF object id.
        cls: the source's class (Zenodo ``classALeRCE``; BTS ``type``).
        ra, dec: the source's position in degrees, ``None`` if it gives none.
        name: Zenodo's ``id_source`` or BTS's ``IAUID`` (informational).
    """

    row: int
    oid: str
    cls: str
    ra: float | None
    dec: float | None
    name: str


def _write_record(dest: Path, name: str, record: dict[str, Any]) -> None:
    path = dest / f"{name}.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(dest: Path, name: str, data: bytes) -> Path:
    path = dest / f"{name}.csv"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return path


def _stream_to(
    stream: Stream,
    url: str,
    path: Path,
    max_bytes: int,
    deadline: float,
    clock: Callable[[], float],
) -> tuple[int, str, str]:
    """Copy ``url`` to ``path``; return (size, md5, sha256)."""
    md5, sha = hashlib.md5(usedforsecurity=False), hashlib.sha256()
    size = 0
    start = clock()
    with stream(url) as body, path.open("wb") as out:
        while chunk := body.read(_CHUNK):
            size += len(chunk)
            if size > max_bytes:
                raise DataError(f"{url}: larger than the expected {max_bytes} bytes")
            if clock() - start > deadline:
                raise DataError(f"{url}: not complete after {deadline:g} s")
            md5.update(chunk)
            sha.update(chunk)
            out.write(chunk)
    return size, md5.hexdigest(), sha.hexdigest()


def _member_bytes(archive: Path, member_name: str) -> bytes:
    """The bytes of the regular file ``member_name`` (by base name) of a .tar.gz, unextracted."""
    try:
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                if Path(member.name).name != member_name:
                    continue
                if not member.isfile() or member.size > MAX_CSV_BYTES:
                    raise DataError(f"{archive}: {member.name} is not a file of at most 64 MB")
                handle = tar.extractfile(member)
                if handle is None:
                    raise DataError(f"{archive}: cannot read {member.name}")
                return handle.read(MAX_CSV_BYTES + 1)
    except (tarfile.TarError, OSError, EOFError) as err:
        raise DataError(f"{archive}: not a readable .tar.gz ({err})") from None
    raise DataError(f"{archive}: no member named {member_name}")


def download_zenodo(
    dest: Path,
    *,
    stream: Stream = urllib_stream,
    now: Callable[[], str] = utc_now,
    clock: Callable[[], float] = time.monotonic,
    expected_size: int = ZENODO_ARCHIVE_SIZE,
    expected_md5: str = ZENODO_ARCHIVE_MD5,
    keep_archive: bool = False,
) -> Path:
    """Snapshot the Zenodo labelled set into ``dest`` (module docstring); return the CSV path.

    Raises:
        DataError: if the archive's size or MD5 differs from Zenodo's, the archive is not a
            readable .tar.gz, or it lacks the labelled set.
        OSError, urllib.error.HTTPError: if the download or a write fails.
    """
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / f".{ZENODO_ARCHIVE}.partial"
    retrieved_at = now()
    try:
        size, md5, sha = _stream_to(
            stream, ZENODO_ARCHIVE_URL, archive, expected_size, ZENODO_DEADLINE, clock
        )
        if size != expected_size or md5 != expected_md5:
            raise DataError(
                f"{ZENODO_ARCHIVE_URL}: {size} bytes, MD5 {md5}; Zenodo says "
                f"{expected_size} bytes, MD5 {expected_md5}"
            )
        data = _member_bytes(archive, ZENODO_MEMBER)
        if len(data) > MAX_CSV_BYTES:
            raise DataError(f"{ZENODO_MEMBER} is larger than {MAX_CSV_BYTES} bytes")
        path = _write_csv(dest, ZENODO, data)
        if keep_archive:
            os.replace(archive, dest / ZENODO_ARCHIVE)
    finally:
        archive.unlink(missing_ok=True)
    _write_record(
        dest,
        ZENODO,
        {
            "name": ZENODO,
            "url": ZENODO_ARCHIVE_URL,
            "record_url": ZENODO_RECORD_URL,
            "retrieved_at": retrieved_at,
            "archive": {"name": ZENODO_ARCHIVE, "size": size, "md5": md5, "sha256": sha},
            "member": ZENODO_MEMBER,
            "file": path.name,
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "licence": ZENODO_LICENCE,
            "citation": ZENODO_CITATION,
        },
    )
    return path


def download_bts(
    dest: Path, *, get: Callable[[str, int], bytes], now: Callable[[], str] = utc_now
) -> Path:
    """Snapshot the BTS explorer's SN table into ``dest``; return the CSV path.

    ``get(url, max_bytes)`` is the request, normally a rate-limited ``Fetcher.get``. The
    body is checked to parse (:func:`read_bts`) before it is written.

    Raises:
        HttpError: if the request fails.
        DataError: if the body is not the expected CSV.
        OSError: if a write fails.
    """
    dest.mkdir(parents=True, exist_ok=True)
    retrieved_at = now()
    data = get(BTS_URL, MAX_CSV_BYTES)
    rows = _parse_bts(data, BTS_URL)
    path = _write_csv(dest, BTS, data)
    _write_record(
        dest,
        BTS,
        {
            "name": BTS,
            "url": BTS_URL,
            "retrieved_at": retrieved_at,
            "file": path.name,
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "rows": len(rows),
            "terms": BTS_TERMS,
            "citation": BTS_CITATION,
        },
    )
    return path


def load_source(sources: Path, name: str) -> tuple[Path, dict[str, Any]]:
    """The CSV of source ``name`` under ``sources`` and its record, after the SHA-256 check.

    Raises:
        DataError: if the record or the CSV is missing or invalid, or the hash differs.
    """
    record_path = sources / f"{name}.json"
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise DataError(f"{record_path}: cannot read the source record ({err})") from None
    if not isinstance(record, dict) or record.get("name") != name:
        raise DataError(f"{record_path}: not the record of {name}")
    path = sources / f"{name}.csv"
    if not path.is_file():
        raise DataError(f"{path}: missing; download it with `astro-simclr r5-sources`")
    if sha256_file(path) != record.get("sha256"):
        raise DataError(f"{path}: SHA-256 differs from {record_path.name}; the file changed")
    return path, record


def _read_text(path: Path) -> str:
    try:
        if path.stat().st_size > MAX_CSV_BYTES:
            raise DataError(f"{path}: larger than {MAX_CSV_BYTES} bytes")
        return path.read_bytes().decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as err:
        raise DataError(f"{path}: cannot read ({err})") from None


def _float(text: str | None) -> float | None:
    if text is None or not text.strip():
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _rows(text: str, where: str, required: tuple[str, ...]) -> csv.DictReader[str]:
    reader = csv.DictReader(io.StringIO(text.lstrip("\r\n"), newline=""))
    missing = [c for c in required if c not in (reader.fieldnames or ())]
    if missing:
        raise DataError(f"{where}: missing columns {missing}")
    return reader


def read_zenodo(path: Path) -> list[SourceRow]:
    """The rows of the Zenodo labelled set (columns ``oid``, ``classALeRCE``, ``ra``, ``dec``).

    Raises:
        DataError: if the file cannot be read, lacks a column, or a row has an invalid
            ``oid`` or an empty class.
    """
    text = _read_text(path)
    rows: list[SourceRow] = []
    try:
        reader = _rows(text, str(path), ("oid", "classALeRCE", "ra", "dec"))
        for i, record in enumerate(reader):
            oid, cls = (record.get("oid") or "").strip(), (record.get("classALeRCE") or "").strip()
            if not is_oid(oid) or not cls or len(cls) > 32:
                raise DataError(f"{path}:{reader.line_num}: invalid oid or class")
            ra, dec = _float(record.get("ra")), _float(record.get("dec"))
            rows.append(SourceRow(i, oid, cls, ra, dec, (record.get("id_source") or "")[:64]))
    except csv.Error as err:
        raise DataError(f"{path}: invalid CSV ({err})") from None
    return rows


def _sexagesimal(text: str | None, scale: float) -> float | None:
    """``hh:mm:ss.s`` or ``±dd:mm:ss.s`` -> degrees (``scale`` 15 for RA, 1 for Dec)."""
    parts = (text or "").strip().split(":")
    if len(parts) != 3:
        return None
    sign = -1.0 if parts[0].startswith("-") else 1.0
    try:
        values = [abs(float(parts[0])), float(parts[1]), float(parts[2])]
    except ValueError:
        return None
    if not all(math.isfinite(v) and v >= 0 for v in values):
        return None
    return sign * scale * (values[0] + values[1] / 60 + values[2] / 3600)


def _parse_bts(data: bytes, where: str) -> list[SourceRow]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as err:
        raise DataError(f"{where}: not UTF-8 ({err})") from None
    rows: list[SourceRow] = []
    try:
        reader = _rows(text, where, ("ZTFID", "IAUID", "RA", "Dec", "type"))
        for i, record in enumerate(reader):
            oid = (record.get("ZTFID") or "").strip()
            if not is_oid(oid):
                raise DataError(f"{where}:{reader.line_num}: invalid ZTFID {oid!r}")
            rows.append(
                SourceRow(
                    i,
                    oid,
                    (record.get("type") or "").strip()[:32],
                    _sexagesimal(record.get("RA"), 15.0),
                    _sexagesimal(record.get("Dec"), 1.0),
                    (record.get("IAUID") or "").strip()[:64],
                )
            )
    except csv.Error as err:
        raise DataError(f"{where}: invalid CSV ({err})") from None
    return rows


def read_bts(path: Path) -> list[SourceRow]:
    """The rows of the BTS explorer CSV (``ZTFID``, ``IAUID``, ``RA``, ``Dec``, ``type``).

    The explorer starts the file with a blank line, which is skipped; RA and Dec are
    sexagesimal and converted to degrees.

    Raises:
        DataError: if the file cannot be read, lacks a column, or a row has an invalid
            ``ZTFID``.
    """
    try:
        if path.stat().st_size > MAX_CSV_BYTES:
            raise DataError(f"{path}: larger than {MAX_CSV_BYTES} bytes")
        data = path.read_bytes()
    except OSError as err:
        raise DataError(f"{path}: cannot read ({err})") from None
    return _parse_bts(data, str(path))
