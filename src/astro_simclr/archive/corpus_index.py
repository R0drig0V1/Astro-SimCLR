"""The corpus index: the archive rows that pretraining may read (phase 5, task 5.1; ADR 0019).

``ingest-archive`` keeps first alerts and drops the ``oid`` list it was given, nothing more
(ADR 0019, point 5). The index applies the **exclusion union** of the plan (task 5.1, plan
review F5) when it is built, so a grown list never needs a re-ingest:

1. ``candid_list``: every row whose ``candid`` is in a ``--exclude-candids`` file;
2. ``oid_list``: every row whose ``oid`` is in a ``--exclude-oids`` file (R5
   Validation/Test, the CD21 objects ``fetch-alerts`` resolved);
3. ``position``: every row of an ``oid`` with an alert within ``radius_arcsec`` (1.5",
   inclusive) of **any** alert of
   the CD21 layout, Train included (CD21 CV evaluates Train), Validation and Test. The
   positions are the layout's ``ra`` and ``dec`` features de-standardised with M1's
   statistics (``protocol.groups.split_positions``), matched on the sphere with the RA wrap
   (``protocol.groups.match_any_within``). This catches the "ghosts": another ``oid`` of
   the same object (another field or readout channel) and artefacts at a fixed position;
4. ``near_duplicate``: the pixel check of :mod:`.near_duplicates` against every CD21 stamp,
   on the rows left by 1-3; a match drops **every** row of its ``oid``;
5. ``duplicate_oid``: one alert per ``oid`` (an object can come back on another night):
   ``first`` (default) keeps the earliest by ``jd``, then ``candid``, which is the first
   alert that CD21 uses (ADR 0019, point 2); ``random`` keeps one drawn from
   ``default_rng([seed, 0, corpus_duplicates])``.

A row is counted under the **first** rule that removes it (``counts["excluded"]``);
``counts["matched"]`` also gives, for rules 1-3, every row each one matches on its own.

**Provenance.** The index records the SHA-256 of every exclusion input (each list file;
the layout's ``meta.json`` and every ``features.npy`` and ``images_f32.npy`` it read,
checked against that ``meta.json``), one ``exclusion_sha256`` over all of them (the value
a pretraining manifest records), the ingest's own exclusion record (``corpus.json``) and
the SHA-256 of ``corpus.json`` and of every night's ``meta.json``, so a reader can tell
that the archive is the one indexed (:func:`check_archive`).

**The file** (``build-corpus-index --out FILE``; not committed: it names local paths) is
an ``.npz`` without pickles: ``rows``, the kept rows
(:data:`~astro_simclr.archive.layout.CORPUS_INDEX_DTYPE`, sorted by night and row, the
read order), and ``meta``, the JSON record above as a 0-d string. It is written through a
temporary file and a rename.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt

from astro_simclr.archive.layout import (
    CORPUS_INDEX_DTYPE,
    CORPUS_NAME,
    META_NAME,
    list_nights,
    load_night,
    night_dir,
)
from astro_simclr.archive.near_duplicates import (
    NearDuplicateRules,
    ReferenceStamps,
    find_near_duplicates,
)
from astro_simclr.archive.reader import CorpusStamps
from astro_simclr.archive.selection import MAX_EXCLUSION_BYTES, Exclusions, load_exclusions
from astro_simclr.data.files import sha256_file
from astro_simclr.data.prepared import load_prepared, read_meta
from astro_simclr.data.schema import SPLITS, DataError
from astro_simclr.engine.randomness import RESERVED_STREAMS, host_generator
from astro_simclr.protocol.groups import DEFAULT_RADIUS_ARCSEC, match_any_within, split_positions

__all__ = [
    "DEFAULT_NEAR",
    "DUPLICATE_POLICIES",
    "INDEX_FORMAT",
    "INDEX_VERSION",
    "RULES",
    "CandidExclusions",
    "CorpusIndex",
    "build_corpus_index",
    "check_archive",
    "load_candid_exclusions",
    "read_corpus_index",
    "write_corpus_index",
]

#: ``format`` of the index's ``meta``; its version is :data:`INDEX_VERSION`.
INDEX_FORMAT: Final = "astro-simclr-corpus-index"
INDEX_VERSION: Final = 1
#: The rules, in the order they are applied (module docstring).
RULES: Final = ("candid_list", "oid_list", "position", "near_duplicate", "duplicate_oid")
DuplicatePolicy = Literal["first", "random"]
DUPLICATE_POLICIES: Final = ("first", "random")
#: The near-duplicate thresholds of ``build-corpus-index`` (:mod:`.near_duplicates`).
DEFAULT_NEAR: Final = NearDuplicateRules()
#: At most this many near-duplicate pairs are listed in the index (all are counted).
_MAX_LISTED: Final = 1000
_PROGRESS = Callable[[str], None]
_INT64_MAX: Final = 2**63 - 1


@dataclass(frozen=True, slots=True)
class CandidExclusions:
    """The ``candid`` values no corpus row may have, read from one file.

    Attributes:
        candids: the excluded alert ids.
        sha256: hex SHA-256 of the file.
        source: the file's name.
    """

    candids: frozenset[int]
    sha256: str
    source: str


def load_candid_exclusions(path: Path) -> CandidExclusions:
    """Read a candid exclusion file: UTF-8 text, one positive integer per line.

    Blank lines and lines starting with ``#`` are ignored; surrounding whitespace is
    stripped; a repeated candid is fine.

    Raises:
        DataError: if the file is larger than the exclusion cap, is not UTF-8, or a line
            is not a positive integer.
        OSError: if the file cannot be read.
    """
    if path.stat().st_size > MAX_EXCLUSION_BYTES:
        raise DataError(f"{path}: exclusion file larger than {MAX_EXCLUSION_BYTES} bytes")
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise DataError(f"{path}: the exclusion file is not UTF-8") from None
    candids: set[int] = set()
    for number, line in enumerate(text.splitlines(), start=1):
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        if not value.isascii() or not value.isdigit() or not 0 < int(value) <= _INT64_MAX:
            raise DataError(f"{path}:{number}: not a candid: {value[:40]!r}")
        candids.add(int(value))
    return CandidExclusions(frozenset(candids), hashlib.sha256(data).hexdigest(), path.name)


@dataclass(frozen=True, slots=True)
class CorpusIndex:
    """A built or read corpus index.

    Attributes:
        rows: the kept rows (:data:`~astro_simclr.archive.layout.CORPUS_INDEX_DTYPE`),
            sorted by night and row.
        meta: the record of the module docstring.
    """

    rows: npt.NDArray[np.void]
    meta: dict[str, Any]

    def __len__(self) -> int:
        return int(self.rows.shape[0])


def _all_rows(
    archive: Path, nights: Sequence[int]
) -> tuple[npt.NDArray[np.void], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Every row of ``nights``: the index table, its ``ra`` and its ``dec``."""
    tables, ras, decs = [], [], []
    for night in nights:
        index = load_night(night_dir(archive, night)).index
        part = np.empty(len(index), dtype=CORPUS_INDEX_DTYPE)
        part["night"], part["row"] = night, np.arange(len(index))
        for name in ("oid", "candid", "jd"):
            part[name] = index[name]
        tables.append(part)
        ras.append(np.asarray(index["ra"], dtype=np.float64))
        decs.append(np.asarray(index["dec"], dtype=np.float64))
    if not tables:
        empty = np.empty(0, dtype=np.float64)
        return np.empty(0, dtype=CORPUS_INDEX_DTYPE), empty, empty
    return np.concatenate(tables), np.concatenate(ras), np.concatenate(decs)


def _archive_record(archive: Path, nights: Sequence[int]) -> dict[str, Any]:
    """The archive's identity: ``corpus.json`` (content and SHA) and the meta SHA of ``nights``."""
    corpus = archive / CORPUS_NAME
    if not corpus.is_file():
        raise DataError(f"{archive}: no {CORPUS_NAME}; not an archive corpus")
    record = {}
    for night in nights:
        meta = night_dir(archive, night) / META_NAME
        record[f"{night:08d}"] = {
            "n": int(json.loads(meta.read_text(encoding="utf-8"))["n"]),
            "meta_sha256": sha256_file(meta),
        }
    return {
        "path": str(archive),
        "corpus_sha256": sha256_file(corpus),
        "corpus": json.loads(corpus.read_text(encoding="utf-8")),
        "nights": record,
    }


def check_archive(index: CorpusIndex, archive: Path) -> None:
    """Check that ``archive`` holds the nights the index was built from, unchanged.

    Compares the SHA-256 of ``corpus.json`` and of the ``meta.json`` of every night the
    index uses (each lists its files' SHA-256; ``archive-summary --verify`` checks those).
    Nights added to the archive later are allowed: the index does not read them.

    Raises:
        DataError: if ``corpus.json`` or a used night is missing or differs.
    """
    recorded = index.meta["archive"]
    corpus = archive / CORPUS_NAME
    if not corpus.is_file() or sha256_file(corpus) != recorded["corpus_sha256"]:
        raise DataError(f"{archive}: {CORPUS_NAME} differs from the one the index was built on")
    for night in np.unique(index.rows["night"]).tolist():
        name = f"{int(night):08d}"
        meta = night_dir(archive, int(night)) / META_NAME
        entry = recorded["nights"].get(name)
        if entry is None or not meta.is_file() or sha256_file(meta) != entry["meta_sha256"]:
            raise DataError(f"{archive}: night {name} differs from the one the index was built on")


def _reference_layout(
    layout: Path, *, images: bool
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], ReferenceStamps | None, dict[str, Any]
]:
    """CD21's positions (every split), its stamps when ``images``, and their record.

    Raises:
        DataError: if the layout is invalid, a split has no features, a file it reads
            differs from its ``meta.json``, or ``images`` and it has no float32 stamps.
    """
    meta = read_meta(layout)
    data = load_prepared(layout)
    names = [name for name in SPLITS if len(data.splits[name])]
    ra, dec = split_positions(data, names)
    files: dict[str, str] = {}
    keys = ["features.npy"] + (["images_f32.npy"] if images else [])
    for name in names:
        listed = meta["splits"][name]["files"]
        for key in keys:
            if key not in listed:
                raise DataError(
                    f"{layout}: {name} has no {key}; the near-duplicate check needs the "
                    "float32 stamps (`astro-simclr prepare-data PKL --images float32`)"
                )
            digest = sha256_file(layout / name / key)
            if digest != listed[key]["sha256"]:
                raise DataError(f"{layout / name / key}: SHA-256 differs from meta.json")
            files[f"{name}/{key}"] = digest
    reference = None
    if images:
        stamps = []
        for name in names:
            split_images = data.splits[name].images_f32
            assert split_images is not None  # noqa: S101 - checked through meta.json above
            stamps.append(split_images)
        reference = ReferenceStamps(tuple(names), tuple(stamps))
    record = {
        "kind": "cd21-layout",
        "path": str(layout),
        "source_sha256": meta["source"]["sha256"],
        "meta_sha256": sha256_file(layout / "meta.json"),
        "splits": {name: len(data.splits[name]) for name in names},
        "files": files,
    }
    return ra, dec, reference, record


def _one_per_oid(
    rows: npt.NDArray[np.void], policy: DuplicatePolicy, seed: int
) -> npt.NDArray[np.int64]:
    """Positions (into ``rows``) of the one alert kept per ``oid``, sorted."""
    if rows.size == 0:
        return np.empty(0, dtype=np.int64)
    if policy == "first":
        order = np.lexsort((rows["candid"], rows["jd"], rows["oid"]))
    else:
        stream = RESERVED_STREAMS["corpus_duplicates"]
        draw = host_generator(seed, 0, stream).permutation(rows.size)
        order = np.lexsort((draw, rows["oid"]))
    _, first = np.unique(rows["oid"][order], return_index=True)
    return np.sort(order[first]).astype(np.int64)


def build_corpus_index(
    archive: Path,
    *,
    layout: Path,
    oid_files: Sequence[Path] = (),
    candid_files: Sequence[Path] = (),
    radius_arcsec: float = DEFAULT_RADIUS_ARCSEC,
    duplicates: DuplicatePolicy = "first",
    seed: int = 0,
    near: NearDuplicateRules | None = DEFAULT_NEAR,
    chunk_rows: int = 4096,
    progress: _PROGRESS | None = None,
) -> CorpusIndex:
    """Build the corpus index of ``archive`` against the CD21 ``layout`` (module docstring).

    ``near=None`` skips the near-duplicate check (recorded as not done). ``progress``
    receives one line per stage.

    Raises:
        ValueError: for an unknown ``duplicates`` policy, a negative seed, a non-positive
            radius or ``chunk_rows < 1``.
        DataError: if the archive or the layout is invalid, an exclusion file is
            unreadable, or a position is impossible.
        OSError: if a file cannot be read.
    """
    if duplicates not in DUPLICATE_POLICIES:
        raise ValueError(f"duplicates must be one of {DUPLICATE_POLICIES}, got {duplicates!r}")
    if seed < 0 or not radius_arcsec > 0 or chunk_rows < 1:
        raise ValueError("need seed >= 0, radius_arcsec > 0 and chunk_rows >= 1")
    say = progress or (lambda _line: None)
    # One listing: an ingest may complete a night meanwhile; it is left for the next index.
    nights = list_nights(archive)
    record = _archive_record(archive, nights)
    rows, ra, dec = _all_rows(archive, nights)
    say(f"{len(rows)} archive rows in {len(record['nights'])} nights")
    oid_lists: list[Exclusions] = [load_exclusions(path) for path in oid_files]
    candid_lists = [load_candid_exclusions(path) for path in candid_files]
    ref_ra, ref_dec, reference, layout_record = _reference_layout(layout, images=near is not None)

    candids = frozenset().union(*(c.candids for c in candid_lists))
    oids = frozenset().union(*(o.oids for o in oid_lists))
    by_candid = np.isin(rows["candid"], np.fromiter(candids, dtype=np.int64, count=len(candids)))
    by_oid = np.isin(rows["oid"], np.array(sorted(oids), dtype=rows.dtype["oid"]))
    by_position, _ = match_any_within(ra, dec, ref_ra, ref_dec, radius_arcsec)
    say(f'position: {int(by_position.sum())} rows within {radius_arcsec}" of a CD21 alert')
    # A matched row is a CD21 object: every alert of its oid goes, as for near-duplicates.
    position_oids = np.isin(rows["oid"], np.unique(rows["oid"][by_position]))

    removed = np.zeros(len(rows), dtype=bool)
    excluded: dict[str, int] = {}
    rules_masks = (("candid_list", by_candid), ("oid_list", by_oid), ("position", position_oids))
    for rule, mask in rules_masks:
        excluded[rule] = int((mask & ~removed).sum())
        removed |= mask

    near_record: dict[str, Any] | None = None
    if near is not None and reference is not None:
        alive = np.flatnonzero(~removed)
        stamps = CorpusStamps(archive, rows)
        found = find_near_duplicates(stamps.chunks(alive, chunk_rows), reference, near)
        dup_oids = np.unique(rows["oid"][np.asarray(found.matched, dtype=np.int64)])
        by_near = np.isin(rows["oid"], dup_oids) & ~removed
        excluded["near_duplicate"] = int(by_near.sum())
        removed |= by_near
        listed = []
        for match in found.matches[:_MAX_LISTED]:
            row = rows[match.pop("position")]
            listed.append(
                {
                    "night": int(row["night"]),
                    "row": int(row["row"]),
                    "oid": str(row["oid"]),
                    "candid": int(row["candid"]),
                    **match,
                }
            )
        near_record = {
            "checked": found.checked,
            "candidates": found.candidates,
            "matched_rows": len(found.matched),
            "matched_oids": int(dup_oids.size),
            "matches": listed,
            "listed": len(listed),
        }
        say(
            f"near-duplicates: {found.checked} stamps checked, {found.candidates} candidates, "
            f"{len(found.matched)} confirmed"
        )
    else:
        excluded["near_duplicate"] = 0

    alive = np.flatnonzero(~removed)
    kept = alive[_one_per_oid(rows[alive], duplicates, seed)]
    excluded["duplicate_oid"] = int(alive.size - kept.size)
    kept_rows = rows[kept]

    inputs: list[dict[str, Any]] = [layout_record]
    inputs += [{"kind": "oids", **o.to_json()} for o in oid_lists]
    inputs += [
        {"kind": "candids", "source": c.source, "sha256": c.sha256, "count": len(c.candids)}
        for c in candid_lists
    ]
    rules = {
        "order": list(RULES),
        "radius_arcsec": radius_arcsec,
        "position_reference": "every CD21 alert (Train, Validation, Test), M1-destandardised; "
        "a match drops its whole oid",
        "duplicates": duplicates,
        "seed": seed,
        "near_duplicates": None if near is None else near.to_json(),
    }
    # One digest over what decides the exclusion: every input (labelled) and the rules.
    lines = [f"layout/{name} {digest}" for name, digest in sorted(layout_record["files"].items())]
    lines.append(f"layout/meta.json {layout_record['meta_sha256']}")
    lines += [f"{item['kind']}/{item['source']} {item['sha256']}" for item in inputs[1:]]
    lines.append(f"rules {json.dumps(rules, sort_keys=True)}")
    meta: dict[str, Any] = {
        "format": INDEX_FORMAT,
        "format_version": INDEX_VERSION,
        "created_by": f"astro-simclr {version('astro-simclr')}",
        "archive": record,
        "rules": rules,
        "exclusion_inputs": inputs,
        "exclusion_sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest(),
        "counts": {
            "rows": len(rows),
            "excluded": excluded,
            "matched": {
                "candid_list": int(by_candid.sum()),
                "oid_list": int(by_oid.sum()),
                "position": int(by_position.sum()),
            },
            "kept": len(kept_rows),
        },
        "near_duplicates": near_record,
        "n": len(kept_rows),
    }
    return CorpusIndex(kept_rows, meta)


def write_corpus_index(index: CorpusIndex, path: Path, *, overwrite: bool = False) -> None:
    """Write ``index`` to ``path`` (an ``.npz``; module docstring), atomically.

    Raises:
        DataError: if ``path`` exists and ``overwrite`` is False.
        OSError: if the file cannot be written.
    """
    if path.exists() and not overwrite:
        raise DataError(f"{path} already exists; pass overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            meta = np.array(json.dumps(index.meta, sort_keys=True))
            np.savez(handle, rows=index.rows, meta=meta)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def read_corpus_index(path: Path) -> CorpusIndex:
    """Read a corpus index written by :func:`write_corpus_index` (no pickles).

    Raises:
        DataError: if the file is unreadable, of another format or version, or its rows
            are not a sorted corpus index table of ``meta["n"]`` rows.
    """
    try:
        with np.load(path, allow_pickle=False) as archive:
            rows = archive["rows"]
            meta = json.loads(str(archive["meta"]))
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as err:
        raise DataError(f"{path}: not a corpus index ({err})") from None
    if (
        not isinstance(meta, dict)
        or meta.get("format") != INDEX_FORMAT
        or meta.get("format_version") != INDEX_VERSION
    ):
        raise DataError(f"{path}: not an {INDEX_FORMAT} (version {INDEX_VERSION})")
    if rows.dtype != CORPUS_INDEX_DTYPE or rows.ndim != 1 or meta.get("n") != len(rows):
        raise DataError(f"{path}: the rows are not {meta.get('n')!r} corpus index rows")
    order = np.lexsort((rows["row"], rows["night"]))
    if not np.array_equal(order, np.arange(len(rows))):
        raise DataError(f"{path}: the rows are not sorted by night and row")
    return CorpusIndex(rows, meta)
