"""``ingest-archive``: nightly ZTF archive tarballs to the unlabelled corpus, resumable per night.

For each tarball (:mod:`~astro_simclr.archive.stream`; a file, or a download piped in so
that no tarball is stored), each avro member is parsed as an untrusted alert
(:func:`~astro_simclr.alerts.parse.parse_alert`, with its size limits, ``repair=False``)
and kept only if it passes, in this order (an alert is counted under the first rule it
fails, :data:`REASONS`):

1. ``malformed``: not one valid alert (bad avro, missing fields, invalid cutouts);
2. ``not_63x63``: valid cutouts of another shape (CCD edge);
3. ``invalid_values``: an identity value the index cannot hold (oid, candid, a ``jd``
   outside the ZTF era :data:`JD_RANGE`, fid, ra/dec ranges, ``ndethist``,
   ``isdiffpos``);
4. ``not_first:previous``, ``not_first:oid_year``, ``not_first:ndethist``: the
   first-detection rules of :mod:`~astro_simclr.archive.selection`;
5. ``excluded``: the oid is in the exclusion list (counted only for alerts that would
   otherwise be kept, so the count is what the exclusion removed);
6. ``duplicate_candid``, ``duplicate_oid``: a candid, or an object, already kept this
   night (the first one in tar order is kept).

The kept rows go to the night's shard (:class:`~astro_simclr.archive.layout.NightWriter`),
committed atomically with the tarball's name, size and SHA-256 and the counts.

One ingest at a time holds a corpus (``corpus_lock``); a second one fails at once.

**Resume.** A night whose directory is complete is skipped without reading its tarball,
after checking that the tarball has the recorded name and (for a file) size; another file
for the same night is refused unless ``overwrite``. A night cut short leaves only its
``.partial`` directory, which the next run removes and redoes. ``corpus.json`` pins the
selection rules and the exclusion list, so a resume cannot mix nights ingested under
other rules.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Final, Literal

from astro_simclr.alerts.parse import Alert, AlertFormatError, CutoutShapeError, parse_alert
from astro_simclr.archive.layout import (
    NightWriter,
    check_corpus,
    corpus_lock,
    night_dir,
    read_night_meta,
    recover_night,
)
from astro_simclr.archive.selection import (
    FIRST_DETECTION_RULES,
    Exclusions,
    Selection,
    first_detection_failure,
)
from astro_simclr.archive.stream import (
    MAX_DECOMPRESSED_BYTES,
    MAX_MEMBER_BYTES,
    MAX_MEMBERS,
    TarballStream,
    night_of,
    tarball_name,
)
from astro_simclr.data.raw_alerts import OID_PATTERN, OID_WIDTH, ZTF_FIDS
from astro_simclr.data.schema import DataError

__all__ = [
    "JD_RANGE",
    "REASONS",
    "Limits",
    "NightResult",
    "ingest_archive",
    "ingest_stream",
    "ingest_tarball",
]

#: The outcome of an avro member, in the order the rules are applied (module docstring).
REASONS: Final = (
    "kept",
    "malformed",
    "not_63x63",
    "invalid_values",
    *(f"not_first:{rule}" for rule in FIRST_DETECTION_RULES),
    "excluded",
    "duplicate_candid",
    "duplicate_oid",
)
_POSITIVE: Final = {"t": True, "1": True, "f": False, "0": False}
#: Accepted ``jd``: 2017-09-04 (before ZTF's first light) to 2050-01-01 UT. An alert outside
#: it is garbage, and date arithmetic on it would overflow.
JD_RANGE: Final = (2458000.5, 2469807.5)


@dataclass(frozen=True, slots=True)
class Limits:
    """The stream's limits (:mod:`~astro_simclr.archive.stream`); tests lower them."""

    max_member_bytes: int = MAX_MEMBER_BYTES
    max_decompressed_bytes: int = MAX_DECOMPRESSED_BYTES
    max_members: int = MAX_MEMBERS


@dataclass(slots=True)
class NightResult:
    """The outcome of one tarball.

    Attributes:
        night: ``YYYYMMDD``.
        status: ``ingested`` (written now) or ``done`` (already complete, skipped).
        counts: the per-reason counts (:data:`REASONS`).
        path: the night's directory.
    """

    night: int
    status: Literal["ingested", "done"]
    counts: dict[str, int] = field(default_factory=dict)
    path: Path | None = None

    @property
    def kept(self) -> int:
        """The rows the night holds."""
        return self.counts.get("kept", 0)


@dataclass(frozen=True, slots=True)
class Rules:
    """The settings of one ingest (what :func:`ingest_tarball` and the others take)."""

    selection: Selection
    exclusions: Exclusions
    rows_per_file: int = 4096
    overwrite: bool = False
    limits: Limits = field(default_factory=Limits)

    def __post_init__(self) -> None:
        if self.rows_per_file < 1:
            raise ValueError(f"rows_per_file must be >= 1, got {self.rows_per_file}")


def ingest_archive(
    tarballs: Sequence[Path],
    out: Path,
    rules: Rules,
    *,
    progress: Callable[[NightResult], None] | None = None,
) -> list[NightResult]:
    """Ingest every tarball file into the corpus at ``out``, one night each.

    Raises:
        DataError: if two tarballs are for the same night, a name is not an archive
            tarball's, ``corpus.json`` records other rules, a complete night came from
            another file, or a tarball is unreadable (``TarballError``); the nights
            committed before the error stay.
        OSError: if a file cannot be read or written.
    """
    nights = [night_of(path) for path in tarballs]
    repeated = sorted({n for n in nights if nights.count(n) > 1})
    if repeated:
        raise DataError(f"several tarballs for the nights {repeated}")
    results = []
    for tarball in tarballs:
        result = ingest_tarball(tarball, out, rules)
        results.append(result)
        if progress is not None:
            progress(result)
    return results


def ingest_tarball(tarball: Path, out: Path, rules: Rules) -> NightResult:
    """Ingest one tarball file as its night (module docstring); see :func:`ingest_archive`."""
    night = night_of(tarball)
    if not tarball.is_file():
        raise DataError(f"{tarball}: no such file")
    with corpus_lock(out):
        done = _done(out, night, tarball.name, tarball.stat().st_size, rules)
        if done is not None:
            return done
        with tarball.open("rb") as handle:
            return _ingest(handle, tarball.name, night, out, rules)


def ingest_stream(handle: BinaryIO, night: int, out: Path, rules: Rules) -> NightResult:
    """Ingest the tarball of ``night`` read from ``handle`` (a download piped in).

    A complete night is skipped without reading ``handle`` (the writer of the pipe then
    sees it closed). The source is recorded under the archive's name of the night.

    Raises:
        DataError: as :func:`ingest_archive`, or if ``night`` is not a valid date.
    """
    name = tarball_name(night)
    night_of(Path(name))  # validates the night
    with corpus_lock(out):
        done = _done(out, night, name, None, rules)
        if done is not None:
            return done
        return _ingest(handle, name, night, out, rules)


def _done(out: Path, night: int, name: str, size: int | None, rules: Rules) -> NightResult | None:
    """Check ``corpus.json``; the result of a complete night to skip, or ``None``."""
    check_corpus(
        out, {"selection": rules.selection.to_json(), "exclusions": rules.exclusions.to_json()}
    )
    recover_night(out, night)
    final = night_dir(out, night)
    if not final.exists() or rules.overwrite:
        return None
    meta = read_night_meta(final)
    source = meta.get("source", {})
    recorded = source.get("compressed_bytes")
    if source.get("tarball") != name or (size is not None and recorded != size):
        raise DataError(
            f"{final}: night already ingested from {source.get('tarball')!r} ({recorded} "
            f"bytes), not {name} ({size} bytes); pass --overwrite to replace it"
        )
    return NightResult(night, "done", dict(meta.get("counts", {})), final)


def _ingest(handle: BinaryIO, name: str, night: int, out: Path, rules: Rules) -> NightResult:
    stream = TarballStream(
        handle,
        name=name,
        max_member_bytes=rules.limits.max_member_bytes,
        max_decompressed_bytes=rules.limits.max_decompressed_bytes,
        max_members=rules.limits.max_members,
    )
    writer = NightWriter(out, night, rules.rows_per_file)
    counts: Counter[str] = Counter({reason: 0 for reason in REASONS})
    candids: set[int] = set()
    oids: set[str] = set()
    for member in stream:
        try:
            alert = parse_alert(member.data, repair=False)
        except CutoutShapeError:
            counts["not_63x63"] += 1
            continue
        except AlertFormatError:
            counts["malformed"] += 1
            continue
        reason, row = _decide(alert, night, member.position, rules)
        if row is not None and alert.candid in candids:
            reason, row = "duplicate_candid", None
        elif row is not None and alert.oid in oids:
            reason, row = "duplicate_oid", None
        if row is None:
            counts[reason] += 1
            continue
        candids.add(alert.candid)
        oids.add(alert.oid)
        writer.add(row, alert.cutouts)
        counts["kept"] += 1
    path = writer.commit(
        {
            "source": {"tarball": name, **stream.stats.to_json()},
            "selection": rules.selection.to_json(),
            "exclusions": rules.exclusions.to_json(),
            "counts": dict(sorted(counts.items())),
        }
    )
    return NightResult(night, "ingested", dict(counts), path)


def _decide(
    alert: Alert, night: int, member: int, rules: Rules
) -> tuple[str, tuple[object, ...] | None]:
    """``(reason, None)`` for a dropped alert, ``("kept", index row)`` for a kept one."""
    row = _index_row(alert, night, member)
    if row is None:
        return "invalid_values", None
    failed = first_detection_failure(alert, rules.selection)
    if failed is not None:
        return f"not_first:{failed}", None
    if alert.oid in rules.exclusions.oids:
        return "excluded", None
    return "kept", row


def _index_row(alert: Alert, night: int, member: int) -> tuple[object, ...] | None:
    """The row of :data:`~astro_simclr.archive.layout.INDEX_DTYPE`, or ``None`` if invalid."""
    c = alert.candidate
    jd, fid, ra, dec = c["jd"], c["fid"], c["ra"], c["dec"]
    ndethist, isdiffpos = c.get("ndethist"), c.get("isdiffpos")
    if not (isinstance(jd, float) and isinstance(ra, float) and isinstance(dec, float)):
        return None  # pragma: no cover - parse_alert checks these types
    if (
        not isinstance(ndethist, int)
        or isinstance(ndethist, bool)
        or not isinstance(isdiffpos, str)
    ):
        return None
    valid = (
        len(alert.oid) <= OID_WIDTH
        and OID_PATTERN.fullmatch(alert.oid) is not None
        and alert.candid > 0
        and math.isfinite(jd)
        and JD_RANGE[0] <= jd <= JD_RANGE[1]
        and fid in ZTF_FIDS
        and 0 <= ra < 360
        and -90 <= dec <= 90
        and ndethist >= 0
        and isdiffpos in _POSITIVE
    )
    if not valid:
        return None
    return (
        alert.oid,
        alert.candid,
        jd,
        fid,
        ra,
        dec,
        night,
        ndethist,
        _POSITIVE[isdiffpos],
        _score(c.get("rb")),
        _score(c.get("drb")),
        member,
    )


def _score(value: object) -> float:
    """A real-bogus score as a float; NaN when it is missing or not a number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return math.nan
    return float(value)
