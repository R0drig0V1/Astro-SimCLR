"""Which archive alerts enter the unlabelled corpus: first detections and exclusions.

The corpus is pretraining data for the stamp classifier (Carrasco-Davis et al. 2021, CD21),
so it keeps what that classifier sees: **the first alert of an object**, as a full 63x63
science / reference / difference triplet. ADR 0016 records the rules and their sources.

**First detection** (:func:`first_detection_failure`): the object's first alert by ``oid``
(phase 5, task 5.1a). CD21 uses "only the first alert of an astronomical object" (Sec. 1),
ALeRCE's first detection of the ``oid``. An archive alert does not carry its object's whole
history, so the rule uses what it does carry:

1. ``previous``: no detection in ``prv_candidates`` (an entry with a ``candid``; upper
   limits have none). That history covers the 30 days before the alert, so an earlier
   alert of the object within 30 days drops it.
2. ``oid_year``: the two digits after ``ZTF`` in the ``oid`` equal the UT year of the
   alert's ``jd``. ZTF names an object after the year it was first seen (``ZTF18...`` for
   2018), so an object returning from an earlier year (a variable star after its seasonal
   gap) is dropped. *Assumption*, measured by the pilot (task 5.1a): an alert of the year's
   first days may belong to an object first seen on the last night of the previous year.
3. ``ndethist`` (optional, off by default): ``candidate.ndethist <= max_ndethist``. ZTF's
   schema defines ``ndethist`` as the detections within 1.5" since the start of the
   survey, **on the same field and readout channel only**, down to S/N ~3, and does not say
   whether the alert's own detection counts. It counts sub-threshold detections that never
   made an alert, so a bound would drop the first *alerts* of long-lived variables (AGN, VS):
   CD21 clips ``ndethist`` at 20 (its Table A1), so its first alerts reach such values.

What remains undetected: an alert more than 30 days after an earlier one of the same year.
The corpus index (:func:`~astro_simclr.archive.layout.corpus_index`) keeps one alert per
``oid`` over all the ingested nights (the earliest), which removes such repeats among the
sampled nights.

**Quality cuts: none on the candidate fields.** CD21 states no cut on ``rb``, ``drb``,
``nbad``, ``fwhm``, ``elong`` or ``magdiff``: its alerts are those of the ZTF stream,
5-sigma detections that passed ZTF's real-bogus filter (CD21 Sec. 2). A real-bogus
threshold would also remove the Bogus class, which the classifier must see (task 5.1a). The
index stores ``rb``, ``drb``, ``ndethist`` and ``isdiffpos``, so a cut can still be applied
when the corpus is read.

**63x63 only.** CD21 removed its 650 non-square stamps (Sec. 3.1); the parser refuses them
here (``parse_alert(repair=False)``) rather than padding them as ``fetch-alerts`` does.

**Exclusions** (:class:`Exclusions`): every ``oid`` of an evaluation set (the union the
integrator builds: CD21's objects, R5 Validation/Test and every later benchmark) is dropped
at ingest, so pretraining never sees an object it is evaluated on; the corpus index can
apply a longer list later without re-ingesting.

No ALeRCE code is used: the rules are re-stated from the papers and documentation cited.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

from astro_simclr.alerts.parse import Alert
from astro_simclr.data.raw_alerts import OID_PATTERN, OID_WIDTH
from astro_simclr.data.schema import DataError

__all__ = [
    "FIRST_DETECTION_RULES",
    "MAX_EXCLUSION_BYTES",
    "Exclusions",
    "Selection",
    "first_detection_failure",
    "jd_year",
    "load_exclusions",
    "oid_year",
]

#: The first-detection rules, in the order they are checked (module docstring).
FIRST_DETECTION_RULES: Final = ("previous", "oid_year", "ndethist")
#: Largest exclusion file read (about 4M oids).
MAX_EXCLUSION_BYTES: Final = 64 * 1024 * 1024
#: A ZTF object id: ``ZTF``, the two-digit year, seven lower-case letters.
_ZTF_OID: Final = re.compile(r"ZTF(\d{2})[a-z]{7}")
_JD_UNIX_EPOCH: Final = 2440587.5
_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class Selection:
    """The selection rules of one corpus (recorded in ``corpus.json`` and every night).

    Attributes:
        oid_year: apply the ``oid_year`` rule (module docstring).
        max_ndethist: the optional ``ndethist`` bound (>= 0), ``None`` for none.
    """

    oid_year: bool = True
    max_ndethist: int | None = None

    def __post_init__(self) -> None:
        if self.max_ndethist is not None and self.max_ndethist < 0:
            raise ValueError(f"max_ndethist must be >= 0, got {self.max_ndethist}")

    def to_json(self) -> dict[str, object]:
        """The rules as JSON (so a record is self-contained)."""
        return {
            "first_detection": {
                "previous_detections": 0,
                "oid_year": self.oid_year,
                "max_ndethist": self.max_ndethist,
            },
            "candidate_cuts": [],
            "stamp_shape": [63, 63, 3],
        }


def oid_year(oid: str) -> int | None:
    """The year a ZTF ``oid`` was named in (``ZTF18abcdefg`` -> 2018), or ``None``."""
    match = _ZTF_OID.fullmatch(oid)
    return None if match is None else 2000 + int(match.group(1))


def jd_year(jd: float) -> int | None:
    """The UT calendar year of the Julian date ``jd``, or ``None`` outside years 1..9999."""
    try:
        return (_EPOCH + timedelta(days=jd - _JD_UNIX_EPOCH)).year
    except (OverflowError, ValueError):
        return None


def first_detection_failure(alert: Alert, selection: Selection) -> str | None:
    """The first rule of :data:`FIRST_DETECTION_RULES` that ``alert`` fails, or ``None``.

    ``ndethist`` fails when the bound is set and the field is missing or not an integer;
    ``oid_year`` fails for an ``oid`` that is not a ZTF object id.
    """
    if alert.previous_detections:
        return "previous"
    jd = alert.candidate["jd"]
    if selection.oid_year and (not isinstance(jd, float) or oid_year(alert.oid) != jd_year(jd)):
        return "oid_year"
    if selection.max_ndethist is not None:
        ndethist = alert.candidate.get("ndethist")
        if isinstance(ndethist, bool) or not isinstance(ndethist, int):
            return "ndethist"
        if ndethist > selection.max_ndethist:
            return "ndethist"
    return None


@dataclass(frozen=True, slots=True)
class Exclusions:
    """The oids no corpus row may have.

    Attributes:
        oids: the excluded object ids.
        sha256: hex SHA-256 of the file they were read from (empty for :meth:`none`).
        source: the file's name (``""`` for :meth:`none`).
    """

    oids: frozenset[str]
    sha256: str
    source: str

    @classmethod
    def none(cls) -> Exclusions:
        """No exclusion (``ingest-archive --no-exclusions``)."""
        return cls(frozenset(), "", "")

    def to_json(self) -> dict[str, object]:
        """The record of ``corpus.json`` and of every night: file name, SHA-256, count."""
        return {"source": self.source, "sha256": self.sha256, "count": len(self.oids)}


def load_exclusions(path: Path) -> Exclusions:
    """Read an exclusion file: UTF-8 text, one ``oid`` per line.

    Blank lines and lines starting with ``#`` are ignored; surrounding whitespace is
    stripped; a repeated oid is fine.

    Raises:
        DataError: if the file is larger than :data:`MAX_EXCLUSION_BYTES`, is not UTF-8, or
            a line is not a valid oid (:data:`~astro_simclr.data.raw_alerts.OID_PATTERN`,
            at most :data:`~astro_simclr.data.raw_alerts.OID_WIDTH` characters).
        OSError: if the file cannot be read.
    """
    if path.stat().st_size > MAX_EXCLUSION_BYTES:
        raise DataError(f"{path}: exclusion file larger than {MAX_EXCLUSION_BYTES} bytes")
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise DataError(f"{path}: the exclusion file is not UTF-8") from None
    oids: set[str] = set()
    for number, line in enumerate(text.splitlines(), start=1):
        oid = line.strip()
        if not oid or oid.startswith("#"):
            continue
        if len(oid) > OID_WIDTH or OID_PATTERN.fullmatch(oid) is None:
            raise DataError(f"{path}:{number}: not an oid: {oid[:40]!r}")
        oids.add(oid)
    return Exclusions(frozenset(oids), hashlib.sha256(data).hexdigest(), path.name)
