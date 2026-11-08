"""The R5 recipe: the pre-registered constants and the seeded per-object alert draw (task 3.5).

R5 is the 5-class, object-level benchmark of phase 3 (``docs/phases/03-newer-benchmarks.md``,
task 3.5), rebuilt under RJ23's recipe (Reyes-Jainaga et al. 2023) from public sources only.
Everything here is fixed *before* any data is seen (the plan's pre-registration); ADR 0018
gives the reasons and the interpretations.

**The time cutoff.** :data:`JD_MAX` is the Julian date of 2022-12-31T23:59:59 UTC, an
inferred stand-in for RJ23's build date: only detections at or before it count, so a
rebuild in 2027 sees the same objects as one in 2026.

**The draw** (:func:`draw_alert`). For one object, the *eligible* detections are those with
a stamp (``has_stamp``) and ``jd <= JD_MAX``, sorted by ``candid``. One is drawn with
``numpy.random.default_rng([seed, oid_hash(oid)]).integers(n)``: independent of every other
object, of the order objects are visited, and of the machine. :func:`oid_hash` is the first
8 bytes of the SHA-256 of the ``oid`` (big-endian), a stable integer. RJ23's own draw was an
unseeded ``sample()``, so no seed reproduces their alerts; this one makes ours reproducible.

**ndet** (:func:`ndet`) counts the detections in the stored list with ``jd <= JD_MAX`` (with
or without a stamp), not today's API ``ndet``, which grows as the object keeps being
detected.

**The first-stamp view** (:func:`first_stamp_order`): the eligible detections in time order
(``jd``, then ``candid``); the view takes the first whose alert is a full 63x63 stamp
(RJ23's ``strategy='first'``).

**Per-class sources** (:data:`SUBCLASSES`): each R5 subclass names its source, the source's
class names, RJ23's count (the target) and RJ23's sampling cap. AGN and VS subclasses come
from the Zenodo light-curve labelled set (Sánchez-Sáez et al. 2021); SN from Zenodo, topped
up from the Bright Transient Survey (Fremling et al. 2020, Perley et al. 2020); Asteroid and
Bogus from CD21 (Carrasco-Davis et al. 2021) through task 3.1's raw-alert store.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

from astro_simclr.alerts.alerce import DetectionRecord

__all__ = [
    "BOGUS_MAX_NDET",
    "JD_MAX",
    "JD_MAX_UTC",
    "LABEL_NAMES",
    "SEED",
    "SN_TARGET",
    "SUBCLASSES",
    "Draw",
    "Source",
    "Subclass",
    "bts_subclass",
    "draw_alert",
    "eligible",
    "first_stamp_order",
    "ndet",
    "oid_hash",
    "stable_hash",
    "year_of_jd",
]

#: The cutoff, as UTC text and as a Julian date: 2023-01-01T00:00 UTC is JD 2459945.5.
JD_MAX_UTC: Final = "2022-12-31T23:59:59+00:00"
JD_MAX: Final = 2459945.5 - 1.0 / 86400.0
#: The default seed of every stream (draw, sampling, splits): RJ23's ``random_state=0``.
SEED: Final = 0
#: R5's classes, in label order (the phase-2 five, ``protocol.alerts.BENCHMARK_LABEL_NAMES``).
LABEL_NAMES: Final = ("AGN", "SN", "VS", "Asteroid", "Bogus")
#: RJ23's SN count, the cap of the Zenodo plus BTS SN.
SN_TARGET: Final = 3615
#: RJ23's Bogus cut: objects with this many detections or more are removed.
BOGUS_MAX_NDET: Final = 10

Source = Literal["zenodo", "zenodo+bts", "cd21"]


@dataclass(frozen=True, slots=True)
class Subclass:
    """One row of the plan's per-class table.

    Attributes:
        label: the R5 class (one of :data:`LABEL_NAMES`).
        name: the subclass, in RJ23's spelling.
        source: where the objects come from.
        source_classes: the source's class names for it (Zenodo's ``classALeRCE``; CD21's
            label name). Empty when the source has none (recorded as absent).
        target: RJ23's count (paper §II.1), or ``None`` where RJ23 gives only the class
            total (SN subclasses, which share :data:`SN_TARGET`).
        cap: RJ23's sampling cap (``sample(min(n, cap))``), ``None`` when RJ23 took all.
    """

    label: str
    name: str
    source: Source
    source_classes: tuple[str, ...]
    target: int | None
    cap: int | None


#: The plan's table, subclass by subclass (RJ23 §II.1 counts; research report §1.5). The
#: ``E`` subclass stands for RJ23's ``EB`` (``EB/EW`` + ``EA``, two caps of 1,200); Zenodo
#: lacks ``RSCVn`` and ``ZZ`` (recorded as absent) and spells ``Ceph`` as ``CEP``.
SUBCLASSES: Final[tuple[Subclass, ...]] = (
    Subclass("AGN", "AGN", "zenodo", ("AGN",), 3916, 4000),
    Subclass("AGN", "Blazar", "zenodo", ("Blazar",), 1932, 4000),
    Subclass("AGN", "QSO", "zenodo", ("QSO",), 3926, 4000),
    Subclass("SN", "SNIa", "zenodo+bts", ("SNIa",), None, None),
    Subclass("SN", "SNII", "zenodo+bts", ("SNII",), None, None),
    Subclass("SN", "SNIbc", "zenodo+bts", ("SNIbc",), None, None),
    Subclass("SN", "SNIIn", "zenodo+bts", (), None, None),
    Subclass("SN", "SNIIb", "zenodo+bts", (), None, None),
    Subclass("SN", "SLSN", "zenodo+bts", ("SLSN",), None, None),
    Subclass("VS", "EB", "zenodo", ("E",), 2339, 2400),
    Subclass("VS", "LPV", "zenodo", ("LPV",), 1167, 1200),
    Subclass("VS", "RRL", "zenodo", ("RRL",), 1158, 1200),
    Subclass("VS", "RSCVn", "zenodo", (), 1171, 1200),
    Subclass("VS", "YSO", "zenodo", ("YSO",), 1168, 1200),
    Subclass("VS", "DSCT", "zenodo", ("DSCT",), 1175, 1200),
    Subclass("VS", "Ceph", "zenodo", ("CEP",), 973, 1200),
    Subclass("VS", "ZZ", "zenodo", (), 9, 1200),
    Subclass("VS", "Periodic-Other", "zenodo", ("Periodic-Other",), 1051, 1200),
    Subclass("Asteroid", "Asteroid", "cd21", ("Asteroid",), 9180, None),
    Subclass("Bogus", "Bogus", "cd21", ("Bogus",), 15950, None),
)

#: BTS ``type`` prefixes -> R5 SN subclass (RJ23's SN classes), first match wins, so
#: ``SN IIn`` and ``SN IIb`` come before ``SN II``.
_BTS_TYPES: Final = (
    ("SLSN", "SLSN"),
    ("SN IIn", "SNIIn"),
    ("SN IIb", "SNIIb"),
    ("SN II", "SNII"),
    ("SN Ia", "SNIa"),
    ("SN Ib", "SNIbc"),
    ("SN Ic", "SNIbc"),
)


def bts_subclass(bts_type: str) -> str | None:
    """RJ23's SN subclass of a BTS ``type`` (``SN Ia-91T`` -> ``SNIa``), ``None`` if not a SN.

    ``SN Iax`` and the other ``SN Ia-*`` are ``SNIa``; ``SN Ib/c``, ``SN Ibn``, ``SN Icn``
    and ``SN Ic-BL`` are ``SNIbc``; ``SN IIP`` and ``SN II-pec`` are ``SNII``.
    """
    text = bts_type.strip()
    for prefix, name in _BTS_TYPES:
        if text.startswith(prefix):
            return name
    return None


def stable_hash(text: str) -> int:
    """The first 8 bytes of SHA-256(``text``, UTF-8) as a big-endian unsigned integer."""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def oid_hash(oid: str) -> int:
    """The draw's per-object stream: :func:`stable_hash` of the ``oid``."""
    return stable_hash(oid)


def eligible(records: Sequence[DetectionRecord], jd_max: float = JD_MAX) -> list[DetectionRecord]:
    """The detections a draw may pick: with a stamp, ``jd <= jd_max``, sorted by ``candid``.

    A ``candid`` listed twice counts once (the first record).
    """
    seen: dict[int, DetectionRecord] = {}
    for record in records:
        if record.has_stamp and record.jd <= jd_max:
            seen.setdefault(record.candid, record)
    return [seen[candid] for candid in sorted(seen)]


def ndet(records: Sequence[DetectionRecord], jd_max: float = JD_MAX) -> int:
    """The detections of the stored list with ``jd <= jd_max`` (distinct candids)."""
    return len({r.candid for r in records if r.jd <= jd_max})


@dataclass(frozen=True, slots=True)
class Draw:
    """The drawn detection of one object: its position in the eligible list and the record."""

    index: int
    n_eligible: int
    record: DetectionRecord


def draw_alert(
    oid: str, records: Sequence[DetectionRecord], *, seed: int = SEED, jd_max: float = JD_MAX
) -> Draw | None:
    """Draw one eligible detection of ``oid`` (module docstring); ``None`` if none is eligible.

    Raises:
        ValueError: if ``seed`` is negative (``SeedSequence`` needs non-negative entropy).
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    pool = eligible(records, jd_max)
    if not pool:
        return None
    index = int(np.random.default_rng([seed, oid_hash(oid)]).integers(len(pool)))
    return Draw(index, len(pool), pool[index])


def first_stamp_order(
    records: Sequence[DetectionRecord], jd_max: float = JD_MAX
) -> list[DetectionRecord]:
    """The eligible detections in time order (``jd``, then ``candid``): the first-stamp walk."""
    return sorted(eligible(records, jd_max), key=lambda r: (r.jd, r.candid))


def year_of_jd(jd: float) -> int:
    """The UTC calendar year of Julian date ``jd`` (the era table's year)."""
    # JD 2440587.5 is 1970-01-01T00:00 UTC; numpy's datetime64 handles the calendar.
    seconds = round((jd - 2440587.5) * 86400.0)
    return int(str(np.datetime64(seconds, "s").astype("datetime64[Y]")))
