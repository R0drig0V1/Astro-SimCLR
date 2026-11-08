"""ALeRCE's public endpoints used by ``fetch-alerts``, and the validation of their JSON (task 3.1).

Three requests resolve and download one alert (``docs/data.md``, "Raw alerts"):

1. ``GET {API_URL}/objects/?ra=..&dec=..&radius=1.5``: the objects within 1.5 arcsec. The
   trailing slash matters: without it the server answers with a redirect.
2. ``GET {API_URL}/objects/{oid}/detections``: every detection of one object, with its
   ``candid``, ``magpsf`` and ``sigmapsf``.
3. ``GET {AVRO_URL}/get_avro?oid=..&candid=..``: the whole alert as an avro file (about
   50 KB: the candidate record, the previous candidates and the three cutouts).

Every request goes through a :class:`Getter`, normally the
:class:`~astro_simclr.alerts.snapshot.Recorder` (rate-limited fetch, the cache of the cone
searches and detection lists, and the snapshot of every raw response).

The JSON is untrusted. :class:`AlerceClient` checks its shape and types and raises
:class:`ResponseError` for anything unexpected; an ``oid`` must look like a ZTF object id
before it is put into a URL. Detections without a numeric ``magpsf``/``sigmapsf`` or an
integer ``candid`` (forced photometry, upper limits) are skipped, since they cannot match.

The phase-3 reconstruction (task 3.5) reads more of each detection than the photometric
match: :func:`parse_detection_records` turns the same detection list into
:class:`DetectionRecord` items (``candid``, ``jd``, ``fid``, ``has_stamp``, position), the
input of its seeded draw. ALeRCE reports ``mjd``; ``jd = mjd + 2400000.5`` (or ``jd``
itself when present). A detection that is not usable for a stamp draw (no integer
``candid``, no finite time) is skipped and counted, never guessed.
"""

from __future__ import annotations

import json
import math
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any, Protocol

from astro_simclr.alerts.snapshot import Recorder, online_fetch
from astro_simclr.alerts.transport import Fetcher

__all__ = [
    "API_URL",
    "AVRO_URL",
    "MAX_AVRO_BYTES",
    "MAX_JSON_BYTES",
    "MJD_OFFSET",
    "RADIUS_ARCSEC",
    "AlerceClient",
    "Detection",
    "DetectionRecord",
    "DetectionRecords",
    "Getter",
    "ObjectsPage",
    "ResponseError",
    "avro_url",
    "detections_url",
    "is_oid",
    "make_client",
    "parse_detection_records",
]

#: ALeRCE's ZTF API and avro service (checked 2026-11-08).
API_URL = "https://api.alerce.online/ztf/v1"
AVRO_URL = "https://avro.alerce.online"
#: Cone-search radius in arcsec: the integrator's probe radius, about 2x ZTF's astrometric
#: scatter, so neighbours are found and resolved by the photometric match.
RADIUS_ARCSEC = 1.5
#: Response size limits: a detections list of a long-lived object is a few hundred KB, an
#: alert about 50 KB (its decompressed size is capped separately, ``parse``).
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_AVRO_BYTES = 1024 * 1024

#: A ZTF object id: ``ZTF``, two digits of the year, seven lowercase letters.
_OID = re.compile(r"ZTF\d{2}[a-z]{7}")


class ResponseError(ValueError):
    """A response is not the JSON the endpoint documents; the message says what is wrong."""


def is_oid(value: object) -> bool:
    """True if ``value`` is a string shaped like a ZTF object id (``ZTF18abcdefg``)."""
    return isinstance(value, str) and _OID.fullmatch(value) is not None


class Getter(Protocol):
    """GET with a response-kind tag and a cache hint.

    Implemented by :class:`~astro_simclr.alerts.snapshot.Recorder`.
    """

    @property
    def requests(self) -> int:
        """Network requests started so far (cache hits are not requests)."""
        ...

    def get(self, url: str, *, max_bytes: int, kind: str, cache: bool) -> bytes:
        """The body of ``url``; ``kind`` tags it in the snapshot; ``cache`` allows reuse."""
        ...


@dataclass(frozen=True, slots=True)
class Detection:
    """One detection of an object: what the photometric match needs."""

    candid: int
    magpsf: float
    sigmapsf: float


@dataclass(frozen=True, slots=True)
class DetectionRecord:
    """One detection of an object as the stamp draw of task 3.5 needs it.

    ``jd`` is the Julian date (``mjd + 2400000.5``); ``has_stamp`` is ALeRCE's flag that the
    detection's alert (and so its cutouts) exists, false for the earlier detections an alert
    only lists as previous candidates. ``fid``, ``ra`` and ``dec`` are ``None`` when the
    list lacks a valid value.
    """

    candid: int
    jd: float
    has_stamp: bool
    fid: int | None
    ra: float | None
    dec: float | None


@dataclass(frozen=True, slots=True)
class DetectionRecords:
    """The detection list of one object: its usable records and how many were skipped."""

    records: tuple[DetectionRecord, ...]
    skipped: int


@dataclass(frozen=True, slots=True)
class ObjectsPage:
    """The answer of a cone search: the object ids, and whether the server has more."""

    oids: tuple[str, ...]
    has_next: bool


class AlerceClient:
    """The three requests of the module docstring, through one :class:`Getter`.

    Cone searches and detection lists are cacheable (several rows share an object); the
    avro files are not (each candid is downloaded once per row).
    """

    def __init__(self, getter: Getter, *, api_url: str = API_URL, avro_url: str = AVRO_URL) -> None:
        self.getter = getter
        self.api_url = api_url.rstrip("/")
        self.avro_url = avro_url.rstrip("/")

    @property
    def requests(self) -> int:
        """Network requests started so far (retries included, cache hits excluded)."""
        return self.getter.requests

    def objects_near(self, ra: float, dec: float, radius: float = RADIUS_ARCSEC) -> ObjectsPage:
        """The objects within ``radius`` arcsec of (``ra``, ``dec``) degrees.

        Raises:
            HttpError: if the request fails (:class:`Fetcher`).
            ResponseError: if the answer is not ``{"items": [{"oid": ...}, ...]}``.
        """
        query = urllib.parse.urlencode({"ra": repr(ra), "dec": repr(dec), "radius": radius})
        url = f"{self.api_url}/objects/?{query}"
        payload = _json(self.getter.get(url, max_bytes=MAX_JSON_BYTES, kind="objects", cache=True))
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise ResponseError("objects: expected an object with an 'items' list")
        oids: list[str] = []
        for item in payload["items"]:
            oid = item.get("oid") if isinstance(item, dict) else None
            if not is_oid(oid):
                raise ResponseError(f"objects: invalid oid {oid!r}")
            assert isinstance(oid, str)  # noqa: S101 - narrowed by is_oid, for mypy
            if oid not in oids:
                oids.append(oid)
        return ObjectsPage(tuple(oids), bool(payload.get("has_next")))

    def detections(self, oid: str) -> list[Detection]:
        """The detections of ``oid`` that carry a candid, magpsf and sigmapsf.

        Accepts a bare list or ``{"items": [...]}``.

        Raises:
            ValueError: if ``oid`` is not a ZTF object id (it would go into the URL).
            HttpError: if the request fails.
            ResponseError: if the answer is not a list of objects.
        """
        url = detections_url(oid, self.api_url)
        payload = _json(
            self.getter.get(url, max_bytes=MAX_JSON_BYTES, kind="detections", cache=True)
        )
        if isinstance(payload, dict):
            payload = payload.get("items")
        if not isinstance(payload, list):
            raise ResponseError(f"detections of {oid}: expected a list")
        found: list[Detection] = []
        for item in payload:
            if not isinstance(item, dict):
                raise ResponseError(f"detections of {oid}: expected objects in the list")
            detection = _detection(item)
            if detection is not None:
                found.append(detection)
        return found

    def detection_records(self, oid: str) -> DetectionRecords:
        """The detection list of ``oid`` as :class:`DetectionRecord` items (task 3.5).

        The same request (and cache entry) as :meth:`detections`.

        Raises:
            ValueError: if ``oid`` is not a ZTF object id.
            HttpError: if the request fails.
            ResponseError: if the answer is not a list of objects.
        """
        url = detections_url(oid, self.api_url)
        body = self.getter.get(url, max_bytes=MAX_JSON_BYTES, kind="detections", cache=True)
        return parse_detection_records(body, oid)

    def alert_avro(self, oid: str, candid: int) -> bytes:
        """The avro bytes of alert (``oid``, ``candid``); parse them with ``parse_alert``.

        Raises:
            ValueError: if ``oid`` is not a ZTF object id.
            HttpError: if the request fails.
        """
        url = avro_url(oid, candid, self.avro_url)
        return self.getter.get(url, max_bytes=MAX_AVRO_BYTES, kind="avro", cache=False)


def detections_url(oid: str, base: str = API_URL) -> str:
    """The URL of ``oid``'s detection list (the key of its response in a snapshot).

    Raises:
        ValueError: if ``oid`` is not a ZTF object id.
    """
    _check_oid(oid)
    return f"{base.rstrip('/')}/objects/{oid}/detections"


def avro_url(oid: str, candid: int, base: str = AVRO_URL) -> str:
    """The URL of alert (``oid``, ``candid``)'s avro file (its key in a snapshot).

    Raises:
        ValueError: if ``oid`` is not a ZTF object id.
    """
    _check_oid(oid)
    query = urllib.parse.urlencode({"oid": oid, "candid": str(int(candid))})
    return f"{base.rstrip('/')}/get_avro?{query}"


#: ``jd - mjd``.
MJD_OFFSET = 2400000.5


def parse_detection_records(body: bytes, oid: str) -> DetectionRecords:
    """Parse a detection-list response (untrusted JSON) into :class:`DetectionRecords`.

    Accepts a bare list or ``{"items": [...]}``. A record needs an integer ``candid`` and a
    finite ``jd`` (or ``mjd``); ``has_stamp`` must be the JSON ``true`` to count as true.
    Records without a candid or time are skipped and counted.

    Raises:
        ResponseError: if the body is not JSON, or not a list of objects.
    """
    payload = _json(body)
    if isinstance(payload, dict):
        payload = payload.get("items")
    if not isinstance(payload, list):
        raise ResponseError(f"detections of {oid}: expected a list")
    records: list[DetectionRecord] = []
    skipped = 0
    for item in payload:
        if not isinstance(item, dict):
            raise ResponseError(f"detections of {oid}: expected objects in the list")
        record = _record(item)
        if record is None:
            skipped += 1
        else:
            records.append(record)
    return DetectionRecords(tuple(records), skipped)


def _record(item: dict[str, Any]) -> DetectionRecord | None:
    candid = _candid(item.get("candid"))
    jd = _number(item.get("jd"))
    if jd is None:
        mjd = _number(item.get("mjd"))
        jd = None if mjd is None else mjd + MJD_OFFSET
    if candid is None or jd is None or not math.isfinite(jd):
        return None
    fid = item.get("fid")
    ra, dec = _number(item.get("ra")), _number(item.get("dec"))
    return DetectionRecord(
        candid=candid,
        jd=jd,
        has_stamp=item.get("has_stamp") is True,
        fid=fid if isinstance(fid, int) and not isinstance(fid, bool) else None,
        ra=ra if ra is not None and math.isfinite(ra) else None,
        dec=dec if dec is not None and math.isfinite(dec) else None,
    )


def make_client(rate: float) -> AlerceClient:
    """The production client: a :class:`Recorder` over a urllib :class:`Fetcher` at ``rate``.

    The CLI builds its online client here, so tests can replace it.
    """
    fetcher = Fetcher(rate=rate)
    return AlerceClient(Recorder(online_fetch(fetcher), requests=lambda: fetcher.requests))


def _check_oid(oid: str) -> None:
    if not is_oid(oid):
        raise ValueError(f"not a ZTF object id: {oid!r}")


def _json(body: bytes) -> Any:  # JSON is untyped until validated
    try:
        return json.loads(body)
    # ValueError covers JSONDecodeError, UnicodeDecodeError and Python's limit on integer
    # digits (a 5,000-digit number in a hostile answer).
    except (ValueError, RecursionError) as err:
        raise ResponseError(f"invalid JSON ({type(err).__name__})") from None


#: A candid is a positive int64 (the shards store it as one).
_MAX_CANDID = 2**63 - 1


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        return float(value)
    except OverflowError:  # an integer beyond the float range
        return None


def _candid(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and value.isascii() and value.isdigit() and len(value) <= 19:
        value = int(value)
    if isinstance(value, int) and 0 < value <= _MAX_CANDID:
        return value
    return None


def _detection(item: dict[str, Any]) -> Detection | None:
    candid = _candid(item.get("candid"))
    magpsf, sigmapsf = _number(item.get("magpsf")), _number(item.get("sigmapsf"))
    if candid is None or magpsf is None or sigmapsf is None:
        return None
    return Detection(candid, magpsf, sigmapsf)
