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
"""

from __future__ import annotations

import json
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
    "RADIUS_ARCSEC",
    "AlerceClient",
    "Detection",
    "Getter",
    "ObjectsPage",
    "ResponseError",
    "is_oid",
    "make_client",
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
        _check_oid(oid)
        url = f"{self.api_url}/objects/{oid}/detections"
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

    def alert_avro(self, oid: str, candid: int) -> bytes:
        """The avro bytes of alert (``oid``, ``candid``); parse them with ``parse_alert``.

        Raises:
            ValueError: if ``oid`` is not a ZTF object id.
            HttpError: if the request fails.
        """
        _check_oid(oid)
        query = urllib.parse.urlencode({"oid": oid, "candid": str(int(candid))})
        url = f"{self.avro_url}/get_avro?{query}"
        return self.getter.get(url, max_bytes=MAX_AVRO_BYTES, kind="avro", cache=False)


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
