"""Offline stand-ins for the alert tests: FITS and avro writers, and a fake ALeRCE.

Everything here is made by the tests themselves: :func:`fits_bytes` writes a minimal FITS
image, :func:`avro_bytes` an avro alert with our own small schema (fastavro's writer), and
:class:`FakeAlerce` answers the three endpoints from an in-memory sky, so the suite never
touches the network. :func:`make_world` builds a prepared split whose stored stamps are
ALeRCE's normalisation of the sky's raw cutouts, with one row per scenario of task 3.1.
"""

from __future__ import annotations

import gzip
import io
import json
import math
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fastavro
import numpy as np
import numpy.typing as npt

from astro_simclr.alerts.alerce import API_URL, AVRO_URL
from astro_simclr.alerts.fetch import alerce_normalise
from astro_simclr.alerts.transport import HttpError
from astro_simclr.data.m1_features import M1_FEATURE_NORM_STATS
from astro_simclr.data.prepared import write_prepared
from astro_simclr.data.preprocessing import repair_non_squared_stamp
from astro_simclr.data.schema import FEATURE_NAMES, SPLITS, SplitData, StampData

_BLOCK = 2880


def fits_bytes(
    image: npt.NDArray[np.floating[Any]],
    *,
    bitpix: int = -32,
    cards: tuple[str, ...] = (),
    compress: bool = True,
) -> bytes:
    """A FITS file with ``image`` (``(NAXIS2, NAXIS1)``) as its primary array."""
    height, width = image.shape
    header = [
        "SIMPLE  =                    T / conforms to FITS",
        f"BITPIX  = {bitpix:>20}",
        "NAXIS   =                    2",
        f"NAXIS1  = {width:>20}",
        f"NAXIS2  = {height:>20}",
        "OBJECT  = 'test / stamp'       / a quoted value with a slash",
        *cards,
        "END",
    ]
    text = "".join(card.ljust(80) for card in header)
    text += " " * (-len(text) % _BLOCK)
    dtype = {-32: ">f4", -64: ">f8"}.get(bitpix, ">f4")
    data = np.asarray(image).astype(dtype).tobytes()
    data += b"\0" * (-len(data) % _BLOCK)
    raw = text.encode("ascii") + data
    return gzip.compress(raw) if compress else raw


#: Our own alert schema: the fields the parser reads, plus extras it must skip.
_CUTOUT = {
    "type": "record",
    "name": "cutout",
    "fields": [
        {"name": "fileName", "type": "string"},
        {"name": "stampData", "type": "bytes"},
    ],
}
ALERT_SCHEMA: dict[str, Any] = {
    "type": "record",
    "name": "alert",
    "fields": [
        {"name": "objectId", "type": ["string", "null"]},
        {
            "name": "candidate",
            "type": [
                {
                    "type": "record",
                    "name": "candidate",
                    "fields": [
                        {"name": "candid", "type": ["long", "null"]},
                        {"name": "jd", "type": ["double", "null"]},
                        {"name": "fid", "type": ["int", "null"]},
                        {"name": "magpsf", "type": ["float", "null"]},
                        {"name": "sigmapsf", "type": ["float", "null"]},
                        {"name": "isdiffpos", "type": ["string", "null"]},
                        {"name": "rb", "type": ["float", "null"]},
                        {"name": "drb", "type": ["float", "null"]},
                        {"name": "ra", "type": ["double", "null"]},
                        {"name": "dec", "type": ["double", "null"]},
                        {"name": "xpos", "type": ["float", "null"]},
                        {"name": "ypos", "type": ["float", "null"]},
                        {"name": "chipsf", "type": ["float", "null"]},
                    ],
                },
                "null",
            ],
        },
        {"name": "prv_candidates", "type": ["null", {"type": "array", "items": "string"}]},
        {"name": "cutoutScience", "type": ["null", _CUTOUT]},
        {"name": "cutoutTemplate", "type": ["null", "cutout"]},
        {"name": "cutoutDifference", "type": ["null", "cutout"]},
    ],
}


def candidate_record(candid: int, **overrides: Any) -> dict[str, Any]:
    """A candidate record with plausible values; ``overrides`` replace fields."""
    record: dict[str, Any] = {
        "candid": candid,
        "jd": 2458495.0536111,
        "fid": 2,
        "magpsf": 19.5,
        "sigmapsf": 0.1,
        "isdiffpos": "t",
        "rb": 0.8,
        "drb": 0.9,
        "ra": 202.2342251,
        "dec": 31.8041336,
        "xpos": 1500.0,
        "ypos": 2000.0,
        "chipsf": float("nan"),
    }
    record.update(overrides)
    return record


def avro_bytes(
    oid: str,
    candidate: dict[str, Any] | None,
    cutouts: npt.NDArray[np.floating[Any]] | None,
    *,
    codec: str = "snappy",
    records: int = 1,
) -> bytes:
    """An avro alert file of ``records`` copies of one alert with ``cutouts`` ``(H, W, 3)``."""
    stamps: list[dict[str, Any] | None] = [None, None, None]
    if cutouts is not None:
        stamps = [
            {"fileName": f"{name}.fits.gz", "stampData": fits_bytes(cutouts[:, :, k])}
            for k, name in enumerate(("sci", "ref", "diff"))
        ]
    alert = {
        "objectId": oid,
        "candidate": candidate,
        "prv_candidates": ["ignored"],
        "cutoutScience": stamps[0],
        "cutoutTemplate": stamps[1],
        "cutoutDifference": stamps[2],
    }
    out = io.BytesIO()
    fastavro.writer(out, fastavro.parse_schema(ALERT_SCHEMA), [alert] * records, codec=codec)
    return out.getvalue()


def raw_cutouts(
    seed: int, shape: tuple[int, int] = (63, 63), nans: int = 0
) -> npt.NDArray[np.float32]:
    """Raw-looking float32 cutouts ``(H, W, 3)``: sky noise plus a source, ``nans`` NaNs."""
    rng = np.random.default_rng(seed)
    image = rng.normal(100.0, 5.0, size=(*shape, 3)).astype(np.float32)
    image[shape[0] // 2, shape[1] // 2] += 500.0
    if nans:
        flat = image.reshape(-1)
        flat[rng.choice(flat.size, size=nans, replace=False)] = np.nan
    return image


@dataclass
class SkyObject:
    """One object of the fake sky: its position and detections (candid, magpsf, sigmapsf)."""

    oid: str
    ra: float
    dec: float
    detections: list[tuple[int, float, float]]


@dataclass
class FakeAlerce:
    """ALeRCE's three endpoints, from memory (a :data:`RawFetch` through :meth:`fetch`).

    ``alerts`` maps (oid, candid) to avro bytes; ``failures`` maps a URL substring to the
    exceptions raised, one per request, before it succeeds. ``calls`` lists every URL.
    """

    objects: list[SkyObject] = field(default_factory=list)
    alerts: dict[tuple[str, int], bytes] = field(default_factory=dict)
    failures: dict[str, list[BaseException]] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    def fetch(self, url: str, max_bytes: int) -> tuple[bytes, str]:
        self.calls.append(url)
        for key, errors in self.failures.items():
            if key in url and errors:
                raise errors.pop(0)
        body = self._answer(url)
        return body[: max_bytes + 1], f"2026-11-08T00:00:{len(self.calls) % 60:02d}+00:00"

    @property
    def requests(self) -> int:
        return len(self.calls)

    def _answer(self, url: str) -> bytes:
        parsed = urllib.parse.urlsplit(url)
        query = dict(urllib.parse.parse_qsl(parsed.query))
        if url.startswith(f"{API_URL}/objects/?"):
            ra, dec, radius = (float(query[k]) for k in ("ra", "dec", "radius"))
            items = [
                {"oid": o.oid, "meanra": o.ra}
                for o in self.objects
                if _separation(o.ra, o.dec, ra, dec) <= radius
            ]
            return json.dumps({"items": items, "has_next": False}).encode()
        if url.startswith(f"{API_URL}/objects/") and url.endswith("/detections"):
            oid = url.removeprefix(f"{API_URL}/objects/").removesuffix("/detections")
            obj = next(o for o in self.objects if o.oid == oid)
            return json.dumps(
                [
                    {"candid": str(c), "magpsf": m, "sigmapsf": s, "fid": 1}
                    for c, m, s in obj.detections
                ]
            ).encode()
        if url.startswith(f"{AVRO_URL}/get_avro?"):
            key = (query["oid"], int(query["candid"]))
            if key not in self.alerts:
                raise HttpError(url, 404, "HTTP 404")
            return self.alerts[key]
        raise HttpError(url, 404, "HTTP 404")


def _separation(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Small-angle separation in arcsec."""
    dra = (ra1 - ra2) * math.cos(math.radians((dec1 + dec2) / 2))
    return math.hypot(dra, dec1 - dec2) * 3600


@dataclass
class World:
    """A prepared dataset, its fake ALeRCE, and the expected outcome per Validation row."""

    prepared: Path
    alerce: FakeAlerce
    expected: dict[int, str]
    raw: dict[int, npt.NDArray[np.float32]]


def _standardise(name: str, value: float) -> float:
    mean, std = M1_FEATURE_NORM_STATS[name]
    return (value - mean) / std


def make_world(root: Path) -> World:
    """A 10-row Validation split (2 per class, sorted by class) covering every status.

    Rows: 0 ok; 1 ok after a tie on the same object as row 0 (cache hits); 2 ok among two
    objects in the cone; 3 ghost_candidate (empty cone); 4 no_detection_match; 5 mismatch;
    6 ambiguous (two candids with identical cutouts); 7 ok with CCD-edge cutouts (63x50,
    repaired); 8 http_error on the first request (the fake fails once), ok on a resume; 9 bad_alert.
    Train and Test are copies of Validation.
    """
    alerce = FakeAlerce()
    labels = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    positions = [(10.0 + 0.01 * i, 20.0 + 0.01 * i) for i in range(10)]
    positions[1] = positions[0]  # rows 0 and 1: the same object
    photometry = [(18.0 + 0.1 * i, 0.05 + 0.001 * i) for i in range(10)]
    raw: dict[int, npt.NDArray[np.float32]] = {}
    stored = np.zeros((10, 63, 63, 3), dtype=np.float32)

    def add_alert(row: int, oid: str, candid: int, cutouts: npt.NDArray[np.float32]) -> None:
        ra, dec = positions[row]
        mag, sig = photometry[row]
        record = candidate_record(
            candid, magpsf=mag, sigmapsf=sig, ra=ra, dec=dec, xpos=30.0, ypos=1000.0
        )
        alerce.alerts[(oid, candid)] = avro_bytes(oid, record, cutouts)

    def truth(row: int, cutouts: npt.NDArray[np.float32]) -> None:
        raw[row] = cutouts
        full = (
            repair_non_squared_stamp(cutouts, 30.0, 1000.0)
            if cutouts.shape[0] != 63 or cutouts.shape[1] != 63
            else cutouts
        )
        stored[row] = alerce_normalise(np.asarray(full)).astype(np.float32)

    # Rows 0 and 1: one object, detections A (row 0), B and C (row 1's tie; C is right).
    mag0, sig0 = photometry[0]
    mag1, sig1 = photometry[1]
    alerce.objects.append(
        SkyObject(
            "ZTF18aaaaaaa",
            *positions[0],
            [(100, mag0, sig0), (101, mag1, sig1), (102, mag1 + 5e-5, sig1)],
        )
    )
    truth(0, raw_cutouts(0, nans=7))
    add_alert(0, "ZTF18aaaaaaa", 100, raw[0])
    truth(1, raw_cutouts(1))
    add_alert(1, "ZTF18aaaaaaa", 101, raw_cutouts(99))  # B: a different stamp
    add_alert(1, "ZTF18aaaaaaa", 102, raw[1])
    # Row 2: two objects in the cone, only the second has the detection.
    ra2, dec2 = positions[2]
    alerce.objects.append(SkyObject("ZTF18aaaaaab", ra2 + 0.0001, dec2, [(200, 15.0, 0.2)]))
    alerce.objects.append(SkyObject("ZTF18aaaaaac", ra2, dec2, [(201, *photometry[2])]))
    truth(2, raw_cutouts(2))
    add_alert(2, "ZTF18aaaaaac", 201, raw[2])
    # Row 3: nothing in the cone. Row 4: an object whose photometry does not match.
    alerce.objects.append(SkyObject("ZTF18aaaaaad", *positions[4], [(400, 12.0, 0.3)]))
    # Row 5: the alert's cutouts do not reproduce the stored stamp.
    alerce.objects.append(SkyObject("ZTF18aaaaaae", *positions[5], [(500, *photometry[5])]))
    truth(5, raw_cutouts(5))
    add_alert(5, "ZTF18aaaaaae", 500, raw_cutouts(55))
    # Row 6: two candids, identical cutouts: both pass.
    alerce.objects.append(
        SkyObject("ZTF18aaaaaaf", *positions[6], [(600, *photometry[6]), (601, *photometry[6])])
    )
    truth(6, raw_cutouts(6))
    add_alert(6, "ZTF18aaaaaaf", 600, raw[6])
    add_alert(6, "ZTF18aaaaaaf", 601, raw[6])
    # Row 7: cut at a CCD edge (xpos 30 < 40: padded on the left).
    alerce.objects.append(SkyObject("ZTF18aaaaaag", *positions[7], [(700, *photometry[7])]))
    truth(7, raw_cutouts(7, shape=(63, 50)))
    add_alert(7, "ZTF18aaaaaag", 700, raw[7])
    # Row 8: the first detections request fails, the next succeeds.
    alerce.objects.append(SkyObject("ZTF18aaaaaah", *positions[8], [(800, *photometry[8])]))
    truth(8, raw_cutouts(8))
    add_alert(8, "ZTF18aaaaaah", 800, raw[8])
    alerce.failures["ZTF18aaaaaah/detections"] = [
        HttpError("x", 503, "HTTP 503 (after 5 attempts)")
    ]
    # Row 9: the avro is corrupt.
    alerce.objects.append(SkyObject("ZTF18aaaaaai", *positions[9], [(900, *photometry[9])]))
    truth(9, raw_cutouts(9))
    alerce.alerts[("ZTF18aaaaaai", 900)] = b"Obj\x01 not really avro"

    features = np.zeros((10, len(FEATURE_NAMES)), dtype=np.float64)
    column = {name: i for i, name in enumerate(FEATURE_NAMES)}
    for row in range(10):
        values = {
            "ra": positions[row][0],
            "dec": positions[row][1],
            "magpsf": photometry[row][0],
            "sigmapsf": photometry[row][1],
        }
        for name, value in values.items():
            features[row, column[name]] = _standardise(name, value)
    split = SplitData(
        images=np.round(255 * stored).astype(np.uint8),
        labels=np.asarray(labels, dtype=np.int64),
        features=features,
        images_f32=stored,
    )
    data = StampData("classifier", dict.fromkeys(SPLITS, split))
    prepared = write_prepared(
        data,
        root / "prepared" / "world",
        {"name": "world.pkl", "size_bytes": 0, "sha256": "0" * 64},
    )
    expected = {
        0: "ok",
        1: "ok",
        2: "ok",
        3: "ghost_candidate",
        4: "no_detection_match",
        5: "mismatch",
        6: "ambiguous",
        7: "ok",
        8: "http_error",
        9: "bad_alert",
    }
    return World(prepared, alerce, expected, raw)
