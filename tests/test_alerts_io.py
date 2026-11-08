"""Tests of the alert parsers (task 3.1): FITS cutouts, avro alerts and ALeRCE's JSON.

The FITS files and avro alerts are written by the tests (``_alerts_support``); downloaded
bytes are untrusted, so the error paths and the size limits are tested as much as the
happy path.
"""

from __future__ import annotations

import gzip
import json
from typing import Any

import numpy as np
import pytest
from _alerts_support import avro_bytes, candidate_record, fits_bytes, raw_cutouts

from astro_simclr.alerts.alerce import AlerceClient, ResponseError, is_oid, make_client
from astro_simclr.alerts.fits import FitsError, gunzip, read_fits_image
from astro_simclr.alerts.parse import AlertFormatError, check_container, parse_alert
from astro_simclr.alerts.snapshot import Recorder

OID = "ZTF19aadgxok"


# FITS ---------------------------------------------------------------------------------


def test_fits_round_trip_keeps_orientation_and_nan() -> None:
    image = np.arange(4 * 6, dtype=np.float32).reshape(4, 6)
    image[1, 2] = np.nan
    for compress in (True, False):
        out = read_fits_image(fits_bytes(image, compress=compress))
        assert out.dtype == np.float32 and out.flags.writeable
        np.testing.assert_array_equal(out, image)  # NAXIS2 rows of NAXIS1, no flip


def test_fits_float64_and_neutral_scaling() -> None:
    image = np.linspace(0, 1, 9).reshape(3, 3)
    data = fits_bytes(image, bitpix=-64, cards=("BSCALE  =                  1.0", "BZERO   = 0.0"))
    np.testing.assert_allclose(read_fits_image(data), image.astype(np.float32))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bitpix": 16}, "BITPIX 16"),
        ({"cards": ("BSCALE  =                  2.0",)}, "BSCALE"),
        ({"cards": ("BZERO   =                 1D3",)}, "BZERO"),
        ({"cards": ("BZERO   = 'abc'",)}, "not a number"),
    ],
)
def test_fits_rejects_unsupported_images(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(FitsError, match=message):
        read_fits_image(fits_bytes(np.zeros((2, 2)), **kwargs))  # type: ignore[arg-type]


def _raw_fits(cards: list[str], data: bytes = b"") -> bytes:
    text = "".join(c.ljust(80) for c in [*cards, "END"])
    return (text + " " * (-len(text) % 2880)).encode("ascii") + data


def test_fits_rejects_malformed_headers() -> None:
    simple = "SIMPLE  =                    T"
    with pytest.raises(FitsError, match="SIMPLE"):
        read_fits_image(_raw_fits(["SIMPLE  =                    F"]))
    with pytest.raises(FitsError, match="missing keyword BITPIX"):
        read_fits_image(_raw_fits([simple]))
    with pytest.raises(FitsError, match="not an integer"):
        read_fits_image(_raw_fits([simple, "BITPIX  = 'x'"]))
    with pytest.raises(FitsError, match="NAXIS must be 2"):
        read_fits_image(_raw_fits([simple, "BITPIX  = -32", "NAXIS   = 3"]))
    big = [simple, "BITPIX  = -32", "NAXIS   = 2", "NAXIS1  = 9999", "NAXIS2  = 1"]
    with pytest.raises(FitsError, match="outside"):
        read_fits_image(_raw_fits(big))
    small = [simple, "BITPIX  = -32", "NAXIS   = 2", "NAXIS1  = 4", "NAXIS2  = 4"]
    with pytest.raises(FitsError, match="truncated image"):
        read_fits_image(_raw_fits(small, b"\0" * 8))
    with pytest.raises(FitsError, match="no END"):
        read_fits_image((simple.ljust(80) * 36).encode() * 33)
    with pytest.raises(FitsError, match="truncated header"):
        read_fits_image(simple.encode())
    with pytest.raises(FitsError, match="non-ASCII"):
        read_fits_image("é".encode().ljust(2880))


def test_gunzip_limits_and_errors() -> None:
    bomb = gzip.compress(b"\0" * 100_000)
    with pytest.raises(FitsError, match="more than 1000"):
        gunzip(bomb, 1000)
    assert gunzip(gzip.compress(b"abc") + b"trailing", 10) == b"abc"
    with pytest.raises(FitsError, match="truncated"):
        gunzip(gzip.compress(b"abc" * 100)[:20], 1000)
    with pytest.raises(FitsError, match="corrupt"):
        gunzip(b"\x1f\x8b\x08\x00garbage-garbage", 1000)
    with pytest.raises(FitsError, match="uncompressed"):
        gunzip(b"x" * 11, 10)


# Avro ---------------------------------------------------------------------------------


@pytest.mark.parametrize("codec", ["null", "deflate", "snappy"])
def test_parse_alert_reads_cutouts_and_candidate(codec: str) -> None:
    cutouts = raw_cutouts(3, nans=4)
    alert = parse_alert(
        avro_bytes(OID, candidate_record(7), cutouts, codec=codec), oid=OID, candid=7
    )
    assert alert.oid == OID and alert.candid == 7 and not alert.repaired
    np.testing.assert_array_equal(alert.cutouts, cutouts)
    assert alert.candidate["isdiffpos"] == "t" and alert.candidate["chipsf"] is None
    assert alert.raw_shape == (63, 63)


def test_parse_alert_repairs_ccd_edge_cutouts() -> None:
    cutouts = raw_cutouts(4, shape=(50, 63))
    record = candidate_record(7, ypos=3000.0)  # ypos >= 40: rows padded after
    alert = parse_alert(avro_bytes(OID, record, cutouts), oid=OID, candid=7)
    assert alert.repaired and alert.raw_shape == (50, 63)
    np.testing.assert_array_equal(alert.cutouts[:50], cutouts)
    assert np.isnan(alert.cutouts[50:]).all()


@pytest.mark.parametrize(
    ("oid", "candid", "record", "cutouts", "records", "message"),
    [
        ("ZTF19aaaaaaa", 7, candidate_record(7), raw_cutouts(0), 1, "objectId"),
        (OID, 8, candidate_record(7), raw_cutouts(0), 1, "candid 7, expected 8"),
        (OID, 7, None, raw_cutouts(0), 1, "no candidate"),
        (OID, 7, candidate_record(7, jd=None), raw_cutouts(0), 1, "candidate.jd"),
        (OID, 7, candidate_record(7), None, 1, "cutoutScience: no stampData"),
        (OID, 7, candidate_record(7), raw_cutouts(0), 2, "more than one record"),
        (OID, 7, candidate_record(7), raw_cutouts(0), 0, "no record"),
        (OID, 7, candidate_record(7), raw_cutouts(0, shape=(64, 64)), 1, "larger than 63"),
        (OID, 7, candidate_record(7, xpos=None), raw_cutouts(0, shape=(60, 63)), 1, "xpos"),
    ],
)
def test_parse_alert_rejects_inconsistent_alerts(
    oid: str,
    candid: int,
    record: dict[str, object] | None,
    cutouts: np.ndarray | None,
    records: int,
    message: str,
) -> None:
    data = avro_bytes(oid, record, cutouts, records=records)
    with pytest.raises(AlertFormatError, match=message):
        parse_alert(data, oid=OID, candid=candid)


def test_parse_alert_rejects_cutouts_of_different_shapes_and_bad_fits() -> None:
    data = avro_bytes(OID, candidate_record(7), raw_cutouts(0))
    good = parse_alert(data, oid=OID, candid=7)
    assert good.cutouts.shape == (63, 63, 3)
    other = data.replace(fits_bytes(raw_cutouts(0)[:, :, 0])[:20], b"x" * 20, 1)
    with pytest.raises(AlertFormatError):
        parse_alert(other, oid=OID, candid=7)


def test_check_container_bounds_the_decompressed_size() -> None:
    zeros = np.zeros((63, 63, 3), dtype=np.float32)
    for codec in ("null", "deflate", "snappy"):
        data = avro_bytes(OID, candidate_record(7), zeros, codec=codec)
        check_container(data)
        with pytest.raises(AlertFormatError, match="decompresses to more than 100"):
            check_container(data, max_bytes=100)
    with pytest.raises(AlertFormatError, match="bad magic"):
        check_container(b"PK\x03\x04")
    with pytest.raises(AlertFormatError, match="truncated"):
        check_container(avro_bytes(OID, candidate_record(7), zeros)[:-30])
    with pytest.raises(AlertFormatError, match="codec 'bzip2'"):
        check_container(avro_bytes(OID, candidate_record(7), zeros, codec="bzip2"))
    with pytest.raises(AlertFormatError, match="varint"):
        check_container(b"Obj\x01" + b"\xff" * 12)
    unsynced = bytearray(avro_bytes(OID, candidate_record(7), zeros, codec="null"))
    unsynced[-1] ^= 0xFF  # break the last sync marker
    with pytest.raises(AlertFormatError, match="sync"):
        check_container(bytes(unsynced))
    deflated = bytearray(avro_bytes(OID, candidate_record(7), zeros, codec="deflate"))
    block = deflated.index(deflated[-16:]) + 16  # the header ends with the sync marker
    start = block + 1  # the object count (1) is a one-byte varint
    while deflated[start] & 0x80:  # the block size varint
        start += 1
    deflated[start + 1] = 0xFF  # BFINAL 1, BTYPE 11: a reserved deflate block type
    with pytest.raises(AlertFormatError, match="corrupt deflate"):
        check_container(bytes(deflated))


# ALeRCE JSON --------------------------------------------------------------------------


class _Getter:
    """Serves fixed bodies by URL substring; a minimal Getter."""

    def __init__(self, bodies: dict[str, object]) -> None:
        self.bodies = bodies
        self.urls: list[str] = []

    @property
    def requests(self) -> int:
        return len(self.urls)

    def get(self, url: str, *, max_bytes: int, kind: str, cache: bool) -> bytes:
        self.urls.append(url)
        for key, body in self.bodies.items():
            if key in url:
                return body if isinstance(body, bytes) else json.dumps(body).encode()
        raise AssertionError(url)


def test_objects_near_builds_the_url_and_validates_items() -> None:
    getter = _Getter({"/objects/?": {"items": [{"oid": OID}, {"oid": OID}], "has_next": True}})
    page = AlerceClient(getter).objects_near(202.2342251, 31.8041336)
    assert page.oids == (OID,) and page.has_next
    assert getter.urls == [
        "https://api.alerce.online/ztf/v1/objects/?ra=202.2342251&dec=31.8041336&radius=1.5"
    ]
    bad_bodies: list[object] = [{"items": [{"oid": "../x"}]}, {"items": "x"}, [], b"\xff", b"{"]
    for bad in bad_bodies:
        with pytest.raises(ResponseError):
            AlerceClient(_Getter({"/objects/?": bad})).objects_near(1.0, 2.0)


def test_detections_accepts_lists_and_items_and_skips_unusable_entries() -> None:
    entries = [
        {"candid": "732508412615010002", "magpsf": 20.6911, "sigmapsf": 0.27},
        {"candid": 5, "magpsf": 20, "sigmapsf": 0.1},
        {"candid": None, "magpsf": 1.0, "sigmapsf": 1.0},
        {"candid": True, "magpsf": 1.0, "sigmapsf": 1.0},
        {"candid": "12a", "magpsf": 1.0, "sigmapsf": 1.0},
        {"candid": 6, "magpsf": None, "sigmapsf": 1.0},
        {"candid": 6, "magpsf": False, "sigmapsf": 1.0},
    ]
    for payload in (entries, {"items": entries}):
        found = AlerceClient(_Getter({"/detections": payload})).detections(OID)
        assert [d.candid for d in found] == [732508412615010002, 5]
    with pytest.raises(ResponseError, match="expected a list"):
        AlerceClient(_Getter({"/detections": {"x": 1}})).detections(OID)
    with pytest.raises(ResponseError, match="objects in the list"):
        AlerceClient(_Getter({"/detections": [1]})).detections(OID)
    with pytest.raises(ValueError, match="not a ZTF object id"):
        AlerceClient(_Getter({})).detections("ZTF19/../../x")


def test_alert_avro_url_and_production_client() -> None:
    getter = _Getter({"get_avro": b"bytes"})
    assert AlerceClient(getter).alert_avro(OID, 740553612615010004) == b"bytes"
    assert getter.urls == [
        f"https://avro.alerce.online/get_avro?oid={OID}&candid=740553612615010004"
    ]
    assert is_oid(OID) and not is_oid("ZTF19AADGXOK") and not is_oid(3)
    client = make_client(2.0)
    assert isinstance(client.getter, Recorder) and client.requests == 0


def test_hostile_json_numbers_are_response_errors() -> None:
    huge = b'{"items": [], "n": ' + b"9" * 5000 + b"}"
    with pytest.raises(ResponseError, match="invalid JSON"):
        AlerceClient(_Getter({"/objects/?": huge})).objects_near(1.0, 2.0)
    entries = [
        {"candid": "9" * 20, "magpsf": 1.0, "sigmapsf": 1.0},
        {"candid": 2**63, "magpsf": 1.0, "sigmapsf": 1.0},
        {"candid": 0, "magpsf": 1.0, "sigmapsf": 1.0},
        {"candid": 7, "magpsf": 10**400, "sigmapsf": 1.0},
        {"candid": 8, "magpsf": 1.0, "sigmapsf": 1.0},
    ]
    found = AlerceClient(_Getter({"/detections": entries})).detections(OID)
    assert [d.candid for d in found] == [8]


def _schema_bytes(schema: Any) -> bytes:
    import io

    import fastavro

    out = io.BytesIO()
    fastavro.writer(out, fastavro.parse_schema(schema), [], codec="null")
    return out.getvalue()


@pytest.mark.parametrize(
    "items",
    [
        "null",
        {"type": "record", "name": "empty", "fields": []},
        {"type": "record", "name": "nulls", "fields": [{"name": "a", "type": "null"}]},
        {"type": "fixed", "name": "zero", "size": 0},
    ],
)
def test_check_container_refuses_arrays_of_zero_size_items(items: object) -> None:
    schema = {
        "type": "record",
        "name": "alert",
        "fields": [{"name": "xs", "type": {"type": "array", "items": items}}],
    }
    with pytest.raises(AlertFormatError, match="zero-size items"):
        check_container(_schema_bytes(schema))


def test_check_container_accepts_bounded_schemas_and_refuses_non_records() -> None:
    bounded = {
        "type": "record",
        "name": "alert",
        "fields": [
            {"name": "e", "type": {"type": "enum", "name": "E", "symbols": ["A"]}},
            {"name": "f", "type": {"type": "fixed", "name": "F", "size": 2}},
            {"name": "u", "type": {"type": "array", "items": ["null", "int"]}},
            {"name": "m", "type": {"type": "map", "values": "E"}},
            {"name": "r", "type": {"type": "array", "items": "F"}},
            {"name": "s", "type": {"type": "array", "items": {"type": "string"}}},
        ],
    }
    check_container(_schema_bytes(bounded))
    with pytest.raises(AlertFormatError, match="not a record"):
        check_container(_schema_bytes({"type": "array", "items": "int"}))
    from astro_simclr.alerts.parse import check_schema

    with pytest.raises(AlertFormatError, match="no schema"):
        check_schema(None)
    with pytest.raises(AlertFormatError, match="invalid avro schema"):
        check_schema(b"{")
