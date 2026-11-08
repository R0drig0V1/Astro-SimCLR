"""Tests of the R5 recipe (task 3.5): JD_MAX, the seeded draw, ndet, first stamps, BTS types."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from astro_simclr.alerts.alerce import DetectionRecord, ResponseError, parse_detection_records
from astro_simclr.r5.recipe import (
    JD_MAX,
    SUBCLASSES,
    bts_subclass,
    draw_alert,
    eligible,
    first_stamp_order,
    ndet,
    oid_hash,
    year_of_jd,
)


def rec(candid: int, jd: float, stamp: bool = True) -> DetectionRecord:
    return DetectionRecord(candid, jd, stamp, 1, 10.0, 20.0)


def test_jd_max_is_the_last_second_of_2022() -> None:
    assert pytest.approx(2459945.5 - 1 / 86400, abs=1e-9) == JD_MAX
    assert year_of_jd(JD_MAX) == 2022
    assert year_of_jd(JD_MAX + 2 / 86400) == 2023
    assert year_of_jd(2458500.0) == 2019


def test_oid_hash_is_the_first_8_bytes_of_sha256() -> None:
    digest = hashlib.sha256(b"ZTF18abcdefg").digest()
    assert oid_hash("ZTF18abcdefg") == int.from_bytes(digest[:8], "big")
    assert oid_hash("ZTF18abcdefg") != oid_hash("ZTF18abcdefh")


def test_eligible_keeps_stamped_detections_up_to_jd_max_sorted_by_candid() -> None:
    records = [rec(30, 2459000.0), rec(10, 2459001.0), rec(20, JD_MAX), rec(40, JD_MAX + 1e-4)]
    records += [rec(5, 2458000.0, stamp=False), rec(10, 2459002.0)]  # no stamp; a repeat
    assert [r.candid for r in eligible(records)] == [10, 20, 30]
    assert ndet(records) == 4  # 5, 10, 20, 30: with or without a stamp, distinct candids


def index_of(oid: str, records: list[DetectionRecord], seed: int) -> int:
    draw = draw_alert(oid, records, seed=seed)
    assert draw is not None
    return draw.index


def test_draw_is_seeded_per_object_and_independent_of_the_list_order() -> None:
    records = [rec(c, 2459000.0 + c) for c in range(100, 110)]
    first = draw_alert("ZTF19aaaaaaa", records, seed=0)
    assert first is not None and first.n_eligible == 10
    expected = int(np.random.default_rng([0, oid_hash("ZTF19aaaaaaa")]).integers(10))
    assert first.index == expected and first.record.candid == 100 + expected
    again = draw_alert("ZTF19aaaaaaa", list(reversed(records)), seed=0)
    assert again == first
    draws = {index_of(f"ZTF19aaaaa{a}{b}", records, 0) for a in "ab" for b in "abcdefghij"}
    assert len(draws) > 3  # different objects, different streams
    assert len({index_of("ZTF19aaaaaaa", records, s) for s in range(20)}) > 3


def test_draw_ignores_detections_after_jd_max_and_without_stamps() -> None:
    late = [rec(1, JD_MAX + 1.0), rec(2, 2459000.0, stamp=False)]
    assert draw_alert("ZTF19aaaaaaa", late) is None
    with pytest.raises(ValueError, match="non-negative"):
        draw_alert("ZTF19aaaaaaa", late, seed=-1)


def test_first_stamp_order_is_by_time_then_candid() -> None:
    records = [rec(3, 2459000.0), rec(1, 2459001.0), rec(2, 2459000.0), rec(4, JD_MAX + 1)]
    assert [r.candid for r in first_stamp_order(records)] == [2, 3, 1]


@pytest.mark.parametrize(
    ("bts", "expected"),
    [
        ("SN Ia", "SNIa"),
        ("SN Ia-91T", "SNIa"),
        ("SN Iax", "SNIa"),
        ("SN II", "SNII"),
        ("SN IIP", "SNII"),
        ("SN IIn", "SNIIn"),
        ("SN IIb", "SNIIb"),
        ("SN Ib/c", "SNIbc"),
        ("SN Ic-BL", "SNIbc"),
        ("SN Icn", "SNIbc"),
        ("SLSN-I", "SLSN"),
        ("TDE", None),
        ("", None),
    ],
)
def test_bts_types_map_to_rj23_subclasses(bts: str, expected: str | None) -> None:
    assert bts_subclass(bts) == expected


def test_the_plan_table_caps_and_absences() -> None:
    by_name = {s.name: s for s in SUBCLASSES}
    assert by_name["EB"].source_classes == ("E",) and by_name["EB"].cap == 2400
    assert by_name["EB"].target == 2339
    assert by_name["RSCVn"].source_classes == () and by_name["ZZ"].source_classes == ()
    assert sum(s.target or 0 for s in SUBCLASSES if s.label == "VS") == 10211
    assert sum(s.target or 0 for s in SUBCLASSES if s.label == "AGN") == 9774
    assert all(s.target is None or s.cap is None or s.target <= s.cap for s in SUBCLASSES)
    assert len({s.name for s in SUBCLASSES}) == len(SUBCLASSES)


def test_parse_detection_records_reads_mjd_jd_and_has_stamp() -> None:
    body = (
        b'[{"candid": "123", "mjd": 58500.0, "has_stamp": true, "fid": 2, "ra": 1.5, "dec": -2.0},'
        b' {"candid": 124, "jd": 2459000.5, "has_stamp": "yes", "fid": true, "ra": "x"},'
        b' {"candid": null, "mjd": 58500.0}, {"candid": 125}, {"candid": 126, "mjd": 1e999}]'
    )
    found = parse_detection_records(body, "ZTF18aaaaaaa")
    assert found.skipped == 3
    first, second = found.records
    assert first == DetectionRecord(123, 58500.0 + 2400000.5, True, 2, 1.5, -2.0)
    assert second.jd == 2459000.5 and second.has_stamp is False
    assert second.fid is None and second.ra is None and second.dec is None
    assert parse_detection_records(b'{"items": []}', "ZTF18aaaaaaa").records == ()
    for bad in (b"not json", b'{"items": 3}', b"[1]"):
        with pytest.raises(ResponseError):
            parse_detection_records(bad, "ZTF18aaaaaaa")
