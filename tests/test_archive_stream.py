"""Tests of the archive tarball stream and of the selection rules (offline, synthetic data).

The tarballs are written by the tests (``_archive_support``), including hostile members
(links, devices, ``..`` and absolute names), oversized members, a gzip bomb, truncated
and non-gzip files: the stream must never extract anything and must stop at its limits.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import tarfile
from pathlib import Path

import numpy as np
import pytest
from _archive_support import JD_2020, Spec, archive_alert, write_tarball

from astro_simclr.alerts.parse import Alert, AlertFormatError, CutoutShapeError, parse_alert
from astro_simclr.archive.selection import (
    Exclusions,
    Selection,
    first_detection_failure,
    jd_year,
    load_exclusions,
    oid_year,
)
from astro_simclr.archive.stream import TarballError, TarballStream, night_of, tarball_name
from astro_simclr.data.schema import DataError

OID = "ZTF20aaaaaaa"


# stream -------------------------------------------------------------------------------


def test_stream_yields_avro_members_and_hashes_the_whole_file(tmp_path: Path) -> None:
    alert = archive_alert(Spec(OID, 11))
    path = write_tarball(
        tmp_path / "ztf_public_20200101.tar.gz",
        [
            ("a/11.avro", alert),
            ("dir", "dir"),
            ("link.avro", "symlink"),
            ("hard.avro", "hardlink"),
            ("pipe.avro", "fifo"),
            ("readme.txt", b"not an alert"),
            ("../../escape.avro", b"junk"),
            ("/abs/12.avro", alert),
        ],
    )
    before = sorted(os.listdir(tmp_path))
    stream = TarballStream(path)
    members = list(stream)
    assert [(m.position, m.name) for m in members] == [
        (0, "a/11.avro"),
        (6, "../../escape.avro"),
        (7, "/abs/12.avro"),
    ]
    assert members[0].data == alert
    stats = stream.stats
    assert (stats.members, stats.avro) == (8, 3)
    assert (stats.skipped_not_file, stats.skipped_not_avro, stats.skipped_oversize) == (4, 1, 0)
    assert stats.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert stats.compressed_bytes == path.stat().st_size
    assert stats.decompressed_bytes == len(gzip.decompress(path.read_bytes()))
    assert stats.to_json()["sha256"] == stats.sha256
    assert sorted(os.listdir(tmp_path)) == before  # nothing extracted
    assert not (tmp_path.parent / "escape.avro").exists()


def test_stream_skips_oversized_members(tmp_path: Path) -> None:
    path = write_tarball(
        tmp_path / "ztf_public_20200101.tar.gz", [("big.avro", b"x" * 100), ("s.avro", b"y")]
    )
    stream = TarballStream(path, max_member_bytes=10)
    assert [m.name for m in stream] == ["s.avro"]
    assert stream.stats.skipped_oversize == 1


def test_stream_stops_a_gzip_bomb_at_the_decompressed_cap(tmp_path: Path) -> None:
    path = write_tarball(tmp_path / "ztf_public_20200101.tar.gz", [("z.avro", b"\0" * 2_000_000)])
    assert path.stat().st_size < 20_000  # it compresses about 1000x
    stream = TarballStream(path, max_member_bytes=10**7, max_decompressed_bytes=100_000)
    with pytest.raises(TarballError, match="decompresses to more than 100000"):
        list(stream)
    assert stream.stats.decompressed_bytes <= 100_001


def test_stream_caps_the_number_of_members(tmp_path: Path) -> None:
    path = write_tarball(
        tmp_path / "ztf_public_20200101.tar.gz", [(f"{i}.avro", b"a") for i in range(5)]
    )
    with pytest.raises(TarballError, match="more than 3 members"):
        list(TarballStream(path, max_members=3))


@pytest.mark.parametrize("damage", ["truncate", "not_gzip", "gzip_not_tar", "corrupt"])
def test_stream_refuses_damaged_files(tmp_path: Path, damage: str) -> None:
    path = write_tarball(
        tmp_path / "ztf_public_20200101.tar.gz",
        [(f"{i}.avro", archive_alert(Spec(OID, i + 1, seed=i))) for i in range(3)],
    )
    data = path.read_bytes()
    if damage == "truncate":
        data = data[: len(data) // 2]
    elif damage == "not_gzip":
        data = b"PK\x03\x04" + data[4:]
    elif damage == "gzip_not_tar":
        data = gzip.compress(b"\x01" * 2048)
    else:
        data = data[:40] + bytes(b ^ 0xFF for b in data[40:80]) + data[80:]
    path.write_bytes(data)
    with pytest.raises(TarballError, match=r"not a readable \.tar\.gz"):
        list(TarballStream(path))


def test_stream_propagates_real_io_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(TarballStream(tmp_path / "ztf_public_20200101.tar.gz"))


def test_stream_file_protocol_refuses_writes_and_seeks(tmp_path: Path) -> None:
    from astro_simclr.archive.stream import StreamStats, _HashingReader, _LimitedReader

    stats = StreamStats()
    raw = _HashingReader(io.BytesIO(gzip.compress(b"abc")), stats)
    with pytest.raises(io.UnsupportedOperation):
        raw.seek(0)
    with gzip.GzipFile(fileobj=raw, mode="rb") as unzipped:
        limited = _LimitedReader(unzipped, stats, 10)
        assert limited.read() == b"abc" and limited.tell() == 3
        with pytest.raises(io.UnsupportedOperation):
            limited.write(b"x")
        with pytest.raises(io.UnsupportedOperation):
            limited.seek(0)
        limited.close()


def test_night_names() -> None:
    assert night_of(Path("x/ztf_public_20190131.tar.gz")) == 20190131
    assert tarball_name(20190131) == "ztf_public_20190131.tar.gz"
    with pytest.raises(TarballError, match="not an archive tarball name"):
        night_of(Path("ztf_public_2019013.tar.gz"))
    with pytest.raises(TarballError, match="not a valid date"):
        night_of(Path("ztf_public_20190231.tar.gz"))


def test_tarfile_writer_of_the_support_makes_regular_members(tmp_path: Path) -> None:
    path = write_tarball(tmp_path / "t.tar.gz", [("a.avro", b"1"), ("d", "dir")])
    with tarfile.open(path) as tar:
        assert [m.isreg() for m in tar.getmembers()] == [True, False]


# parser additions ---------------------------------------------------------------------


def test_parse_without_an_expected_identity_counts_previous_detections() -> None:
    data = archive_alert(Spec(OID, 42, previous=[None, 7, None, 9], nans=5))
    alert = parse_alert(data, repair=False)
    assert (alert.oid, alert.candid, alert.previous_detections) == (OID, 42, 2)
    assert alert.cutouts.shape == (63, 63, 3) and np.isnan(alert.cutouts).sum() == 5
    assert parse_alert(archive_alert(Spec(OID, 43))).previous_detections == 0
    with pytest.raises(AlertFormatError, match="objectId"):
        parse_alert(data, oid="ZTF20bbbbbbb")
    with pytest.raises(AlertFormatError, match="candid 42, expected 41"):
        parse_alert(data, candid=41)


def test_parse_refuses_or_repairs_an_edge_cutout() -> None:
    data = archive_alert(Spec(OID, 42, shape=(50, 63)))
    with pytest.raises(CutoutShapeError, match=r"\(50, 63\)"):
        parse_alert(data, repair=False)
    repaired = parse_alert(data)
    assert repaired.repaired and repaired.cutouts.shape == (63, 63, 3)


# selection ----------------------------------------------------------------------------


def _alert(oid: str = OID, previous: list[int | None] | None = None, **candidate: object) -> Alert:
    return parse_alert(archive_alert(Spec(oid, 5, candidate=candidate, previous=previous or [])))


def test_first_detection_rules() -> None:
    default = Selection()
    assert first_detection_failure(_alert(), default) is None
    assert first_detection_failure(_alert(ndethist=40), default) is None  # no bound by default
    assert first_detection_failure(_alert(previous=[None, None]), default) is None
    assert first_detection_failure(_alert(previous=[None, 3]), default) == "previous"
    assert first_detection_failure(_alert("ZTF19aaaaaaa"), default) == "oid_year"
    assert first_detection_failure(_alert("XYZ20aaaaaaa"), default) == "oid_year"
    assert first_detection_failure(_alert("ZTF19aaaaaaa"), Selection(oid_year=False)) is None
    bounded = Selection(max_ndethist=1)
    assert first_detection_failure(_alert(ndethist=1), bounded) is None
    assert first_detection_failure(_alert(ndethist=2), bounded) == "ndethist"
    assert first_detection_failure(_alert(ndethist=None), bounded) == "ndethist"
    assert first_detection_failure(_alert(ndethist=None), default) is None
    record = bounded.to_json()
    assert record["first_detection"] == {
        "previous_detections": 0,
        "oid_year": True,
        "max_ndethist": 1,
    }
    assert record["candidate_cuts"] == []
    with pytest.raises(ValueError, match="max_ndethist"):
        Selection(max_ndethist=-1)


def test_oid_and_jd_years() -> None:
    assert oid_year("ZTF18abcdefg") == 2018
    assert oid_year("ZTF18abcdef") is None and oid_year("ZTF18ABCDEFG") is None
    assert jd_year(2458119.5) == 2018  # 2018-01-01 00:00 UT
    assert jd_year(2458119.4999) == 2017
    assert jd_year(JD_2020) == 2020


def test_exclusion_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file = tmp_path / "exclude.txt"
    file.write_text("# CD21 + R5 Validation/Test\nZTF18aaaaaaa\n\n  ZTF19bbbbbbb \nZTF18aaaaaaa\n")
    exclusions = load_exclusions(file)
    assert exclusions.oids == {"ZTF18aaaaaaa", "ZTF19bbbbbbb"}
    assert exclusions.sha256 == hashlib.sha256(file.read_bytes()).hexdigest()
    assert exclusions.to_json() == {
        "source": "exclude.txt",
        "sha256": exclusions.sha256,
        "count": 2,
    }
    assert Exclusions.none().to_json() == {"source": "", "sha256": "", "count": 0}
    for bad in ("ZTF18 aaa\n", "x" * 17 + "\n", "oid,label\n"):
        file.write_text(bad)
        with pytest.raises(DataError, match="not an oid"):
            load_exclusions(file)
    file.write_bytes(b"\xff\xfe")
    with pytest.raises(DataError, match="not UTF-8"):
        load_exclusions(file)
    monkeypatch.setattr("astro_simclr.archive.selection.MAX_EXCLUSION_BYTES", 1)
    with pytest.raises(DataError, match="larger than"):
        load_exclusions(file)


# review fixes: tar metadata members, the gzip trailer, extreme dates ----------------


def _tar_with(path: Path, info: tarfile.TarInfo, fmt: int) -> Path:
    with tarfile.open(path, "w:gz", format=fmt) as tar:
        tar.addfile(info, io.BytesIO(b"a"))
    return path


@pytest.mark.parametrize("kind", ["pax", "gnu_longname"])
def test_oversized_tar_metadata_members_are_refused(tmp_path: Path, kind: str) -> None:
    if kind == "pax":
        info = tarfile.TarInfo("a.avro")
        info.pax_headers = {"comment": "x" * 2_000_000}
        fmt = tarfile.PAX_FORMAT
    else:
        info = tarfile.TarInfo("d/" * 1_000_000 + "a.avro")
        fmt = tarfile.GNU_FORMAT
    info.size = 1
    path = _tar_with(tmp_path / "ztf_public_20200101.tar.gz", info, fmt)
    assert path.stat().st_size < 50_000
    with pytest.raises(TarballError, match="metadata member of"):
        list(TarballStream(path))


def test_small_tar_metadata_members_are_read(tmp_path: Path) -> None:
    info = tarfile.TarInfo("n" * 300 + ".avro")  # a long name: a pax header
    info.size = 1
    path = _tar_with(tmp_path / "ztf_public_20200101.tar.gz", info, tarfile.PAX_FORMAT)
    assert [m.data for m in TarballStream(path)] == [b"a"]


def test_sparse_members_are_refused(tmp_path: Path) -> None:
    info = tarfile.TarInfo("s.avro")
    info.size = 1
    path = tmp_path / "ztf_public_20200101.tar.gz"
    with tarfile.open(path, "w:gz", format=tarfile.GNU_FORMAT) as tar:
        tar.addfile(info, io.BytesIO(b"a"))
    data = bytearray(gzip.decompress(path.read_bytes()))
    data[156:157] = tarfile.GNUTYPE_SPARSE  # the type flag of the first header
    data[148:156] = b" " * 8  # the checksum over the header with blanks in its field
    data[148:155] = f"{sum(data[:512]):06o}\0".encode()
    path.write_bytes(gzip.compress(bytes(data)))
    with pytest.raises(TarballError, match="sparse"):
        list(TarballStream(path))


@pytest.mark.parametrize("damage", ["crc", "length", "cut_trailer"])
def test_the_gzip_trailer_is_checked(tmp_path: Path, damage: str) -> None:
    path = write_tarball(
        tmp_path / "ztf_public_20200101.tar.gz", [("1.avro", archive_alert(Spec(OID, 1)))]
    )
    assert len(list(TarballStream(path))) == 1
    data = bytearray(path.read_bytes())
    if damage == "crc":
        data[-8] ^= 0xFF
    elif damage == "length":
        data[-1] ^= 0xFF
    else:
        del data[-4:]
    path.write_bytes(bytes(data))
    with pytest.raises(TarballError, match="not a readable"):
        list(TarballStream(path))


def test_jd_year_outside_the_calendar_is_none() -> None:
    assert jd_year(1.0) is None and jd_year(1e300) is None
