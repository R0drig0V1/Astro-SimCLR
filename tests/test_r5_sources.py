"""Tests of the R5 source snapshots (task 3.5): Zenodo archive checks, BTS parsing, records."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
from _r5_support import NOW, bts_csv, gz, stream_of

from astro_simclr.data.schema import DataError
from astro_simclr.r5.sources import (
    BTS,
    BTS_URL,
    ZENODO,
    ZENODO_MEMBER,
    download_bts,
    download_zenodo,
    load_source,
    read_bts,
    read_zenodo,
)

CSV = (
    b"oid,classALeRCE,ra,dec,source,id_source\n"
    b"ZTF18aaaaaaa,AGN,10.5,-3.25,x,1\nZTF19bbbbbbb,E,,nan,y,2\n"
)


def archive(members: dict[str, bytes], kinds: dict[str, bytes] | None = None) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            if kinds and name in kinds:
                info.type = kinds[name]
                info.size = 0
                tar.addfile(info)
                continue
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def zenodo(
    dest: Path,
    data: bytes,
    *,
    expected_size: int | None = None,
    expected_md5: str | None = None,
    keep_archive: bool = False,
) -> Path:
    return download_zenodo(
        dest,
        stream=lambda _u: stream_of(data),
        now=lambda: NOW,
        expected_size=len(data) if expected_size is None else expected_size,
        expected_md5=expected_md5 or hashlib.md5(data, usedforsecurity=False).hexdigest(),
        keep_archive=keep_archive,
    )


def test_zenodo_snapshot_reads_the_member_without_extracting(tmp_path: Path) -> None:
    data = archive({"../evil/x.csv": b"no", f"dir/{ZENODO_MEMBER}": CSV})
    path = zenodo(tmp_path, data)
    assert path == tmp_path / f"{ZENODO}.csv" and path.read_bytes() == CSV
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"{ZENODO}.csv", f"{ZENODO}.json"]
    record = json.loads((tmp_path / f"{ZENODO}.json").read_text())
    assert record["sha256"] == hashlib.sha256(CSV).hexdigest()
    assert record["archive"]["sha256"] == hashlib.sha256(data).hexdigest()
    assert record["licence"] == "CC-BY-4.0" and record["retrieved_at"] == NOW
    _, loaded = load_source(tmp_path, ZENODO)
    assert loaded == record
    rows = read_zenodo(path)
    assert [(r.row, r.oid, r.cls, r.ra, r.dec) for r in rows] == [
        (0, "ZTF18aaaaaaa", "AGN", 10.5, -3.25),
        (1, "ZTF19bbbbbbb", "E", None, None),
    ]


def test_zenodo_keep_archive(tmp_path: Path) -> None:
    data = archive({ZENODO_MEMBER: CSV})
    zenodo(tmp_path, data, keep_archive=True)
    assert (tmp_path / "files_for_lc_classifier_SanchezSaez2020.tar.gz").read_bytes() == data


@pytest.mark.parametrize(
    ("data", "size", "md5", "message"),
    [
        (archive({ZENODO_MEMBER: CSV}), None, "0" * 32, "MD5"),
        (archive({ZENODO_MEMBER: CSV}), 10, None, "larger than"),
        (archive({"other.csv": CSV}), None, None, "no member"),
        (archive({ZENODO_MEMBER: b""}, {ZENODO_MEMBER: tarfile.SYMTYPE}), None, None, "not a file"),
        (gz(b"not a tar file at all" * 100), None, None, "readable"),
    ],
    ids=["md5", "size", "no-member", "symlink", "not-tar"],
)
def test_zenodo_refuses_a_bad_archive(
    tmp_path: Path, data: bytes, size: int | None, md5: str | None, message: str
) -> None:
    with pytest.raises(DataError, match=message):
        zenodo(tmp_path, data, expected_size=size, expected_md5=md5)
    assert not list(tmp_path.iterdir())  # the partial archive is removed, nothing written


def test_bts_snapshot_and_parse(tmp_path: Path) -> None:
    body = bts_csv([("ZTF18aaaaaaa", "SN Ia", 193.3, -6.385), ("ZTF19bbbbbbb", "TDE", 0.5, 45.0)])
    urls: list[str] = []

    def get(url: str, _max_bytes: int) -> bytes:
        urls.append(url)
        return body

    path = download_bts(tmp_path, get=get, now=lambda: NOW)
    assert urls == [BTS_URL] and "subsample=sn" in BTS_URL and "format=csv" in BTS_URL
    record = json.loads((tmp_path / f"{BTS}.json").read_text())
    assert (
        record["rows"] == 2 and "Fremling" in record["citation"] and "Perley" in record["citation"]
    )
    assert "No terms of use" in record["terms"]
    rows = read_bts(path)
    assert [r.oid for r in rows] == ["ZTF18aaaaaaa", "ZTF19bbbbbbb"]
    assert rows[0].ra == pytest.approx(193.3, abs=1e-4)
    assert rows[0].dec == pytest.approx(-6.385, abs=1e-4)
    assert rows[0].cls == "SN Ia" and rows[0].name == "SN2019x0"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"ZTFID,IAUID\nZTF18aaaaaaa,x\n", "missing columns"),
        (b"ZTFID,IAUID,RA,Dec,type\nnot-an-oid,x,1:2:3,4:5:6,SN Ia\n", "invalid ZTFID"),
        (b"\xff\xfe", "not UTF-8"),
    ],
    ids=["columns", "oid", "utf8"],
)
def test_bts_refuses_malformed_tables(tmp_path: Path, body: bytes, message: str) -> None:
    with pytest.raises(DataError, match=message):
        download_bts(tmp_path, get=lambda _u, _n: body)
    assert not (tmp_path / f"{BTS}.csv").exists()


def test_bts_tolerates_unparseable_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "bts.csv"
    path.write_bytes(b"ZTFID,IAUID,RA,Dec,type\nZTF18aaaaaaa,x,12:xx:00,1:2,SN Ia\n")
    (row,) = read_bts(path)
    assert row.ra is None and row.dec is None


def test_load_source_refuses_a_changed_or_missing_file(tmp_path: Path) -> None:
    zenodo(tmp_path, archive({ZENODO_MEMBER: CSV}))
    (tmp_path / f"{ZENODO}.csv").write_bytes(CSV + b"ZTF18ccccccc,AGN,1,1,x,3\n")
    with pytest.raises(DataError, match="SHA-256 differs"):
        load_source(tmp_path, ZENODO)
    (tmp_path / f"{ZENODO}.csv").unlink()
    with pytest.raises(DataError, match="missing"):
        load_source(tmp_path, ZENODO)
    with pytest.raises(DataError, match="cannot read"):
        load_source(tmp_path, BTS)
    (tmp_path / f"{BTS}.json").write_text('{"name": "other"}')
    with pytest.raises(DataError, match="not the record"):
        load_source(tmp_path, BTS)


def test_read_zenodo_refuses_bad_rows(tmp_path: Path) -> None:
    path = tmp_path / "z.csv"
    path.write_bytes(b"oid,classALeRCE,ra,dec\nZTF18aaaaaaa,,1,2\n")
    with pytest.raises(DataError, match="invalid oid or class"):
        read_zenodo(path)
    path.write_bytes(b"oid,ra\nZTF18aaaaaaa,1\n")
    with pytest.raises(DataError, match="missing columns"):
        read_zenodo(path)
    with pytest.raises(DataError, match="cannot read"):
        read_zenodo(tmp_path / "absent.csv")
