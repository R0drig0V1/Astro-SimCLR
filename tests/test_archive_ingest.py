"""Tests of ``ingest-archive``: filters, exclusions, the night shards, resume and the CLI.

Every tarball is synthetic (``_archive_support``); nothing is downloaded. One night holds
an alert per rule of :data:`~astro_simclr.archive.ingest.REASONS`, so the counts, the kept
rows and their raw cutouts (NaN kept) are checked exactly.
"""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from _archive_support import SCHEMA_ERAS, Spec, archive_alert, night_tarball, write_tarball

from astro_simclr.alerts.parse import parse_alert
from astro_simclr.archive.ingest import (
    REASONS,
    Limits,
    Rules,
    ingest_archive,
    ingest_stream,
    ingest_tarball,
)
from astro_simclr.archive.layout import (
    CORPUS_NAME,
    META_NAME,
    NightWriter,
    corpus_index,
    excluded_rows,
    list_nights,
    load_night,
    night_dir,
    verify_night,
)
from astro_simclr.archive.selection import Exclusions, Selection, load_exclusions
from astro_simclr.archive.stream import TarballError, TarballStream
from astro_simclr.cli import main
from astro_simclr.data.schema import DataError, SchemaError

NIGHT = 20200115
EXCLUDED = "ZTF20exclude"
OPEN = Rules(Selection(), Exclusions.none())


def _specs() -> list[Spec]:
    """One alert per rule; the kept ones are candids 1, 2 and 3."""
    return [
        Spec("ZTF20aaaaaaa", 1, seed=1, nans=7),  # kept, with NaN pixels
        Spec("ZTF20aaaaaab", 2, seed=2, candidate={"ndethist": 5, "rb": 0.2}),  # kept: no cuts
        Spec("ZTF20aaaaaac", 3, seed=3, previous=[None, None]),  # kept: upper limits only
        Spec("ZTF19aaaaaad", 4),  # not first: named in 2019
        Spec("ZTF20aaaaaae", 5, previous=[None, 99]),  # not first: an earlier detection
        Spec("ZTF20aaaaaaf", 6, shape=(40, 63)),  # CCD edge
        Spec("ZTF20aaaaaag", 7, candidate={"fid": 7}),  # invalid values
        Spec("ZTF20aaaaaah", 8, candidate={"isdiffpos": "x"}),  # invalid values
        Spec("ZTF20aaaaaai", 9, candidate={"ra": 360.0}),  # invalid values
        Spec(EXCLUDED, 10),  # excluded
        Spec("ZTF20aaaaaaa", 11),  # duplicate oid in the night
    ]


def _tarball(directory: Path, night: int = NIGHT, specs: list[Spec] | None = None) -> Path:
    specs = _specs() if specs is None else specs
    members: list[tuple[str, bytes | str]] = [(f"{s.candid}.avro", archive_alert(s)) for s in specs]
    members += [
        ("dup/1.avro", archive_alert(specs[0])),  # duplicate candid
        ("bad.avro", b"Obj\x01garbage"),  # malformed
        ("listing.txt", b"x"),
        ("lnk.avro", "symlink"),
    ]
    return write_tarball(directory / f"ztf_public_{night}.tar.gz", members)


def _exclusions(tmp_path: Path) -> Exclusions:
    file = tmp_path / "exclude.txt"
    file.write_text(f"{EXCLUDED}\nZTF18zzzzzzz\n")
    return load_exclusions(file)


def test_one_night_end_to_end(tmp_path: Path) -> None:
    tarball = _tarball(tmp_path / "in")
    out = tmp_path / "corpus"
    rules = Rules(Selection(), _exclusions(tmp_path), rows_per_file=2)
    [result] = ingest_archive([tarball], out, rules)
    assert result.status == "ingested" and result.kept == 3
    assert result.counts == dict.fromkeys(REASONS, 0) | {
        "kept": 3,
        "malformed": 1,
        "not_63x63": 1,
        "invalid_values": 3,
        "not_first:previous": 1,
        "not_first:oid_year": 1,
        "excluded": 1,
        "duplicate_candid": 1,
        "duplicate_oid": 1,
    }
    night = load_night(night_dir(out, NIGHT))
    verify_night(night.path)
    assert list(night.index["candid"]) == [1, 2, 3]
    assert list(night.index["oid"]) == ["ZTF20aaaaaaa", "ZTF20aaaaaab", "ZTF20aaaaaac"]
    assert set(night.index["night"]) == {NIGHT} and night.index["isdiffpos"].all()
    assert list(night.index["member"]) == [0, 1, 2]
    assert list(night.index["ndethist"]) == [1, 5, 1]
    np.testing.assert_allclose(night.index["rb"], [0.9, 0.2, 0.9], rtol=1e-6)
    assert [len(c) for c in night.cutouts] == [2, 1]  # rows_per_file=2
    for row, spec in enumerate(_specs()[:3]):
        expected = parse_alert(archive_alert(spec)).cutouts
        np.testing.assert_array_equal(night.cutout(row), expected)  # NaN equal to NaN
    assert int(np.isnan(night.cutout(0)).sum()) == 7
    with pytest.raises(IndexError):
        night.cutout(3)
    source = night.meta["source"]
    assert source["tarball"] == tarball.name and source["avro"] == 13
    assert source["compressed_bytes"] == tarball.stat().st_size and len(source["sha256"]) == 64
    assert night.meta["exclusions"]["count"] == 2
    corpus = json.loads((out / CORPUS_NAME).read_text())
    assert corpus["kind"] == "archive-corpus" and corpus["selection"]["candidate_cuts"] == []
    assert excluded_rows(night.index, frozenset({"ZTF20aaaaaab"})).tolist() == [False, True, False]
    assert not excluded_rows(night.index, frozenset()).any()


def test_optional_rules(tmp_path: Path) -> None:
    rules = Rules(Selection(oid_year=False, max_ndethist=1), Exclusions.none())
    [result] = ingest_archive([_tarball(tmp_path)], tmp_path / "c", rules)
    # candid 2 fails ndethist <= 1; the 2019 oid and the excluded object are kept
    assert result.counts["not_first:ndethist"] == 1 and result.counts["not_first:oid_year"] == 0
    assert result.kept == 4 and result.counts["excluded"] == 0


@pytest.mark.parametrize("era", sorted(SCHEMA_ERAS))
def test_every_schema_era_is_read(tmp_path: Path, era: int) -> None:
    specs = [Spec(f"ZTF20aaaaaa{c}", i + 1, seed=i, era=era) for i, c in enumerate("abc")]
    [result] = ingest_archive([night_tarball(tmp_path, NIGHT, specs)], tmp_path / "c", OPEN)
    assert result.kept == 3
    drb = load_night(night_dir(tmp_path / "c", NIGHT)).index["drb"]
    assert np.isnan(drb).all() if era == 2018 else (drb > 0).all()


def test_a_piped_night(tmp_path: Path) -> None:
    tarball, out = _tarball(tmp_path), tmp_path / "c"
    with tarball.open("rb") as handle:
        result = ingest_stream(handle, NIGHT, out, OPEN)
    assert result.status == "ingested" and result.kept == 4
    meta = load_night(night_dir(out, NIGHT)).meta
    assert meta["source"]["tarball"] == tarball.name
    assert meta["source"]["compressed_bytes"] == tarball.stat().st_size

    class Unread(io.RawIOBase):
        def read(self, size: int = -1) -> bytes:
            raise AssertionError("a done night must not be read")

    again = ingest_stream(Unread(), NIGHT, out, OPEN)  # type: ignore[arg-type]
    assert again.status == "done" and again.kept == 4
    with pytest.raises(TarballError, match="not a valid date"):
        ingest_stream(Unread(), 20200231, out, OPEN)  # type: ignore[arg-type]


def test_resume_skips_complete_nights_without_reading_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = _tarball(tmp_path, 20200115), _tarball(tmp_path, 20200116)
    out = tmp_path / "c"
    ingest_archive([first], out, OPEN)
    meta_before = (night_dir(out, 20200115) / META_NAME).read_bytes()

    opened: list[str] = []

    def spy(source: object, *, name: str, **limits: int) -> TarballStream:
        opened.append(name)
        return TarballStream(source, name=name, **limits)  # type: ignore[arg-type]

    monkeypatch.setattr("astro_simclr.archive.ingest.TarballStream", spy)
    results = ingest_archive([first, second], out, OPEN)
    assert [(r.night, r.status) for r in results] == [(20200115, "done"), (20200116, "ingested")]
    assert results[0].kept == 4 and opened == [second.name]
    assert (night_dir(out, 20200115) / META_NAME).read_bytes() == meta_before
    assert list_nights(out) == [20200115, 20200116]


def test_a_night_cut_short_is_redone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tarball, out = _tarball(tmp_path), tmp_path / "c"
    rules = Rules(Selection(), Exclusions.none(), rows_per_file=1)
    calls = {"n": 0}

    def dying_parse(data: bytes, **kw: object) -> object:
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return parse_alert(data, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr("astro_simclr.archive.ingest.parse_alert", dying_parse)
    with pytest.raises(KeyboardInterrupt):
        ingest_archive([tarball], out, rules)
    partial = out / "nights" / f".{NIGHT}.partial"
    assert partial.is_dir() and list(partial.glob("cutouts-*.npy"))  # files written, no meta
    assert list_nights(out) == [] and not night_dir(out, NIGHT).exists()
    monkeypatch.setattr("astro_simclr.archive.ingest.parse_alert", parse_alert)
    [result] = ingest_archive([tarball], out, rules)
    assert result.status == "ingested" and result.kept == 4 and not partial.exists()


def test_a_damaged_tarball_commits_nothing(tmp_path: Path) -> None:
    tarball = _tarball(tmp_path)
    tarball.write_bytes(tarball.read_bytes()[:-200])
    out = tmp_path / "c"
    with pytest.raises(TarballError):
        ingest_archive([tarball], out, OPEN)
    assert list_nights(out) == []


def test_size_limits_reach_the_ingest(tmp_path: Path) -> None:
    out = tmp_path / "c"
    small = Rules(Selection(), Exclusions.none(), limits=Limits(max_member_bytes=100))
    [result] = ingest_archive([_tarball(tmp_path)], out, small)
    meta = load_night(night_dir(out, NIGHT)).meta
    # every alert is over 100 bytes; only the 15-byte malformed member is read
    assert result.kept == 0 and meta["n"] == 0 and meta["source"]["skipped_oversize"] == 12
    assert result.counts["malformed"] == 1
    bomb = Rules(Selection(), Exclusions.none(), limits=Limits(max_decompressed_bytes=1000))
    with pytest.raises(TarballError, match="decompresses"):
        ingest_tarball(_tarball(tmp_path, 20200116), out, bomb)
    with pytest.raises(ValueError, match="rows_per_file"):
        Rules(Selection(), Exclusions.none(), rows_per_file=0)


def test_another_file_for_a_done_night_is_refused_unless_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "c"
    ingest_archive([_tarball(tmp_path / "a")], out, OPEN)
    other = night_tarball(tmp_path / "b", NIGHT, [Spec("ZTF20zzzzzzz", 50)])
    with pytest.raises(DataError, match="already ingested from"):
        ingest_archive([other], out, OPEN)
    overwrite = Rules(Selection(), Exclusions.none(), overwrite=True)
    [result] = ingest_archive([other], out, overwrite)
    assert result.status == "ingested" and result.kept == 1
    assert list(load_night(night_dir(out, NIGHT)).index["candid"]) == [50]


def test_a_corpus_keeps_its_rules(tmp_path: Path) -> None:
    out, tarball = tmp_path / "c", _tarball(tmp_path)
    ingest_archive([tarball], out, Rules(Selection(), _exclusions(tmp_path)))
    with pytest.raises(DataError, match="other exclusions"):
        ingest_archive([tarball], out, OPEN)
    with pytest.raises(DataError, match="other selection"):
        ingest_archive([tarball], out, Rules(Selection(max_ndethist=3), _exclusions(tmp_path)))
    with pytest.raises(DataError, match="several tarballs"):
        ingest_archive([tarball, tarball], out, OPEN)
    (out / CORPUS_NAME).write_text("[]")
    with pytest.raises(DataError, match="not a JSON object"):
        ingest_archive([tarball], out, OPEN)


def test_the_writer_is_deterministic(tmp_path: Path) -> None:
    tarball = _tarball(tmp_path)
    for out in (tmp_path / "x", tmp_path / "y"):
        ingest_archive([tarball], out, OPEN)
    x, y = night_dir(tmp_path / "x", NIGHT), night_dir(tmp_path / "y", NIGHT)
    assert sorted(p.name for p in x.iterdir()) == sorted(p.name for p in y.iterdir())
    for file in x.iterdir():
        assert file.read_bytes() == (y / file.name).read_bytes()


def test_the_corpus_index_keeps_one_alert_per_object(tmp_path: Path) -> None:
    out = tmp_path / "c"
    assert len(corpus_index(out)) == 0
    early = [Spec("ZTF20aaaaaaa", 1), Spec("ZTF20aaaaaab", 2)]
    late = [
        Spec("ZTF20aaaaaab", 3, candidate={"jd": 2458900.5}),  # again, 37 days later
        Spec("ZTF20aaaaaac", 4, candidate={"jd": 2458900.5}),
    ]
    tarballs = [night_tarball(tmp_path, 20200301, late), night_tarball(tmp_path, 20200115, early)]
    ingest_archive(tarballs, out, OPEN)
    index = corpus_index(out)
    assert [(int(r["night"]), int(r["row"]), str(r["oid"])) for r in index] == [
        (20200115, 0, "ZTF20aaaaaaa"),
        (20200115, 1, "ZTF20aaaaaab"),
        (20200301, 1, "ZTF20aaaaaac"),
    ]
    trimmed = corpus_index(out, exclude=frozenset({"ZTF20aaaaaab"}))
    assert list(trimmed["candid"]) == [1, 4]


# layout checks ------------------------------------------------------------------------


@pytest.fixture
def night_path(tmp_path: Path) -> Path:
    out = tmp_path / "c"
    ingest_archive(
        [_tarball(tmp_path)], out, Rules(Selection(), Exclusions.none(), rows_per_file=2)
    )
    return night_dir(out, NIGHT)


def _edit_meta(path: Path, **changes: object) -> None:
    meta = json.loads((path / META_NAME).read_text())
    meta.update(changes)
    (path / META_NAME).write_text(json.dumps(meta))


def test_verify_detects_a_modified_file(night_path: Path) -> None:
    file = night_path / "cutouts-00001.npy"
    data = bytearray(file.read_bytes())
    data[-1] ^= 1
    file.write_bytes(bytes(data))
    with pytest.raises(DataError, match="SHA-256 differs"):
        verify_night(night_path)
    file.unlink()
    with pytest.raises(DataError, match="missing"):
        verify_night(night_path)


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        ({"kind": "raw-alerts"}, DataError, "not an archive-night"),
        ({"night": 20200116}, DataError, "is not 20200115"),
        ({"n": 99}, SchemaError, "meta n=99"),
        ({"files": {}}, SchemaError, "lists no index.npy"),
    ],
)
def test_load_refuses_a_meta_that_disagrees(
    night_path: Path, change: dict[str, object], error: type[Exception], message: str
) -> None:
    _edit_meta(night_path, **change)
    with pytest.raises(error, match=message):
        load_night(night_path)


def test_load_refuses_files_that_disagree(night_path: Path) -> None:
    meta = json.loads((night_path / META_NAME).read_text())
    files = meta["files"]
    del files["cutouts-00000.npy"]
    _edit_meta(night_path, files=files)
    with pytest.raises(SchemaError, match="must list the cutout files"):
        load_night(night_path)
    shutil.copy(night_path / "cutouts-00001.npy", night_path / "cutouts-00000.npy")
    files["cutouts-00000.npy"] = dict(files["cutouts-00001.npy"])
    _edit_meta(night_path, files=files)
    load_night(night_path)  # consistent again (two 2-row files)
    np.save(night_path / "cutouts-00000.npy", np.zeros((1, 63, 63, 3), dtype=np.float32))
    with pytest.raises(SchemaError, match=r"cutouts-00000.npy is float32 \[1, 63, 63, 3\]"):
        load_night(night_path)
    files["cutouts-00000.npy"]["shape"] = [1, 63, 63, 3]
    _edit_meta(night_path, files=files)
    with pytest.raises(SchemaError, match="4 index rows, 3 cutout rows"):
        load_night(night_path)
    np.save(night_path / "cutouts-00000.npy", np.zeros((2, 3), dtype=np.float32))
    files["cutouts-00000.npy"] = {"dtype": "float32", "shape": [2, 3], "sha256": ""}
    _edit_meta(night_path, files=files)
    with pytest.raises(SchemaError, match="must be float32"):
        load_night(night_path)
    (night_path / "index.npy").write_bytes(b"\x93NUMPY junk")
    with pytest.raises(SchemaError, match=r"cannot load index\.npy"):
        load_night(night_path)


def test_load_checks_the_index_values(tmp_path: Path) -> None:
    good = (
        "ZTF20aaaaaaa", 1, 2458850.5, 1, 10.0, 0.0, NIGHT, 1, True, 0.9, 0.9, 0,
    )  # fmt: skip
    stamp = np.zeros((63, 63, 3), dtype=np.float32)
    for bad, message in (
        ({1: 1, 6: NIGHT + 1}, "a night other than"),
        ({1: 0}, "a candid <= 0"),
        ({3: 4}, "a fid not in"),
        ({4: 400.0}, "an ra outside"),
        ({5: -91.0}, "a dec outside"),
        ({2: float("nan")}, "a jd"),
        ({7: -1}, "an ndethist < 0"),
        ({0: ""}, "an empty oid"),
        ({}, "duplicated"),
    ):
        writer = NightWriter(tmp_path / "c", NIGHT)
        row = list(good)
        for key, value in bad.items():
            row[key] = value
        writer.add(tuple(good), stamp)
        writer.add(tuple(row), stamp)
        path = writer.commit({})
        with pytest.raises(SchemaError, match=message):
            load_night(path)
    with pytest.raises(ValueError, match="must be float32"):
        NightWriter(tmp_path / "c", NIGHT).add(good, stamp.astype(np.float64))
    with pytest.raises(ValueError, match="rows_per_file"):
        NightWriter(tmp_path / "c", NIGHT, rows_per_file=0)
    with pytest.raises(DataError, match="missing"):
        load_night(tmp_path / "c" / "nights" / "20990101")
    assert list_nights(tmp_path / "nowhere") == []


# CLI ----------------------------------------------------------------------------------


def test_cli_ingest_summary_and_verify(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exclude = tmp_path / "exclude.txt"
    exclude.write_text(f"{EXCLUDED}\n")
    tarballs = [str(_tarball(tmp_path, n)) for n in (20200115, 20200116)]
    out = tmp_path / "c"
    args = ["ingest-archive", *tarballs, "--out", str(out), "--exclude", str(exclude)]
    assert main(args) == 0
    captured = capsys.readouterr()
    assert "2 nights (2 ingested, 0 already done), 6 stamps kept" in captured.out
    assert "20200115 ingested: kept 3 (duplicate_candid 1" in captured.err
    assert main([*args, "--quiet"]) == 0
    captured = capsys.readouterr()
    assert "(0 ingested, 2 already done)" in captured.out and captured.err == ""

    assert main(["archive-summary", str(out), "--verify"]) == 0
    text = capsys.readouterr().out
    assert "2 nights, 3 objects (one alert each), every file verified" in text
    assert "excluded" in text
    assert main(["archive-summary", str(out), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["totals"]["kept"] == 6 and payload["nights"][0]["night"] == 20200115
    assert payload["objects"] == 3

    (night_dir(out, 20200116) / "index.npy").write_bytes(b"x")
    assert main(["archive-summary", str(out), "--verify"]) == 1
    assert "cannot load index.npy" in capsys.readouterr().err


def test_cli_ingest_usage_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tarball = str(_tarball(tmp_path))
    out = str(tmp_path / "c")
    with pytest.raises(SystemExit) as exit_info:
        main(["ingest-archive", tarball, "--out", out])  # neither --exclude nor --no-exclusions
    assert exit_info.value.code == 2
    base = ["ingest-archive", tarball, "--out", out, "--no-exclusions"]
    assert main([*base, "--rows-per-file", "0"]) == 2
    assert main([*base, "--max-ndethist", "-1"]) == 2
    capsys.readouterr()
    assert main(["ingest-archive", "-", tarball, "--out", out, "--no-exclusions"]) == 2
    assert main([*base, "--night", str(NIGHT)]) == 2  # --night only with '-'
    assert main(["ingest-archive", "-", "--out", out, "--no-exclusions"]) == 2  # no --night
    capsys.readouterr()
    assert main([*base, "--no-oid-year", "--max-ndethist", "1"]) == 0
    assert (
        main(["ingest-archive", str(tmp_path / "x.tar.gz"), "--out", out, "--no-exclusions"]) == 1
    )
    assert "not an archive tarball name" in capsys.readouterr().err


def test_cli_reads_a_piped_night(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tarball = _tarball(tmp_path)
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(tarball.read_bytes())))
    args = ["ingest-archive", "-", "--night", str(NIGHT), "--out", str(tmp_path / "c")]
    assert main([*args, "--no-exclusions"]) == 0
    captured = capsys.readouterr()
    assert "1 nights (1 ingested, 0 already done), 4 stamps kept" in captured.out
    assert f"{NIGHT} ingested: kept 4" in captured.err


# review fixes: extreme dates, the lock, the replace window, missing files ------------


def test_extreme_dates_are_invalid_values(tmp_path: Path) -> None:
    specs = [
        Spec("ZTF20aaaaaaa", 1, candidate={"jd": 1.0}),
        Spec("ZTF20aaaaaab", 2, candidate={"jd": 1e300}),
        Spec("ZTF20aaaaaac", 3, candidate={"jd": 2470000.5}),
        Spec("ZTF20aaaaaad", 4),
    ]
    [result] = ingest_archive([night_tarball(tmp_path, NIGHT, specs)], tmp_path / "c", OPEN)
    assert result.counts["invalid_values"] == 3 and result.kept == 1


def test_one_ingest_at_a_time(tmp_path: Path) -> None:
    from astro_simclr.archive.layout import corpus_lock

    out = tmp_path / "c"
    with corpus_lock(out), pytest.raises(DataError, match="another ingest"):
        ingest_archive([_tarball(tmp_path)], out, OPEN)
    with tmp_path.joinpath("t.tar.gz").open("wb") as _, corpus_lock(out):
        pass  # released: the lock can be taken again
    ingest_archive([_tarball(tmp_path)], out, OPEN)
    assert not list(out.glob("*.tmp"))  # corpus.json written through a temporary file


def test_a_night_set_aside_by_a_crash_is_put_back(tmp_path: Path) -> None:
    out, tarball = tmp_path / "c", _tarball(tmp_path)
    ingest_archive([tarball], out, OPEN)
    final = night_dir(out, NIGHT)
    aside = final.with_name(f".{final.name}.replaced")
    final.rename(aside)  # the state between the two renames of an overwrite
    [result] = ingest_archive([tarball], out, OPEN)
    assert result.status == "done" and final.is_dir() and not aside.exists()
    overwrite = Rules(Selection(), Exclusions.none(), overwrite=True)
    [again] = ingest_archive([tarball], out, overwrite)
    assert again.status == "ingested" and not aside.exists()
    verify_night(final)


def test_a_missing_tarball_is_a_data_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "ztf_public_20200101.tar.gz"
    with pytest.raises(DataError, match="no such file"):
        ingest_archive([missing], tmp_path / "c", OPEN)
    assert (
        main(["ingest-archive", str(missing), "--out", str(tmp_path / "c"), "--no-exclusions"]) == 1
    )
    assert "no such file" in capsys.readouterr().err
