"""An offline world for the R5 tests: tiny source CSVs, a CD21 store, detection JSON, avro.

Everything is made here: the Zenodo archive is a small ``.tar.gz`` built in memory, the BTS
table a few CSV lines, task 3.1's store of CD21 three split directories with index lines and
one shard of raw responses each, and every alert an avro written by ``_alerts_support``.
:class:`FakeService` answers the by-oid requests from a URL -> body map (404 otherwise), so
the suite never touches the network.

The world (:func:`make_r5_world`), per class, with the case each object covers:

- AGN: 12 good Zenodo objects (6 ``AGN``, 6 ``QSO``), one of them also a CD21 Train AGN
  (pinned to R5 Train), and one more within 1.5" of a CD21 Train row *and* a CD21 Test
  ghost (dropped: ``cd21_train_and_eval``); a ``Blazar`` with every stamp after ``JD_MAX``
  (``no_stamp_before_jd_max``); a repeated Zenodo row (``duplicate_oid``); an AGN that BTS
  calls a SN (``label_conflict``);
- VS: 12 good (6 ``E``, 6 ``RRL``), one of them a CD21 Test VS (kept out of R5 Train); an
  ``RRL`` whose drawn alert is a 404 (``avro_unavailable``); an ``LPV`` whose alert is cut
  at a CCD edge (``not_63x63``); a ``CV/Nova`` (``not_an_r5_subclass``);
- SN: 6 Zenodo ``SNIa`` and 6 BTS SN; a BTS repeat of a Zenodo SN (``duplicate_oid``); a
  BTS ``TDE`` (``not_an_r5_subclass``);
- Asteroid: 12 CD21 rows with one detection (8 Train, 2 Validation, 2 Test); one with two
  (``asteroid_ndet_not_1``);
- Bogus: 12 CD21 rows (8 Train, 2 Validation, 2 Test) with 1-3 detections, whose drawn
  alert is downloaded by the build when it is not CD21's own; a ghost row
  (``cd21_ghost``); an unresolved row (``cd21_unresolved``); one with 12 detections
  (``bogus_ndet_ge_10``); one whose oid is a Zenodo AGN (``crossmatch_known_object``).
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import numpy as np
from _alerts_support import avro_bytes, candidate_record, raw_cutouts

from astro_simclr.alerts.alerce import AlerceClient, avro_url, detections_url
from astro_simclr.alerts.snapshot import Recorder, Response
from astro_simclr.alerts.store import IndexRow, IndexWriter, Shard, write_meta, write_shard
from astro_simclr.alerts.transport import HttpError
from astro_simclr.data.schema import LABEL_NAMES, SPLITS
from astro_simclr.r5.sources import ZENODO_MEMBER, download_bts, download_zenodo

#: MJDs of the detections: before JD_MAX (2022-12-31), and one after it.
MJD_BEFORE = (58500.25, 58600.5, 58700.75)
MJD_AFTER = 60000.0
_OIDS = iter(
    f"ZTF18{a}{b}{c}aaaa" for a in "abcdefghij" for b in "abcdefghij" for c in "abcdefghij"
)
NOW = "2026-11-08T12:00:00+00:00"


def new_oid() -> str:
    """A fresh valid ZTF object id."""
    return next(_OIDS)


@dataclass
class Obj:
    """One object of the world: position, detections (candid, mjd, has_stamp) and alerts."""

    oid: str
    ra: float
    dec: float
    detections: list[tuple[int, float, bool]]
    shapes: dict[int, tuple[int, int]] = field(default_factory=dict)
    missing_avro: set[int] = field(default_factory=set)


@dataclass
class FakeService:
    """ALeRCE by URL: bodies from a map, HTTP 404 for anything else; counts the calls."""

    bodies: dict[str, bytes] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    def fetch(self, url: str, max_bytes: int) -> tuple[bytes, str]:
        self.calls.append(url)
        if url not in self.bodies:
            raise HttpError(url, 404, "HTTP 404")
        return self.bodies[url], NOW

    def client(self) -> AlerceClient:
        return AlerceClient(Recorder(self.fetch, requests=lambda: len(self.calls)))


@dataclass
class R5World:
    """The world's directories, its service, and the expected outcome per oid."""

    sources: Path
    cd21: Path
    service: FakeService
    expect: dict[str, str]
    objects: dict[str, Obj]
    pinned_train: str
    kept_from_train: str
    dropped_cross: str


def detections_json(obj: Obj) -> bytes:
    """The detection list of ``obj`` as ALeRCE serves it (``mjd``, string ``candid``)."""
    items = [
        {
            "candid": str(c),
            "mjd": mjd,
            "has_stamp": stamp,
            "fid": 1 + i % 2,
            "ra": obj.ra,
            "dec": obj.dec,
            "magpsf": 18.0,
            "sigmapsf": 0.1,
        }
        for i, (c, mjd, stamp) in enumerate(obj.detections)
    ]
    return json.dumps(items).encode()


def alert_of(obj: Obj, candid: int, mjd: float) -> bytes:
    shape = obj.shapes.get(candid, (63, 63))
    record = candidate_record(
        candid, jd=mjd + 2400000.5, ra=obj.ra, dec=obj.dec, fid=1, xpos=30.0, ypos=1000.0
    )
    return avro_bytes(obj.oid, record, raw_cutouts(candid % 1000, shape=shape))


class _Candids:
    def __init__(self) -> None:
        self.next = 1000

    def __call__(self) -> int:
        self.next += 1
        return self.next


def _make_obj(
    candid: _Candids,
    index: int,
    *,
    before: int = 3,
    after: bool = True,
    prv: bool = True,
    oid: str | None = None,
) -> Obj:
    dets: list[tuple[int, float, bool]] = []
    if prv:
        dets.append((candid(), 58400.0, False))  # a previous candidate, no stamp
    dets += [(candid(), MJD_BEFORE[k % 3] + k, True) for k in range(before)]
    if after:
        dets.append((candid(), MJD_AFTER, True))
    return Obj(oid or new_oid(), 10.0 + 0.02 * index, 20.0 + 0.01 * (index % 7), dets)


def _serve(service: FakeService, obj: Obj) -> None:
    service.bodies[detections_url(obj.oid)] = detections_json(obj)
    for c, mjd, stamp in obj.detections:
        if stamp and c not in obj.missing_avro:
            service.bodies[avro_url(obj.oid, c)] = alert_of(obj, c, mjd)


def _zenodo_archive(rows: list[tuple[str, str, float, float]]) -> bytes:
    text = "oid,classALeRCE,ra,dec,source,id_source\n" + "".join(
        f"{o},{c},{ra!r},{dec!r},test,x{i}\n" for i, (o, c, ra, dec) in enumerate(rows)
    )
    data = text.encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        readme = tarfile.TarInfo("files/README.txt")
        readme.size = 5
        tar.addfile(readme, io.BytesIO(b"hello"))
        info = tarfile.TarInfo(f"files_for_lc_classifier/{ZENODO_MEMBER}")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def bts_csv(rows: list[tuple[str, str, float, float]]) -> bytes:
    """The BTS explorer's CSV: a blank first line, sexagesimal RA/Dec."""

    def sexa(value: float, scale: float) -> str:
        sign = "-" if value < 0 else ""
        v = abs(value) / scale
        h = int(v)
        m = int((v - h) * 60)
        s = (v - h - m / 60) * 3600
        return f"{sign}{h:02d}:{m:02d}:{s:05.2f}"

    lines = ["", "ZTFID,IAUID,RA,Dec,peakt,peakfilt,peakmag,type"]
    for i, (oid, kind, ra, dec) in enumerate(rows):
        lines.append(f"{oid},SN2019x{i},{sexa(ra, 15.0)},{sexa(dec, 1.0)},600.0,g,18.0,{kind}")
    return ("\r\n".join(lines) + "\r\n").encode()


@contextmanager
def stream_of(data: bytes) -> Iterator[IO[bytes]]:
    yield io.BytesIO(data)


def write_sources(dest: Path, zenodo: bytes, bts: bytes) -> None:
    """Snapshot both sources through the real download functions (fake transports)."""
    download_zenodo(
        dest,
        stream=lambda _url: stream_of(zenodo),
        now=lambda: NOW,
        expected_size=len(zenodo),
        expected_md5=hashlib.md5(zenodo, usedforsecurity=False).hexdigest(),
    )
    download_bts(dest, get=lambda _url, _n: bts, now=lambda: NOW)


@dataclass
class _Cd21Row:
    split: str
    label: str
    status: str
    obj: Obj | None
    ra: float
    dec: float


def write_cd21(dest: Path, rows: list[_Cd21Row]) -> None:
    """Task 3.1's store: per split, fetch.json, index.csv, one shard of raw responses.

    An ``ok`` row's shard holds its object's detection list and the avro of its own alert
    (the first stamped detection), as task 3.1 stores them.
    """
    for split in SPLITS:
        split_dir = dest / split
        split_dir.mkdir(parents=True)
        mine = [r for r in rows if r.split == split]
        write_meta(split_dir, {"dataset": "world", "split": split, "n_rows": len(mine)})
        responses: list[Response] = []
        lines: list[IndexRow] = []
        for i, r in enumerate(mine):
            line = IndexRow(i, LABEL_NAMES.index(r.label), r.status, r.ra, r.dec, 18.0, 0.1)
            if r.status == "ok" and r.obj is not None:
                own = next(c for c, _, stamp in r.obj.detections if stamp)
                own_mjd = next(m for c, m, _ in r.obj.detections if c == own)
                for url, body in (
                    (detections_url(r.obj.oid), detections_json(r.obj)),
                    (avro_url(r.obj.oid, own), alert_of(r.obj, own, own_mjd)),
                ):
                    sha = hashlib.sha256(body).hexdigest()
                    responses.append(
                        Response(url, "avro" if "avro" in url else "detections", NOW, sha, body)
                    )
                line = IndexRow(
                    i,
                    LABEL_NAMES.index(r.label),
                    "ok",
                    r.ra,
                    r.dec,
                    18.0,
                    0.1,
                    oid=r.obj.oid,
                    candid=own,
                    shard=0,
                )
            lines.append(line)
        write_shard(
            split_dir,
            0,
            Shard(
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty((0, 63, 63, 3), dtype=np.float32),
                [],
                responses,
            ),
        )
        writer = IndexWriter(split_dir)
        writer.append(lines)
        writer.close()


@dataclass
class _Builder:
    """The world under construction (:func:`make_r5_world`)."""

    candid: _Candids = field(default_factory=_Candids)
    service: FakeService = field(default_factory=FakeService)
    expect: dict[str, str] = field(default_factory=dict)
    objects: dict[str, Obj] = field(default_factory=dict)
    zenodo: list[tuple[str, str, float, float]] = field(default_factory=list)
    bts: list[tuple[str, str, float, float]] = field(default_factory=list)
    cd21: list[_Cd21Row] = field(default_factory=list)
    index: int = 0

    def obj(self, **kwargs: Any) -> Obj:
        self.index += 1
        made = _make_obj(self.candid, self.index, **kwargs)
        self.objects[made.oid] = made
        return made

    def z(self, o: Obj, cls: str, outcome: str = "accepted") -> None:
        self.zenodo.append((o.oid, cls, o.ra, o.dec))
        self.expect.setdefault(o.oid, outcome)
        _serve(self.service, o)

    def c21(
        self, o: Obj | None, split: str, label: str, status: str = "ok", outcome: str = "accepted"
    ) -> None:
        ra, dec = (o.ra, o.dec) if o is not None else (50.0 + 0.01 * len(self.cd21), -10.0)
        self.cd21.append(_Cd21Row(split, label, status, o, ra, dec))
        if o is not None:
            self.expect.setdefault(o.oid, outcome)


def _zenodo_and_bts(w: _Builder) -> tuple[str, str, Obj]:
    """AGN, VS and SN (module docstring); returns the pinned, kept-out and cross objects."""
    for k in range(12):
        w.z(w.obj(), "AGN" if k < 6 else "QSO")
    pinned = w.zenodo[0][0]
    w.z(w.obj(before=0, after=True), "Blazar", "no_stamp_before_jd_max")
    w.zenodo.append((w.zenodo[1][0], "AGN", 0.0, 0.0))  # a repeated row
    conflict = w.obj()
    w.z(conflict, "AGN", "label_conflict")
    w.bts.append((conflict.oid, "SN Ia", conflict.ra, conflict.dec))
    cross = w.obj()
    w.z(cross, "QSO", "cd21_train_and_eval")
    for k in range(12):
        w.z(w.obj(), "E" if k < 6 else "RRL")
    kept_out = w.zenodo[-1][0]
    no_avro = w.obj(before=1, after=False)
    no_avro.missing_avro = {no_avro.detections[1][0]}
    w.z(no_avro, "RRL", "avro_unavailable")
    edge = w.obj(before=1, after=False)
    edge.shapes = {edge.detections[1][0]: (63, 50)}
    w.z(edge, "LPV", "not_63x63")
    w.z(w.obj(), "CV/Nova", "not_an_r5_subclass")
    for _ in range(6):
        w.z(w.obj(), "SNIa")
    for k in range(6):
        o = w.obj()
        w.bts.append((o.oid, "SN II" if k % 2 else "SN Ia-91T", o.ra, o.dec))
        w.expect[o.oid] = "accepted"
        _serve(w.service, o)
    w.bts.append((w.zenodo[-1][0], "SN Ia", 0.0, 0.0))  # a repeat of a Zenodo SN
    tde = w.obj()
    w.bts.append((tde.oid, "TDE", tde.ra, tde.dec))
    w.expect[tde.oid] = "not_an_r5_subclass"
    return pinned, kept_out, cross


def _cd21_part(w: _Builder, pinned: str, kept_out: str, cross: Obj) -> None:
    """Asteroid, Bogus and the cross-benchmark cases (module docstring)."""
    plan = ["Train"] * 8 + ["Validation"] * 2 + ["Test"] * 2
    for split in plan:
        w.c21(w.obj(before=1, after=False, prv=False), split, "Asteroid")
    w.c21(
        w.obj(before=2, after=False, prv=False), "Train", "Asteroid", outcome="asteroid_ndet_not_1"
    )
    for k, split in enumerate(plan):
        o = w.obj(before=1 + k % 3, after=False)
        w.c21(o, split, "Bogus")
        _serve(w.service, o)
    w.c21(None, "Train", "Bogus", status="ghost_candidate")
    w.c21(None, "Validation", "Bogus", status="mismatch")
    w.c21(w.obj(before=12, after=False), "Train", "Bogus", outcome="bogus_ndet_ge_10")
    known = w.objects[w.zenodo[2][0]]
    w.cd21.append(_Cd21Row("Test", "Bogus", "ok", known, known.ra, known.dec))
    # The cross-benchmark cases: an R5 AGN that is a CD21 Train AGN, an R5 VS that is a CD21
    # Test VS, and an R5 QSO near both a CD21 Train AGN row and a CD21 Test ghost.
    pin, out = w.objects[pinned], w.objects[kept_out]
    w.cd21.append(_Cd21Row("Train", "AGN", "ok", pin, pin.ra, pin.dec))
    w.cd21.append(_Cd21Row("Test", "VS", "ok", out, out.ra, out.dec))
    w.cd21.append(_Cd21Row("Train", "AGN", "no_object", None, cross.ra + 0.5 / 3600, cross.dec))
    w.cd21.append(
        _Cd21Row("Test", "Bogus", "ghost_candidate", None, cross.ra, cross.dec + 0.5 / 3600)
    )


def make_r5_world(root: Path) -> R5World:
    """The world of the module docstring under ``root``."""
    w = _Builder()
    pinned, kept_out, cross = _zenodo_and_bts(w)
    _cd21_part(w, pinned, kept_out, cross)
    sources = root / "sources"
    write_sources(sources, _zenodo_archive(w.zenodo), bts_csv(w.bts))
    write_cd21(root / "cd21", w.cd21)
    return R5World(
        sources, root / "cd21", w.service, w.expect, w.objects, pinned, kept_out, cross.oid
    )


def gz(data: bytes) -> bytes:
    """gzip ``data`` (for a malformed-archive test)."""
    return gzip.compress(data, mtime=0)
