"""Tests of the by-oid fetch (task 3.5's extension of ``fetch-alerts``): statuses, resume,
snapshot, the items file and the store checks. Offline: a fake service answers from memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from _alerts_support import avro_bytes, candidate_record, raw_cutouts
from _r5_support import FakeService

from astro_simclr.alerts.alerce import AlerceClient, avro_url, detections_url
from astro_simclr.alerts.by_oid import (
    FetchItem,
    fetch_by_oid,
    final_statuses,
    read_items,
    read_oid_index,
)
from astro_simclr.alerts.snapshot import Recorder
from astro_simclr.alerts.store import Snapshot
from astro_simclr.alerts.transport import HttpError
from astro_simclr.data.schema import DataError

A, B, C = "ZTF18aaaaaaa", "ZTF18bbbbbbb", "ZTF18ccccccc"


class Flaky(FakeService):
    """Raises the listed errors for a URL (one per request) before serving it."""

    def __init__(self) -> None:
        super().__init__()
        self.errors: dict[str, list[HttpError]] = {}

    def fetch(self, url: str, max_bytes: int) -> tuple[bytes, str]:
        if self.errors.get(url):
            self.calls.append(url)
            raise self.errors[url].pop(0)
        return super().fetch(url, max_bytes)


def service() -> Flaky:
    s = Flaky()
    s.bodies[detections_url(A)] = b'[{"candid": "5", "mjd": 58500.0, "has_stamp": true}]'
    s.bodies[avro_url(A, 5)] = avro_bytes(A, candidate_record(5), raw_cutouts(5))
    s.bodies[avro_url(A, 6)] = avro_bytes(A, candidate_record(6), raw_cutouts(6, shape=(63, 40)))
    s.bodies[avro_url(B, 7)] = b"Obj\x01 not avro"
    s.bodies[detections_url(C)] = b"not json"
    return s


def client(s: FakeService) -> AlerceClient:
    return AlerceClient(Recorder(s.fetch, requests=lambda: len(s.calls)))


ITEMS = [
    FetchItem("detections", A),
    FetchItem("avro", A, 5),
    FetchItem("avro", A, 6),
    FetchItem("avro", B, 7),
    FetchItem("detections", B),
    FetchItem("detections", C),
]


def test_statuses_snapshot_and_resume(tmp_path: Path) -> None:
    s = service()
    s.errors[avro_url(B, 9)] = [HttpError("x", 400, "HTTP 400")]
    out = tmp_path / "fetched"
    run = fetch_by_oid(
        [*ITEMS, FetchItem("avro", B, 9), ITEMS[0]], out, client=client(s), batch_size=3
    )
    assert run.attempted == 7 and run.skipped == 0
    lines = {line.item: line for line in read_oid_index(out)}
    assert lines[ITEMS[0]].status == "ok" and lines[ITEMS[0]].n_detections == 1
    assert lines[ITEMS[1]].status == "ok" and lines[ITEMS[1]].raw_shape == "63x63"
    assert lines[ITEMS[2]].raw_shape == "63x40"
    assert lines[ITEMS[3]].status == "bad_alert"
    assert lines[ITEMS[4]].status == "not_found"
    assert lines[ITEMS[5]].status == "http_error"  # malformed JSON: transient
    assert lines[FetchItem("avro", B, 9)].status == "refused"
    assert all(line.shard is not None and line.requests == 1 for line in lines.values())
    snapshot = Snapshot(out)
    assert detections_url(A) in snapshot and avro_url(A, 5) in snapshot
    body, _, _ = snapshot.read_many([avro_url(A, 5)])[avro_url(A, 5)]
    assert body == s.bodies[avro_url(A, 5)]
    assert len(snapshot) == 5  # every response received, bad alert and bad JSON included
    finals = final_statuses(out)
    assert ITEMS[5] not in finals and finals[ITEMS[4]] == "not_found"
    s.bodies[detections_url(C)] = b"[]"
    again = fetch_by_oid(ITEMS, out, client=client(s), resume=True)
    assert (again.attempted, again.skipped) == (1, 5)
    assert final_statuses(out)[ITEMS[5]] == "ok"
    with pytest.raises(DataError, match="--resume"):
        fetch_by_oid(ITEMS, out, client=client(s))
    elsewhere = AlerceClient(Recorder(s.fetch), api_url="https://other.example/ztf/v1")
    with pytest.raises(DataError, match="api_url"):
        fetch_by_oid(ITEMS, out, client=elsewhere, resume=True)


def test_cooldown_after_consecutive_transient_failures(tmp_path: Path) -> None:
    s = service()
    items = [FetchItem("detections", f"ZTF18aaaaab{c}") for c in "abcde"]
    for item in items:
        s.errors[detections_url(item.oid)] = [HttpError("x", 503, "HTTP 503")]
    slept: list[float] = []
    run = fetch_by_oid(items, tmp_path / "o", client=client(s), cooldown=7.0, sleep=slept.append)
    assert run.counts["http_error"] == 5 and run.cooldowns == 1 and slept == [7.0]


def test_stop_commits_the_batch(tmp_path: Path) -> None:
    s = service()
    seen: list[str] = []
    calls = iter([False, True])
    run = fetch_by_oid(
        ITEMS,
        tmp_path / "o",
        client=client(s),
        stop=lambda: next(calls, True),
        progress=lambda i, n, line: seen.append(line.status),
    )
    assert run.attempted == 1 and seen == ["ok"]
    assert len(read_oid_index(tmp_path / "o")) == 1


def test_fetch_item_validation_and_order() -> None:
    cone: Any = "cone"
    with pytest.raises(ValueError, match="kind"):
        FetchItem(cone, A)
    with pytest.raises(ValueError, match="object id"):
        FetchItem("detections", "ZTF18")
    with pytest.raises(ValueError, match="needs a candid"):
        FetchItem("avro", A)
    with pytest.raises(ValueError, match="needs a candid"):
        FetchItem("detections", A, 5)
    with pytest.raises(ValueError, match="positive"):
        FetchItem("avro", A, 0)
    ordered = sorted(
        [FetchItem("avro", A, 2), FetchItem("detections", A), FetchItem("avro", A, 1)],
        key=FetchItem.sort_key,
    )
    assert [(i.kind, i.candid) for i in ordered] == [("avro", 1), ("avro", 2), ("detections", None)]


def test_read_items(tmp_path: Path) -> None:
    path = tmp_path / "items.txt"
    path.write_text(f"# objects\n{A}\n\n{A} 5\n{A}\n{B}  7\n")
    assert read_items(path) == [
        FetchItem("detections", A),
        FetchItem("avro", A, 5),
        FetchItem("avro", B, 7),
    ]
    path.write_text(f"{A} x\n")
    with pytest.raises(DataError, match=":1:"):
        read_items(path)
    path.write_text("ZTF1 5\n")
    with pytest.raises(DataError, match="object id"):
        read_items(path)
    with pytest.raises(DataError, match="cannot read"):
        read_items(tmp_path / "absent")


def test_store_checks(tmp_path: Path) -> None:
    s = service()
    with pytest.raises(ValueError, match="batch_size"):
        fetch_by_oid([], tmp_path / "o", client=client(s), batch_size=0)
    with pytest.raises(ValueError, match="cooldown"):
        fetch_by_oid([], tmp_path / "o", client=client(s), cooldown=-1)
    other = tmp_path / "other"
    other.mkdir()
    (other / "fetch.json").write_text('{"format": "astro-simclr-raw-alerts"}')
    with pytest.raises(DataError, match="not an astro-simclr-oid-fetch"):
        fetch_by_oid([], other, client=client(s), resume=True)
    (other / "fetch.json").write_text('{"format": "astro-simclr-oid-fetch", "format_version": 9}')
    with pytest.raises(DataError, match="format version"):
        fetch_by_oid([], other, client=client(s), resume=True)
    (other / "fetch.json").write_text("{")
    with pytest.raises(DataError, match="cannot read"):
        fetch_by_oid([], other, client=client(s), resume=True)


def test_index_tolerates_a_truncated_line_and_refuses_bad_ones(tmp_path: Path) -> None:
    s = service()
    out = tmp_path / "o"
    fetch_by_oid(ITEMS[:2], out, client=client(s))
    index = out / "index.csv"
    text = index.read_text()
    index.write_text(text + "detections,ZTF18ddddddd,,ok")  # an interrupted append
    assert len(read_oid_index(out)) == 2
    fetch_by_oid(ITEMS[:3], out, client=client(s), resume=True)  # the writer cuts the tail
    assert len(read_oid_index(out)) == 3
    index.write_text(text + "detections,ZTF18ddddddd,,weird,,,,,0,0.0,\n")
    with pytest.raises(DataError, match="unknown status"):
        read_oid_index(out)
    index.write_text("a,b\n1,2\n")
    with pytest.raises(DataError, match="unexpected columns"):
        read_oid_index(out)
