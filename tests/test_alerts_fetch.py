"""Tests of the fetch-alerts run (task 3.1): matching, ties, verification, resume, snapshot.

All offline: the fake ALeRCE of ``_alerts_support`` answers from memory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from _alerts_support import FakeAlerce, SkyObject, World, make_world

from astro_simclr.alerts.alerce import AlerceClient, Detection
from astro_simclr.alerts.fetch import (
    COOLDOWN_AFTER,
    MATCH_TOLERANCE,
    Target,
    alerce_normalise,
    fetch_alerts,
    interleaved_order,
    match_detections,
    verify_cutouts,
)
from astro_simclr.alerts.snapshot import Recorder
from astro_simclr.alerts.store import (
    Snapshot,
    duplicates,
    latest_rows,
    load_recovered,
    read_index,
    read_shard,
    shard_path,
    summarize,
)
from astro_simclr.alerts.transport import HttpError
from astro_simclr.data.prepared import write_prepared
from astro_simclr.data.schema import SPLITS, DataError, SplitData, StampData


@pytest.fixture
def world(tmp_path: Path) -> World:
    return make_world(tmp_path)


def client_for(alerce: FakeAlerce) -> AlerceClient:
    return AlerceClient(Recorder(alerce.fetch, requests=lambda: alerce.requests))


def statuses(split_dir: Path) -> dict[int, str]:
    return {row: line.status for row, line in latest_rows(read_index(split_dir)).items()}


def test_interleaved_order_round_robins_over_classes() -> None:
    assert interleaved_order([0, 0, 0, 1, 2, 2]) == [0, 3, 4, 1, 5, 2]
    assert interleaved_order([]) == []


def test_match_detections_counts_distinct_candids_within_the_tolerance() -> None:
    target = Target(0, 0, 1.0, 2.0, 19.0, 0.1)
    near = 19.0 + MATCH_TOLERANCE * 0.9
    detections = {
        "ZTF18aaaaaaa": [Detection(1, near, 0.1), Detection(2, 19.0 + 2 * MATCH_TOLERANCE, 0.1)],
        "ZTF18aaaaaab": [Detection(1, 19.0, 0.1), Detection(3, 19.0, 0.1 + 2e-4)],
    }
    found = match_detections(target, detections)
    assert [(oid, d.candid) for oid, d in found] == [("ZTF18aaaaaaa", 1)]
    detections["ZTF18aaaaaab"].append(Detection(4, 19.0, 0.1))
    assert [d.candid for _, d in match_detections(target, detections)] == [1, 4]


def test_alerce_normalise_is_minmax_with_nan_to_zero() -> None:
    raw = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
    raw[0, 0, 0] = np.nan
    raw[:, :, 2] = 7.0  # a constant channel: 0 / 0 -> 0
    out = alerce_normalise(raw)
    assert out.dtype == np.float64
    assert out[0, 0, 0] == 0.0 and out[..., 0].max() == 1.0
    np.testing.assert_allclose(out[..., 1].ravel(), [0, 1 / 3, 2 / 3, 1])
    assert (out[..., 2] == 0).all()
    assert verify_cutouts(raw, out.astype(np.float32)) <= 1e-7
    with pytest.raises(ValueError, match="differ in shape"):
        verify_cutouts(raw, out[:1])


def test_every_status_and_the_store(world: World, tmp_path: Path) -> None:
    out = tmp_path / "raw"
    run = fetch_alerts(world.prepared, "Validation", out, client=client_for(world.alerce))
    split_dir = out / "Validation"
    assert run.attempted == 10 and run.skipped == 0
    assert statuses(split_dir) == world.expected
    lines = latest_rows(read_index(split_dir))
    assert lines[1].candid == 102 and lines[1].n_matches == 2
    assert lines[1].candidates == "ZTF18aaaaaaa:101 ZTF18aaaaaaa:102"
    assert lines[2].n_objects == 2 and lines[2].oid == "ZTF18aaaaaac"
    assert lines[0].nan_pixels == 7 and lines[0].raw_shape == "63x63"
    assert lines[7].raw_shape == "63x50" and lines[7].nan_pixels == 0
    assert lines[5].max_abs_diff is not None and lines[5].max_abs_diff > 1e-3
    assert lines[5].candid == 500  # the closest candidate is recorded, not stored
    assert "several candidates pass" in lines[6].note
    assert lines[8].oid == "" and "503" in lines[8].note
    assert "not a readable avro" in lines[9].note or "avro" in lines[9].note
    assert lines[0].isdiffpos == "t" and lines[0].fid == 2 and lines[0].drb == pytest.approx(0.9)
    # Only ok rows are stored, and their cutouts are the raw ones, NaN kept.
    recovered = load_recovered(split_dir)
    assert recovered.rows.tolist() == [0, 1, 2, 7]
    assert recovered.candid.tolist() == [100, 102, 201, 700]
    np.testing.assert_array_equal(recovered.cutouts[0], world.raw[0])
    assert np.isnan(recovered.cutouts[3][:, :13]).all()  # the CCD-edge padding
    np.testing.assert_array_equal(recovered.cutouts[3][:, 13:], world.raw[7])
    assert recovered.candidates[0]["candid"] == 100 and recovered.candidates[0]["chipsf"] is None
    summary = summarize(split_dir)
    assert summary.counts["AGN"]["ok"] == 2 and summary.edge["Asteroid"] == 1
    assert not summary.accepted()


def test_an_object_without_detections_is_no_object(world: World, tmp_path: Path) -> None:
    world.alerce.objects.append(SkyObject("ZTF18aaaaaaz", 10.03, 20.03, []))  # row 3's cone
    fetch_alerts(world.prepared, "Validation", tmp_path / "raw", client=client_for(world.alerce))
    lines = latest_rows(read_index(tmp_path / "raw" / "Validation"))
    assert lines[3].status == "no_object" and lines[3].n_objects == 1


def test_a_ghost_keeps_its_position_and_is_final(world: World, tmp_path: Path) -> None:
    out = tmp_path / "raw"
    fetch_alerts(world.prepared, "Validation", out, client=client_for(world.alerce))
    ghost = latest_rows(read_index(out / "Validation"))[3]
    assert ghost.status == "ghost_candidate" and ghost.n_objects == 0
    assert ghost.target_ra == pytest.approx(10.03) and ghost.target_dec == pytest.approx(20.03)
    before = len(world.alerce.calls)
    fetch_alerts(world.prepared, "Validation", out, client=client_for(world.alerce), resume=True)
    assert not any("ra=10.03" in u for u in world.alerce.calls[before:])  # not retried


def test_shared_objects_are_served_from_the_cache(world: World, tmp_path: Path) -> None:
    fetch_alerts(world.prepared, "Validation", tmp_path / "raw", client=client_for(world.alerce))
    cones = [u for u in world.alerce.calls if "/objects/?" in u and "ra=10.0&" in u]
    detections = [u for u in world.alerce.calls if u.endswith("ZTF18aaaaaaa/detections")]
    assert len(cones) == 1 and len(detections) == 1
    lines = latest_rows(read_index(tmp_path / "raw" / "Validation"))
    assert lines[1].requests == 2  # two avro files; the cone and detections were cached
    assert lines[1].responses.count("objects:") == 1


def test_resume_skips_finished_rows_and_retries_http_errors(world: World, tmp_path: Path) -> None:
    out = tmp_path / "raw"
    first = fetch_alerts(
        world.prepared, "Validation", out, client=client_for(world.alerce), limit=5
    )
    assert first.attempted == 5
    assert statuses(out / "Validation")[8] == "http_error"
    with pytest.raises(DataError, match="--resume"):
        fetch_alerts(world.prepared, "Validation", out, client=client_for(world.alerce))
    calls_before = len(world.alerce.calls)
    second = fetch_alerts(
        world.prepared, "Validation", out, client=client_for(world.alerce), resume=True
    )
    assert second.skipped == 4 and second.attempted == 6
    assert statuses(out / "Validation")[8] == "ok"
    new_calls = world.alerce.calls[calls_before:]
    assert not any("candid=100" in u for u in new_calls)  # row 0 was not fetched again
    assert len(read_index(out / "Validation")) == 11  # row 8 has two lines
    summary = summarize(out / "Validation")
    assert summary.attempts == 11 and summary.totals["http_error"] == 0


def test_resume_refuses_another_dataset(world: World, tmp_path: Path) -> None:
    out = tmp_path / "raw"
    fetch_alerts(world.prepared, "Validation", out, client=client_for(world.alerce), limit=1)
    meta_file = out / "Validation" / "fetch.json"
    meta = json.loads(meta_file.read_text())
    meta["dataset"] = "other"
    meta_file.write_text(json.dumps(meta))
    with pytest.raises(DataError, match="dataset is 'other'"):
        fetch_alerts(
            world.prepared, "Validation", out, client=client_for(world.alerce), resume=True
        )


def test_batches_shards_and_orphans(world: World, tmp_path: Path) -> None:
    split_dir = tmp_path / "raw" / "Validation"
    split_dir.mkdir(parents=True)
    (split_dir / "shard-00007.npz").write_bytes(b"an orphan from a crashed commit")
    fetch_alerts(
        world.prepared,
        "Validation",
        tmp_path / "raw",
        client=client_for(world.alerce),
        batch_size=4,
    )
    lines = latest_rows(read_index(split_dir))
    assert {line.shard for line in lines.values()} == {8, 9, 10}
    shard = read_shard(shard_path(split_dir, 8))
    assert shard.rows.tolist() == [0, 2]  # rows 0, 2, 4, 6: the ok ones
    kinds = {r.kind for r in shard.responses}
    assert kinds == {"objects", "detections", "avro"}
    assert load_recovered(split_dir).rows.tolist() == [0, 1, 2, 7]


def test_an_interrupted_run_commits_its_batch(world: World, tmp_path: Path) -> None:
    out = tmp_path / "raw"
    world.alerce.failures["ZTF18aaaaaaf/detections"] = [KeyboardInterrupt()]
    with pytest.raises(KeyboardInterrupt):
        fetch_alerts(world.prepared, "Validation", out, client=client_for(world.alerce))
    assert sorted(statuses(out / "Validation")) == [0, 2, 4]  # row 6 was in progress
    fetch_alerts(world.prepared, "Validation", out, client=client_for(world.alerce), resume=True)
    assert statuses(out / "Validation")[6] == "ambiguous"


def test_a_truncated_index_line_is_dropped_and_repaired(world: World, tmp_path: Path) -> None:
    out = tmp_path / "raw"
    fetch_alerts(world.prepared, "Validation", out, client=client_for(world.alerce), limit=2)
    index = out / "Validation" / "index.csv"
    index.write_text(index.read_text() + "4,2,ok,1.0")
    assert sorted(statuses(out / "Validation")) == [0, 2]
    fetch_alerts(world.prepared, "Validation", out, client=client_for(world.alerce), resume=True)
    assert statuses(out / "Validation") == world.expected  # row 8 fails on its first try


def test_cooldown_after_consecutive_http_errors(world: World, tmp_path: Path) -> None:
    error = HttpError("x", None, "TimeoutError (after 5 attempts)")
    world.alerce.failures["/objects/?"] = [error] * 5
    sleeps: list[float] = []
    run = fetch_alerts(
        world.prepared,
        "Validation",
        tmp_path / "raw",
        client=client_for(world.alerce),
        cooldown=42.0,
        sleep=sleeps.append,
    )
    assert run.counts["http_error"] >= COOLDOWN_AFTER
    assert sleeps == [42.0] and run.cooldowns == 1


def test_offline_rebuild_from_the_snapshot(world: World, tmp_path: Path) -> None:
    out = tmp_path / "raw"
    fetch_alerts(world.prepared, "Validation", out, client=client_for(world.alerce))
    fetch_alerts(world.prepared, "Validation", out, client=client_for(world.alerce), resume=True)
    snapshot = Snapshot(out / "Validation")
    assert len(snapshot) > 10
    rebuilt = tmp_path / "rebuilt"
    fetch_alerts(
        world.prepared, "Validation", rebuilt, client=AlerceClient(Recorder(snapshot.fetch))
    )
    assert statuses(rebuilt / "Validation") == statuses(out / "Validation")
    original = load_recovered(out / "Validation")
    copy = load_recovered(rebuilt / "Validation")
    np.testing.assert_array_equal(copy.cutouts, original.cutouts)
    with pytest.raises(HttpError, match="not in the snapshot"):
        snapshot.fetch("https://example.org/", 10)


def test_duplicated_candids_across_splits(world: World, tmp_path: Path) -> None:
    out = tmp_path / "raw"
    for split in ("Validation", "Test"):
        fetch_alerts(world.prepared, split, out, client=client_for(world.alerce))
    found = duplicates([out / "Validation", out / "Test"])
    assert found[100] == [("Test", 0), ("Validation", 0)]


def test_invalid_inputs(world: World, tmp_path: Path) -> None:
    client = client_for(world.alerce)
    with pytest.raises(ValueError, match="split"):
        fetch_alerts(world.prepared, "Dev", tmp_path, client=client)
    with pytest.raises(ValueError, match="limit"):
        fetch_alerts(world.prepared, "Test", tmp_path, client=client, limit=0)
    with pytest.raises(ValueError, match="batch_size"):
        fetch_alerts(world.prepared, "Test", tmp_path, client=client, batch_size=0)
    with pytest.raises(ValueError, match="cooldown"):
        fetch_alerts(world.prepared, "Test", tmp_path, client=client, cooldown=-1)
    split = SplitData(
        np.zeros((2, 63, 63, 3), np.uint8), np.zeros(2, np.int64), np.zeros((2, 26)), None
    )
    no_f32 = write_prepared(
        StampData("classifier", dict.fromkeys(SPLITS, split)),
        tmp_path / "no_f32",
        {"name": "x.pkl", "size_bytes": 0, "sha256": "0" * 64},
    )
    with pytest.raises(DataError, match="images_f32"):
        fetch_alerts(no_f32, "Test", tmp_path / "raw", client=client)


def test_a_failed_commit_is_not_retried_and_keeps_its_responses(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from astro_simclr.alerts.store import write_shard as real

    calls: list[int] = []

    def failing(split_dir: Path, shard_id: int, shard: object) -> Path:
        calls.append(shard_id)
        if len(calls) == 1:
            raise OSError(28, "No space left on device")
        return real(split_dir, shard_id, shard)  # type: ignore[arg-type]

    monkeypatch.setattr("astro_simclr.alerts.fetch.write_shard", failing)
    client = client_for(world.alerce)
    out = tmp_path / "raw"
    with pytest.raises(OSError, match="No space"):
        fetch_alerts(world.prepared, "Validation", out, client=client, batch_size=3)
    assert calls == [0]  # the finally did not commit the failed batch again
    assert read_index(out / "Validation") == []
    assert isinstance(client.getter, Recorder) and client.getter.pending()  # kept
    fetch_alerts(world.prepared, "Validation", out, client=client, resume=True, batch_size=3)
    assert statuses(out / "Validation") == world.expected
    # Every response a line references is in the snapshot (the kept ones included).
    snapshot = Snapshot(out / "Validation")
    stored = {
        r.ref for s in (out / "Validation").glob("shard-*.npz") for r in read_shard(s).responses
    }
    for line in read_index(out / "Validation"):
        assert set(line.responses.split()) <= stored
    assert len(snapshot) > 0


def test_stop_ends_the_run_between_rows(world: World, tmp_path: Path) -> None:
    seen: list[int] = []
    run = fetch_alerts(
        world.prepared,
        "Validation",
        tmp_path / "raw",
        client=client_for(world.alerce),
        progress=lambda position, total, line: seen.append(position),
        stop=lambda: len(seen) >= 3,
    )
    assert run.attempted == 3 and len(statuses(tmp_path / "raw" / "Validation")) == 3


def test_an_ok_row_whose_shard_is_missing_is_fetched_again(world: World, tmp_path: Path) -> None:
    out = tmp_path / "raw"
    fetch_alerts(world.prepared, "Validation", out, client=client_for(world.alerce), batch_size=4)
    shard_path(out / "Validation", 0).unlink()  # rows 0, 2, 4, 6 (ok: 0 and 2)
    run = fetch_alerts(
        world.prepared, "Validation", out, client=client_for(world.alerce), resume=True
    )
    assert run.attempted == 3  # rows 0 and 2, and row 8 (http_error, ok this time)
    assert load_recovered(out / "Validation").rows.tolist() == [0, 1, 2, 7, 8]


def test_final_client_errors_do_not_trigger_the_cooldown(world: World, tmp_path: Path) -> None:
    world.alerce.failures["/objects/?"] = [HttpError("x", 404, "HTTP 404")] * 5
    sleeps: list[float] = []
    run = fetch_alerts(
        world.prepared,
        "Validation",
        tmp_path / "raw",
        client=client_for(world.alerce),
        sleep=sleeps.append,
    )
    assert run.counts["http_error"] >= 5 and sleeps == [] and run.cooldowns == 0


def test_an_unread_page_of_objects_makes_a_miss_ambiguous(world: World, tmp_path: Path) -> None:
    original = world.alerce._answer

    def paged(url: str) -> bytes:
        body = original(url)
        if "/objects/?" in url:
            payload = json.loads(body)
            payload["has_next"] = True
            return json.dumps(payload).encode()
        return body

    world.alerce._answer = paged  # type: ignore[method-assign]
    fetch_alerts(world.prepared, "Validation", tmp_path / "raw", client=client_for(world.alerce))
    got = statuses(tmp_path / "raw" / "Validation")
    assert got[4] == "ambiguous" and got[5] == "ambiguous" and got[0] == "ok"
    assert got[3] == "ghost_candidate"  # an empty first page has no second one


def test_verify_cutouts_handles_non_finite_values() -> None:
    raw = np.ones((63, 63, 3), dtype=np.float32)
    raw[0, 0, 0] = np.inf  # inf / inf is NaN, then 0, as in ALeRCE's arithmetic
    assert verify_cutouts(raw, np.zeros((63, 63, 3), dtype=np.float32)) == 0.0
    stored = np.zeros((63, 63, 3), dtype=np.float32)
    stored[1, 1, 1] = np.nan  # a NaN difference never passes as "not larger"
    assert verify_cutouts(raw, stored) == float("inf")
