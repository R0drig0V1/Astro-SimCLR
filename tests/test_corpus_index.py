"""Tests of the corpus index of task 5.1 and of the pilot reading it.

Offline, on synthetic data made here: a small CD21-like prepared layout (float32 stamps,
positions set per row) and a synthetic archive corpus written with the archive's own
``NightWriter`` (no tarballs: ``tests/test_archive_ingest.py`` covers the ingest). Covered:
the cross-match (inside and outside 1.5", the RA wrap, the poles), every exclusion rule and
its count, one alert per ``oid`` (``first`` and the seeded ``random``), the near-duplicate
check (a renormalised copy dropped, a noisy look-alike only counted), the SHA-256 record,
the index file's round trip and refusals, the reader, and the pilot running a few steps
from an index on the ``memmap`` and ``host`` tiers (the same model on both).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch
from _corpus_support import (
    ARCSEC,
    CD21_POSITIONS,
    archive_row,
    cd21_split,
    oid_name,
    sky_stamps,
    write_archive,
    write_layout,
)

from astro_simclr.alerts.fetch import alerce_normalise
from astro_simclr.archive.corpus_index import (
    CorpusIndex,
    build_corpus_index,
    check_archive,
    load_candid_exclusions,
    read_corpus_index,
    write_corpus_index,
)
from astro_simclr.archive.layout import (
    CORPUS_INDEX_DTYPE,
    night_dir,
)
from astro_simclr.archive.near_duplicates import (
    NearDuplicateRules,
    ReferenceStamps,
    descriptors,
    find_near_duplicates,
)
from astro_simclr.archive.reader import CorpusStamps, normalise_stamps
from astro_simclr.cli import main
from astro_simclr.config import ConfigError
from astro_simclr.data.prepared import write_prepared
from astro_simclr.data.schema import (
    STAMP_SHAPE,
    DataError,
    SplitData,
    StampData,
)
from astro_simclr.protocol.groups import match_any_within
from astro_simclr.protocol.splits import make_splits, write_splits
from astro_simclr.simclr.pilot import (
    MemmapStamps,
    PilotConfig,
    load_checkpoint,
    load_pilot_data,
    train_pilot,
)

# --- synthetic data ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A CD21 layout, an archive exercising every rule, and the exclusion files."""
    root = tmp_path_factory.mktemp("corpus")
    layout = write_layout(root / "cd21")
    train_f32 = np.load(layout / "Train" / "images_f32.npy")
    rng = np.random.default_rng(1)
    raw = sky_stamps(rng, 13).astype(np.float32)
    raw[0, :3, :3, 2] = np.nan  # a raw cutout with NaNs (normalised to finite values)
    copy = (train_f32[3].astype(np.float64) * 37.5 + 120.0).astype(np.float32)  # same stamp
    lookalike = (train_f32[4] + rng.normal(0, 0.02, STAMP_SHAPE)).astype(np.float32)
    n1, n2 = 20190101, 20190102
    jd1, jd2 = 2458484.7, 2458485.7
    rows = {
        "inside": archive_row(
            oid_name(1), 11, jd1, 150.0 + 1.4 * ARCSEC / np.cos(np.radians(20)), 20.0, n1
        ),
        # Another alert of the matched object, far away (a bad coordinate): it goes too.
        "inside_elsewhere": archive_row(oid_name(1), 24, jd2, 80.0, 30.0, n2),
        "outside": archive_row(oid_name(2), 12, jd1, 150.0, 20.0 + 1.6 * ARCSEC, n1),
        "wrap": archive_row(
            oid_name(3), 13, jd1, 0.0002, 10.0, n1
        ),  # 1.06" from Test 1 across RA 0
        "oid_listed": archive_row(oid_name(4), 14, jd1, 10.0, 5.0, n1),
        "candid_listed": archive_row(oid_name(5), 15, jd1, 20.0, 5.0, n1),
        "dup_late": archive_row(oid_name(6), 21, jd2, 30.0, 5.0, n2),
        "dup_early": archive_row(oid_name(6), 16, jd1, 30.0, 5.0, n1),
        "near": archive_row(oid_name(7), 17, jd1, 40.0, -5.0, n1),
        "near_again": archive_row(oid_name(7), 22, jd2, 40.0, -5.0, n2),
        "lookalike": archive_row(oid_name(8), 18, jd1, 50.0, 0.0, n1),
        "plain_a": archive_row(oid_name(9), 19, jd1, 60.0, 0.0, n1),
        "plain_b": archive_row(oid_name(10), 23, jd2, 70.0, 0.0, n2),
    }
    cutouts = dict(zip(rows, raw, strict=True))
    cutouts["near"], cutouts["near_again"], cutouts["lookalike"] = copy, copy, lookalike
    nights: dict[int, list[tuple[tuple[Any, ...], npt.NDArray[np.float32]]]] = {n1: [], n2: []}
    for name, row in rows.items():
        nights[row[6]].append((row, cutouts[name]))
    archive = write_archive(root / "archive", nights)
    oids = root / "oids.txt"
    oids.write_text(f"# R5 Validation/Test\n{oid_name(4)}\n\n")
    candids = root / "candids.txt"
    candids.write_text("15\n# comment\n 15 \n")
    return {
        "root": root,
        "layout": layout,
        "archive": archive,
        "oids": oids,
        "candids": candids,
        "rows": rows,
    }


def _build(world: dict[str, Any], **changes: Any) -> CorpusIndex:
    options: dict[str, Any] = {
        "layout": world["layout"],
        "oid_files": [world["oids"]],
        "candid_files": [world["candids"]],
    }
    return build_corpus_index(world["archive"], **(options | changes))


def _candids(index: CorpusIndex) -> set[int]:
    return {int(c) for c in index.rows["candid"]}


# --- the cross-match -----------------------------------------------------------------------


def test_match_any_within_radius_wrap_and_poles() -> None:
    ref_ra, ref_dec = [150.0, 359.9999, 0.0], [20.0, 10.0, 89.9999]
    ra = [150.0, 150.0, 0.0002, 180.0, 10.0, 150.0 + 1.5 * ARCSEC]
    dec = [20.0 + 1.4 * ARCSEC, 20.0 + 1.6 * ARCSEC, 10.0, 89.9999, 10.0, 20.0]
    matched, nearest = match_any_within(ra, dec, ref_ra, ref_dec, 1.5)
    # 1.4" in, 1.6" out, across RA 0 in, at the pole (0.36" apart) in, far out; exactly
    # 1.5" in RA at dec 20 is 1.41" on the sky: in.
    assert matched.tolist() == [True, False, True, True, False, True]
    assert nearest.tolist() == [0, -1, 1, 2, -1, 0]


def test_match_any_within_takes_the_nearest_and_checks_inputs() -> None:
    matched, nearest = match_any_within([10.0], [0.0], [10.0, 10.0, 10.0], [1e-4, 1e-5, 1e-5], 1)
    assert matched.tolist() == [True] and nearest.tolist() == [1]  # ties: the lower row
    empty, where = match_any_within([], [], [1.0], [1.0], 1.5)
    assert empty.size == 0 and where.size == 0
    none, _ = match_any_within([1.0], [1.0], [], [], 1.5)
    assert none.tolist() == [False]
    with pytest.raises(ValueError, match="positive"):
        match_any_within([1.0], [1.0], [1.0], [1.0], 0)
    with pytest.raises(DataError, match="non-finite"):
        match_any_within([np.nan], [1.0], [1.0], [1.0], 1.5)


# --- the index -----------------------------------------------------------------------------


def test_index_applies_every_rule_and_counts_it(world: dict[str, Any]) -> None:
    index = _build(world)
    rows = world["rows"]
    kept = {name for name, row in rows.items() if row[1] in _candids(index)}
    assert kept == {"outside", "dup_early", "lookalike", "plain_a", "plain_b"}
    counts = index.meta["counts"]
    assert counts["rows"] == len(rows) and counts["kept"] == len(kept) == len(index)
    assert counts["excluded"] == {
        "candid_list": 1,
        "oid_list": 1,
        "position": 3,  # inside 1.4", its object's other alert, and across the RA wrap
        "near_duplicate": 2,  # both alerts of the copied object
        "duplicate_oid": 1,  # the later alert of oid 6
    }
    assert counts["matched"] == {"candid_list": 1, "oid_list": 1, "position": 2}  # rows within
    near = index.meta["near_duplicates"]
    assert near["matched_rows"] == 2 and near["matched_oids"] == 1
    assert near["candidates"] >= 3  # the copies and the look-alike reach the cosine
    match = near["matches"][0]
    assert match["set"] == "Train" and match["row"] == 3 and match["max_abs_diff"] < 1e-5
    assert index.rows.dtype == CORPUS_INDEX_DTYPE
    assert np.array_equal(np.lexsort((index.rows["row"], index.rows["night"])), np.arange(5))


def test_index_records_the_sha256_of_every_input(world: dict[str, Any]) -> None:
    index = _build(world)
    inputs = index.meta["exclusion_inputs"]
    layout = world["layout"]
    assert inputs[0]["kind"] == "cd21-layout" and inputs[0]["path"] == str(layout)
    files = inputs[0]["files"]
    assert set(files) == {
        f"{s}/{f}" for s in CD21_POSITIONS for f in ("features.npy", "images_f32.npy")
    }
    for name, digest in files.items():
        assert digest == hashlib.sha256((layout / name).read_bytes()).hexdigest()
    by_kind = {item["kind"]: item for item in inputs[1:]}
    assert by_kind["oids"]["sha256"] == hashlib.sha256(world["oids"].read_bytes()).hexdigest()
    assert by_kind["candids"]["count"] == 1
    assert len(index.meta["exclusion_sha256"]) == 64
    without = _build(world, candid_files=[])
    assert without.meta["exclusion_sha256"] != index.meta["exclusion_sha256"]
    assert world["rows"]["candid_listed"][1] in _candids(without)
    wider = _build(world, radius_arcsec=2.0)  # the rules are part of the digest
    assert wider.meta["exclusion_sha256"] != index.meta["exclusion_sha256"]
    nights = index.meta["archive"]["nights"]
    assert set(nights) == {"20190101", "20190102"} and nights["20190102"]["n"] == 4


def test_one_alert_per_oid_first_or_seeded(world: dict[str, Any]) -> None:
    early, late = world["rows"]["dup_early"][1], world["rows"]["dup_late"][1]
    first = _build(world, near=None)
    assert early in _candids(first) and late not in _candids(first)
    assert first.meta["near_duplicates"] is None
    assert first.meta["counts"]["excluded"]["near_duplicate"] == 0
    picks = set()
    for seed in range(8):
        drawn = _build(world, near=None, duplicates="random", seed=seed)
        chosen = _candids(drawn) & {early, late}
        assert len(chosen) == 1
        picks |= chosen
        again = _build(world, near=None, duplicates="random", seed=seed)
        assert np.array_equal(again.rows, drawn.rows)  # seeded
    assert picks == {early, late}
    with pytest.raises(ValueError, match="duplicates"):
        _build(world, duplicates="last")
    with pytest.raises(ValueError, match="seed"):
        _build(world, seed=-1)


def test_index_file_round_trip_and_archive_check(world: dict[str, Any], tmp_path: Path) -> None:
    index = _build(world)
    path = tmp_path / "corpus-index.npz"
    write_corpus_index(index, path)
    back = read_corpus_index(path)
    assert np.array_equal(back.rows, index.rows) and back.meta == json.loads(json.dumps(index.meta))
    with pytest.raises(DataError, match="already exists"):
        write_corpus_index(index, path)
    write_corpus_index(index, path, overwrite=True)
    check_archive(back, world["archive"])
    # A night that changed (here: its meta.json) is refused.
    changed = dict(back.meta, archive=dict(back.meta["archive"], nights={}))
    with pytest.raises(DataError, match="night 20190101 differs"):
        check_archive(CorpusIndex(back.rows, changed), world["archive"])
    other = dict(back.meta, archive=dict(back.meta["archive"], corpus_sha256="0" * 64))
    with pytest.raises(DataError, match=r"corpus\.json differs"):
        check_archive(CorpusIndex(back.rows, other), world["archive"])


def test_read_corpus_index_refuses_bad_files(world: dict[str, Any], tmp_path: Path) -> None:
    index = _build(world, near=None)
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not an npz")
    with pytest.raises(DataError, match="not a corpus index"):
        read_corpus_index(bad)
    write_corpus_index(index, tmp_path / "good.npz")
    bad.write_bytes((tmp_path / "good.npz").read_bytes()[:200])  # a truncated zip
    with pytest.raises(DataError, match="not a corpus index"):
        read_corpus_index(bad)
    np.savez(bad, rows=index.rows, meta=np.array(json.dumps({"format": "other"})))
    with pytest.raises(DataError, match="not an astro-simclr-corpus-index"):
        read_corpus_index(bad)
    unsorted = CorpusIndex(index.rows[::-1].copy(), index.meta)
    write_corpus_index(unsorted, tmp_path / "unsorted.npz")
    with pytest.raises(DataError, match="sorted"):
        read_corpus_index(tmp_path / "unsorted.npz")
    short = CorpusIndex(index.rows[:1].copy(), index.meta)
    write_corpus_index(short, tmp_path / "short.npz")
    with pytest.raises(DataError, match="corpus index rows"):
        read_corpus_index(tmp_path / "short.npz")


def test_build_refuses_a_layout_without_float32_or_an_archive(
    world: dict[str, Any], tmp_path: Path
) -> None:
    rng = np.random.default_rng(0)
    splits = {}
    for name in CD21_POSITIONS:
        split = cd21_split(rng, 1, {})
        splits[name] = SplitData(split.images, split.labels, split.features)
    source = {"name": "u8.pkl", "size_bytes": 0, "sha256": "1" * 64}
    uint8 = write_prepared(StampData("classifier", splits), tmp_path / "u8", source)
    with pytest.raises(DataError, match="float32 stamps"):
        _build(world, layout=uint8)
    assert len(_build(world, layout=uint8, near=None)) > 0  # positions only
    with pytest.raises(DataError, match="not an archive corpus"):
        build_corpus_index(tmp_path, layout=world["layout"])


def test_load_candid_exclusions(tmp_path: Path) -> None:
    good = tmp_path / "c.txt"
    good.write_text("# header\n12\n\n 12\n3400000000000000001\n")
    loaded = load_candid_exclusions(good)
    assert loaded.candids == {12, 3400000000000000001} and loaded.source == "c.txt"
    for text in ("ZTF18abc\n", "-3\n", "0\n", "1.5\n", f"{2**63}\n"):
        good.write_text(text)
        with pytest.raises(DataError, match="not a candid"):
            load_candid_exclusions(good)
    good.write_bytes(b"\xff\n")
    with pytest.raises(DataError, match="UTF-8"):
        load_candid_exclusions(good)


# --- near duplicates and the reader ---------------------------------------------------------


def test_descriptors_and_rules() -> None:
    flat = np.full((2, *STAMP_SHAPE), 0.5, dtype=np.float32)
    assert not descriptors(flat).any()  # no structure: the zero vector
    rng = np.random.default_rng(0)
    some = rng.random((3, *STAMP_SHAPE)).astype(np.float32)
    values = descriptors(some)
    assert values.shape == (3, 243) and np.allclose(np.linalg.norm(values, axis=1), 1)
    with pytest.raises(ValueError, match="63, 63, 3"):
        descriptors(some[:, :21])
    for bad in ({"cosine": 0.0}, {"cosine": 1.0}, {"max_abs_diff": -1.0}, {"candidates": 0}):
        with pytest.raises(ValueError, match="need"):
            NearDuplicateRules(**bad)
    empty = ReferenceStamps((), ())
    found = find_near_duplicates([(np.arange(3), some)], empty, NearDuplicateRules())
    assert found.checked == 3 and found.matched == [] and found.candidates == 0
    with pytest.raises(ValueError, match="one name"):
        ReferenceStamps(("a",), ())


def test_reader_serves_any_order_across_files(world: dict[str, Any]) -> None:
    index = _build(world, near=None)
    stamps = CorpusStamps(world["archive"], index.rows)
    assert len(stamps) == len(index) and stamps.nbytes == len(index) * 63 * 63 * 3 * 4
    raw_rows = {}
    for night in (20190101, 20190102):
        path = night_dir(world["archive"], night)
        cutouts = np.concatenate([np.load(f) for f in sorted(path.glob("cutouts-*.npy"))])
        raw_rows[night] = cutouts
    order = np.array([4, 0, 2, 1, 3, 0])
    got = stamps.read(order)
    for k, position in enumerate(order):
        row = index.rows[position]
        np.testing.assert_array_equal(got[k], raw_rows[int(row["night"])][int(row["row"])])
    normalised = stamps.normalised(order)
    assert np.isfinite(normalised).all() and normalised.min() >= 0 and normalised.max() <= 1
    np.testing.assert_array_equal(normalised, alerce_normalise(got).astype(np.float32))
    np.testing.assert_array_equal(normalise_stamps(got), normalised)
    assert [p.tolist() for p, _ in stamps.chunks(np.arange(5), 2)] == [[0, 1], [2, 3], [4]]
    with pytest.raises(ValueError, match="chunk_rows"):
        next(iter(stamps.chunks(np.arange(5), 0)))
    with pytest.raises(IndexError):
        stamps.read([len(stamps)])
    tampered = index.rows.copy()
    tampered["oid"][0] = "ZTF18zzzzzzz"
    with pytest.raises(DataError, match="another corpus"):
        CorpusStamps(world["archive"], tampered)
    beyond = index.rows.copy()
    beyond["row"][0] = 99
    with pytest.raises(DataError, match="outside"):
        CorpusStamps(world["archive"], beyond)
    with pytest.raises(DataError, match="corpus index rows"):
        CorpusStamps(world["archive"], np.zeros(2, dtype=[("night", "<i8")]))


# --- the pilot on an index -----------------------------------------------------------------


@pytest.fixture(scope="module")
def pilot_world(world: dict[str, Any], tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """An index of the archive (positions only) and a splits file of the CD21 layout."""
    root = tmp_path_factory.mktemp("pilot")
    index = root / "index.npz"
    write_corpus_index(_build(world), index)
    splits = root / "splits.json"
    write_splits(make_splits(world["layout"], selection_per_class=1, screen_per_class=2), splits)
    return {"index": index, "splits": splits, "layout": world["layout"]}


def _pilot(pilot_world: dict[str, Path], **changes: Any) -> PilotConfig:
    base = PilotConfig(
        data=str(pilot_world["index"]),
        splits=str(pilot_world["splits"]),
        epochs=None,
        steps=3,
        batch_size=2,
        warmup_epochs=0.5,
        device="cpu",
        tier="memmap",
        log_every=1,
        checkpoint_every=2,
        diagnostics_every=3,
        keep_every_epochs=None,
        diag_samples=4,
        knn_bank_per_class=2,
        embed_batch=8,
    )
    return replace(base, **changes)


def test_pilot_trains_from_an_index_on_memmap_and_host(
    pilot_world: dict[str, Path], tmp_path: Path
) -> None:
    memmap = train_pilot(_pilot(pilot_world), tmp_path / "memmap")
    host = train_pilot(_pilot(pilot_world, tier="host"), tmp_path / "host")
    assert memmap.steps_done == host.steps_done == 3
    assert memmap.order.n == 5  # the index's kept rows
    a = load_checkpoint(tmp_path / "memmap" / "ckpt" / "final.pt")["model"]
    b = load_checkpoint(tmp_path / "host" / "ckpt" / "final.pt")["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)  # the tier does not change the numbers
    record = json.loads((tmp_path / "memmap" / "pilot.json").read_text())
    pretrain = record["data"]["pretrain"]
    assert pretrain["kind"] == "archive-corpus" and pretrain["n"] == 5
    assert pretrain["index_sha256"] == hashlib.sha256(pilot_world["index"].read_bytes()).hexdigest()
    assert len(pretrain["exclusion_sha256"]) == 64
    assert record["data"]["path"] == str(pilot_world["layout"])  # evaluated on CD21
    diagnostics = [json.loads(x) for x in (tmp_path / "memmap" / "diagnostics.jsonl").open()]
    assert diagnostics[-1]["knn_selection_bacc"] is not None
    train_h = np.load(tmp_path / "memmap" / "embeddings" / "train_h.npy")
    assert train_h.shape == (5 * 4, 512)  # CD21 Train, not the corpus


def test_pilot_refuses_corpus_misuse(pilot_world: dict[str, Path], tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not kept on the training device"):
        train_pilot(_pilot(pilot_world, tier="device"), tmp_path / "a")
    with pytest.raises(ConfigError, match="host cap"):
        train_pilot(_pilot(pilot_world, tier="host", host_cap_gb=1e-6), tmp_path / "b")
    with pytest.raises(ConfigError, match="stamp fits"):
        load_pilot_data(_pilot(pilot_world, noise_scale=1.0))
    with pytest.raises(ConfigError, match="fewer than a batch"):
        load_pilot_data(_pilot(pilot_world, batch_size=6))
    with pytest.raises(ConfigError, match="memmap tier reads a corpus"):
        train_pilot(_pilot(pilot_world, data=str(pilot_world["layout"])), tmp_path / "c")
    other = write_layout(tmp_path / "other", seed=5)
    with pytest.raises(DataError, match="is not the layout"):
        load_pilot_data(_pilot(pilot_world, eval_data=str(other), splits=None))
    with pytest.raises(ConfigError, match="host_cap_gb"):
        _pilot(pilot_world, host_cap_gb=0)


def test_pilot_resumes_a_checkpoint_from_before_the_corpus_fields(
    pilot_world: dict[str, Path], tmp_path: Path
) -> None:
    config = _pilot(pilot_world, data=str(pilot_world["layout"]), tier="host", steps=2)
    train_pilot(config, tmp_path, stop_at=1)
    last = tmp_path / "ckpt" / "last.pt"
    payload = load_checkpoint(last)
    for name in ("archive", "eval_data", "host_cap_gb"):
        del payload["config"][name]
    torch.save(payload, last)
    assert train_pilot(config, tmp_path, resume=True).steps_done == 2


def test_cli_builds_an_index_and_pilots_from_it(
    world: dict[str, Any],
    pilot_world: dict[str, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "index.npz"
    build = ("build-corpus-index", world["archive"], "--out", out, "--cd21", world["layout"])
    lists = ("--exclude-oids", world["oids"], "--exclude-candids", world["candids"])
    assert main([str(a) for a in (*build, *lists, "--quiet")]) == 0
    printed = capsys.readouterr().out
    assert "excluded by position" in printed and "kept" in printed
    assert len(read_corpus_index(out)) == 5
    assert main([str(a) for a in build]) == 2  # exists
    assert main([str(a) for a in (*build, "--overwrite", "--json", "--no-near-duplicates")]) == 0
    record = json.loads(capsys.readouterr().out)
    assert (
        record["rules"]["near_duplicates"] is None and record["counts"]["kept"] == 8
    )  # no lists given
    assert main([str(a) for a in (*build, "--overwrite", "--radius-arcsec", "0")]) == 2
    run = tmp_path / "run"
    pilot = ("simclr-pilot", out, "--output-dir", run, "--device", "cpu", "--batch-size", 2)
    splits = ("--splits", pilot_world["splits"])
    assert main([str(a) for a in (*pilot, "--steps", 2, *splits)]) == 2  # the index is weak
    assert main([str(a) for a in (*pilot, "--steps", 2, *splits, "--allow-weak-index")]) == 0
    record = json.loads((run / "pilot.json").read_text())
    assert record["config"]["tier"] == "memmap"  # the default for an index
    assert record["config"]["allow_weak_index"] is True


def test_memmap_stamps_prefetch_one_batch_ahead(world: dict[str, Any]) -> None:
    index = _build(world, near=None)
    stamps = MemmapStamps(CorpusStamps(world["archive"], index.rows), torch.device("cpu"))
    direct = stamps.take(np.array([3, 1]))
    assert direct.shape == (2, 3, 63, 63) and not direct.is_pinned()
    stamps.prefetch(np.array([3, 1]))
    other = stamps.take(np.array([0]))  # another read: the prefetch stays pending
    assert stamps._pending is not None and other.shape == (1, 3, 63, 63)
    assert torch.equal(stamps.take(np.array([3, 1])), direct) and stamps._pending is None
    stamps.prefetch(np.array([2]))
    stamps.close()
    assert stamps._pending is None and torch.equal(stamps.take(np.array([0])), other)


def test_pilot_refuses_a_weak_index_unless_allowed(
    world: dict[str, Any], pilot_world: dict[str, Path], tmp_path: Path
) -> None:
    weak = tmp_path / "weak.npz"
    write_corpus_index(_build(world, near=None, radius_arcsec=1.0), weak)
    with pytest.raises(ConfigError, match="weaker exclusions"):
        load_pilot_data(_pilot(pilot_world, data=str(weak)))
    data = load_pilot_data(_pilot(pilot_world, data=str(weak), allow_weak_index=True))
    assert data.record["pretrain"]["weak_rules"] == [
        'radius 1.0" < 1.5"',
        "no near-duplicate check",
    ]


def test_pilot_refuses_to_resume_on_a_rebuilt_index(
    world: dict[str, Any], pilot_world: dict[str, Path], tmp_path: Path
) -> None:
    index = tmp_path / "index.npz"
    write_corpus_index(_build(world), index)
    config = _pilot(pilot_world, data=str(index))
    train_pilot(config, tmp_path / "run", stop_at=1)
    write_corpus_index(_build(world, candid_files=[]), index, overwrite=True)  # same path
    with pytest.raises(ConfigError, match="another corpus index"):
        train_pilot(config, tmp_path / "run", resume=True)
