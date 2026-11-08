"""Tests of ``engine.bench``: the data-tier throughput microbenchmark runs and records itself."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from astro_simclr.engine.bench import bench_tiers, main
from astro_simclr.engine.data import DataTier


def test_bench_times_every_tier_in_interleaved_rounds(tmp_path: Path) -> None:
    results = bench_tiers(
        ["cpu"],
        n=200,
        image_shape=(3, 8, 8),
        batch_size=16,
        steps=5,
        warmup=1,
        rounds=3,
        workdir=tmp_path,
    )
    assert [r.tier for r in results] == [str(tier) for tier in DataTier]
    for result in results:
        assert result.device == "cpu"
        assert len(result.rounds) == 3
        assert result.batches_per_s > 0
        assert result.mb_per_s == pytest.approx(result.batches_per_s * 16 * 3 * 8 * 8 * 4 / 1e6)
        assert result.describe["tier"] == result.tier


def test_bench_validates(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="steps"):
        bench_tiers(["cpu"], steps=0, workdir=tmp_path)


def test_main_prints_and_writes_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "bench" / "tiers.json"
    argv = ["--device", "cpu", "--n", "100", "--shape", "3,4,4", "--batch-size", "8"]
    results = main([*argv, "--steps", "3", "--warmup", "1", "--rounds", "2", "--out", str(out)])
    assert len(results) == 3
    printed = capsys.readouterr().out
    assert "batches/s" in printed and "memmap" in printed
    record = json.loads(out.read_text())
    assert record["machine"]["torch"]
    assert [r["tier"] for r in record["results"]] == ["device", "pinned", "memmap"]
    assert record["results"][0]["image_shape"] == [3, 4, 4]
