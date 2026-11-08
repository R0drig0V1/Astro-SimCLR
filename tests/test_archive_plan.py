"""Tests of the night sampling plan and its size estimate (``plan-archive-nights``)."""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from _archive_support import Spec, night_tarball

from astro_simclr.archive.ingest import Rules, ingest_archive
from astro_simclr.archive.layout import INDEX_DTYPE
from astro_simclr.archive.plan import (
    ARCHIVE_FIRST_NIGHT,
    ARCHIVE_LAST_NIGHT,
    ASSUMED_STAMPS_PER_NIGHT,
    STAMP_BYTES,
    estimate,
    format_estimate,
    plan_nights,
)
from astro_simclr.archive.selection import Exclusions, Selection
from astro_simclr.cli import main

#: 2019-03-01 00:00 UT, for alerts of ``ZTF19`` objects.
JD_2019 = 2458543.5


def test_the_default_plan_covers_every_year_in_strata() -> None:
    plan = plan_nights(0)
    primaries = plan.primaries()
    per_year = Counter(n.night.year for n in primaries)
    assert per_year == {
        2018: 24, 2019: 24, 2020: 24, 2021: 8, 2022: 8, 2023: 8, 2024: 8, 2025: 8, 2026: 8,
    }  # fmt: skip
    assert all(ARCHIVE_FIRST_NIGHT <= n.night <= ARCHIVE_LAST_NIGHT for n in plan.nights)
    assert len({n.night for n in plan.nights}) == len(plan.nights)  # primaries != spares
    for year in per_year:
        nights = [n for n in plan.nights if n.night.year == year]
        strata = Counter((n.stratum, n.rank) for n in nights)
        assert set(strata.values()) == {1} and {r for _, r in strata} == {0, 1, 2}
        ordered = sorted(nights, key=lambda n: n.night)
        assert [n.stratum for n in ordered] == sorted(n.stratum for n in ordered)
    record = plan.to_json()
    assert record["nights"][0]["tarball"].startswith("ztf_public_2018")
    assert {n["role"] for n in record["nights"]} == {"primary", "spare1", "spare2"}
    assert primaries[0].yyyymmdd // 10_000 == 2018
    assert [n.night.year for n in plan.pilot()] == [2018, 2020, 2024]
    assert all(n.rank == 0 for n in plan.pilot())
    assert {n.use for n in primaries if n.night.year <= 2022} == {"h1"}
    assert {n.use for n in primaries if n.night.year > 2022} == {"ablation"}


def test_the_plan_is_seeded_and_years_are_independent() -> None:
    assert plan_nights(0) == plan_nights(0)
    assert plan_nights(0).nights != plan_nights(1).nights
    labelled = [n for n in plan_nights(0).nights if n.night.year <= 2020]
    other = [n for n in plan_nights(0, later_per_year=3).nights if n.night.year <= 2020]
    assert labelled == other
    shorter = plan_nights(0, last=date(2024, 12, 31))
    assert list(shorter.nights) == [n for n in plan_nights(0).nights if n.night.year <= 2024]
    no_spares = plan_nights(0, spares=0)
    assert len(no_spares.nights) == len(no_spares.primaries()) == len(plan_nights(0).primaries())


def test_short_ranges_and_invalid_arguments() -> None:
    plan = plan_nights(3, first=date(2021, 1, 1), last=date(2021, 1, 7), later_per_year=8)
    assert len(plan.primaries()) == 2  # 7 nights: 2 strata of a primary and two spares
    assert plan_nights(0, first=date(2021, 1, 1), last=date(2021, 1, 2)).nights == ()
    invalid: list[dict[str, Any]] = [
        {"labelled_per_year": -1},
        {"spares": -1},
        {"first": date(2022, 1, 2), "last": date(2022, 1, 1)},
    ]
    for kwargs in invalid:
        with pytest.raises(ValueError):
            plan_nights(0, **kwargs)
    with pytest.raises(ValueError):
        plan_nights(-1)


def test_the_assumed_estimate() -> None:
    plan = plan_nights(0, labelled_per_year=2, later_per_year=1)
    result = estimate(plan)
    total = result["total"]
    assert total["nights"] == 3 * 2 + 6 * 1
    assert total["stamps"] == total["nights"] * ASSUMED_STAMPS_PER_NIGHT == 12 * 17_500
    row_bytes = STAMP_BYTES + INDEX_DTYPE.itemsize
    assert result["bytes_per_stamp"] == row_bytes and STAMP_BYTES == 47_628
    assert total["float32_gb"] == pytest.approx(total["stamps"] * row_bytes / 1e9)
    assert total["float16_gb"] < total["float32_gb"] / 1.9
    assert total["download_gb"] == pytest.approx(12 * 8.0)
    assert total["download_hours"] == pytest.approx(12 * 0.5)
    assert {row["basis"] for row in result["years"].values()} == {"assumed"}
    assert result["uses"]["h1"]["nights"] == 3 * 2 + 2 and result["uses"]["ablation"]["nights"] == 4
    text = format_estimate(result)
    assert text.splitlines()[-2].lstrip().startswith("total") and "bytes per stamp" in text
    assert "h1, assumed" in text and "ablation, assumed" in text


def test_the_measured_estimate(tmp_path: Path) -> None:
    out = tmp_path / "corpus"
    first = [Spec(f"ZTF19aaaaaa{c}", i + 1, candidate={"jd": JD_2019}) for i, c in enumerate("ab")]
    second = [Spec(f"ZTF19aaaaaa{c}", i + 3, candidate={"jd": JD_2019}) for i, c in enumerate("cd")]
    tarballs = [night_tarball(tmp_path, 20190301, first), night_tarball(tmp_path, 20190302, second)]
    ingest_archive(tarballs, out, Rules(Selection(), Exclusions.none()))
    plan = plan_nights(0, labelled_per_year=2, later_per_year=1)
    result = estimate(plan, measured=out)
    assert result["years"]["2019"]["stamps_per_night"] == 2
    assert result["years"]["2019"]["basis"] == "measured (2 nights of 2019)"
    assert result["years"]["2020"]["basis"] == "measured (2 nights of other years)"
    size = sum(t.stat().st_size for t in tarballs) / 2
    assert result["years"]["2021"]["download_gb"] == pytest.approx(size / 1e9)
    assert result["years"]["2021"]["download_hours"] == pytest.approx(size / 1e9 / 8.0 * 0.5)
    assert result["total"]["stamps"] == 12 * 2


def test_cli_plan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "plans" / "nights.json"
    args = ["plan-archive-nights", "--seed", "7", "--later-per-year", "2"]
    assert main([*args, "--out", str(out)]) == 0
    table = capsys.readouterr().out
    assert " total " in table and "assumed" in table
    written = json.loads(out.read_text())
    assert written["plan"]["seed"] == 7 and written["estimate"]["total"]["nights"] == 3 * 24 + 6 * 2
    assert main([*args, "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == written
    assert main(["plan-archive-nights", "--first", "2022-01-02", "--last", "2022-01-01"]) == 2
    assert main(["plan-archive-nights", "--stamps-per-night", "0"]) == 2
    assert main(["plan-archive-nights", "--download-hours", "0"]) == 2


def test_the_committed_plan_is_the_code_s() -> None:
    """``configs/archive/nights-seed0.json`` is what ``plan-archive-nights --seed 0`` writes."""
    committed = Path(__file__).parents[1] / "configs" / "archive" / "nights-seed0.json"
    payload = json.loads(committed.read_text(encoding="utf-8"))
    plan = plan_nights(0)
    assert payload["plan"] == plan.to_json()
    assert payload["estimate"] == estimate(plan)
