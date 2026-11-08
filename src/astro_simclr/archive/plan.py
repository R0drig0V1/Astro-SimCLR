"""The night sampling plan of the archive corpus, and its size estimate (``plan-archive-nights``).

**Rule** (seeded, deterministic; ADR 0019). For each calendar year from the archive's first
night (2018-06-01) to the last one considered, the year's nights within that range are cut
into ``k`` contiguous strata of near-equal length (``k`` = ``labelled_per_year`` for the
labelled era 2018-2020, ``later_per_year`` after it), and ``1 + spares`` different nights
are drawn in each stratum with ``numpy.random.default_rng([seed, year])``: the
**primary** night, then the spares in order. Strata spread the nights over the seasons
(the sky ZTF observes, the seeing and the moon change with them). A spare replaces its
primary when the archive's tarball for it is empty or missing: 758 of the 3,087 nightly
files were empty on 2026-11-08 (phase 5, task 5.1a), so two spares by default leave about
a 1.5 % chance that a stratum has no night. Seeding by ``(seed, year)`` keeps each year's
draw independent of the others, so changing one year's count or the last night changes
no other year.

Every night carries its **use**: ``h1`` for 2018-2022 (the H1 corpus, task 5.1a) and
``ablation`` after (the 2018-2026 ablation, the later-years domain shift). The **pilot**
(task 5.1a: one night of 2018, 2020 and 2024 first, to measure the yield, the composition
and the download time) is the primary of one stratum per pilot year, drawn from the same
generator after the strata.

**Estimate.** The stored size per stamp is exact: 63 x 63 x 3 float32 = 47,628 bytes plus
one index row (:data:`~astro_simclr.archive.layout.INDEX_DTYPE`). The stamps kept per
night are not known before a night is ingested, so :func:`estimate` takes either an
*assumed* number per night (:data:`ASSUMED_STAMPS_PER_NIGHT`) or the counts *measured*
on the nights of an existing corpus (per year where a year has ingested nights, else
their mean). The download volume and time use the measured tarball sizes, or the figures
of task 5.1a (:data:`ASSUMED_TARBALL_GB`, :data:`ASSUMED_DOWNLOAD_HOURS`). The tarballs are
streamed (``ingest-archive -``), so they need no disk space.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np

from astro_simclr.archive.layout import INDEX_DTYPE, iter_nights
from astro_simclr.archive.stream import tarball_name
from astro_simclr.data.schema import STAMP_SHAPE

__all__ = [
    "ARCHIVE_FIRST_NIGHT",
    "ARCHIVE_LAST_NIGHT",
    "ARCHIVE_URL",
    "ASSUMED_DOWNLOAD_HOURS",
    "ASSUMED_FIRST_FRACTION",
    "ASSUMED_STAMPS_PER_NIGHT",
    "ASSUMED_TARBALL_GB",
    "H1_LAST_YEAR",
    "LABELLED_YEARS",
    "MEASURED_ALERTS_PER_NIGHT",
    "PILOT_YEARS",
    "STAMP_BYTES",
    "NightPlan",
    "PlannedNight",
    "estimate",
    "format_estimate",
    "plan_nights",
]

#: Where the nightly tarballs are (``ztf_public_YYYYMMDD.tar.gz``).
ARCHIVE_URL: Final = "https://ztf.uw.edu/alerts/public/"
#: The archive's first night, and the last one listed on 2026-11-08 (integrator probe).
ARCHIVE_FIRST_NIGHT: Final = date(2018, 6, 1)
ARCHIVE_LAST_NIGHT: Final = date(2026, 11, 7)
#: The years of CD21's labelled set (the labelled era).
LABELLED_YEARS: Final = (2018, 2019, 2020)
#: The last year of the H1 corpus (task 5.1a: 2018-2022; later years are an ablation).
H1_LAST_YEAR: Final = 2022
#: The pilot's years (task 5.1a).
PILOT_YEARS: Final = (2018, 2020, 2024)
#: Stored bytes of one stamp: float32 (63, 63, 3).
STAMP_BYTES: Final = int(np.prod(STAMP_SHAPE)) * 4
_ROW_BYTES: Final = STAMP_BYTES + INDEX_DTYPE.itemsize
#: Alerts in a non-empty night (task 5.1a, measured on 2026-11-08).
MEASURED_ALERTS_PER_NIGHT: Final = 175_000
#: ASSUMPTION (not measured): the fraction of a night's alerts that are kept (first alerts
#: of non-excluded objects, 63x63). The pilot replaces it (``--measured``).
ASSUMED_FIRST_FRACTION: Final = 0.10
ASSUMED_STAMPS_PER_NIGHT: Final = round(MEASURED_ALERTS_PER_NIGHT * ASSUMED_FIRST_FRACTION)
#: A 2020 night's tarball (task 5.1a: 6.5-9.8 GB per night, 8.0 GB in 2020), in GB.
ASSUMED_TARBALL_GB: Final = 8.0
#: Download time of one night (task 5.1a: about 30 min), in hours.
ASSUMED_DOWNLOAD_HOURS: Final = 0.5
_GB: Final = 1e9

type Use = Literal["h1", "ablation"]


@dataclass(frozen=True, slots=True)
class PlannedNight:
    """One night of the plan: its date, stratum, rank (0 primary, then spares), pilot flag."""

    night: date
    stratum: int
    rank: int
    pilot: bool = False

    @property
    def yyyymmdd(self) -> int:
        """The night as ``YYYYMMDD`` (the corpus's night id)."""
        return int(self.night.strftime("%Y%m%d"))

    @property
    def use(self) -> Use:
        """``h1`` for the H1 corpus years, ``ablation`` after them."""
        return "h1" if self.night.year <= H1_LAST_YEAR else "ablation"

    def to_json(self) -> dict[str, Any]:
        """The night as a JSON object, with its tarball name."""
        return {
            "night": self.yyyymmdd,
            "year": self.night.year,
            "stratum": self.stratum,
            "role": "primary" if self.rank == 0 else f"spare{self.rank}",
            "use": self.use,
            "pilot": self.pilot,
            "tarball": tarball_name(self.yyyymmdd),
        }


@dataclass(frozen=True, slots=True)
class NightPlan:
    """A sampling plan: its parameters and its nights (primaries and spares)."""

    seed: int
    first: date
    last: date
    labelled_per_year: int
    later_per_year: int
    spares: int
    nights: tuple[PlannedNight, ...]

    def primaries(self) -> list[PlannedNight]:
        """The primary nights, in date order."""
        return [n for n in self.nights if n.rank == 0]

    def pilot(self) -> list[PlannedNight]:
        """The pilot nights, in date order."""
        return [n for n in self.nights if n.pilot]

    def to_json(self) -> dict[str, Any]:
        """The plan as JSON: the rule's parameters, then every night."""
        return {
            "rule": "per year: k equal strata, 1 + spares distinct nights per stratum "
            "(primary, then spares) from numpy default_rng([seed, year]); then one pilot "
            "stratum per pilot year from the same generator",
            "seed": self.seed,
            "first": self.first.isoformat(),
            "last": self.last.isoformat(),
            "labelled_years": list(LABELLED_YEARS),
            "h1_last_year": H1_LAST_YEAR,
            "pilot_years": list(PILOT_YEARS),
            "labelled_per_year": self.labelled_per_year,
            "later_per_year": self.later_per_year,
            "spares": self.spares,
            "archive_url": ARCHIVE_URL,
            "nights": [n.to_json() for n in self.nights],
        }


def plan_nights(
    seed: int = 0,
    *,
    labelled_per_year: int = 24,
    later_per_year: int = 8,
    spares: int = 2,
    first: date = ARCHIVE_FIRST_NIGHT,
    last: date = ARCHIVE_LAST_NIGHT,
) -> NightPlan:
    """Draw the sampling plan (module docstring).

    A year with fewer than ``(1 + spares) k`` nights in range gets
    ``k = days // (1 + spares)`` strata.

    Raises:
        ValueError: if ``seed`` < 0, a count < 0, or ``first`` is after ``last``.
    """
    if min(seed, labelled_per_year, later_per_year, spares) < 0:
        raise ValueError("seed, the nights per year and the spares must be >= 0")
    if first > last:
        raise ValueError(f"first ({first}) is after last ({last})")
    draws = 1 + spares
    nights: list[PlannedNight] = []
    for year in range(first.year, last.year + 1):
        start, end = max(first, date(year, 1, 1)), min(last, date(year, 12, 31))
        days = (end - start).days + 1
        wanted = labelled_per_year if year in LABELLED_YEARS else later_per_year
        k = min(wanted, days // draws)
        rng = np.random.default_rng([seed, year])
        bounds = [i * days // k for i in range(k + 1)] if k else [0]
        picks = [
            [
                bounds[s] + int(x)
                for x in rng.choice(bounds[s + 1] - bounds[s], draws, replace=False)
            ]
            for s in range(k)
        ]
        # Drawn after the strata, so the nights do not depend on the pilot years.
        pilot = int(rng.integers(k)) if k and year in PILOT_YEARS else -1
        for stratum, offsets in enumerate(picks):
            for rank, offset in enumerate(offsets):
                day = start + timedelta(offset)
                nights.append(PlannedNight(day, stratum, rank, rank == 0 and stratum == pilot))
    nights.sort(key=lambda n: (n.night, n.rank))
    return NightPlan(seed, first, last, labelled_per_year, later_per_year, spares, tuple(nights))


def estimate(
    plan: NightPlan,
    *,
    stamps_per_night: float = ASSUMED_STAMPS_PER_NIGHT,
    tarball_gb: float = ASSUMED_TARBALL_GB,
    download_hours: float = ASSUMED_DOWNLOAD_HOURS,
    measured: Path | None = None,
) -> dict[str, Any]:
    """The stamps, storage and download of ``plan``'s primary nights, per year, use and total.

    With ``measured`` (an ingested corpus), the kept stamps and the tarball sizes of its
    nights replace the assumptions: per year where that year has nights, else their mean
    over all years. Every year records where its numbers came from. The download time
    scales with the tarball size from ``download_hours`` per ``tarball_gb``.

    Raises:
        DataError: if ``measured`` holds an invalid night.
    """
    kept: dict[int, list[int]] = defaultdict(list)
    sizes: dict[int, list[int]] = defaultdict(list)
    if measured is not None:
        for night, meta in iter_nights(measured):
            counts, source = meta.get("counts", {}), meta.get("source", {})
            kept[night // 10_000].append(int(counts.get("kept", 0)))
            sizes[night // 10_000].append(int(source.get("compressed_bytes", 0)))
    all_kept = [k for values in kept.values() for k in values]
    all_sizes = [s for values in sizes.values() for s in values]
    years: dict[int, dict[str, Any]] = {}
    for planned in plan.primaries():
        row = years.setdefault(planned.night.year, {"nights": 0, "use": planned.use})
        row["nights"] += 1
    for year, row in sorted(years.items()):
        if kept.get(year):
            per_night, size = statistics.fmean(kept[year]), statistics.fmean(sizes[year]) / _GB
            basis = f"measured ({len(kept[year])} nights of {year})"
        elif all_kept:
            per_night, size = statistics.fmean(all_kept), statistics.fmean(all_sizes) / _GB
            basis = f"measured ({len(all_kept)} nights of other years)"
        else:
            per_night, size, basis = stamps_per_night, tarball_gb, "assumed"
        stamps = row["nights"] * per_night
        row.update(
            stamps_per_night=per_night,
            stamps=round(stamps),
            float32_gb=stamps * _ROW_BYTES / _GB,
            float16_gb=stamps * (STAMP_BYTES / 2 + INDEX_DTYPE.itemsize) / _GB,
            download_gb=row["nights"] * size,
            download_hours=row["nights"] * size / tarball_gb * download_hours,
            basis=basis,
        )
    keys = ("nights", "stamps", "float32_gb", "float16_gb", "download_gb", "download_hours")
    uses: dict[str, dict[str, Any]] = {}
    for row in years.values():
        bucket = uses.setdefault(row["use"], dict.fromkeys(keys, 0))
        for key in keys:
            bucket[key] += row[key]
    return {
        "bytes_per_stamp": _ROW_BYTES,
        "years": {str(y): row for y, row in years.items()},
        "uses": uses,
        "total": {key: sum(row[key] for row in years.values()) for key in keys},
    }


def format_estimate(result: dict[str, Any]) -> str:
    """The estimate as a text table (``plan-archive-nights`` prints it)."""
    head = (
        f"{'year':>8} {'nights':>6} {'stamps/night':>12} {'stamps':>11} {'f32 GB':>8} "
        f"{'f16 GB':>8} {'dl GB':>8} {'dl h':>6}  basis"
    )
    lines = [head]

    def line(label: str, row: dict[str, Any], per_night: str, basis: str) -> str:
        return (
            f"{label:>8} {row['nights']:>6} {per_night:>12} {row['stamps']:>11,} "
            f"{row['float32_gb']:>8.1f} {row['float16_gb']:>8.1f} {row['download_gb']:>8.0f} "
            f"{row['download_hours']:>6.1f}  {basis}"
        ).rstrip()

    for year, row in result["years"].items():
        basis = f"{row['use']}, {row['basis']}"
        lines.append(line(year, row, f"{row['stamps_per_night']:,.0f}", basis))
    for use, row in result["uses"].items():
        lines.append(line(use, row, "", ""))
    lines.append(line("total", result["total"], "", ""))
    lines.append(f"({result['bytes_per_stamp']:,} bytes per stamp with its index row)")
    return "\n".join(lines) + "\n"
