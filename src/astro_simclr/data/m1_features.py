"""M1's feature normalisation statistics, and the de-standardisation check (task 2.1).

The pickles store the 26 alert features already standardised (``docs/data.md``). The paper's
model, M1, standardised its 23 features with the per-feature mean and std (``np.mean``,
``np.std`` with ddof 0) of its Train split, saved next to the checkpoint as
``feature_norm_stats.pkl``. :data:`M1_FEATURE_NORM_STATS` records those values.

Provenance: ``alercebroker/pipeline`` @ ``a3705acb58ce2e45c1f9b240b824c51260295a99``,
``stamp_classifier_step/model/results/staging_model/DeepHits_EntropyRegBeta0.5000_batch64_
lr0.00100_droprate0.5000_inputsize21_filtersize5_0_20200708-160759/feature_norm_stats.pkl``
(1,418 bytes, SHA-256
``65a5611c0ca6762622788b546d82097e0bc6a0c95798eb525c0da77253ff8064``). The file was never
unpickled: ``scripts/characterize/data/m1_feature_stats.py`` downloads it, checks the
SHA-256 and reads the float64 bytes with ``pickletools``. The values below are copied from
its output, at full precision (``repr``). ``isdiffpos`` has mean 0 (stored as an int64
scalar): ALeRCE only scaled it.

:func:`check_destandardisation` asks whether a file's features were standardised with
these statistics: after ``x * std + mean`` the integer features must be integers, the
clipped ``sgscore``/``distpsnr`` features must have a floor of -1 (ALeRCE's -999 sentinels
clipped by Table A1), and ``isdiffpos`` must be 0 or 1. Statistics that differ even in the
sixth digit break the integer test (``ncovhist`` has a std of 425). On ``red``, all three
splits pass with integer deviations of about 1e-13 (``docs/data.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from astro_simclr.data.schema import DISCARDED_FEATURES, FEATURE_NAMES

__all__ = [
    "FLOOR_FEATURES",
    "INTEGER_FEATURES",
    "M1_FEATURE_NORM_STATS",
    "FeatureCheck",
    "check_destandardisation",
    "destandardise",
]

#: Feature name -> (mean, std) of M1's Train split (see the module docstring), in the order
#: of the file, which is the order of the 23 kept features.
M1_FEATURE_NORM_STATS: dict[str, tuple[float, float]] = {
    "sgscore1": (0.7041255430924432, 0.3799197334571002),
    "distpsnr1": (3.941122253808356, 5.949694365436986),
    "sgscore2": (0.462221382971118, 0.4261297122366326),
    "distpsnr2": (10.893805420307675, 7.159838762869845),
    "sgscore3": (0.39136703661723216, 0.528184045264236),
    "distpsnr3": (13.665451778592196, 8.042496801134273),
    "isdiffpos": (0.0, 0.47113361884526406),
    "fwhm": (2.617771189810078, 1.6291465825377438),
    "magpsf": (18.91661276279921, 1.408401373184911),
    "sigmapsf": (0.14152754134129397, 0.060091724758989724),
    "ra": (181.0820148804342, 96.23590348616496),
    "dec": (22.153968169154407, 24.427197378027234),
    "diffmaglim": (20.21664878176659, 0.601868085019363),
    "classtar": (0.856637110442036, 0.2031539876796047),
    "ndethist": (5.344072419654505, 6.552626738216146),
    "ncovhist": (291.40510732497927, 424.5186709286612),
    "ecl_lat": (21.683213451976407, 27.232781418645576),
    "ecl_long": (166.73433625354693, 95.03070518723176),
    "gal_lat": (16.996460092087215, 38.88456173989733),
    "gal_long": (140.10421536618335, 84.23470064762579),
    "non_detections": (273.5501838162628, 390.1727524999179),
    "chinr": (1.179799640266478, 1.5123727927432673),
    "sharpnr": (-0.028653200003162668, 0.20231731177535472),
}

#: Counts: integers before standardisation.
INTEGER_FEATURES: tuple[str, ...] = ("ndethist", "ncovhist", "non_detections")
#: Clipped at -1 from below (the -999 "no source" sentinel, Table A1).
FLOOR_FEATURES: tuple[str, ...] = (
    "sgscore1",
    "distpsnr1",
    "sgscore2",
    "distpsnr2",
    "sgscore3",
    "distpsnr3",
)
#: Binary: 0 or 1 before scaling.
_BINARY = "isdiffpos"

#: Absolute tolerance of the integer, floor and binary tests. float64 standardisation and
#: its inverse lose about 1e-13 on these magnitudes; a wrong mean or std moves them by far
#: more than this.
_ATOL = 1e-6


@dataclass(frozen=True, slots=True)
class FeatureCheck:
    """One test of :func:`check_destandardisation` on one feature.

    Attributes:
        feature: the feature name.
        test: ``integer``, ``floor -1`` or ``binary``.
        passed: whether every value passed.
        detail: the measured quantity (max deviation from an integer, the minimum, or
            the distinct values), for the report.
    """

    feature: str
    test: str
    passed: bool
    detail: str


def destandardise(features: npt.NDArray[np.floating[Any]]) -> dict[str, npt.NDArray[np.float64]]:
    """Undo M1's standardisation: ``x * std + mean`` for each of the 23 kept features.

    ``features`` is the stored ``(N, 26)`` array (all :data:`FEATURE_NAMES` columns); the
    discarded columns (``rb``, ``distnr``, ``magnr``) have no M1 statistics and are left
    out. Returns feature name -> float64 ``(N,)`` values, in M1's order.

    Raises:
        ValueError: if ``features`` is not ``(N, 26)``.
    """
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"features must be (N, {len(FEATURE_NAMES)}), got {features.shape}")
    kept = [name for i, name in enumerate(FEATURE_NAMES) if i not in DISCARDED_FEATURES]
    assert kept == list(M1_FEATURE_NORM_STATS)  # noqa: S101 - a schema invariant
    columns = {name: i for i, name in enumerate(FEATURE_NAMES)}
    out: dict[str, npt.NDArray[np.float64]] = {}
    for name, (mean, std) in M1_FEATURE_NORM_STATS.items():
        out[name] = np.asarray(features[:, columns[name]], dtype=np.float64) * std + mean
    return out


def check_destandardisation(features: npt.NDArray[np.floating[Any]]) -> list[FeatureCheck]:
    """Check that ``features`` (stored ``(N, 26)``) were standardised with M1's statistics.

    Runs the tests of the module docstring on the de-standardised values
    (:func:`destandardise`) and returns one :class:`FeatureCheck` per test: integers for
    :data:`INTEGER_FEATURES`, a minimum of -1 or more for :data:`FLOOR_FEATURES`, and
    {0, 1} for ``isdiffpos``, all within 1e-6.

    Raises:
        ValueError: if ``features`` is not ``(N, 26)`` or is empty.
    """
    if len(features) == 0:
        raise ValueError("features is empty")
    values = destandardise(features)
    checks: list[FeatureCheck] = []
    for name in INTEGER_FEATURES:
        deviation = float(np.abs(values[name] - np.round(values[name])).max())
        checks.append(
            FeatureCheck(name, "integer", deviation <= _ATOL, f"max |x - round(x)| {deviation:.1e}")
        )
    for name in FLOOR_FEATURES:
        low = float(values[name].min())
        checks.append(FeatureCheck(name, "floor -1", low >= -1 - _ATOL, f"min {low:.6g}"))
    binary = values[_BINARY]
    near = np.minimum(np.abs(binary), np.abs(binary - 1))
    distinct = sorted({round(float(v), 6) for v in np.unique(np.round(binary, 6))})
    shown = distinct if len(distinct) <= 4 else [*distinct[:4], "..."]
    checks.append(FeatureCheck(_BINARY, "binary", bool(near.max() <= _ATOL), f"values {shown}"))
    return checks
