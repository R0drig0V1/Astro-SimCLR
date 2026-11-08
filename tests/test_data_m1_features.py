"""Tests of astro_simclr.data.m1_features and ``astro-simclr check-features`` (task 2.1)."""

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from astro_simclr.cli import main
from astro_simclr.data import FEATURE_NAMES, SplitData, StampData
from astro_simclr.data.m1_features import (
    FLOOR_FEATURES,
    INTEGER_FEATURES,
    M1_FEATURE_NORM_STATS,
    check_destandardisation,
    destandardise,
)
from astro_simclr.data.prepared import write_prepared
from astro_simclr.data.schema import DISCARDED_FEATURES

COLUMN = {name: i for i, name in enumerate(FEATURE_NAMES)}


def _physical(n: int, seed: int = 0) -> dict[str, npt.NDArray[np.float64]]:
    """Plausible raw values of the 23 features: counts, -1 floors, a binary isdiffpos."""
    rng = np.random.default_rng(seed)
    values = {name: rng.normal(mean, std, n) for name, (mean, std) in M1_FEATURE_NORM_STATS.items()}
    for name in INTEGER_FEATURES:
        values[name] = rng.integers(0, 3000, n).astype(np.float64)
    for name in FLOOR_FEATURES:
        values[name] = np.where(rng.random(n) < 0.1, -1.0, rng.uniform(0, 1, n))
    values["isdiffpos"] = rng.integers(0, 2, n).astype(np.float64)
    return values


def _standardised(
    values: dict[str, npt.NDArray[np.float64]],
    stats: dict[str, tuple[float, float]] = M1_FEATURE_NORM_STATS,
) -> npt.NDArray[np.float64]:
    n = len(next(iter(values.values())))
    features = np.random.default_rng(1).normal(size=(n, len(FEATURE_NAMES)))  # 13-15 random
    for name, (mean, std) in stats.items():
        features[:, COLUMN[name]] = (values[name] - mean) / std
    return features


def test_the_statistics_cover_the_23_kept_features_in_order() -> None:
    kept = [name for i, name in enumerate(FEATURE_NAMES) if i not in DISCARDED_FEATURES]
    assert list(M1_FEATURE_NORM_STATS) == kept
    assert M1_FEATURE_NORM_STATS["isdiffpos"][0] == 0.0  # scaled only, not centred
    assert all(std > 0 for _, std in M1_FEATURE_NORM_STATS.values())


def test_destandardise_inverts_the_standardisation() -> None:
    values = _physical(200)
    restored = destandardise(_standardised(values))
    assert list(restored) == list(M1_FEATURE_NORM_STATS)
    for name, column in restored.items():
        np.testing.assert_allclose(column, values[name], rtol=0, atol=1e-9)
    with pytest.raises(ValueError, match="26"):
        destandardise(np.zeros((3, 23)))


def test_check_passes_on_m1_standardised_features() -> None:
    checks = check_destandardisation(_standardised(_physical(500)))
    assert [(c.feature, c.test) for c in checks] == [
        *((name, "integer") for name in INTEGER_FEATURES),
        *((name, "floor -1") for name in FLOOR_FEATURES),
        ("isdiffpos", "binary"),
    ]
    assert all(c.passed for c in checks), checks
    assert checks[-1].detail == "values [0.0, 1.0]"


def test_check_detects_other_statistics() -> None:
    # A std off by one part in a million breaks the integer test of ncovhist.
    stats = dict(M1_FEATURE_NORM_STATS)
    mean, std = stats["ncovhist"]
    stats["ncovhist"] = (mean, std * (1 + 1e-6))
    failed = {
        c.feature
        for c in check_destandardisation(_standardised(_physical(500), stats))
        if not c.passed
    }
    assert failed == {"ncovhist"}

    values = _physical(500)
    values["sgscore2"][0] = -1.5  # below the floor
    values["isdiffpos"][0] = 0.5  # not binary
    failed = {c.feature for c in check_destandardisation(_standardised(values)) if not c.passed}
    assert failed == {"sgscore2", "isdiffpos"}


def test_check_rejects_empty_features() -> None:
    with pytest.raises(ValueError, match="empty"):
        check_destandardisation(np.zeros((0, 26)))


def _layout(tmp_path: Path, features: npt.NDArray[np.float64], *, pretraining: bool) -> Path:
    n = len(features)
    images = np.zeros((n, 63, 63, 3), dtype=np.uint8)
    labels = np.arange(n, dtype=np.int64) % 5
    labelled = SplitData(images, labels, features)
    train = SplitData(images, None, None) if pretraining else labelled
    data = StampData(
        "pretraining" if pretraining else "classifier",
        {"Train": train, "Validation": labelled, "Test": labelled},
    )
    source = {"name": "x.pkl", "size_bytes": 0, "sha256": "0" * 64}
    return write_prepared(data, tmp_path / ("pre" if pretraining else "cls"), source)


def test_cli_check_features(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = _layout(tmp_path, _standardised(_physical(50)), pretraining=False)
    capsys.readouterr()
    assert main(["check-features", str(good)]) == 0
    out = capsys.readouterr().out
    assert "PASS  ncovhist        integer" in out and out.endswith("PASS\n")
    assert "Validation:" in out

    assert main(["check-features", "--json", str(good)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is True and set(report["splits"]) == {"Train", "Validation", "Test"}

    values = _physical(50)
    values["ndethist"] += 0.25
    bad = _layout(tmp_path, _standardised(values), pretraining=True)
    assert main(["check-features", str(bad)]) == 1
    out = capsys.readouterr().out
    assert "FAIL  ndethist" in out and out.endswith("FAIL\n")
    assert "Train:" not in out  # the unlabelled pretraining Train split has no features

    assert main(["check-features", str(tmp_path / "missing")]) == 1  # a DataError
