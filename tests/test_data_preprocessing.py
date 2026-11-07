"""Characterization tests of astro_simclr.data.preprocessing against the legacy fixtures.

The fixtures (tests/fixtures/data/preprocessing.npz) hold the outputs of the legacy
utils/dataset.py functions, run with the legacy numpy by
scripts/characterize/data/generate.py.
"""

from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from astro_simclr.data.preprocessing import (
    img_float2int,
    preprocess_stamps,
    repair_non_squared_stamp,
    stamp_to_uint8,
)

FIXTURES = Path(__file__).parent / "fixtures" / "data"
ATOL = 1e-6


@pytest.fixture(scope="module")
def legacy() -> dict[str, np.ndarray]:
    with np.load(FIXTURES / "preprocessing.npz") as npz:
        return {key: npz[key] for key in npz.files}


def _cases(legacy: dict[str, np.ndarray], prefix: str) -> list[str]:
    return sorted({key.split("__")[1] for key in legacy if key.startswith(prefix + "__")})


def test_fixture_records_its_provenance(legacy: dict[str, np.ndarray]) -> None:
    assert len(str(legacy["legacy_commit"])) == 40
    assert str(legacy["torch_version"]) == "not used"
    assert _cases(legacy, "repair")  # the file is not empty


def test_repair_matches_legacy(legacy: dict[str, np.ndarray]) -> None:
    names = _cases(legacy, "repair")
    assert {"short_rows_low", "short_rows_high", "short_cols_low", "short_cols_high"} <= set(names)
    for name in names:
        stamp = legacy[f"repair__{name}__in"]
        xpos, ypos = legacy[f"repair__{name}__pos"]
        expected = legacy[f"repair__{name}__out"]
        result = repair_non_squared_stamp(stamp, float(xpos), float(ypos))
        assert result.shape == (63, 63, 3), name
        assert result.dtype == expected.dtype, name
        np.testing.assert_allclose(result, expected, atol=ATOL, equal_nan=True, err_msg=name)


def test_repair_pads_on_the_side_given_by_the_position() -> None:
    stamp = np.zeros((60, 57, 3), dtype=np.float32)
    low = repair_non_squared_stamp(stamp, xpos=10.0, ypos=10.0)
    assert np.isnan(low[:3]).all() and np.isnan(low[:, :6]).all()
    assert not np.isnan(low[3:, 6:]).any()
    high = repair_non_squared_stamp(stamp, xpos=40.0, ypos=40.0)  # 40 is "high"
    assert np.isnan(high[60:]).all() and np.isnan(high[:, 57:]).all()
    assert not np.isnan(high[:60, :57]).any()


def test_repair_returns_a_full_stamp_unchanged() -> None:
    stamp = np.ones((63, 63, 3), dtype=np.float32)
    assert repair_non_squared_stamp(stamp, 0.0, 0.0) is stamp


@pytest.mark.parametrize("shape", [(64, 63, 3), (63, 70, 3), (63, 63, 2), (63, 63)])
def test_repair_rejects_stamps_it_cannot_pad(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match=r"stamp"):
        repair_non_squared_stamp(np.zeros(shape), 0.0, 0.0)


def test_preprocess_matches_legacy(legacy: dict[str, np.ndarray]) -> None:
    names = _cases(legacy, "preprocess")
    assert {"nan_inf_float32", "nan_inf_float64", "constant_float64", "all_nan_float32"} <= set(
        names
    )
    for name in names:
        stamp = legacy[f"preprocess__{name}__in"]
        for index, nan_val in enumerate((0.0, 0.5)):
            expected = legacy[f"preprocess__{name}__out{index}"]
            # An all-NaN channel makes numpy warn "All-NaN slice", as in the legacy code.
            warns: AbstractContextManager[Any] = (
                pytest.warns(RuntimeWarning) if "all_nan" in name else nullcontext()
            )
            with warns:
                result = preprocess_stamps(stamp, nan_val)
            assert result.dtype == stamp.dtype, name
            assert np.isfinite(result).all(), name
            np.testing.assert_allclose(result, expected, atol=ATOL, err_msg=f"{name} {nan_val}")


def test_preprocess_maps_nan_and_inf_to_nan_val() -> None:
    stamp = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    stamp[0, 0, 0], stamp[1, 1, 1], stamp[2, 2, 2] = np.nan, np.inf, -np.inf
    result = preprocess_stamps(stamp, nan_val=0.25)
    assert result[0, 0, 0] == result[1, 1, 1] == result[2, 2, 2] == 0.25
    assert result.min() >= 0 and result.max() < 1


def test_float2int_matches_legacy(legacy: dict[str, np.ndarray]) -> None:
    for dtype in ("float32", "float64"):
        values = legacy[f"float2int__{dtype}__in"]
        assert values.dtype == dtype
        np.testing.assert_array_equal(img_float2int(values), legacy[f"float2int__{dtype}__out"])
        np.testing.assert_array_equal(stamp_to_uint8(values), legacy[f"float2int__{dtype}__out"])


def test_float2int_depends_on_the_stored_dtype(legacy: dict[str, np.ndarray]) -> None:
    # Why the loader converts in the stored dtype: a float32 detour changes some pixels.
    values = legacy["float2int__float64__in"]
    assert (img_float2int(values) != img_float2int(values.astype(np.float32))).any()


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (np.nan, "NaN or inf"),
        (np.inf, "NaN or inf"),
        (1.01, r"in \[0, 1\]"),
        (-0.01, r"in \[0, 1\]"),
    ],
)
def test_stamp_to_uint8_rejects_invalid_values(value: float, match: str) -> None:
    stamp = np.zeros((63, 63, 3), dtype=np.float32)
    stamp[1, 2, 0] = value
    with pytest.raises(ValueError, match=match):
        stamp_to_uint8(stamp)


def test_stamp_to_uint8_rejects_integer_stamps() -> None:
    with pytest.raises(ValueError, match=r"floating point"):
        stamp_to_uint8(np.zeros((63, 63, 3), dtype=np.uint8))


def test_stamp_to_uint8_accepts_the_rounding_margin() -> None:
    stamp = np.array([-0.0019, 1.0019])
    np.testing.assert_array_equal(stamp_to_uint8(stamp), [0, 255])
