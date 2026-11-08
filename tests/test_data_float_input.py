"""Tests of astro_simclr.data.float_input against ALeRCE's own preprocessing (task 2.1).

The golden fixture tests/fixtures/data/alerce_crop.npz holds synthetic raw stamps (NaNs,
constant channels, all-NaN crops), the stored form of each (ALeRCE's normalisation over
63x63 and NaN -> 0, as in our pickles), and the outputs of ALeRCE's
image_crop_at_center -> image_normalize_by_image -> image_nan_to_num at crop 21 on both,
produced once by scripts/characterize/data/alerce_crop.py running ALeRCE's code.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from astro_simclr.augmentations import get_augmentation
from astro_simclr.data.float_input import (
    PAPER_CROP,
    approximated_stamps,
    crop_bounds,
    crop_split,
    crop_then_normalise,
    former_nan_mask,
)
from astro_simclr.data.preprocessing import stamp_to_uint8

FIXTURE = Path(__file__).parent / "fixtures" / "data" / "alerce_crop.npz"
ATOL = 1e-6

#: The fixture's arrays by key (float32 stamps, string metadata).
type Golden = dict[str, Any]

#: The one named case where the approximation is knowingly inexact: the raw channel has a
#: NaN *and* its minimum inside the crop, so the masked zeros include the real minimum.
INEXACT = "nan_and_min_inside_crop"


@pytest.fixture(scope="module")
def alerce() -> Golden:
    with np.load(FIXTURE, allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


def _index(alerce: Golden, name: str) -> int:
    return list(alerce["names"]).index(name)


def test_fixture_records_its_provenance(alerce: Golden) -> None:
    assert str(alerce["alerce_commit"]) == "a3705acb58ce2e45c1f9b240b824c51260295a99"
    assert str(alerce["alerce_path"]).endswith("data_loaders/ztf_preprocessor.py")
    assert len(str(alerce["alerce_sha256"])) == 64
    names = set(alerce["names"].tolist())
    assert {"nan_inside_crop", "constant_channel", "all_nan_crop", "all_nan_channel"} <= names
    raw = alerce["raw"]
    assert raw.dtype == np.float32 and raw.shape[1:] == (63, 63, 3)
    assert np.isnan(raw).any()
    assert alerce["alerce_crop21"].shape == (len(raw), 21, 21, 3)


def test_raw_stamps_match_alerce(alerce: Golden) -> None:
    # On ALeRCE's own input (raw stamps with NaNs) the port is exact: same ops, same dtype.
    raw = alerce["raw"]
    result = crop_then_normalise(raw, nan_approximation=False)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, alerce["alerce_crop21"], rtol=0, atol=ATOL)


def test_stored_stamps_without_the_approximation_match_alerce_on_them(
    alerce: Golden,
) -> None:
    result = crop_then_normalise(alerce["stored"], nan_approximation=False)
    np.testing.assert_allclose(result, alerce["alerce_crop21_of_stored"], rtol=0, atol=ATOL)


def test_the_approximation_recovers_alerce_from_the_stored_stamps(
    alerce: Golden,
) -> None:
    # The float path's real job: from the stored (63x63-normalised, NaN -> 0) stamps, give
    # what ALeRCE gave from the raw stamps. Min-max is affine-invariant, so only rounding
    # (one float32 ulp) separates them, except for the documented inexact case.
    stored, expected = alerce["stored"], alerce["alerce_crop21"]
    result = crop_then_normalise(stored)
    error = np.abs(result - expected).max(axis=(1, 2, 3))
    inexact = _index(alerce, INEXACT)
    assert error[inexact] > 1e-3
    assert approximated_stamps(stored[inexact : inexact + 1]).all()
    others = np.delete(error, inexact)
    assert others.max() <= ATOL

    # And the approximation is needed: without it, NaNs inside the crop give another input.
    plain = crop_then_normalise(stored, nan_approximation=False)
    case = _index(alerce, "nan_inside_crop")
    assert np.abs(plain[case] - expected[case]).max() > 1e-3


def test_the_mask_covers_every_former_nan(alerce: Golden) -> None:
    raw, stored = alerce["raw"], alerce["stored"]
    mask = former_nan_mask(stored)
    assert mask.shape == stored.shape
    nan = np.isnan(raw)
    assert (mask | ~nan).all()  # every raw NaN is masked
    clean = _index(alerce, "clean")
    assert not mask[clean].any()  # one zero per channel: nothing masked


@pytest.mark.parametrize("case", ["constant_channel", "constant_crop", "all_nan_crop"])
def test_degenerate_crops_become_zero(case: str, alerce: Golden) -> None:
    i = _index(alerce, case)
    channel = {"constant_channel": 1, "constant_crop": 2, "all_nan_crop": 1}[case]
    for stamps in (alerce["raw"], alerce["stored"]):
        result = crop_then_normalise(stamps[i], nan_approximation=stamps is alerce["stored"])
        assert (result[..., channel] == 0).all()
        assert np.isclose(result[..., channel - 1].max(), 1)  # the other channels are kept


def test_output_properties(alerce: Golden) -> None:
    stored = alerce["stored"]
    before = stored.copy()
    result = crop_then_normalise(stored)
    np.testing.assert_array_equal(stored, before)  # the input is not modified
    assert result.shape == (len(stored), PAPER_CROP, PAPER_CROP, 3)
    assert not np.isnan(result).any()
    lows, highs = result.min(axis=(1, 2)), result.max(axis=(1, 2))
    assert (lows == 0).all()
    assert ((highs == 0) | np.isclose(highs, 1, rtol=0, atol=ATOL)).all()
    assert (highs == 0).any()  # the degenerate cases are in the fixture
    # One stamp gives the same as its row of the batch, and float64 stays float64.
    np.testing.assert_array_equal(crop_then_normalise(stored[3]), result[3])
    as64 = crop_then_normalise(stored.astype(np.float64))
    assert as64.dtype == np.float64
    np.testing.assert_allclose(as64, result, rtol=0, atol=ATOL)


def test_crop_bounds() -> None:
    assert crop_bounds(63, 21) == (21, 42)
    assert crop_bounds(63, 27) == (18, 45)
    assert crop_bounds(63, 63) == (0, 63)
    assert crop_bounds(64, 22) == (21, 43)
    for side, crop in ((63, 20), (63, 0), (63, 65)):
        with pytest.raises(ValueError, match="crop"):
            crop_bounds(side, crop)


def test_crop_then_normalise_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="floating point"):
        integers: Any = np.zeros((63, 63, 3), dtype=np.uint8)  # rejected at run time
        crop_then_normalise(integers)
    with pytest.raises(ValueError, match="H == W"):
        crop_then_normalise(np.zeros((63, 61, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="H == W"):
        crop_then_normalise(np.zeros((63, 63), dtype=np.float32))
    with pytest.raises(ValueError, match="parity"):
        crop_then_normalise(np.zeros((63, 63, 3), dtype=np.float32), crop=20)


@pytest.mark.parametrize("chunk", [1, 5, 4096])
def test_crop_split_matches_one_call(chunk: int, alerce: Golden) -> None:
    stored = alerce["stored"]
    for approx in (True, False):
        expected = crop_then_normalise(stored, nan_approximation=approx)
        result = crop_split(stored, nan_approximation=approx, chunk=chunk)
        np.testing.assert_array_equal(result, expected)
    assert crop_split(stored, 27).shape == (len(stored), 27, 27, 3)


def test_crop_split_rejects_bad_input(alerce: Golden) -> None:
    stored = alerce["stored"]
    with pytest.raises(ValueError, match="float32"):
        crop_split(stored.astype(np.float64))
    with pytest.raises(ValueError, match="float32"):
        crop_split(stored[:, :21])
    with pytest.raises(ValueError, match="chunk"):
        crop_split(stored, chunk=0)
    with pytest.raises(ValueError, match="parity"):
        crop_split(stored, 20)


def test_approximated_stamps(alerce: Golden) -> None:
    stored = alerce["stored"]
    flags = approximated_stamps(stored)
    assert flags.shape == (len(stored),) and flags.any() and not flags.all()
    np.testing.assert_array_equal(approximated_stamps(stored, chunk=3), flags)
    # Only flagged stamps can differ between the two settings.
    on, off = crop_then_normalise(stored), crop_then_normalise(stored, nan_approximation=False)
    differ = np.abs(on - off).max(axis=(1, 2, 3)) > 0
    assert (flags | ~differ).all()
    assert not flags[_index(alerce, "nan_outside_crop")]  # masked zeros, but outside
    with pytest.raises(ValueError, match="H == W"):
        approximated_stamps(stored[0])
    with pytest.raises(ValueError, match="chunk"):
        approximated_stamps(stored, chunk=0)


def test_uint8_and_float_crops_cover_the_same_pixels(
    alerce: Golden,
) -> None:
    # The alignment check of 2.1: the phase-1 uint8 path (center_crop at 27, torchvision's
    # CenterCrop) and the float crop of 21 are concentric, 3 pixels apart on every side,
    # which is the paper's zero padding back to 27.
    view = get_augmentation("center_crop", 27)
    begin, end = crop_bounds(63, PAPER_CROP)
    for stamp in alerce["stored"]:
        uint8 = stamp_to_uint8(stamp)
        crop27 = np.round(view.view(uint8).numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        np.testing.assert_array_equal(crop27[3:24, 3:24], uint8[begin:end, begin:end])
        # Same pixels, so the float crop and the re-normalised uint8 crop agree to within
        # the uint8 quantisation (half a step of 1/255 on each end, rescaled).
        floats = crop_then_normalise(stamp, nan_approximation=False)
        quantised = crop_then_normalise(
            crop27[3:24, 3:24].astype(np.float64), PAPER_CROP, nan_approximation=False
        )
        span = np.ptp(uint8[begin:end, begin:end].astype(np.float64), axis=(0, 1)) / 255
        ok = span > 0.05  # skip near-constant crops, where quantisation dominates
        bound = 2 / 255 / np.where(ok, span, 1)
        assert (np.abs(floats - quantised).max(axis=(0, 1))[ok] <= bound[ok]).all()

    # The check is discriminating: one pixel off and the uint8 crop no longer matches.
    uint8 = stamp_to_uint8(alerce["stored"][0])
    crop27 = np.round(view.view(uint8).numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    assert not np.array_equal(crop27[4:25, 3:24], uint8[begin:end, begin:end])
