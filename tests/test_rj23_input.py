"""Tests for the RJ23 input path (``astro_simclr.data.rj23_input``, task 3.3).

Hand-computed normalisation cases (NaN, inf, constant channels, ``|x| > 1e10`` in both
modes), the mask, the shape and exact values of each view on small synthetic stamps, the
multi-scale ordering and fields of view, TF 'same' pooling against a hand-computed array
and against torch, the position, and the rotation and flip behaviour of every view.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
import torch

from astro_simclr.data import rj23_input as rj

F64 = npt.NDArray[np.float64]
AnyFloat = npt.NDArray[np.floating]

# The eight elements of the dihedral group D4 on (N, H, W, C) arrays.
D4: dict[str, Callable[[AnyFloat], AnyFloat]] = {
    "identity": lambda a: a,
    "rot90": lambda a: np.rot90(a, 1, axes=(1, 2)),
    "rot180": lambda a: np.rot90(a, 2, axes=(1, 2)),
    "rot270": lambda a: np.rot90(a, 3, axes=(1, 2)),
    "flip_rows": lambda a: a[:, ::-1],
    "flip_cols": lambda a: a[:, :, ::-1],
    "transpose": lambda a: np.swapaxes(a, 1, 2),
    "anti_transpose": lambda a: np.rot90(np.swapaxes(a, 1, 2), 2, axes=(1, 2)),
}


def raw_stamps(n: int = 4, seed: int = 0) -> F64:
    """Random raw cutouts with signs, a wide dynamic range and a few invalid pixels."""
    rng = np.random.default_rng(seed)
    stamps = rng.normal(size=(n, 63, 63, 3)) * rng.uniform(0.5, 500.0, size=(n, 1, 1, 3))
    stamps[0, :4, :6, 0] = np.nan
    stamps[min(1, n - 1), 10, 20, 1] = np.inf
    stamps[min(1, n - 1), 11, 20, 2] = -np.inf
    return stamps


def channel_from(values: F64) -> F64:
    """One raw stamp whose three channels are ``values`` (3969 pixels) laid out row-major."""
    channel = np.asarray(values, dtype=np.float64).reshape(63, 63)
    return np.repeat(channel[None, :, :, None], 3, axis=3)


def with_padding(values: list[float]) -> F64:
    """``values`` followed by NaN up to 3969 pixels."""
    flat = np.full(63 * 63, np.nan)
    flat[: len(values)] = values
    return flat


# ----------------------------------------------------------------------------------------
# Normalisation and mask


def test_normalise_linear_ramp_hand_computed() -> None:
    # |x| = 0, 1, ..., 3968 with alternating signs: min 0, p99 at the virtual index
    # 0.99 * 3968 = 3928.32, so between 3928 and 3929.
    k = np.arange(63 * 63, dtype=np.float64)
    values = np.where(k % 2 == 0, k, -k)
    stamps = channel_from(values)
    minimum, p99 = rj.channel_statistics(stamps)
    np.testing.assert_allclose(minimum, 0.0)
    np.testing.assert_allclose(p99, 3928.32, rtol=1e-12)
    out = rj.normalise(stamps)
    expected = np.clip(values / (3928.32 + 1e-8), -2, 2).reshape(63, 63)
    np.testing.assert_allclose(out[0, :, :, 0], expected, rtol=1e-6)
    assert out.dtype == np.float32
    assert out.shape == (1, 63, 63, 6)
    assert not out[..., 3:].any()


def test_normalise_skips_nan_and_subtracts_min_abs() -> None:
    # 101 valid pixels 5, 6, ..., 105 (the rest NaN): min|x| = 5, p99 at index 99 = 104.
    values = with_padding([float(v) for v in range(5, 106)])
    values[0] = -5.0  # |x| is still 5, but (x - 5) / 99 is now -10 / 99
    stamps = channel_from(values)
    minimum, p99 = rj.channel_statistics(stamps)
    np.testing.assert_allclose(minimum, 5.0)
    np.testing.assert_allclose(p99, 104.0)
    out = rj.normalise(stamps)[0, ..., 0].reshape(-1)
    scale = 99.0 + 1e-8
    np.testing.assert_allclose(out[0], -10.0 / scale, rtol=1e-6)
    np.testing.assert_allclose(out[1], (6.0 - 5.0) / scale, rtol=1e-6)
    np.testing.assert_allclose(out[100], (105.0 - 5.0) / scale, rtol=1e-6)
    assert np.all(out[101:] == 0.0)  # NaN -> 0
    mask = rj.normalise(stamps)[0, ..., 3].reshape(-1)
    assert np.all(mask[:101] == 0.0)
    assert np.all(mask[101:] == 1.0)


def test_normalise_clips_to_two() -> None:
    values = with_padding([0.0] * 99 + [1.0, 50.0])  # p99 = 0.99*100 -> index 99 = 1.0
    out = rj.normalise(channel_from(values))[0, ..., 0].reshape(-1)
    assert out[99] == pytest.approx(1.0, rel=1e-6)
    assert out[100] == 2.0
    negative = with_padding([0.0] * 99 + [1.0, -50.0])
    assert rj.normalise(channel_from(negative))[0, ..., 0].reshape(-1)[100] == -2.0


def test_infinities_are_masked_zeroed_and_skipped() -> None:
    values = with_padding([float(v) for v in range(101)])
    values[101] = np.inf
    values[102] = -np.inf
    stamps = channel_from(values)
    minimum, p99 = rj.channel_statistics(stamps)
    np.testing.assert_allclose(p99, 99.0)
    np.testing.assert_allclose(minimum, 0.0)
    out = rj.normalise(stamps)[0].reshape(-1, 6)
    assert out[101, 0] == 0.0
    assert out[102, 0] == 0.0
    assert out[101, 3] == 1.0
    assert out[102, 3] == 1.0
    assert out[100, 0] == pytest.approx(100.0 / 99.0, rel=1e-6)


def test_constant_channels() -> None:
    stamps = np.zeros((3, 63, 63, 3))
    stamps[0] = 3.0  # min = p99 = 3: (3 - 3) / 1e-8 = 0
    stamps[1] = -3.0  # |x| = 3: (-3 - 3) / 1e-8 saturates at -2
    stamps[2] = 0.0
    out = rj.normalise(stamps)
    assert np.all(out[0, ..., :3] == 0.0)
    assert np.all(out[1, ..., :3] == -2.0)
    assert np.all(out[2, ..., :3] == 0.0)
    assert not out[..., 3:].any()


def test_all_invalid_channel_is_zero_and_fully_masked() -> None:
    stamps = raw_stamps(2)
    stamps[1, ..., 2] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "All-NaN slice" or division warnings
        minimum, p99 = rj.channel_statistics(stamps)
        out = rj.normalise(stamps)
    assert np.isnan(minimum[1, 2])
    assert np.isnan(p99[1, 2])
    assert np.all(out[1, ..., 2] == 0.0)
    assert np.all(out[1, ..., 5] == 1.0)


@pytest.mark.parametrize("large_values", ["paper", "code"])
def test_large_values_both_modes(large_values: rj.LargeValues) -> None:
    values = with_padding([float(v) for v in range(101)] + [1e11, -1e12])
    stamps = channel_from(values)
    minimum, p99 = rj.channel_statistics(stamps, large_values=large_values)
    # Out of the statistics in both modes.
    np.testing.assert_allclose(minimum, 0.0)
    np.testing.assert_allclose(p99, 99.0)
    out = rj.normalise(stamps, large_values=large_values)[0].reshape(-1, 6)
    mask = rj.invalid_mask(stamps, large_values=large_values)[0].reshape(-1, 3)
    if large_values == "paper":
        assert out[101, 0] == 0.0
        assert out[102, 0] == 0.0
        assert out[101, 3] == 1.0
        assert out[102, 3] == 1.0
        assert mask[101, 0]
    else:  # RJ23's code: normalised and clipped, not masked
        assert out[101, 0] == 2.0
        assert out[102, 0] == -2.0
        assert out[101, 3] == 0.0
        assert out[102, 3] == 0.0
        assert not mask[101, 0]
    # A value just at the threshold is a normal pixel in both modes.
    assert not rj.invalid_mask(channel_from(with_padding([1e10])), large_values=large_values)[
        0, 0, 0, 0
    ]


def test_large_values_all_large_channel_is_zero_in_code_mode() -> None:
    stamps = np.full((1, 63, 63, 3), 1e11)
    out = rj.normalise(stamps, large_values="code")
    assert np.all(out == 0.0)
    assert np.all(rj.normalise(stamps)[..., 3:] == 1.0)


def test_mask_channels_and_order() -> None:
    stamps = raw_stamps(2)
    mask = rj.invalid_mask(stamps)
    expected = ~np.isfinite(stamps)
    np.testing.assert_array_equal(mask, expected)
    out = rj.normalise(stamps)
    np.testing.assert_array_equal(out[..., 3:], expected.astype(np.float32))
    assert rj.CHANNEL_NAMES == ("sci", "ref", "diff", "m_sci", "m_ref", "m_diff")
    # Each normalised channel uses its own statistics, in sci, ref, diff order.
    for c in range(3):
        alone = np.repeat(stamps[..., c : c + 1], 3, axis=3)
        np.testing.assert_array_equal(rj.normalise(alone)[..., 0], out[..., c])


def test_statistics_match_numpy_nanpercentile() -> None:
    stamps = raw_stamps(6, seed=3)
    stamps[2, 5:9, 5:9, 1] = 2e10
    minimum, p99 = rj.channel_statistics(stamps)
    absolute = np.abs(stamps).reshape(6, -1, 3)
    absolute[~np.isfinite(absolute) | (absolute > 1e10)] = np.nan
    np.testing.assert_array_equal(p99, np.nanpercentile(absolute, 99, axis=1))
    np.testing.assert_array_equal(minimum, np.nanmin(absolute, axis=1))


def test_normalise_is_chunk_independent_and_deterministic() -> None:
    stamps = raw_stamps(7, seed=1).astype(np.float32)
    whole = rj.normalise(stamps)
    np.testing.assert_array_equal(whole, rj.normalise(stamps, chunk_size=2))
    np.testing.assert_array_equal(whole, rj.normalise(stamps, chunk_size=1))
    views = rj.rj23_views(stamps)
    chunked = rj.rj23_views(stamps, chunk_size=3)
    for name in rj.VIEW_NAMES:
        np.testing.assert_array_equal(views[name], chunked[name])


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: rj.normalise(np.zeros((2, 63, 63))), "shape"),
        (lambda: rj.normalise(np.zeros((2, 64, 63, 3))), "shape"),
        (lambda: rj.normalise(np.zeros((2, 63, 63, 3), dtype=np.int16)), "float"),
        (lambda: rj.normalise(np.zeros((2, 63, 63, 3)), chunk_size=0), "chunk_size"),
        (lambda: rj.invalid_mask(np.zeros((1, 63, 63, 3)), large_values="x"), "large_values"),  # type: ignore[arg-type]
        (lambda: rj.rj23_views(np.zeros((1, 63, 63, 3)), ["full", "full"]), "distinct"),
        (lambda: rj.rj23_views(np.zeros((1, 63, 63, 3)), ["big"]), "distinct"),  # type: ignore[list-item]
        (lambda: rj.full_view(np.zeros((1, 63, 63, 3))), "normalised"),
        (lambda: rj.crop_offset(63, 64), "crop"),
        (lambda: rj.tf_same_avg_pool(np.zeros((63, 63, 3))), "images"),
        (lambda: rj.tf_same_avg_pool(np.zeros((1, 63, 63, 3)), 0), "pool"),
        (lambda: rj.channels_first(np.zeros((63, 63, 3), dtype=np.float32)), "images"),
    ],
)
def test_input_errors(call: Callable[[], object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


# ----------------------------------------------------------------------------------------
# TF 'same' average pooling


def test_tf_same_pool_hand_computed() -> None:
    # 5x5 -> 3x3: TF pads one row and column at the end, and averages valid pixels only.
    image = np.arange(25, dtype=np.float64).reshape(1, 5, 5, 1)
    expected = np.array(
        [
            [(0 + 1 + 5 + 6) / 4, (2 + 3 + 7 + 8) / 4, (4 + 9) / 2],
            [(10 + 11 + 15 + 16) / 4, (12 + 13 + 17 + 18) / 4, (14 + 19) / 2],
            [(20 + 21) / 2, (22 + 23) / 2, 24.0],
        ]
    )
    pooled = rj.tf_same_avg_pool(image, 2)
    assert pooled.dtype == np.float32
    np.testing.assert_allclose(pooled[0, :, :, 0], expected, rtol=1e-7)
    # 6 pooled by 4: out 2, 2 pixels of padding split 1 before and 1 after.
    row = np.arange(6, dtype=np.float64).reshape(1, 6, 1, 1) * np.ones((1, 1, 4, 1))
    np.testing.assert_allclose(rj.tf_same_avg_pool(row, 4)[0, :, 0, 0], [1.0, 4.0])


@pytest.mark.parametrize(("size", "pool"), [(63, 2), (32, 2), (16, 2), (63, 4), (64, 4)])
def test_tf_same_pool_matches_torch_when_padding_is_at_the_end(size: int, pool: int) -> None:
    # torch's ceil_mode windows run off the end and divide by the valid count, which is TF
    # 'same' whenever TF pads at most one pixel (all at the end).
    images = np.random.default_rng(size + pool).normal(size=(3, size, size, 6))
    reference = torch.nn.functional.avg_pool2d(
        torch.from_numpy(images.transpose(0, 3, 1, 2).copy()), pool, pool, ceil_mode=True
    )
    np.testing.assert_allclose(
        rj.tf_same_avg_pool(images, pool), reference.numpy().transpose(0, 2, 3, 1), atol=1e-6
    )


def test_tf_same_pool_differs_from_torch_when_tf_pads_before() -> None:
    images = np.random.default_rng(0).normal(size=(1, 62, 62, 1))
    reference = torch.nn.functional.avg_pool2d(
        torch.from_numpy(images.transpose(0, 3, 1, 2).copy()), 4, 4, ceil_mode=True
    )
    assert not np.allclose(rj.tf_same_avg_pool(images, 4), reference.numpy().transpose(0, 2, 3, 1))


def test_one_4x4_pool_differs_from_two_2x2_only_in_the_last_row_and_column() -> None:
    images = np.random.default_rng(5).normal(size=(2, 63, 63, 6))
    once = rj.tf_same_avg_pool(images, 4)
    twice = rj.tf_same_avg_pool(rj.tf_same_avg_pool(images, 2), 2)
    np.testing.assert_allclose(once[:, :15, :15], twice[:, :15, :15], atol=1e-6)
    assert not np.allclose(once[:, 15, :], twice[:, 15, :])
    # RJ23's low-resolution view is the single 4x4 pooling.
    normalised = rj.normalise(raw_stamps(2))
    np.testing.assert_array_equal(rj.lowres_view(normalised), rj.tf_same_avg_pool(normalised, 4))


# ----------------------------------------------------------------------------------------
# Views


def coordinate_stamp() -> F64:
    """A normalised-shaped (1, 63, 63, 6) stamp with value ``100 * row + col + channel / 10``."""
    rows, cols, chans = np.meshgrid(np.arange(63), np.arange(63), np.arange(6), indexing="ij")
    return (100.0 * rows + cols + chans / 10.0)[None].astype(np.float64)


def test_view_shapes_and_dtypes() -> None:
    views = rj.rj23_views(raw_stamps(3))
    for name in rj.VIEW_NAMES:
        assert views[name].shape == (3, *rj.VIEW_SHAPES[name])
        assert views[name].dtype == np.float32
        assert np.all(np.isfinite(views[name]))
    subset = rj.rj23_views(raw_stamps(3), ["multiscale"])
    assert list(subset) == ["multiscale"]
    np.testing.assert_array_equal(subset["multiscale"], views["multiscale"])


def test_full_and_cropped_view_values() -> None:
    stamp = coordinate_stamp()
    full = rj.full_view(stamp)
    np.testing.assert_array_equal(full, stamp.astype(np.float32))
    assert rj.crop_offset(63, 16) == rj.CROP16[0] == 24
    assert rj.crop_offset(63, 8) == rj.CROP8[0] == 28
    cropped = rj.cropped_view(stamp)
    assert cropped.shape == (1, 16, 16, 6)
    assert cropped[0, 0, 0, 0] == 100 * 24 + 24
    assert cropped[0, 15, 15, 5] == pytest.approx(100 * 39 + 39 + 0.5)


def test_lowres_view_values() -> None:
    lowres = rj.lowres_view(coordinate_stamp())[0, :, :, 0]
    # Block (i, j) averages rows 4i..4i+3 and cols 4j..4j+3: 100 * (4i + 1.5) + 4j + 1.5.
    assert lowres[0, 0] == pytest.approx(100 * 1.5 + 1.5)
    assert lowres[3, 7] == pytest.approx(100 * 13.5 + 29.5)
    # The last block has 3 valid rows and columns (60, 61, 62): mean 61.
    assert lowres[15, 15] == pytest.approx(100 * 61 + 61)
    assert lowres[15, 0] == pytest.approx(100 * 61 + 1.5)


def test_multiscale_view_values_and_scale_order() -> None:
    stamp = coordinate_stamp()
    view = rj.multiscale_view(stamp)
    assert view.shape == (1, 8, 8, 24)
    assert rj.MULTISCALE_SCALES == (1, 2, 4, 8)
    one, two, four, eight = (view[0, :, :, 6 * s : 6 * s + 6] for s in range(4))
    np.testing.assert_array_equal(one, stamp[0, 28:36, 28:36].astype(np.float32))
    pooled32 = rj.tf_same_avg_pool(stamp, 2)
    pooled16 = rj.tf_same_avg_pool(pooled32, 2)
    pooled8 = rj.tf_same_avg_pool(pooled16, 2)
    np.testing.assert_array_equal(two, pooled32[0, 12:20, 12:20])
    np.testing.assert_array_equal(four, pooled16[0, 4:12, 4:12])
    np.testing.assert_array_equal(eight, pooled8[0])
    # Hand values: 2" pixel (0, 0) is rows 24-25, cols 24-25. 8" pixel (7, 7) nests the
    # means of rows 56-62: 4" row 14 is mean(56..59) = 57.5, 4" row 15 is
    # mean(mean(60, 61), 62) = 61.25 (2" row 31 is row 62 alone), so (57.5 + 61.25) / 2.
    assert two[0, 0, 0] == pytest.approx(100 * 24.5 + 24.5)
    last = (57.5 + 61.25) / 2
    assert eight[7, 7, 0] == pytest.approx(100 * last + last, rel=1e-6)
    # Each scale's 6 channels are in CHANNEL_NAMES order.
    np.testing.assert_allclose(view[0, 0, 0, 0:6] - view[0, 0, 0, 0], np.arange(6) / 10, atol=1e-3)


@pytest.mark.parametrize(
    ("pixel", "scales"),
    [((31, 31), (0, 1, 2, 3)), ((26, 26), (1, 2, 3)), ((20, 20), (2, 3)), ((0, 0), (3,))],
)
def test_multiscale_fields_of_view(pixel: tuple[int, int], scales: tuple[int, ...]) -> None:
    # A single bright pixel appears exactly in the scales whose field of view covers it:
    # 8" (rows 28-35), 16" (24-39), 32" (16-47), 63" (all).
    stamp = np.zeros((1, 63, 63, 6))
    stamp[0, pixel[0], pixel[1], 2] = 1.0
    view = rj.multiscale_view(stamp)
    seen = tuple(s for s in range(4) if view[0, :, :, 6 * s + 2].any())
    assert seen == scales


def test_crops_hold_the_source_pixel_half_a_pixel_off_centre() -> None:
    # The source is pixel 31; every crop's centre is half a pixel below and right of it, at
    # the same local position: (7, 7) of 16, (3, 3) of each 8x8 scale.
    stamp = np.zeros((1, 63, 63, 6))
    stamp[0, 31, 31, :] = 1.0
    cropped = rj.cropped_view(stamp)[0, :, :, 0]
    assert np.argwhere(cropped).tolist() == [[7, 7]]
    view = rj.multiscale_view(stamp)
    for s in range(4):
        scale = view[0, :, :, 6 * s]
        assert tuple(int(i) for i in np.unravel_index(np.argmax(scale), scale.shape)) == (3, 3)


# ----------------------------------------------------------------------------------------
# Rotations and flips


@pytest.mark.parametrize("name", list(D4))
def test_normalisation_mask_and_full_view_are_equivariant(name: str) -> None:
    g = D4[name]
    stamps = raw_stamps(3, seed=7)
    np.testing.assert_array_equal(rj.normalise(g(stamps)), g(rj.normalise(stamps)))
    full = rj.rj23_views(g(stamps), ["full"])["full"]
    np.testing.assert_array_equal(full, g(rj.rj23_views(stamps, ["full"])["full"]))


def mirrored_window(name: str, start: int, size: int, n: int = 63) -> tuple[slice, slice]:
    """The window of ``x`` whose transform is the crop of the transformed stamp."""
    mirrored = n - start - size  # 23 for 16 at 24 on 63, 27 for 8 at 28
    rows, cols = slice(start, start + size), slice(start, start + size)
    flip_r, flip_c = slice(mirrored, mirrored + size), slice(mirrored, mirrored + size)
    # Which axes of the *source* stamp a transform reverses.
    reversed_axes = {
        "identity": (False, False),
        "rot90": (False, True),
        "rot180": (True, True),
        "rot270": (True, False),
        "flip_rows": (True, False),
        "flip_cols": (False, True),
        "transpose": (False, False),
        "anti_transpose": (True, True),
    }[name]
    return (flip_r if reversed_axes[0] else rows, flip_c if reversed_axes[1] else cols)


@pytest.mark.parametrize("name", list(D4))
@pytest.mark.parametrize(("size", "start"), [(16, 24), (8, 28)])
def test_crops_are_equivariant_up_to_a_one_pixel_shift(name: str, size: int, start: int) -> None:
    # The 16 and 8 crops are centred on 31.5, not 31: crop(g(x)) is g of the window
    # mirrored about the stamp's centre (23:39 or 27:35 along each reversed axis).
    g = D4[name]
    stamp = coordinate_stamp()
    crop = rj.cropped_view if size == 16 else (lambda a: rj.multiscale_view(a)[..., :6])
    rows, cols = mirrored_window(name, start, size)
    np.testing.assert_array_equal(crop(g(stamp)), g(stamp[:, rows, cols]).astype(np.float32))
    if name != "identity" and name != "transpose":
        assert not np.array_equal(crop(g(stamp)), g(crop(stamp)))


@pytest.mark.parametrize("name", list(D4))
def test_pooling_is_equivariant_without_padding(name: str) -> None:
    # On 64x64 no padding happens and pooling commutes with D4 exactly (float64 sums); the
    # 63x63 stamp pads at the end, so the coarse views are not equivariant there.
    g = D4[name]
    images = np.random.default_rng(2).normal(size=(2, 64, 64, 6))
    for pool in (2, 4):
        np.testing.assert_array_equal(
            rj.tf_same_avg_pool(g(images), pool), g(rj.tf_same_avg_pool(images, pool))
        )
    stamp = rj.normalise(raw_stamps(1))
    if name not in ("identity", "transpose"):
        assert not np.array_equal(rj.lowres_view(g(stamp)), g(rj.lowres_view(stamp)))


# ----------------------------------------------------------------------------------------
# Position


def test_unit_sphere() -> None:
    xyz = rj.unit_sphere([0.0, 90.0, 180.0, 45.0, 10.0], [0.0, 0.0, 0.0, 90.0, -30.0])
    assert xyz.dtype == np.float32
    assert xyz.shape == (5, 3)
    np.testing.assert_allclose(xyz[0], [1, 0, 0], atol=1e-7)
    np.testing.assert_allclose(xyz[1], [0, 1, 0], atol=1e-7)
    np.testing.assert_allclose(xyz[2], [-1, 0, 0], atol=1e-7)
    np.testing.assert_allclose(xyz[3], [0, 0, 1], atol=1e-7)
    ra, dec = np.deg2rad(10.0), np.deg2rad(-30.0)
    np.testing.assert_allclose(
        xyz[4], [np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)], rtol=1e-6
    )
    np.testing.assert_allclose(np.linalg.norm(xyz, axis=1), 1.0, rtol=1e-6)


@pytest.mark.parametrize(
    ("ra", "dec", "match"),
    [
        ([0.0, 1.0], [0.0], "equal length"),
        ([[0.0]], [[0.0]], "1-D"),
        ([np.nan], [0.0], "finite"),
        ([0.0], [90.5], r"\[-90, 90\]"),
    ],
)
def test_unit_sphere_errors(ra: list[object], dec: list[object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        rj.unit_sphere(ra, dec)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------------------
# Layout


def test_channels_first() -> None:
    images = rj.rj23_views(raw_stamps(2), ["multiscale"])["multiscale"]
    first = rj.channels_first(images)
    assert first.shape == (2, 24, 8, 8)
    assert first.flags.c_contiguous
    np.testing.assert_array_equal(first[:, 5], images[..., 5])
    tensor = torch.from_numpy(first)
    assert tensor.dtype == torch.float32


def test_module_does_not_import_torch() -> None:
    import subprocess
    import sys

    code = (
        "import sys; import astro_simclr.data.rj23_input; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603
