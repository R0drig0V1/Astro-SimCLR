"""Tests of the optional SimCLR views of task 5.2 (``astro_simclr.simclr.views``).

Scale jitter, blur, the intensity (gamma) remap and the small-angle rotation: their draws
(own streams, ranges, determinism, nothing drawn when off), the rotation bound and the
no-fill rule of the window, the identity at strength 0, the same transform on the three
channels, and :func:`make_views` with each of them on.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from astro_simclr.engine.randomness import RESERVED_STREAMS, host_generator
from astro_simclr.simclr.views import (
    CENTRE_OFFSET,
    MAX_BLUR_SIGMA,
    STAMP_SIDE,
    VIEW_SIDE,
    ExtraViewParams,
    ViewConfig,
    crop,
    draw_extra_params,
    draw_view_params,
    gamma_remap,
    gaussian_blur,
    make_views,
    resample,
    rotation_bound,
    window_transform,
)
from astro_simclr.stamp_relation import masked_pixels

ALL_ON = ViewConfig(scale_jitter=0.3, blur_sigma=1.5, intensity_jitter=0.4, rotation_degrees=5.0)


def _stamps(n: int, channels: int = 3, seed: int = 0) -> torch.Tensor:
    """Min-max normalised stamps (each channel has minimum 0 and maximum 1)."""
    raw = np.random.default_rng(seed).random((n, channels, STAMP_SIDE, STAMP_SIDE))
    low = raw.min(axis=(2, 3), keepdims=True)
    high = (raw - low).max(axis=(2, 3), keepdims=True)
    return torch.from_numpy(((raw - low) / high).astype(np.float32))


def _coordinates(matrix: np.ndarray, shift: np.ndarray, side: int = VIEW_SIDE) -> np.ndarray:
    """The unclamped source coordinates (B, 2, side, side) of :func:`resample`, float64."""
    q = np.arange(side) - (side - 1) / 2
    grid = np.stack(np.meshgrid(q, q, indexing="ij"))  # (2, side, side): row, column
    centre = (STAMP_SIDE - 1) / 2
    coords: np.ndarray = centre + shift[:, :, None, None] + np.einsum("bij,jrc->birc", matrix, grid)
    return coords


# --- draws ---------------------------------------------------------------------------------


def test_nothing_is_drawn_with_every_optional_view_off() -> None:
    assert not ViewConfig().extra and not ViewConfig(noise_scale=3.0).extra
    assert draw_extra_params(0, 0, 4, ViewConfig()) == ExtraViewParams()


def test_extra_params_ranges_streams_and_determinism() -> None:
    params = draw_extra_params(3, 7, 500, ALL_ON)
    assert params.scale is not None and params.blur_sigma is not None
    assert params.gamma is not None and params.angle is not None
    for values in (params.scale, params.blur_sigma, params.gamma, params.angle):
        assert values.shape == (2, 500) and values.dtype == np.float64
    assert params.scale.min() >= 1.0 and params.scale.max() <= 1.3
    assert params.blur_sigma.min() >= 0.0 and params.blur_sigma.max() <= 1.5
    assert np.abs(np.log(params.gamma)).max() <= 0.4 + 1e-12
    assert np.abs(params.angle).max() <= 5.0
    # Both signs of the symmetric ranges are drawn.
    assert params.angle.min() < 0 < params.angle.max()
    assert np.log(params.gamma).min() < 0 < np.log(params.gamma).max()
    # Each view's u comes from [seed, step, its stream].
    u = host_generator(3, 7, RESERVED_STREAMS["simclr_view_rotation"]).random((2, 500))
    np.testing.assert_array_equal(params.angle, 5.0 * (2.0 * u - 1.0))
    u = host_generator(3, 7, RESERVED_STREAMS["simclr_view_scale"]).random((2, 500))
    np.testing.assert_array_equal(params.scale, 1.3**u)
    again = draw_extra_params(3, 7, 500, ALL_ON)
    np.testing.assert_array_equal(params.blur_sigma, again.blur_sigma)
    other = draw_extra_params(3, 8, 500, ALL_ON)
    assert other.gamma is not None and not np.array_equal(params.gamma, other.gamma)
    with pytest.raises(ValueError, match="batch_size"):
        draw_extra_params(0, 0, 0, ALL_ON)


def test_one_view_on_or_off_changes_no_other_views_draws() -> None:
    alone = draw_extra_params(1, 2, 16, ViewConfig(rotation_degrees=5.0))
    assert alone.scale is None and alone.blur_sigma is None and alone.gamma is None
    np.testing.assert_array_equal(alone.angle, draw_extra_params(1, 2, 16, ALL_ON).angle)
    # The always-drawn dihedral elements, offsets and noise strengths are not touched.
    before = draw_view_params(1, 2, 16)
    draw_extra_params(1, 2, 16, ALL_ON)
    np.testing.assert_array_equal(before.offsets, draw_view_params(1, 2, 16).offsets)


def test_view_config_validation() -> None:
    for name in ("scale_jitter", "blur_sigma", "intensity_jitter", "rotation_degrees"):
        with pytest.raises(ValueError, match=name):
            ViewConfig(**{name: -0.1})
        with pytest.raises(ValueError, match=name):
            ViewConfig(**{name: math.nan})
    with pytest.raises(ValueError, match="blur_sigma must be <="):
        ViewConfig(blur_sigma=MAX_BLUR_SIGMA + 1)
    with pytest.raises(ValueError, match="rotation_degrees must be <="):
        ViewConfig(rotation_degrees=6.6)
    config = ViewConfig(blur_sigma=1.5)
    assert config.blur and config.extra and not config.geometric and config.blur_radius == 5
    assert ViewConfig(rotation_degrees=rotation_bound()).geometric


# --- the rotation bound and the window ------------------------------------------------------


def test_rotation_bound_is_the_corner_reaching_the_stamp_edge() -> None:
    bound = rotation_bound()
    assert bound == pytest.approx(6.5239, abs=1e-4)
    theta = math.radians(bound)
    assert 28 * (math.cos(theta) + math.sin(theta)) == pytest.approx(31.0, abs=1e-12)
    assert rotation_bound(63, 21) == 45.0  # a small window turns freely
    assert rotation_bound(5, 1) == 45.0
    with pytest.raises(ValueError, match="window"):
        rotation_bound(57, 63)


def test_window_transform_off_is_the_integer_shift() -> None:
    offsets = np.array([[0, 6], [3, 3], [6, 0]])
    matrix, shift = window_transform(offsets)
    np.testing.assert_array_equal(matrix, np.broadcast_to(np.eye(2), (3, 2, 2)))
    np.testing.assert_array_equal(shift, offsets - CENTRE_OFFSET)
    # A zoom in leaves more margin: the shift is still the integer one.
    _, zoomed = window_transform(offsets, scale=np.array([1.0, 1.2, 1.5]))
    np.testing.assert_array_equal(zoomed, offsets - CENTRE_OFFSET)


def test_window_transform_reads_no_fill_and_the_shift_gives_way() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    offsets = rng.integers(0, 7, size=(n, 2))
    angle = rng.uniform(-rotation_bound(), rotation_bound(), n)
    scale = np.exp(rng.uniform(0, np.log(1.5), n))
    for s in (None, scale):
        matrix, shift = window_transform(offsets, s, angle)
        coords = _coordinates(matrix, shift)
        assert coords.min() >= -1e-9 and coords.max() <= STAMP_SIDE - 1 + 1e-9
        assert np.abs(shift).max() <= CENTRE_OFFSET
    # At the bound the window touches the edge and cannot move.
    corners = np.array([[0, 0], [6, 6]])
    matrix, shift = window_transform(corners, angle=np.full(2, rotation_bound()))
    np.testing.assert_allclose(shift, 0.0, atol=1e-12)
    coords = _coordinates(matrix, shift)
    assert coords.min() == pytest.approx(0.0, abs=1e-9)
    assert coords.max() == pytest.approx(STAMP_SIDE - 1, abs=1e-9)
    # Half way, the shift is scaled by the margin left over 3 pixels.
    theta = math.radians(3.0)
    factor = (31 - 28 * (math.cos(theta) + math.sin(theta))) / 3
    _, shift = window_transform(np.array([[0, 6]]), angle=np.array([3.0]))
    np.testing.assert_allclose(shift, [[-3 * factor, 3 * factor]])


def test_window_transform_refusals() -> None:
    with pytest.raises(ValueError, match="offsets"):
        window_transform(np.array([[7, 0]]))
    with pytest.raises(ValueError, match="offsets"):
        window_transform(np.array([0, 0]))
    with pytest.raises(ValueError, match="zoom out"):
        window_transform(np.array([[3, 3]]), scale=np.array([0.9]))
    with pytest.raises(ValueError, match="rotation_bound"):
        window_transform(np.array([[3, 3]]), angle=np.array([7.0]))
    with pytest.raises(ValueError, match=r"angle must be \(1,\)"):
        window_transform(np.array([[3, 3]]), angle=np.array([1.0, 2.0]))


# --- resampling ----------------------------------------------------------------------------


def test_resample_of_the_identity_is_the_crop_bit_for_bit() -> None:
    x = _stamps(5, channels=4)
    offsets = np.random.default_rng(1).integers(0, 7, size=(5, 2))
    matrix, shift = window_transform(offsets, scale=np.ones(5), angle=np.zeros(5))
    assert torch.equal(resample(x, matrix, shift), crop(x, offsets))
    whole = resample(x, np.broadcast_to(np.eye(2), (5, 2, 2)), np.zeros((5, 2)), side=STAMP_SIDE)
    assert torch.equal(whole, x)


def test_resample_samples_every_channel_alike_and_interpolates() -> None:
    one = _stamps(3, channels=1)
    x = one.expand(3, 4, STAMP_SIDE, STAMP_SIDE).contiguous()
    matrix, shift = window_transform(
        np.full((3, 2), 3), scale=np.array([1.0, 1.2, 1.4]), angle=np.array([-5.0, 2.0, 6.0])
    )
    out = resample(x, matrix, shift)
    assert out.shape == (3, 4, VIEW_SIDE, VIEW_SIDE)
    for c in range(1, 4):
        assert torch.equal(out[:, c], out[:, 0])
    # Bilinear weights are convex: the view stays within the stamp's range.
    assert out.min() >= 0.0 and out.max() <= 1.0
    # A half-pixel shift is the mean of two neighbours.
    ramp = torch.arange(STAMP_SIDE, dtype=torch.float32).expand(1, 1, STAMP_SIDE, STAMP_SIDE)
    half = resample(ramp, np.eye(2)[None], np.array([[0.0, 0.5]]), side=3)
    np.testing.assert_allclose(half[0, 0, 0].numpy(), [30.5, 31.5, 32.5])
    constant = torch.full((1, 3, STAMP_SIDE, STAMP_SIDE), 0.25)
    rotated = resample(constant, *window_transform(np.array([[3, 3]]), angle=np.array([4.0])))
    assert torch.allclose(rotated, torch.full_like(rotated, 0.25))


def test_resample_clamps_outside_the_stamp_and_validates() -> None:
    x = _stamps(1)
    out = resample(x, np.eye(2)[None], np.array([[3.0, 0.0]]), side=STAMP_SIDE)
    # The rows shifted past the edge repeat the last row (the admission test's outer ring).
    assert torch.equal(out[0, :, -1], x[0, :, -1]) and torch.equal(out[0, :, -4], x[0, :, -1])
    with pytest.raises(ValueError, match="images"):
        resample(x[0], np.eye(2)[None], np.zeros((1, 2)))
    with pytest.raises(ValueError, match="matrix"):
        resample(x, np.eye(2), np.zeros((1, 2)))


# --- blur -----------------------------------------------------------------------------------


def test_blur_at_sigma_zero_is_the_identity_and_rows_are_independent() -> None:
    x = _stamps(3, channels=4)
    out = gaussian_blur(x, np.array([0.0, 1.0, 0.0]), radius=3)
    assert torch.equal(out[0], x[0]) and torch.equal(out[2], x[2])
    assert not torch.equal(out[1, :3], x[1, :3])
    assert torch.equal(out[:, 3], x[:, 3])  # a fourth channel is never touched
    alone = gaussian_blur(x[1:2], np.array([1.0]), radius=3)
    assert torch.equal(alone[0], out[1])


def test_blur_is_the_same_on_the_three_channels_and_keeps_the_statistics() -> None:
    one = _stamps(2, channels=1, seed=3)
    x = one.expand(2, 3, STAMP_SIDE, STAMP_SIDE).contiguous()
    out = gaussian_blur(x, np.array([0.8, 2.0]), radius=6)
    assert torch.equal(out[:, 1], out[:, 0]) and torch.equal(out[:, 2], out[:, 0])
    assert torch.allclose(out.amin(dim=(-2, -1)), torch.zeros(2, 3))
    assert torch.allclose(out.amax(dim=(-2, -1)), torch.ones(2, 3))
    # Larger sigma, smoother: neighbouring pixels differ less.
    roughness = (out[:, 0, :, 1:] - out[:, 0, :, :-1]).abs().mean(dim=(-2, -1))
    assert roughness[1] < roughness[0]


def test_blur_spreads_a_point_symmetrically_and_leaves_masked_pixels_out() -> None:
    x = torch.zeros(1, 3, STAMP_SIDE, STAMP_SIDE)
    x[0, :, 31, 31] = 1.0
    x[0, :, 0, 0] = 0.0
    x[0, :, 40, 40] = 0.5
    out = gaussian_blur(x, np.array([1.0]), radius=3)[0, 0]
    # Every 0 of a channel with many zeros is a former NaN: they stay 0 and get no weight,
    # so the two valid pixels keep their values (then the min-max: 0.5 -> 0, 1 -> 1).
    assert torch.count_nonzero(out) == 1 and out[31, 31] == 1.0
    # Without masked pixels (a background > 0), a point spreads into a symmetric blob.
    x = torch.full((1, 3, STAMP_SIDE, STAMP_SIDE), 0.1)
    x[0, :, 31, 31] = 1.0
    x[0, :, 10, 10] = 0.0  # the minimum, a single 0: not masked
    assert not masked_pixels(x).any()
    blob = gaussian_blur(x, np.array([1.5]), radius=5)[0, 0]
    assert blob[31, 31] == 1.0
    assert torch.allclose(blob[31, 28:31], blob[31, 32:35].flip(0))
    assert torch.allclose(blob[28:31, 31], blob[31, 28:31])


def test_blur_validates() -> None:
    x = _stamps(2)
    with pytest.raises(ValueError, match="images"):
        gaussian_blur(x[:, :2], np.zeros(2), radius=1)
    with pytest.raises(ValueError, match="sigma must be"):
        gaussian_blur(x, np.zeros(3), radius=1)
    with pytest.raises(ValueError, match="radius"):
        gaussian_blur(x, np.zeros(2), radius=0)
    with pytest.raises(ValueError, match=">= 0"):
        gaussian_blur(x, np.array([-1.0, 0.0]), radius=1)


# --- intensity ------------------------------------------------------------------------------


def test_gamma_is_one_monotone_map_of_the_three_channels() -> None:
    x = _stamps(4, channels=4, seed=2)
    gamma = np.array([1.0, 0.7, 1.4, 1.0])
    out = gamma_remap(x, gamma)
    assert torch.equal(out[0], x[0]) and torch.equal(out[3], x[3])  # gamma 1: unchanged
    assert torch.equal(out[:, 3], x[:, 3])
    for b in (1, 2):
        exponent = torch.tensor(gamma[b], dtype=torch.float32)
        for c in range(3):
            # The same map on every channel: each channel is x ** gamma[b].
            assert torch.equal(out[b, c], x[b, c] ** exponent)
        assert torch.equal(out[b, :3].amin(dim=(-2, -1)), torch.zeros(3))
        assert torch.equal(out[b, :3].amax(dim=(-2, -1)), torch.ones(3))
        flat_in, flat_out = x[b, 0].flatten(), out[b, 0].flatten()
        order = flat_in.argsort()
        assert (flat_out[order].diff() >= 0).all()  # monotone
    assert (out[1, :3] >= x[1, :3]).all() and (out[2, :3] <= x[2, :3]).all()
    same = gamma_remap(x[:, :1].expand(4, 3, STAMP_SIDE, STAMP_SIDE), gamma)
    assert torch.equal(same[:, 1], same[:, 0]) and torch.equal(same[:, 2], same[:, 0])
    with pytest.raises(ValueError, match="gamma must be finite"):
        gamma_remap(x, np.array([1.0, 0.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="images"):
        gamma_remap(x[:, :2], gamma)


# --- make_views -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        ViewConfig(scale_jitter=0.3),
        ViewConfig(blur_sigma=1.5),
        ViewConfig(intensity_jitter=0.4),
        ViewConfig(rotation_degrees=6.0),
        ALL_ON,
    ],
    ids=["scale", "blur", "intensity", "rotation", "all"],
)
def test_make_views_with_an_optional_view(config: ViewConfig) -> None:
    x = _stamps(6)
    params = draw_view_params(4, 2, 6)
    extra = draw_extra_params(4, 2, 6, config)
    v1, v2 = make_views(x, params, config, extra=extra)
    assert v1.shape == v2.shape == (6, 3, VIEW_SIDE, VIEW_SIDE)
    assert v1.min() >= 0 and v1.max() <= 1 and torch.isfinite(v2).all()
    plain = make_views(x, params, ViewConfig())
    assert not torch.equal(v1, plain[0]) and not torch.equal(v2, plain[1])
    again = make_views(x, params, config, extra=draw_extra_params(4, 2, 6, config))
    assert torch.equal(again[0], v1) and torch.equal(again[1], v2)
    with pytest.raises(ValueError, match="parameters"):
        make_views(x, params, config)


def test_make_views_with_the_identity_parameters_is_the_plain_view() -> None:
    # Draws at the centre of every range (s = 1, sigma = 0, gamma = 1, theta = 0) give the
    # plain views bit for bit, through the resampling path.
    x = _stamps(4)
    params = draw_view_params(0, 0, 4)
    ones = np.ones((2, 4))
    extra = ExtraViewParams(scale=ones, blur_sigma=0 * ones, gamma=ones, angle=0 * ones)
    views = make_views(x, params, ALL_ON, extra=extra)
    plain = make_views(x, params, ViewConfig())
    assert torch.equal(views[0], plain[0]) and torch.equal(views[1], plain[1])
