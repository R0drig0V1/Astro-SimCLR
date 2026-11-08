"""Tests of the SimCLR views (``astro_simclr.simclr.views``): dihedral, shift, noise, draws."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from astro_simclr.engine.randomness import RESERVED_STREAMS
from astro_simclr.simclr.views import (
    CENTRE_OFFSET,
    MAX_OFFSET,
    N_DIHEDRAL,
    STAMP_SIDE,
    VIEW_SIDE,
    ViewConfig,
    background_noise,
    centre_crop,
    crop,
    dihedral,
    draw_view_params,
    make_views,
    noise_generator,
)
from astro_simclr.stamp_relation import stamp_fits


def _stamps(n: int, channels: int = 3, seed: int = 0) -> torch.Tensor:
    return torch.from_numpy(
        np.random.default_rng(seed).random((n, channels, STAMP_SIDE, STAMP_SIDE), dtype=np.float32)
    )


def _related_stamps(n: int, seed: int = 0) -> torch.Tensor:
    """Stamps whose difference is exactly an affine function of science and template."""
    rng = np.random.default_rng(seed)
    sci = rng.normal(0.0, 1.0, (n, STAMP_SIDE, STAMP_SIDE))
    sci[:, 31, 31] += 30.0  # a source at the centre
    tmpl = rng.normal(0.0, 1.0, (n, STAMP_SIDE, STAMP_SIDE))
    diff = 0.8 * sci - 0.6 * tmpl
    stack = np.stack([sci, tmpl, diff], axis=1)
    low = stack.min(axis=(2, 3), keepdims=True)
    high = (stack - low).max(axis=(2, 3), keepdims=True)
    return torch.from_numpy(((stack - low) / high).astype(np.float32))


def test_draw_view_params_shapes_ranges_and_determinism() -> None:
    params = draw_view_params(3, 7, 64)
    assert params.batch_size == 64
    assert params.dihedral.shape == (2, 64) and params.dihedral.dtype == np.int64
    assert params.offsets.shape == (2, 64, 2) and params.offsets.dtype == np.int64
    assert params.noise_u.shape == (2, 64, 2)
    assert params.dihedral.min() >= 0 and params.dihedral.max() < N_DIHEDRAL
    assert params.offsets.min() >= 0 and params.offsets.max() <= MAX_OFFSET
    assert set(np.unique(params.dihedral)) == set(range(N_DIHEDRAL))
    again = draw_view_params(3, 7, 64)
    np.testing.assert_array_equal(params.dihedral, again.dihedral)
    np.testing.assert_array_equal(params.offsets, again.offsets)
    other = draw_view_params(3, 8, 64)
    assert not np.array_equal(params.offsets, other.offsets)


def test_draw_view_params_is_the_reserved_host_stream() -> None:
    rng = np.random.default_rng([3, 7, RESERVED_STREAMS["simclr_views"]])
    expected = rng.integers(0, N_DIHEDRAL, size=(2, 5), dtype=np.int64)
    np.testing.assert_array_equal(draw_view_params(3, 7, 5).dihedral, expected)
    diag = draw_view_params(3, 7, 5, stream=RESERVED_STREAMS["simclr_diagnostics"])
    assert not np.array_equal(diag.offsets, draw_view_params(3, 7, 5).offsets)


def test_draw_view_params_rejects_an_empty_batch() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        draw_view_params(0, 0, 0)


def test_dihedral_gives_the_eight_distinct_elements() -> None:
    x = _stamps(1).repeat(N_DIHEDRAL, 1, 1, 1)
    out = dihedral(x, np.arange(N_DIHEDRAL))
    flat = {tuple(out[i].flatten()[:50].tolist()) for i in range(N_DIHEDRAL)}
    assert len(flat) == N_DIHEDRAL
    torch.testing.assert_close(out[0], x[0], rtol=0, atol=0)
    torch.testing.assert_close(out[1], torch.rot90(x[0], 1, dims=(-2, -1)), rtol=0, atol=0)
    torch.testing.assert_close(out[4], torch.flip(x[0], dims=(-1,)), rtol=0, atol=0)
    flipped_rot = torch.flip(torch.rot90(x[0], 3, dims=(-2, -1)), dims=(-1,))
    torch.testing.assert_close(out[7], flipped_rot, rtol=0, atol=0)


def test_dihedral_is_exact_and_invertible() -> None:
    x = _stamps(N_DIHEDRAL)
    k = np.arange(N_DIHEDRAL)
    out = dihedral(x, k)
    # Rotations invert by 4 - r quarter turns; every reflection is its own inverse.
    inverse = np.array([0, 3, 2, 1, 4, 5, 6, 7])
    torch.testing.assert_close(dihedral(out, inverse), x, rtol=0, atol=0)
    # Values are permuted, never interpolated: the same multiset per channel.
    torch.testing.assert_close(
        out.flatten(2).sort(dim=-1).values, x.flatten(2).sort(dim=-1).values, rtol=0, atol=0
    )


def test_dihedral_does_not_modify_its_input_and_validates() -> None:
    x = _stamps(2)
    before = x.clone()
    dihedral(x, [1, 5])
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    with pytest.raises(ValueError, match="0-7"):
        dihedral(x, [0, 8])
    with pytest.raises(ValueError, match="rows"):
        dihedral(x, [0])
    with pytest.raises(ValueError, match="B, C, S, S"):
        dihedral(x[..., :60], [0, 0])


def test_crop_takes_the_same_window_on_every_channel() -> None:
    x = _stamps(3, channels=4)
    offsets = np.array([[0, 6], [3, 3], [6, 0]])
    out = crop(x, offsets)
    assert out.shape == (3, 4, VIEW_SIDE, VIEW_SIDE)
    for b, (dy, dx) in enumerate(offsets):
        expected = x[b, :, dy : dy + VIEW_SIDE, dx : dx + VIEW_SIDE]
        torch.testing.assert_close(out[b], expected, rtol=0, atol=0)


def test_centre_crop_is_the_centred_window() -> None:
    x = _stamps(2)
    centre = np.full((2, 2), CENTRE_OFFSET)
    torch.testing.assert_close(centre_crop(x), crop(x, centre), rtol=0, atol=0)
    assert CENTRE_OFFSET == 3
    # The stamp's central pixel stays the view's central pixel.
    assert centre_crop(x)[0, 0, VIEW_SIDE // 2, VIEW_SIDE // 2] == x[0, 0, 31, 31]
    with pytest.raises(ValueError, match="centred"):
        centre_crop(x, side=58)


def test_crop_rejects_out_of_range_offsets() -> None:
    x = _stamps(1)
    with pytest.raises(ValueError, match="0-6"):
        crop(x, [[0, 7]])
    with pytest.raises(ValueError, match="0-6"):
        crop(x, [[-1, 0]])
    with pytest.raises(ValueError, match="window"):
        crop(x, [[0, 0]], side=64)


def test_make_views_without_noise_is_dihedral_then_crop() -> None:
    x = _stamps(4)
    params = draw_view_params(0, 0, 4)
    v1, v2 = make_views(x, params, ViewConfig())
    assert v1.shape == v2.shape == (4, 3, VIEW_SIDE, VIEW_SIDE)
    for v, view in enumerate((v1, v2)):
        expected = crop(dihedral(x, params.dihedral[v]), params.offsets[v])
        torch.testing.assert_close(view, expected, rtol=0, atol=0)


def test_make_views_validates_its_inputs() -> None:
    x = _stamps(2)
    with pytest.raises(ValueError, match="63, 63"):
        make_views(x[..., :57, :57], draw_view_params(0, 0, 2), ViewConfig())
    with pytest.raises(ValueError, match="view parameters"):
        make_views(x, draw_view_params(0, 0, 3), ViewConfig())
    with pytest.raises(ValueError, match="fits"):
        make_views(x, draw_view_params(0, 0, 2), ViewConfig(noise_scale=1.0))
    with pytest.raises(ValueError, match="noise_scale"):
        ViewConfig(noise_scale=-1.0)


def test_background_noise_keeps_the_relation_and_the_normalisation() -> None:
    x = _related_stamps(4)
    fits = torch.from_numpy(stamp_fits(x.numpy()))
    assert torch.all(fits[:, 3] > 0.999)  # the relation is exact
    u = np.ones((4, 2))
    out = background_noise(x, fits, u, noise_generator(0, 0, x.device), ViewConfig(5.0))
    assert not torch.equal(out, x)
    lows = out.flatten(2).amin(dim=-1)
    highs = out.flatten(2).amax(dim=-1)
    torch.testing.assert_close(lows, torch.zeros_like(lows))
    torch.testing.assert_close(highs, torch.ones_like(highs))
    # The difference is recomputed through the fit, so it is still affine in sci and tmpl.
    refit = stamp_fits(out.numpy())
    assert np.all(refit[:, 3] > 0.999)


def test_background_noise_leaves_poor_fits_and_extra_channels_alone() -> None:
    x = torch.cat([_related_stamps(2), _stamps(2, channels=1)], dim=1)  # a 4th channel
    fits = torch.from_numpy(stamp_fits(x.numpy()))
    fits[1, 3] = 0.1  # below min_r2
    out = background_noise(
        x, fits, np.ones((2, 2)), noise_generator(0, 0, x.device), ViewConfig(5.0)
    )
    torch.testing.assert_close(out[1], x[1], rtol=0, atol=0)
    torch.testing.assert_close(out[:, 3], x[:, 3], rtol=0, atol=0)
    assert not torch.equal(out[0, :3], x[0, :3])


def test_noised_views_are_reproducible_from_the_step() -> None:
    x = _related_stamps(3)
    fits = torch.from_numpy(stamp_fits(x.numpy()))
    config = ViewConfig(noise_scale=2.0)
    params = draw_view_params(1, 5, 3)
    first = make_views(x, params, config, fits=fits, generator=noise_generator(1, 5, x.device))
    again = make_views(x, params, config, fits=fits, generator=noise_generator(1, 5, x.device))
    other = make_views(x, params, config, fits=fits, generator=noise_generator(1, 6, x.device))
    torch.testing.assert_close(first[0], again[0], rtol=0, atol=0)
    torch.testing.assert_close(first[1], again[1], rtol=0, atol=0)
    assert not torch.equal(first[0], other[0])
    assert not torch.equal(first[0], first[1])


def test_class_augment_re_exports_the_stamp_relation() -> None:
    """The move to stamp_relation (ADR 0019) left every class_augment name in place."""
    from astro_simclr import stamp_relation
    from astro_simclr.training import class_augment

    for name in stamp_relation.__all__:
        assert getattr(class_augment, name) is getattr(stamp_relation, name)
