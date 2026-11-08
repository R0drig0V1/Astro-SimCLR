"""The RJ23 input path on raw ZTF cutouts (phase 3, task 3.3).

Reyes-Jainaga et al. 2023 (RJ23, ApJL, arXiv:2304.13080) feed their stamp classifiers the
*raw* 63x63 science, reference and difference cutouts of an alert, not CD21's min-max
normalised stamps. This module is a reimplementation from the paper and from the
description of RJ23's MIT code (``github.com/alercebroker/multiscale_stamps``,
``src/data_loader.py``) in the task 3.0 research; no ALeRCE code is copied. The choices the
sources leave open are in ADR 0008, the formats in ``docs/data.md`` ("RJ23 inputs").
Carvajal et al. 2025 (C25) use the same 6-channel stamps; their FFT branch is not built
(plan review round 2: C25's satellites cannot be rebuilt from public data).

Every function is pure numpy (torch is not imported), deterministic, batched over a leading
axis and chunked for memory, and returns channels-last float32 arrays
(:func:`channels_first` converts; ``torch.from_numpy`` then takes the result as is).

1. **Normalisation** (:func:`normalise`), per stamp and per channel, on the full 63x63
   before any crop or pooling: ``(x - min|x|) / (p99|x| - min|x| + 1e-8)``, clipped to
   ``[-2, 2]``. The statistics skip NaN, +-inf and ``|x| > 1e10``; the percentile is
   numpy's ``linear`` one, as ``np.nanpercentile``. Invalid pixels become 0. What
   "invalid" means for ``|x| > 1e10`` depends on ``large_values``:

   - ``"paper"`` (the default; RJ23 sec. II.1): they are invalid like NaN and inf, so they
     are masked and zeroed;
   - ``"code"`` (RJ23's ``normalize_stamps``): they are only left out of the statistics,
     then normalised and clipped like any other pixel, and not masked.

   A channel with no pixel left for the statistics is all zeros (in RJ23's code its NaN
   statistics turn every pixel into NaN, which ``nan_to_num`` makes 0).
2. **The invalid mask** (:func:`invalid_mask`): 1.0 where the raw pixel is invalid, one
   channel per stamp type. The normalised stamp is 6 channels, :data:`CHANNEL_NAMES`.
3. **The four views** (:func:`full_view`, :func:`cropped_view`, :func:`lowres_view`,
   :func:`multiscale_view`, all from the normalised 63x63x6 stamp):

   - full 63x63x6;
   - cropped 16x16x6, rows and columns 24:40 (RJ23's ``ceil((63 - 16) / 2)``);
   - low-resolution 16x16x6, **one** 4x4 average pooling with stride 4 and TF ``'same'``
     padding (RJ23's ``AveragePooling2D(pool_size=(4, 4), padding='same')``; two 2x2
     poolings would differ in the last row and column, see :func:`tf_same_avg_pool`);
   - multi-scale 8x8x24: the 8x8 crop 28:36 of the full stamp, then the 8x8 crops of the
     stamp after 1, 2 and 3 iterated 2x2 'same' average poolings (63 -> 32 -> 16 -> 8; the
     last is the whole pooled image), ordered by scale 1"/2"/4"/8" with the 6 channels of
     each scale together (scale-major). The masks are pooled too, so they become
     fractional at the coarse scales.

   The crops are not exactly centred: their centre is pixel 31.5 while the source is at
   31 (RJ23's docstring: "The original center might be slightly shifted if crop_size is
   even"), so a rotation or flip of the stamp moves the crop window by one pixel.
4. **The position** (:func:`unit_sphere`): ``(cos ra cos dec, sin ra cos dec, sin dec)``.
:func:`rj23_views` runs steps 1 to 3 over a raw batch in chunks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

__all__ = [
    "CHANNEL_NAMES",
    "CLIP",
    "CROP8",
    "CROP16",
    "EPSILON",
    "INVALID_THRESHOLD",
    "MULTISCALE_SCALES",
    "PERCENTILE",
    "RAW_SHAPE",
    "VIEW_NAMES",
    "VIEW_SHAPES",
    "LargeValues",
    "ViewName",
    "channel_statistics",
    "channels_first",
    "crop_offset",
    "cropped_view",
    "full_view",
    "invalid_mask",
    "lowres_view",
    "multiscale_view",
    "normalise",
    "rj23_views",
    "tf_same_avg_pool",
    "unit_sphere",
]

#: Raw cutouts per alert: rows, columns, then science, reference, difference.
RAW_SHAPE = (63, 63, 3)
#: The channels of the normalised stamp (and of each multi-scale scale).
CHANNEL_NAMES = ("sci", "ref", "diff", "m_sci", "m_ref", "m_diff")
#: ``|x|`` above this is out of the statistics (and invalid with ``large_values="paper"``).
INVALID_THRESHOLD = 1e10
#: The percentile of ``|x|`` that maps to 1.
PERCENTILE = 99.0
#: Added to the denominator of the normalisation.
EPSILON = 1e-8
#: The normalised values are clipped to ``[-CLIP, CLIP]``.
CLIP = 2.0
#: Rows and columns of the cropped view on the 63x63 stamp.
CROP16 = (24, 40)
#: Rows and columns of the 1" multi-scale crop on the 63x63 stamp.
CROP8 = (28, 36)
#: Arcseconds per pixel of the four multi-scale scales, in channel order.
MULTISCALE_SCALES = (1, 2, 4, 8)

ViewName = Literal["full", "cropped", "lowres", "multiscale"]
VIEW_NAMES: tuple[ViewName, ...] = ("full", "cropped", "lowres", "multiscale")
#: The channels-last shape of one stamp's view.
VIEW_SHAPES: dict[ViewName, tuple[int, int, int]] = {
    "full": (63, 63, 6),
    "cropped": (16, 16, 6),
    "lowres": (16, 16, 6),
    "multiscale": (8, 8, 24),
}

LargeValues = Literal["paper", "code"]

FloatArray = npt.NDArray[np.floating]
F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

_DEFAULT_CHUNK = 1000


def _check_raw(stamps: npt.ArrayLike) -> FloatArray:
    """Return ``stamps`` as a float array of shape ``(N, 63, 63, 3)``.

    Raises:
        ValueError: if the shape is not ``(N, 63, 63, 3)`` or the dtype is not a float.
    """
    array = np.asarray(stamps)
    if array.ndim != 4 or array.shape[1:] != RAW_SHAPE:
        raise ValueError(f"expected raw cutouts of shape (N, 63, 63, 3), got {array.shape}")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"expected float cutouts, got dtype {array.dtype}")
    return array


def _check_chunk(chunk_size: int) -> None:
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")


def _excluded_and_invalid(
    stamps: FloatArray, large_values: LargeValues
) -> tuple[BoolArray, BoolArray]:
    """The pixels left out of the statistics and the invalid (masked, zeroed) pixels."""
    if large_values not in ("paper", "code"):
        raise ValueError(f"large_values must be 'paper' or 'code', got {large_values!r}")
    non_finite = ~np.isfinite(stamps)
    with np.errstate(invalid="ignore"):
        large = np.abs(stamps) > INVALID_THRESHOLD  # False for NaN
    excluded = non_finite | large
    invalid = excluded if large_values == "paper" else non_finite
    return excluded, invalid


def invalid_mask(stamps: npt.ArrayLike, *, large_values: LargeValues = "paper") -> BoolArray:
    """The invalid-pixel mask of raw cutouts, ``(N, 63, 63, 3)`` booleans.

    True where the pixel is NaN or +-inf and, with ``large_values="paper"``, where
    ``|x| > 1e10``. The float32 mask channels of :func:`normalise` are this mask as 0/1.

    Raises:
        ValueError: on a wrong shape or dtype, or an unknown ``large_values``.
    """
    return _excluded_and_invalid(_check_raw(stamps), large_values)[1]


def channel_statistics(
    stamps: npt.ArrayLike, *, large_values: LargeValues = "paper"
) -> tuple[F64, F64]:
    """``min|x|`` and ``p99|x|`` per stamp and channel, each ``(N, 3)`` float64.

    The pixels of :func:`invalid_mask`'s statistics exclusion (NaN, +-inf, ``|x| > 1e10``,
    in both ``large_values`` modes) are skipped. The percentile is numpy's ``linear``
    interpolation, the same values as ``np.nanpercentile(|x|, 99)``, computed with one
    sort per chunk instead of numpy's per-row loop. A channel with no pixel left gives NaN
    for both.

    Raises:
        ValueError: on a wrong shape or dtype, or an unknown ``large_values``.
    """
    array = _check_raw(stamps)
    excluded, _ = _excluded_and_invalid(array, large_values)
    return _statistics(array, excluded)


def _statistics(stamps: FloatArray, excluded: BoolArray) -> tuple[F64, F64]:
    n = stamps.shape[0]
    pixels = stamps.shape[1] * stamps.shape[2]
    absolute = np.abs(stamps.astype(np.float64, copy=False)).reshape(n, pixels, -1)
    absolute = np.where(excluded.reshape(n, pixels, -1), np.nan, absolute)
    ordered = np.sort(absolute, axis=1)  # NaN sort last
    count = np.sum(~excluded.reshape(n, pixels, -1), axis=1)
    safe = np.maximum(count, 1)
    # numpy's 'linear' method: virtual index (n - 1) * q, then its _lerp, which
    # interpolates from the upper neighbour when the weight is >= 0.5.
    virtual = (safe - 1) * np.true_divide(PERCENTILE, 100)
    lower = np.floor(virtual).astype(np.intp)
    upper = np.minimum(lower + 1, safe - 1)
    gamma = virtual - lower
    below = np.take_along_axis(ordered, lower[:, None, :], axis=1)[:, 0, :]
    above = np.take_along_axis(ordered, upper[:, None, :], axis=1)[:, 0, :]
    diff = above - below
    p99 = np.where(gamma >= 0.5, above - diff * (1 - gamma), below + diff * gamma)
    minimum = ordered[:, 0, :]
    empty = count == 0
    return np.where(empty, np.nan, minimum), np.where(empty, np.nan, p99)


def _normalise_chunk(stamps: FloatArray, large_values: LargeValues) -> F32:
    excluded, invalid = _excluded_and_invalid(stamps, large_values)
    minimum, p99 = _statistics(stamps, excluded)
    lo = minimum[:, None, None, :]
    scale = p99[:, None, None, :] - lo + EPSILON
    with np.errstate(invalid="ignore", over="ignore"):
        values = np.clip((stamps.astype(np.float64, copy=False) - lo) / scale, -CLIP, CLIP)
    # NaN covers both the invalid pixels and the channels without statistics.
    values = np.where(invalid | np.isnan(values), 0.0, values)
    return np.concatenate([values, invalid.astype(np.float64)], axis=-1).astype(np.float32)


def normalise(
    stamps: npt.ArrayLike,
    *,
    large_values: LargeValues = "paper",
    chunk_size: int = _DEFAULT_CHUNK,
) -> F32:
    """RJ23's normalised stamps with their mask, ``(N, 63, 63, 6)`` float32.

    Channels 0-2 are ``clip((x - min|x|) / (p99|x| - min|x| + 1e-8), -2, 2)`` per stamp and
    channel (statistics of :func:`channel_statistics`, arithmetic in float64), with the
    invalid pixels and the channels without statistics set to 0; channels 3-5 are
    :func:`invalid_mask` as 0.0/1.0. The order is :data:`CHANNEL_NAMES`. ``chunk_size``
    stamps are processed at a time; the result does not depend on it.

    Note that the centre is ``min|x|``, not 0: a negative pixel maps below
    ``-min|x| / (p99 - min)``, and a constant negative channel saturates at -2.

    Raises:
        ValueError: on a wrong shape or dtype, an unknown ``large_values`` or a
            ``chunk_size`` below 1.
    """
    array = _check_raw(stamps)
    _check_chunk(chunk_size)
    out = np.empty((array.shape[0], *VIEW_SHAPES["full"]), dtype=np.float32)
    for start in range(0, array.shape[0], chunk_size):
        stop = start + chunk_size
        out[start:stop] = _normalise_chunk(array[start:stop], large_values)
    return out


def crop_offset(size: int, crop: int) -> int:
    """RJ23's crop start, ``ceil((size - crop) / 2)``: 24 for 16 on 63, 28 for 8 on 63.

    Raises:
        ValueError: if ``crop`` is not in ``[1, size]``.
    """
    if not 1 <= crop <= size:
        raise ValueError(f"crop must be in [1, {size}], got {crop}")
    return (size - crop + 1) // 2


def _crop(images: npt.NDArray[np.float32], crop: int) -> F32:
    """The ``crop`` x ``crop`` window at :func:`crop_offset` of ``(N, H, W, C)`` images."""
    top = crop_offset(images.shape[1], crop)
    left = crop_offset(images.shape[2], crop)
    return images[:, top : top + crop, left : left + crop, :]


def _same_padding(size: int, pool: int) -> tuple[int, int, int]:
    """TF 'same' for a window and stride of ``pool``: (output size, pad before, after)."""
    out = -(-size // pool)
    total = max(out * pool - size, 0)
    return out, total // 2, total - total // 2


def tf_same_avg_pool(images: npt.ArrayLike, pool: int = 2) -> F32:
    """TF's ``AveragePooling2D(pool, strides=pool, padding='same')`` on ``(N, H, W, C)``.

    The output is ``ceil(H / pool) x ceil(W / pool)``. TF pads ``total = out * pool - size``
    pixels, ``total // 2`` before and the rest after, so for ``pool=2`` an odd size gets
    one pixel at the end. Each output is the average of the window's **valid** pixels
    (the padding is not counted), so the last row of 63 pooled by 2 is the input's last row
    alone, and of 63 pooled by 4 the mean of 3 rows. Sums are accumulated in float64 and
    cast to float32, so the result does not depend on the pixel order.

    Two 2x2 poolings equal one 4x4 pooling except in a last, partial block (63: the last row
    and column); RJ23 uses one 4x4 pooling for its low-resolution view and iterated 2x2
    ones for the multi-scale view.

    Raises:
        ValueError: if ``images`` is not 4-D or ``pool`` is below 1.
    """
    array = np.asarray(images, dtype=np.float64)
    if array.ndim != 4:
        raise ValueError(f"expected (N, H, W, C) images, got shape {array.shape}")
    if pool < 1:
        raise ValueError(f"pool must be >= 1, got {pool}")
    height, width = array.shape[1:3]
    out_h, top, bottom = _same_padding(height, pool)
    out_w, left, right = _same_padding(width, pool)
    widths = ((0, 0), (top, bottom), (left, right), (0, 0))
    padded = np.pad(array, widths)
    counts = np.pad(np.ones((1, height, width, 1)), widths)
    sums = padded.reshape(-1, out_h, pool, out_w, pool, padded.shape[-1]).sum(axis=(2, 4))
    valid = counts.reshape(1, out_h, pool, out_w, pool, 1).sum(axis=(2, 4))
    return (sums / valid).astype(np.float32)


def _check_normalised(normalised: npt.ArrayLike) -> npt.NDArray[np.float32]:
    array = np.asarray(normalised, dtype=np.float32)
    if array.ndim != 4 or array.shape[1:] != VIEW_SHAPES["full"]:
        raise ValueError(f"expected normalised stamps of shape (N, 63, 63, 6), got {array.shape}")
    return array


def full_view(normalised: npt.ArrayLike) -> F32:
    """RJ23's "full" view, the normalised stamp itself, ``(N, 63, 63, 6)`` float32.

    Raises:
        ValueError: if ``normalised`` is not ``(N, 63, 63, 6)``.
    """
    return _check_normalised(normalised).copy()


def cropped_view(normalised: npt.ArrayLike) -> F32:
    """RJ23's "cropped" view: rows and columns 24:40, ``(N, 16, 16, 6)`` float32.

    Raises:
        ValueError: if ``normalised`` is not ``(N, 63, 63, 6)``.
    """
    return _crop(_check_normalised(normalised), 16).copy()


def lowres_view(normalised: npt.ArrayLike) -> F32:
    """RJ23's "low-resolution" view: one 4x4 'same' average pooling, ``(N, 16, 16, 6)``.

    Raises:
        ValueError: if ``normalised`` is not ``(N, 63, 63, 6)``.
    """
    return tf_same_avg_pool(_check_normalised(normalised), 4)


def multiscale_view(normalised: npt.ArrayLike) -> F32:
    """RJ23's "multi-scale" view, ``(N, 8, 8, 24)`` float32.

    Channels ``6 * s : 6 * (s + 1)`` hold scale ``MULTISCALE_SCALES[s]`` (1", 2", 4", 8" per
    pixel; fields of view 8", 16", 32", 63"), each in :data:`CHANNEL_NAMES` order:

    - 1": the full stamp's rows and columns 28:36;
    - 2": one 2x2 'same' average pooling (63 -> 32), rows 12:20;
    - 4": two poolings (32 -> 16), rows 4:12;
    - 8": three poolings (16 -> 8), the whole image.

    Raises:
        ValueError: if ``normalised`` is not ``(N, 63, 63, 6)``.
    """
    current = _check_normalised(normalised)
    scales = [_crop(current, 8)]
    for _ in MULTISCALE_SCALES[1:]:
        current = tf_same_avg_pool(current, 2)
        scales.append(_crop(current, 8))
    return np.concatenate(scales, axis=-1)


_VIEW_FUNCTIONS = {
    "full": full_view,
    "cropped": cropped_view,
    "lowres": lowres_view,
    "multiscale": multiscale_view,
}


def rj23_views(
    stamps: npt.ArrayLike,
    views: Sequence[ViewName] = VIEW_NAMES,
    *,
    large_values: LargeValues = "paper",
    chunk_size: int = _DEFAULT_CHUNK,
) -> dict[ViewName, F32]:
    """The requested RJ23 views of raw cutouts ``(N, 63, 63, 3)``, channels last, float32.

    :func:`normalise` and then each view function, ``chunk_size`` stamps at a time, so the
    peak memory is the outputs plus one chunk's intermediates. The result does not depend
    on ``chunk_size``. Shapes per stamp are :data:`VIEW_SHAPES`.

    Raises:
        ValueError: on a wrong shape or dtype, an unknown or repeated view, an unknown
            ``large_values`` or a ``chunk_size`` below 1.
    """
    array = _check_raw(stamps)
    _check_chunk(chunk_size)
    unknown = [view for view in views if view not in _VIEW_FUNCTIONS]
    if unknown or len(set(views)) != len(views):
        raise ValueError(f"views must be distinct names from {VIEW_NAMES}, got {list(views)}")
    n = array.shape[0]
    out = {view: np.empty((n, *VIEW_SHAPES[view]), dtype=np.float32) for view in views}
    for start in range(0, n, chunk_size):
        stop = start + chunk_size
        normalised = _normalise_chunk(array[start:stop], large_values)
        for view in views:
            out[view][start:stop] = _VIEW_FUNCTIONS[view](normalised)
    return out


def unit_sphere(ra: npt.ArrayLike, dec: npt.ArrayLike) -> F32:
    """RJ23's position input: ``(x, y, z)`` on the unit sphere, ``(N, 3)`` float32.

    ``x = cos(ra) cos(dec)``, ``y = sin(ra) cos(dec)``, ``z = sin(dec)``, with ``ra`` and
    ``dec`` in degrees (computed in float64). ``ra`` and ``dec`` are 1-D and of equal length.

    Raises:
        ValueError: on mismatched or non-1-D inputs, non-finite values or ``|dec| > 90``.
    """
    ra_deg = np.asarray(ra, dtype=np.float64)
    dec_deg = np.asarray(dec, dtype=np.float64)
    if ra_deg.ndim != 1 or ra_deg.shape != dec_deg.shape:
        raise ValueError(
            f"expected 1-D ra and dec of equal length, got {ra_deg.shape}, {dec_deg.shape}"
        )
    if not (np.isfinite(ra_deg).all() and np.isfinite(dec_deg).all()):
        raise ValueError("ra and dec must be finite")
    if np.any(np.abs(dec_deg) > 90):
        raise ValueError("dec must lie in [-90, 90] degrees")
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    xyz = np.stack(
        [np.cos(ra_rad) * np.cos(dec_rad), np.sin(ra_rad) * np.cos(dec_rad), np.sin(dec_rad)],
        axis=1,
    )
    return xyz.astype(np.float32)


def channels_first(images: npt.NDArray[np.float32]) -> F32:
    """``(N, H, W, C)`` -> a contiguous ``(N, C, H, W)`` copy (torch's layout).

    Raises:
        ValueError: if ``images`` is not 4-D.
    """
    array = np.asarray(images)
    if array.ndim != 4:
        raise ValueError(f"expected (N, H, W, C) images, got shape {array.shape}")
    return np.ascontiguousarray(np.moveaxis(array, -1, 1))
