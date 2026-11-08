"""Stamp preprocessing, moved from the legacy ``utils/dataset.py``.

The three legacy functions keep their arithmetic exactly (the characterization fixtures in
``tests/fixtures/data/`` pin them, ``atol=1e-6``):

- :func:`repair_non_squared_stamp` pads a stamp cut at the CCD edge back to 63x63 with NaN;
- :func:`preprocess_stamps` min-max normalizes each channel to [0, 1] and replaces
  NaN/inf;
- :func:`img_float2int` turns a [0, 1] float stamp into uint8, as the first transform of
  both legacy datasets did.

The first two build the pickles (the pretraining set and the regeneration scripts of the
backlog); they are not applied when loading. :func:`stamp_to_uint8` is the checked form of
:func:`img_float2int` that the loaders use.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt

from astro_simclr.data.schema import STAMP_SHAPE

__all__ = ["img_float2int", "preprocess_stamps", "repair_non_squared_stamp", "stamp_to_uint8"]

#: Side of a full stamp.
_SIDE = STAMP_SHAPE[0]

#: Pixel coordinate below which a short stamp was cut on its low side (legacy constant).
_EDGE = 40.0


def repair_non_squared_stamp(
    stamp: npt.NDArray[np.floating], xpos: float, ypos: float
) -> npt.NDArray[np.floating]:
    """Pad a stamp cut at the CCD edge back to 63x63 with NaN.

    A stamp of the right size is returned unchanged (the same object). A short side is
    padded with NaN rows/columns: before the stamp when the source position (``ypos`` for
    rows, ``xpos`` for columns) is below 40 px, after it otherwise. The padding is float64,
    so a padded float32 stamp comes back as float64 (legacy behaviour).

    Raises:
        ValueError: if the stamp is not (H, W, 3) with H, W <= 63.
    """
    if stamp.ndim != 3 or stamp.shape[2] != STAMP_SHAPE[2]:
        raise ValueError(f"stamp must have shape (H, W, 3), got {stamp.shape}")
    height, width, _ = stamp.shape
    if height > _SIDE or width > _SIDE:
        raise ValueError(f"stamp is larger than {_SIDE}x{_SIDE}: {stamp.shape}")

    if height == _SIDE and width == _SIDE:
        return stamp

    if height != _SIDE:
        nan_pad = np.full((_SIDE - height, width, 3), np.nan)
        parts = [nan_pad, stamp] if ypos < _EDGE else [stamp, nan_pad]
        stamp = np.concatenate(parts, axis=0)

    if width != _SIDE:
        nan_pad = np.full((_SIDE, _SIDE - width, 3), np.nan)
        parts = [nan_pad, stamp] if xpos < _EDGE else [stamp, nan_pad]
        stamp = np.concatenate(parts, axis=1)
    return stamp


def preprocess_stamps(stamp: npt.NDArray[np.floating], nan_val: float) -> npt.NDArray[np.floating]:
    """Min-max normalize each channel of ``stamp`` to [0, 1], then replace NaN with ``nan_val``.

    ``inf`` and ``-inf`` are treated as NaN: they take no part in the minimum and maximum,
    and become ``nan_val``. The denominator gets ``1e-6`` added, so a constant channel
    becomes 0 and the maximum maps slightly below 1. A channel that is entirely NaN/inf
    becomes ``nan_val`` everywhere (numpy warns "All-NaN slice"). The dtype is preserved.
    """
    stamp_image = np.nan_to_num(stamp, nan=np.nan, posinf=np.nan, neginf=np.nan)
    stamp_image = stamp_image - np.nanmin(stamp_image, axis=(0, 1))
    stamp_image = stamp_image / (np.nanmax(stamp_image, axis=(0, 1)) + 0.000001)
    result: npt.NDArray[np.floating] = np.nan_to_num(
        stamp_image, nan=nan_val, posinf=nan_val, neginf=nan_val
    )
    return result


def img_float2int(img: npt.NDArray[np.floating]) -> npt.NDArray[np.uint8]:
    """Convert a [0, 1] float stamp to uint8: ``uint8(round(255 * img))``.

    The product and the rounding are done in the dtype of ``img``, so float32 and float64
    inputs can round differently near ``(k + 0.5) / 255``: always apply it to the stored
    dtype. Values outside [0, 1] or non-finite give undefined uint8 values (numpy only
    warns); use :func:`stamp_to_uint8` to reject them.
    """
    img_255 = 255 * img
    # np.uint8(array) returns an array; numpy's stubs type it as a scalar.
    return cast("npt.NDArray[np.uint8]", np.uint8(np.round(img_255)))


def stamp_to_uint8(img: npt.NDArray[Any]) -> npt.NDArray[np.uint8]:
    """:func:`img_float2int` with its preconditions checked.

    Gives the same bytes as :func:`img_float2int` for valid input.

    Raises:
        ValueError: if ``img`` is not floating point, has a NaN or an inf, or has a value
            that rounds outside 0-255 (that is, outside about [-0.002, 1.002]).
    """
    if not np.issubdtype(img.dtype, np.floating):
        raise ValueError(f"stamps must be floating point in [0, 1], got dtype {img.dtype}")
    rounded = np.round(255 * img)
    if not np.isfinite(rounded).all():
        raise ValueError("stamp has NaN or inf values")
    low, high = float(rounded.min()), float(rounded.max())
    if low < 0 or high > 255:
        raise ValueError(f"stamp values must be in [0, 1], got {low / 255:.4g}..{high / 255:.4g}")
    return rounded.astype(np.uint8)
