"""The paper's float input path: centre crop, per-crop min-max, NaN -> 0 (task 2.1).

Carrasco-Davis et al. 2021 (ALeRCE's stamp classifier, the model "M1" of phase 2) feeds the
network the central 21x21 pixels of each stamp, re-normalised per channel on the crop. In
ALeRCE's code (``alercebroker/pipeline@a3705ac``,
``stamp_classifier_step/model/modules/data_loaders/ztf_preprocessor.py``) that is the
pipeline ``image_crop_at_center`` -> ``image_normalize_by_image`` -> ``image_nan_to_num``:

1. crop rows and columns ``[c - 10, c + 11)`` with ``c = 63 // 2 = 31``, that is
   ``[21, 42)``;
2. per stamp and channel, ``x -= nanmin(x)``, then ``x = x / nanmax(x)``, with **no
   epsilon**: a constant crop gives ``0 / 0 = NaN`` and an all-NaN crop stays NaN;
3. NaN -> 0.

:func:`crop_then_normalise` is that pipeline, with the same operations in the same order
and dtype, so it is bit-exact on ALeRCE's input (the golden fixtures of
``tests/fixtures/data/alerce_crop.npz``, produced by ALeRCE's own functions).

**The NaN approximation.** Our pickles do not hold the raw stamps: they hold them after
the same normalisation over the full 63x63 stamp and NaN -> 0 (``CROP_SIZE 63``). Min-max
is affine-invariant, so re-normalising a stored crop gives the same result as ALeRCE's,
except where the raw stamp had NaNs: ALeRCE's ``nanmin``/``nanmax`` skipped them, while in
the stored stamp they are 0, the same value as the stamp's minimum. With
``nan_approximation=True`` (the default), the zero pixels of a channel are treated as
former NaNs when that channel, **counted over the full stamp**, has more than one exact
zero (a channel without NaNs has exactly one, its minimum, unless the minimum is tied).
They are set back to NaN before the crop is normalised. The approximation is exact when the
channel's minimum pixel lies outside the crop; when it lies inside, it is masked too, and
the crop's minimum becomes its second-smallest value. A channel whose minimum is tied
without NaNs is masked as well. ``nan_approximation=False`` treats every stored zero as a
real value (a diagnostic of task 2.4).

The uint8 path of phase 1 (``center_crop`` at 27, torchvision's ``CenterCrop``: rows
``[18, 45)``) and this crop are concentric: the 21x21 crop is the 27x27 one minus 3 pixels
per side, which the paper's model zero-pads back to 27 (task 2.2a). ``tests/
test_data_float_input.py`` checks that alignment.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt

from astro_simclr.data.schema import STAMP_SHAPE

__all__ = [
    "PAPER_CROP",
    "approximated_stamps",
    "crop_bounds",
    "crop_split",
    "crop_then_normalise",
    "former_nan_mask",
]

#: The paper's crop side (ALeRCE's ``CROP_SIZE`` of M1: ``inputsize21``).
PAPER_CROP = 21

#: Default number of stamps per chunk in :func:`crop_split` (about 190 MB of float32
#: 63x63x3 temporaries).
_CHUNK = 4096


def crop_bounds(side: int, crop: int) -> tuple[int, int]:
    """Start and end (exclusive) of ALeRCE's centre crop of ``crop`` pixels on ``side``.

    ``image_crop_at_center``'s arithmetic: ``centre = side // 2``, ``half = crop // 2``,
    from ``centre - half`` to ``centre + half`` (+ 1 when ``side`` is odd). For 63 and 21
    that is ``(21, 42)``.

    Raises:
        ValueError: if ``crop`` is not in ``1..side``, or ``side`` and ``crop`` differ in
            parity (ALeRCE asserts the same: the crop could not be centred).
    """
    if not 1 <= crop <= side:
        raise ValueError(f"crop must be in 1..{side}, got {crop}")
    if side % 2 != crop % 2:
        raise ValueError(f"crop {crop} and side {side} must have the same parity")
    centre, half = side // 2, crop // 2
    return centre - half, centre + half + side % 2


def former_nan_mask(stamps: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.bool_]:
    """The pixels that the NaN approximation treats as former NaNs.

    ``stamps`` is ``(..., H, W, C)`` (one stamp or a batch). A pixel is masked when it is
    exactly 0 and its channel has more than one exact zero over the whole ``H x W`` stamp
    (see the module docstring). The result has the shape of ``stamps``.
    """
    zeros = stamps == 0
    flagged = zeros.sum(axis=(-3, -2), keepdims=True) > 1
    mask: npt.NDArray[np.bool_] = zeros & flagged
    return mask


def crop_then_normalise[F: np.floating[Any]](
    stamps: npt.NDArray[F], crop: int = PAPER_CROP, *, nan_approximation: bool = True
) -> npt.NDArray[F]:
    """Centre-crop, min-max normalise each channel on the crop (no epsilon), NaN -> 0.

    ``stamps`` is ``(..., H, W, C)`` floating point, square (``H == W``), stored stamps
    (finite) or raw ones (with NaNs). The result is a new ``(..., crop, crop, C)`` array of
    the same dtype, in [0, 1]: every channel of every crop has minimum 0, and maximum 1
    unless the crop is constant or all NaN, in which case it is all 0. The arithmetic is
    ALeRCE's (module docstring), in the input dtype. With ``nan_approximation``, the
    former NaNs of :func:`former_nan_mask`, counted over the full stamp, are excluded from
    the minimum and maximum and become 0. The input is never modified.

    Raises:
        ValueError: if ``stamps`` is not floating point, has fewer than 3 dimensions, is
            not square, or ``crop`` does not fit (see :func:`crop_bounds`).
    """
    if not np.issubdtype(stamps.dtype, np.floating):
        raise ValueError(f"stamps must be floating point, got dtype {stamps.dtype}")
    if stamps.ndim < 3 or stamps.shape[-3] != stamps.shape[-2]:
        raise ValueError(f"stamps must be (..., H, W, C) with H == W, got {stamps.shape}")
    begin, end = crop_bounds(stamps.shape[-3], crop)
    window = (..., slice(begin, end), slice(begin, end), slice(None))
    cropped: npt.NDArray[F] = np.array(stamps[window], copy=True)
    if nan_approximation:
        cropped[former_nan_mask(stamps)[window]] = np.nan
    axes = (-3, -2)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        # All-NaN crops ("All-NaN slice encountered") and constant crops (0 / 0) give NaN,
        # as in ALeRCE; both become 0 below.
        warnings.simplefilter("ignore", RuntimeWarning)
        cropped -= np.nanmin(cropped, axis=axes, keepdims=True)
        # ALeRCE writes `images = images / nanmax`; in place gives the same values.
        cropped /= np.nanmax(cropped, axis=axes, keepdims=True)
    cropped[np.isnan(cropped)] = 0
    return cropped


def crop_split(
    images: npt.NDArray[np.float32],
    crop: int = PAPER_CROP,
    *,
    nan_approximation: bool = True,
    chunk: int = _CHUNK,
) -> npt.NDArray[np.float32]:
    """:func:`crop_then_normalise` over a whole split, ``chunk`` stamps at a time.

    ``images`` is a float32 ``(N, 63, 63, 3)`` array, usually the read-only memory map of
    ``images_f32.npy``. Working in chunks keeps the temporaries small; the result,
    ``(N, crop, crop, 3)`` float32, is the same as one call on the whole array (the
    operations are per stamp).

    Raises:
        ValueError: if ``images`` is not float32 ``(N, 63, 63, 3)``, ``chunk < 1``, or
            ``crop`` does not fit.
    """
    if images.dtype != np.float32 or images.shape[1:] != STAMP_SHAPE:
        raise ValueError(
            f"images must be float32 (N, {', '.join(map(str, STAMP_SHAPE))}), "
            f"got {images.dtype} {images.shape}"
        )
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    crop_bounds(STAMP_SHAPE[0], crop)
    out = np.empty((len(images), crop, crop, STAMP_SHAPE[2]), dtype=np.float32)
    for start in range(0, len(images), chunk):
        block = np.asarray(images[start : start + chunk])
        out[start : start + chunk] = crop_then_normalise(
            block, crop, nan_approximation=nan_approximation
        )
    return out


def approximated_stamps(
    images: npt.NDArray[np.floating[Any]], crop: int = PAPER_CROP, *, chunk: int = _CHUNK
) -> npt.NDArray[np.bool_]:
    """Which stamps the NaN approximation touches: a masked zero lies inside the crop.

    ``images`` is ``(N, H, W, C)`` (a memory map is read ``chunk`` stamps at a time).
    Returns a boolean ``(N,)`` array. Only these stamps can differ between
    ``nan_approximation=True`` and ``False`` (task 2.2b reports them separately).

    Raises:
        ValueError: if ``images`` is not 4-dimensional and square, ``chunk < 1``, or
            ``crop`` does not fit.
    """
    if images.ndim != 4 or images.shape[1] != images.shape[2]:
        raise ValueError(f"images must be (N, H, W, C) with H == W, got {images.shape}")
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    begin, end = crop_bounds(images.shape[1], crop)
    flags = np.empty(len(images), dtype=np.bool_)
    for start in range(0, len(images), chunk):
        mask = former_nan_mask(np.asarray(images[start : start + chunk]))
        flags[start : start + chunk] = mask[:, begin:end, begin:end, :].any(axis=(1, 2, 3))
    return flags
