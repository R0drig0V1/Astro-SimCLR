"""The cross-channel relation of a stamp, and the pieces of the noise that keeps it (C5).

Engine-free (no Lightning import): the paper model's class-aware augmentation
(:mod:`astro_simclr.training.class_augment`, plan row C5) and the SimCLR views of phase 5
(:mod:`astro_simclr.simclr.views`) both use these functions, so they live outside the
``training`` package, whose ``__init__`` imports Lightning. ``class_augment`` re-exports
every name, so its callers are unchanged; the behaviour is the one ADR 0006 records, moved
without a change (ADR 0018 records the move).

- :func:`stamp_fits`: per stamp, the least-squares fit ``diff ~ a*sci + b*tmpl + c``, its
  R^2 and the background sigmas of the science and the template (1.4826 x the MAD of the
  :data:`BORDER_WIDTH`-pixel border). It works on any square side: the paper's 21-pixel
  crops and the 63-pixel stamps of phase 5.
- the pieces of C5's noise: :func:`perturb` moves the science and the template and the
  difference through the fit, :func:`renormalise` is the per-channel min-max again, with
  the former-NaN pixels (:func:`masked_pixels`) kept out and 0. The paper model's noise
  (``class_augment.noise_augment``, its host plan and device apply) and the SimCLR views'
  noise (``simclr.views.background_noise``) are both built from them.
"""

from __future__ import annotations

import warnings
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import torch

__all__ = [
    "BORDER_WIDTH",
    "DEFAULT_MIN_R2",
    "FIT_COLUMNS",
    "MAD_TO_SIGMA",
    "border_mask",
    "masked_pixels",
    "perturb",
    "renormalise",
    "stamp_fits",
]

#: The columns of :func:`stamp_fits`: ``diff ~ a*sci + b*tmpl + c``, its R^2, and the
#: background sigmas of the science and template crops.
FIT_COLUMNS: Final = ("a", "b", "c", "r2", "sigma_science", "sigma_template")
#: ``aug_min_r2`` default: below it a stamp is not noised (C5) nor used as a source (C6).
DEFAULT_MIN_R2: Final = 0.5
#: Width in pixels of the crop border whose MAD gives the background sigma (152 pixels).
BORDER_WIDTH: Final = 2
#: The MAD of a Gaussian times this is its standard deviation.
MAD_TO_SIGMA: Final = 1.4826

_CHUNK: Final = 4096


def border_mask(side: int) -> npt.NDArray[np.bool_]:
    """(side, side) True on the outer :data:`BORDER_WIDTH` pixels of a square crop."""
    inner = np.zeros((side, side), dtype=bool)
    inner[BORDER_WIDTH:-BORDER_WIDTH, BORDER_WIDTH:-BORDER_WIDTH] = True
    return ~inner


def _background_sigma(
    channel: npt.NDArray[np.float64], masked: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    """1.4826 x the MAD of the unmasked border pixels of (N, S, S) crops; 0 if none."""
    border = border_mask(channel.shape[-1])
    values = np.where(masked, np.nan, channel)[:, border]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows: "All-NaN slice"
        median = np.nanmedian(values, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(values - median), axis=1)
    return np.nan_to_num(MAD_TO_SIGMA * mad, nan=0.0)


def stamp_fits(images: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float32]:
    """``(a, b, c, R^2, sigma_sci, sigma_tmpl)`` of each crop, float32 (N, 6) (:data:`FIT_COLUMNS`).

    ``images`` is (N, C, S, S), channels science, template, difference first (a context
    after them is ignored), normally the [0, 1] crops of a ``TensorSplit``. The fit is the
    least-squares ``diff ~ a*sci + b*tmpl + c`` over the S x S pixels, in float64 (a
    pseudo-inverse, so a constant channel gives finite coefficients); R^2 is
    ``1 - SS_res / SS_tot``, 0 for a constant difference. The sigmas are 1.4826 x the MAD of
    the border pixels (:data:`BORDER_WIDTH`) that are not masked (:func:`masked_pixels`),
    0 when all are. Runs in chunks of 4096 crops.

    Raises:
        ValueError: if ``images`` is not (N, C, S, S) with C >= 3.
    """
    if images.ndim != 4 or images.shape[1] < 3 or images.shape[2] != images.shape[3]:
        raise ValueError(f"images must be (N, C >= 3, S, S), got {images.shape}")
    out = np.empty((images.shape[0], len(FIT_COLUMNS)), dtype=np.float32)
    for start in range(0, images.shape[0], _CHUNK):
        crops = np.asarray(images[start : start + _CHUNK, :3], dtype=np.float64)
        n = crops.shape[0]
        sci, tmpl, diff = (crops[:, k].reshape(n, -1) for k in range(3))
        design = np.stack([sci, tmpl, np.ones_like(sci)], axis=-1)
        gram = design.transpose(0, 2, 1) @ design
        coef = (np.linalg.pinv(gram) @ (design.transpose(0, 2, 1) @ diff[..., None]))[..., 0]
        residual = diff - (design @ coef[..., None])[..., 0]
        total = ((diff - diff.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
        safe = np.where(total > 0, total, 1.0)
        r2 = np.where(total > 0, 1.0 - (residual**2).sum(axis=1) / safe, 0.0)
        zeros = crops == 0
        masked = zeros & (zeros.sum(axis=(-2, -1), keepdims=True) > 1)
        sigmas = [_background_sigma(crops[:, k], masked[:, k]) for k in (0, 1)]
        out[start : start + n] = np.column_stack([coef, r2, *sigmas])
    return out


def masked_pixels(images: torch.Tensor) -> torch.Tensor:
    """The former-NaN pixels of (B, C, S, S) crops: exact zeros of a channel with several.

    A min-max normalised channel has one exact zero, its minimum; more mean NaN pixels set
    to 0 (``data.float_input``'s NaN approximation and NaN -> 0), or a tied minimum.
    """
    zeros = images == 0
    return zeros & (zeros.sum(dim=(-2, -1), keepdim=True) > 1)


def renormalise(images: torch.Tensor, masked: torch.Tensor) -> torch.Tensor:
    """Per-channel min-max of (B, C, S, S) without epsilon, ``masked`` excluded, NaN -> 0.

    ALeRCE's ``x -= min; x /= max`` (``data.float_input``) over the unmasked pixels; masked
    pixels, constant channels (0 / 0) and all-masked channels become 0.
    """
    inf = torch.tensor(float("inf"), dtype=images.dtype, device=images.device)
    low = torch.where(masked, inf, images).amin(dim=(-2, -1), keepdim=True)
    shifted = images - low
    high = torch.where(masked, -inf, shifted).amax(dim=(-2, -1), keepdim=True)
    out = shifted / high
    return torch.where(masked | ~torch.isfinite(out), torch.zeros_like(out), out)


def perturb(
    images: torch.Tensor, fits: torch.Tensor, delta_sci: torch.Tensor, delta_tmpl: torch.Tensor
) -> torch.Tensor:
    """The crops (B, 3, S, S) after ``delta_sci`` and ``delta_tmpl`` (B, S, S), before the min-max.

    Science and template move by their deltas; the difference by ``a*delta_sci +
    b*delta_tmpl`` (``fits`` (B, 6), :data:`FIT_COLUMNS`), so a stamp whose difference is
    exactly ``a*sci + b*tmpl + c`` keeps that relation. Only the first 3 channels are read.
    """
    a = fits[:, 0, None, None]
    b = fits[:, 1, None, None]
    delta_diff = a * delta_sci + b * delta_tmpl
    return images[:, :3] + torch.stack([delta_sci, delta_tmpl, delta_diff], dim=1)
