"""Class-aware training augmentation of the paper model (plan rows C5 and C6; ADR 0006).

Both rows come from a visual inspection of the Train stamps per class and channel
(2026-11-08): SN has 1,396 Train stamps, so the tiling repeats each about 10 times (C5 makes
the copies differ), and about half of the inspected Bogus are a hot pixel or a cosmic ray: a
1-2 pixel spike at the centre of the science and the difference, on an unremarkable template
(C6 makes more of them). Everything here runs in ``PaperClassifierModule.training_step`` on
the device-resident batch; validation and test are never augmented.

**The difference is computed, not augmented on its own.** In the stored crops (each channel
min-max normalised on its own), the difference is still close to an affine function of the
science and the template: a least-squares fit ``diff ~ a*sci + b*tmpl + c`` over the 441
pixels of a crop has a median R^2 of 0.77-0.95 per class on ``red`` Train, and the sign of
``a`` agrees with ``isdiffpos`` for 98-99.6 % of the stamps (``a > 0`` for a positive alert,
whose stored ``isdiffpos`` is 0; a negative alert's difference is stored flipped).
:func:`stamp_fits` computes ``(a, b, c, R^2)`` once per Train stamp, with the background
sigmas of the science and template crops, and every change of the science (``dsci``) or the
template (``dtmpl``) moves the difference by ``a*dsci + b*dtmpl``. Independent noise in the
difference would break a correlation the model can use. There is no independent term for the
difference's own noise: in the raw images it *is* the science's noise minus the (convolved)
template's, which the relation already carries.

- **C5, ``aug_noise``** (:func:`noise_augment`): per stamp, ``dsci`` and ``dtmpl`` are
  Gaussian, of standard deviation ``u * aug_noise * sigma_bg`` with ``u ~ U(0, 1)`` drawn per
  channel and ``sigma_bg`` that crop channel's background sigma (1.4826 x the MAD of its
  two-pixel border, :data:`BORDER_WIDTH`); then the difference follows, and the three
  channels are min-max normalised again (:func:`renormalise`), so training inputs keep the
  [0, 1], minimum 0, maximum 1 statistics of the evaluation inputs. Stamps with
  ``R^2 < aug_min_r2`` (their difference is not explained by the relation) are left
  untouched.
- **C5, ``aug_feature_jitter``** (:func:`jitter_features`): Gaussian jitter, in standardised
  units, on the continuous features only. :data:`DISCRETE_FEATURES` (``isdiffpos``, binary;
  the integer counts ``ndethist``, ``ncovhist``, ``non_detections``, found as exact integer
  lattices in the stored values) are never changed.
- **C6, ``synthetic_bogus``** (:func:`synthetic_bogus`): a fraction of the batch rows is
  replaced by synthetic hot-pixel Bogus, each built from the row's own stamp (the rows are
  a shuffled stream, so the source is a random training stamp): the template is unchanged;
  the science is the static sky the relation implies, ``-(b/a) * tmpl``, plus Gaussian noise
  of the source science's own background sigma, plus a spike at the crop centre
  (:func:`spike_patterns`: one pixel, or a 2-pixel horizontal or vertical line, 10-50 sigma);
  the difference is the relation applied to that science and template,
  ``a * (noise + spike) + c`` (so a negative alert's difference gets a negative spike, and
  the source's own transient, in its residual, is gone); then the min-max. The label is
  Bogus, and the features are those of a real Bogus row of the same batch whose
  ``isdiffpos`` matches the sign of ``a`` (a donor from the batch rather than a separate
  bank: always the run's own training part, also for split runs). A real stamp is never
  relabelled: the replaced row's stamp is only the raw material.

**The spike is sharper than any PSF** (:func:`moment_fwhm`, :data:`MIN_PSF_FWHM_PX`). The
smallest PSF FWHM measured on ``red`` Train is 1.19 px: a pixel-integrated circular Gaussian
fitted to the central source of the difference of every Asteroid and SN (fits with
S/N > 15; minimum 1.21 px for the Asteroids, 1.19 px for the SN;
``scripts/inspect_class_augment.py``). The difference is used because its static sources
are subtracted; in the science, a static neighbour can spoil the fit (its first percentile,
1.22-1.25 px, agrees). The widest spike, a 2-pixel line of equal
pixels, has a circular-Gaussian-equivalent FWHM of 0.48 px with the same estimator (the
pixel second moments less a pixel's own 1/12; 0.96 px even along the line; a single pixel:
0), so a synthetic Bogus cannot pass for an Asteroid, whose template is just as empty. The alert
feature ``fwhm`` cannot replace that measurement: mapped to pixels through the fits
(``FWHM ~ 1.07 * z + 2.29``, correlation 0.86), its smallest values are about 0 px.

**Masked pixels.** A crop channel with more than one exact zero has former NaNs (the NaN
approximation, ``data.float_input``); :func:`masked_pixels` finds them, and they stay out of
the minimum and maximum and stay 0, as in the evaluation inputs. C6 does not use stamps with
masked pixels as sources.

**The context channels** (``context: avgpool63``) are not augmented: the relation and the
sigmas are those of the 21x21 crop. C5 with context noises the crop only; C6 with context is
rejected (``config.Settings``), since it would need a synthetic context.

**Randomness.** Each augmentation has its own NumPy generator seeded with
``[seed, global_step, stream]`` (:data:`NOISE_STREAM`, :data:`JITTER_STREAM`,
:data:`SYNTHETIC_STREAM`; ``rot90_augment`` keeps ``[seed, global_step]``): independent of
the device and of the global generator, the same after a ``--resume``, and turning one
augmentation on does not change the draws of another. Every draw has a fixed size per batch,
whichever rows end up augmented.
"""

from __future__ import annotations

import warnings
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import torch

from astro_simclr.data.schema import DISCARDED_FEATURES, FEATURE_NAMES, LABEL_NAMES
from astro_simclr.paper_model import PAPER_CROP

__all__ = [
    "BORDER_WIDTH",
    "DEFAULT_MIN_R2",
    "DISCRETE_FEATURES",
    "FIT_COLUMNS",
    "JITTER_STREAM",
    "KEPT_FEATURE_NAMES",
    "MIN_PSF_FWHM_PX",
    "NOISE_STREAM",
    "SPIKE_LINE_PROBABILITY",
    "SPIKE_LINE_RATIO",
    "SPIKE_SNR",
    "SYNTHETIC_STREAM",
    "augment_rng",
    "continuous_feature_mask",
    "fit_exclusions",
    "jitter_features",
    "masked_pixels",
    "moment_fwhm",
    "noise_augment",
    "perturb",
    "renormalise",
    "spike_patterns",
    "stamp_fits",
    "synthetic_bogus",
]

#: The columns of :func:`stamp_fits`: ``diff ~ a*sci + b*tmpl + c``, its R^2, and the
#: background sigmas of the science and template crops.
FIT_COLUMNS: Final = ("a", "b", "c", "r2", "sigma_science", "sigma_template")
#: ``aug_min_r2`` default: below it a stamp is not noised (C5) nor used as a source (C6).
DEFAULT_MIN_R2: Final = 0.5
#: Width in pixels of the crop border whose MAD gives the background sigma (152 pixels).
BORDER_WIDTH: Final = 2
#: The MAD of a Gaussian times this is its standard deviation.
_MAD_TO_SIGMA: Final = 1.4826

#: The 23 features the paper model reads, in order (columns 13-15 dropped).
KEPT_FEATURE_NAMES: Final = tuple(
    name for i, name in enumerate(FEATURE_NAMES) if i not in DISCARDED_FEATURES
)
#: Kept features that :func:`jitter_features` never changes: ``isdiffpos`` takes 2 values;
#: the counts are exact integer lattices in the stored (standardised) values.
DISCRETE_FEATURES: Final = ("isdiffpos", "ndethist", "ncovhist", "non_detections")
_ISDIFFPOS: Final = KEPT_FEATURE_NAMES.index("isdiffpos")
#: Stored ``isdiffpos`` is 0 (a positive alert, ``a > 0``) or 2.12 (negative).
_ISDIFFPOS_SPLIT: Final = 1.0
_BOGUS: Final = LABEL_NAMES.index("Bogus")

#: The smallest PSF FWHM in the data, in pixels (measured on the differences; module
#: docstring).
MIN_PSF_FWHM_PX: Final = 1.19
#: The spike's peak, in units of the source science's background sigma: U(low, high).
SPIKE_SNR: Final = (10.0, 50.0)
#: Probability that the spike is a 2-pixel line instead of a single pixel.
SPIKE_LINE_PROBABILITY: Final = 0.5
#: The line's second pixel, as a fraction of the peak: U(low, high).
SPIKE_LINE_RATIO: Final = (0.25, 1.0)
#: The second pixel's offset (row, column): right, left, up, down. No diagonals: along its
#: axis a diagonal pair of equal pixels is 1.52 px wide, more than the PSF minimum.
_LINE_OFFSETS: Final = ((0, 1), (0, -1), (1, 0), (-1, 0))

#: Generator streams (the third seed word) of each augmentation.
NOISE_STREAM: Final = 1
JITTER_STREAM: Final = 2
SYNTHETIC_STREAM: Final = 3

_FWHM_PER_SIGMA: Final = 2.0 * float(np.sqrt(2.0 * np.log(2.0)))
_CHUNK: Final = 4096


def augment_rng(seed: int, step: int, stream: int) -> np.random.Generator:
    """The generator of augmentation ``stream`` at global step ``step`` of a run seeded ``seed``.

    A ``SeedSequence`` of the three words, so every bit of each one matters (a torch seed
    keeps 32 bits) and the draws do not depend on the device.
    """
    return np.random.default_rng([seed, step, stream])


def _border(side: int) -> npt.NDArray[np.bool_]:
    inner = np.zeros((side, side), dtype=bool)
    inner[BORDER_WIDTH:-BORDER_WIDTH, BORDER_WIDTH:-BORDER_WIDTH] = True
    return ~inner


def _background_sigma(
    channel: npt.NDArray[np.float64], masked: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    """1.4826 x the MAD of the unmasked border pixels of (N, S, S) crops; 0 if none."""
    border = _border(channel.shape[-1])
    values = np.where(masked, np.nan, channel)[:, border]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows: "All-NaN slice"
        median = np.nanmedian(values, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(values - median), axis=1)
    return np.nan_to_num(_MAD_TO_SIGMA * mad, nan=0.0)


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


def fit_exclusions(
    fits: npt.NDArray[np.float32], labels: npt.NDArray[np.int64], min_r2: float
) -> dict[str, Any]:
    """The share of each class's stamps whose fit has ``R^2 < min_r2`` (not augmented).

    ``fits`` (N, 6) of :func:`stamp_fits`, ``labels`` (N,). Returns ``{"min_r2": min_r2,
    "excluded": {class name: fraction}, "count": {class name: stamps}}`` (a class with no
    stamps has fraction 0), which a run records in ``metrics.json["paper_data"]``.
    """
    excluded: dict[str, float] = {}
    count: dict[str, int] = {}
    for label, name in enumerate(LABEL_NAMES):
        rows = labels == label
        count[name] = int(rows.sum())
        excluded[name] = float((fits[rows, 3] < min_r2).mean()) if rows.any() else 0.0
    return {"min_r2": min_r2, "excluded": excluded, "count": count}


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


def noise_augment(
    images: torch.Tensor,
    fits: torch.Tensor,
    scale: float,
    rng: np.random.Generator,
    *,
    min_r2: float = DEFAULT_MIN_R2,
    skip: npt.NDArray[np.bool_] | None = None,
) -> tuple[torch.Tensor, npt.NDArray[np.bool_]]:
    """C5's noise: (B, C, S, S) crops with background noise, and which rows were changed.

    Draws ``u`` (B, 2) uniform and ``z`` (B, 2, S, S) standard normal, always, then for the
    rows with ``R^2 >= min_r2`` (and not in ``skip``): ``delta_sci = u0 * scale * sigma_sci
    * z0``, ``delta_tmpl`` likewise, the difference by :func:`perturb`, the masked pixels
    kept masked, and :func:`renormalise`. Other rows, and channels beyond the first 3 (the
    context), are returned unchanged (bit for bit). ``images`` is not modified.
    """
    n, side = images.shape[0], images.shape[-1]
    u = rng.random((n, 2))
    z = rng.standard_normal((n, 2, side, side))
    fits_cpu = fits.detach().cpu().numpy()
    rows = fits_cpu[:, 3] >= min_r2
    if skip is not None:
        rows &= ~skip
    if scale == 0 or not rows.any():
        return images, np.zeros(n, dtype=bool)
    sigma = u * scale * fits_cpu[:, 4:6]
    deltas = torch.from_numpy(sigma[:, :, None, None] * z).to(images.device, images.dtype)
    crops = images[:, :3]
    masked = masked_pixels(crops)
    noisy = renormalise(perturb(crops, fits, deltas[:, 0], deltas[:, 1]), masked)
    keep = torch.from_numpy(rows).to(images.device)[:, None, None, None]
    out = images.clone()
    out[:, :3] = torch.where(keep, noisy, crops)
    return out, rows


def continuous_feature_mask() -> npt.NDArray[np.bool_]:
    """(23,) True for the kept features that :func:`jitter_features` may change."""
    return np.array([name not in DISCRETE_FEATURES for name in KEPT_FEATURE_NAMES])


def jitter_features(features: torch.Tensor, sigma: float, rng: np.random.Generator) -> torch.Tensor:
    """C5's feature jitter: ``features`` (B, 23) plus N(0, sigma^2) on the continuous ones.

    The draw is (B, 23) standard normal; :data:`DISCRETE_FEATURES` columns stay exactly as
    they were. ``features`` is not modified.
    """
    noise = rng.standard_normal(tuple(features.shape)) * sigma * continuous_feature_mask()
    return features + torch.from_numpy(noise).to(features.device, features.dtype)


def spike_patterns(rng: np.random.Generator, n: int, side: int = PAPER_CROP) -> np.ndarray:
    """``n`` unit-peak spikes (n, side, side) float64 at the centre pixel ``side // 2``.

    Each is one pixel of value 1, or, with probability :data:`SPIKE_LINE_PROBABILITY`, a
    2-pixel line: one 4-neighbour (right, left, up or down, uniform) of value
    ``U(*SPIKE_LINE_RATIO)``. Draws ``n`` uniforms (line or not), ``n`` integers (direction)
    and ``n`` uniforms (ratio), always.
    """
    line = rng.random(n) < SPIKE_LINE_PROBABILITY
    direction = rng.integers(0, len(_LINE_OFFSETS), size=n)
    ratio = rng.uniform(*SPIKE_LINE_RATIO, size=n)
    centre = side // 2
    out = np.zeros((n, side, side))
    out[:, centre, centre] = 1.0
    for i in np.flatnonzero(line):
        dr, dc = _LINE_OFFSETS[direction[i]]
        out[i, centre + dr, centre + dc] = ratio[i]
    return out


def moment_fwhm(image: npt.NDArray[np.floating[Any]]) -> float:
    """The circular-Gaussian-equivalent FWHM, in pixels, of a non-negative source image.

    From the intensity-weighted second moments about the centroid, minus each pixel's own
    variance (1/12 per axis, a uniform pixel): ``2 sqrt(2 ln 2) sqrt(max((Ixx + Iyy) / 2 -
    1/12, 0))``. For a pixel-integrated circular Gaussian it is the Gaussian's FWHM (what
    the PSF measurement fits); a single pixel gives 0. Meant for noise-free images such as
    :func:`spike_patterns`.

    Raises:
        ValueError: if ``image`` is not 2-D, has a negative pixel or sums to 0.
    """
    if image.ndim != 2:
        raise ValueError(f"image must be 2-D, got shape {image.shape}")
    weights = np.asarray(image, dtype=np.float64)
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("image must be non-negative with a positive sum")
    weights = weights / weights.sum()
    rows, cols = np.indices(weights.shape)
    variances = []
    for axis in (rows, cols):
        mean = (weights * axis).sum()
        variances.append((weights * (axis - mean) ** 2).sum())
    variance = max(sum(variances) / 2 - 1 / 12, 0.0)
    return _FWHM_PER_SIGMA * float(np.sqrt(variance))


def synthetic_bogus(
    images: torch.Tensor,
    features: torch.Tensor,
    targets: torch.Tensor,
    fits: torch.Tensor,
    fraction: float,
    rng: np.random.Generator,
    *,
    min_r2: float = DEFAULT_MIN_R2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, npt.NDArray[np.bool_]]:
    """C6: replace ``round(fraction * B)`` batch rows with synthetic hot-pixel Bogus.

    ``images`` (B, 3, S, S) crops (no context channels), ``features`` (B, 23), ``targets`` (B,
    5) one-hot, ``fits`` (B, 6) of :func:`stamp_fits`. Returns the new images, features and
    targets, and which rows were replaced. The construction is in the module docstring. A row
    can be a source when ``R^2 >= min_r2``, ``a * b < 0`` (the template enters the static
    science with a positive scale), its science sigma is positive and no pixel is masked; a
    source is used when the batch has a real Bogus row whose ``isdiffpos`` matches the sign of
    ``a`` (the feature donor, chosen uniformly among them; the original rows, before any
    replacement). The rows are taken in a random order, so fewer than ``round(fraction * B)``
    are replaced when sources or donors run out. Draws, always: a permutation of B, B uniforms
    (donors), (B, S, S) normals (noise), B uniforms (spike S/N) and :func:`spike_patterns` of B.
    The inputs are not modified.

    Raises:
        ValueError: if ``images`` is not (B, 3, S, S).
    """
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"synthetic_bogus takes (B, 3, S, S) crops, got {tuple(images.shape)}")
    n, side = images.shape[0], images.shape[-1]
    order = rng.permutation(n)
    donor_u = rng.random(n)
    noise = rng.standard_normal((n, side, side))
    snr = rng.uniform(*SPIKE_SNR, size=n)
    spikes = spike_patterns(rng, n, side)
    wanted = round(fraction * n)
    replaced = np.zeros(n, dtype=bool)
    if wanted == 0:
        return images, features, targets, replaced

    fits_cpu = fits.detach().cpu().numpy().astype(np.float64)
    a, b, c, r2, sigma = (fits_cpu[:, k] for k in (0, 1, 2, 3, 4))
    unmasked = ~masked_pixels(images[:, :3]).flatten(1).any(dim=1).cpu().numpy()
    source = (r2 >= min_r2) & (a * b < 0) & (sigma > 0) & unmasked
    labels = targets.argmax(dim=1).cpu().numpy()
    positive = features[:, _ISDIFFPOS].cpu().numpy() < _ISDIFFPOS_SPLIT
    donors = {
        sign: np.flatnonzero((labels == _BOGUS) & (positive == sign)) for sign in (True, False)
    }
    donor_of = np.full(n, -1)
    for row in order:
        if replaced.sum() == wanted:
            break
        pool = donors[bool(a[row] > 0)]
        if source[row] and pool.size:
            replaced[row] = True
            donor_of[row] = pool[int(donor_u[row] * pool.size)]
    if not replaced.any():
        return images, features, targets, replaced

    rows = np.flatnonzero(replaced)
    tmpl = images[rows, 1].detach().cpu().numpy().astype(np.float64)
    scale = sigma[rows, None, None]
    signal = scale * noise[rows] + (snr[rows] * sigma[rows])[:, None, None] * spikes[rows]
    science = -(b[rows] / a[rows])[:, None, None] * tmpl + signal
    difference = a[rows, None, None] * signal + c[rows, None, None]
    built = torch.from_numpy(np.stack([science, tmpl, difference], axis=1))
    built = built.to(images.device, images.dtype)
    built = renormalise(built, torch.zeros_like(built, dtype=torch.bool))

    index = torch.from_numpy(rows).to(images.device)
    out_images = images.clone()
    out_images[index] = built
    out_features = features.clone()
    out_features[index] = features[torch.from_numpy(donor_of[rows]).to(features.device)]
    out_targets = targets.clone()
    out_targets[index] = torch.nn.functional.one_hot(
        torch.full((rows.size,), _BOGUS, device=targets.device), targets.shape[1]
    ).to(targets.dtype)
    return out_images, out_features, out_targets, replaced
