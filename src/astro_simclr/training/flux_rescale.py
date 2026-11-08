"""Transient flux rescaling of the paper model's training stamps (plan row C7; ADR 0007).

Real SN, AGN, VS and Asteroids are seen at many brightnesses, but each Train stamp shows its
transient at one. C7 makes the transient of a training stamp ``k`` times brighter or fainter,
**keeping the noise**, so its signal-to-noise really changes, and updates the features that
measure it. Training only, in ``PaperClassifierModule.training_step``, never for Bogus.

**Why not scale the difference.** Every crop channel is min-max normalised on its own
(``data.float_input``), and the min-max is scale-invariant: ``k * diff`` normalises to the
same input. The transient must change *relative to the noise*. So the central source of the
difference is modelled, and only the model is scaled:

1. :func:`fit_central_psf`, once per Train stamp (``experiments._build_paper``, next to
   ``class_augment.stamp_fits``): a pixel-integrated circular Gaussian plus a constant,
   fitted by Levenberg-Marquardt (per stamp, float64, the Jacobians by ``torch.func``) to
   the central 11x11 pixels of the normalised difference crop, masked pixels left out. The
   stamp is kept only if the fit is good (:data:`PSF_COLUMNS` ``ok``): finite, positive
   flux, centroid within :data:`MAX_CENTRE_OFFSET` px of the centre, FWHM in
   :data:`FWHM_RANGE` px and S/N (the Gaussian's peak over the residual RMS) at least
   :data:`MIN_FIT_SNR`. The difference of a negative alert is stored flipped, so its
   transient is a peak too.
2. :func:`flux_rescale`: ``diff += (k - 1) * PSF`` and ``sci += (k - 1) * PSF / a``, with
   ``PSF`` the fitted Gaussian without its constant and ``a`` the science coefficient of
   the stamp's cross-channel fit (``class_augment``, ``diff ~ a*sci + b*tmpl + c``): the
   science change that produces that difference change. For a negative alert (``a < 0``)
   a brighter alert is a fainter science, as it should be. The template is unchanged, the
   residual (noise, host, neighbours) is untouched, then the per-channel min-max with the
   masked pixels kept out (``class_augment.renormalise``).

**Features** (:func:`rescaled_features`), de-standardised with M1's statistics
(``data.m1_features``), changed, re-standardised:

- ``magpsf' = magpsf - 2.5 log10 k``;
- ``sigmapsf' = sigmapsf / k``: the magnitude error is ``1.0857 * sigma_flux / flux``, and
  ``sigma_flux`` is dominated by the background, which C7 keeps, so it scales as ``1 / k``.
  An approximation: the source's own Poisson noise, which grows with the flux, is ignored
  (small for the faint transients near the limit, where most of the headroom is used).

**The result is still an alert**: ``magpsf' <= diffmaglim`` (the 5-sigma limit, so
S/N >= 5), that is ``k >= 10^(-0.4 (diffmaglim - magpsf))``. Each rescaled row draws ``k``
log-uniformly (symmetric in magnitudes) in ``[max(k_min, that floor), k_max]``; a row whose
floor exceeds ``k_max`` is left as it is. Median headroom ``diffmaglim - magpsf`` on ``red``
Train: AGN 0.35 mag, SN 0.96, VS 2.18, Asteroid 0.92, so AGN can barely be dimmed.

**Randomness**: :data:`FLUX_STREAM` of ``class_augment.augment_rng`` (``[seed,
global_step, 4]``), two fixed-size draws per batch (which rows, and ``k``).

The context channels are not rescaled; ``config.Settings`` rejects ``flux_rescale`` with
``context`` (the pooled context would keep the old flux).
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import numpy.typing as npt
import torch

from astro_simclr.data.m1_features import M1_FEATURE_NORM_STATS
from astro_simclr.data.schema import LABEL_NAMES
from astro_simclr.training.class_augment import (
    FIT_COLUMNS,
    KEPT_FEATURE_NAMES,
    masked_pixels,
    renormalise,
)

__all__ = [
    "AUGMENT_COLUMNS",
    "FLUX_STREAM",
    "FWHM_RANGE",
    "MAX_CENTRE_OFFSET",
    "MIN_FIT_SNR",
    "PSF_COLUMNS",
    "PSF_WINDOW",
    "apply_flux",
    "fit_central_psf",
    "flux_rescale",
    "k_floor",
    "plan_flux",
    "psf_exclusions",
    "psf_image",
    "rescale_transient",
    "rescaled_features",
]

#: The columns of :func:`fit_central_psf`: total flux of the Gaussian, centroid offset from
#: the crop centre (column, row), sigma (px), and 1.0 if the fit passed the checks, else 0.
PSF_COLUMNS: Final = ("psf_flux", "psf_x0", "psf_y0", "psf_sigma", "psf_ok")
#: The per-stamp table C7 needs: ``class_augment.FIT_COLUMNS`` then :data:`PSF_COLUMNS`.
AUGMENT_COLUMNS: Final = FIT_COLUMNS + PSF_COLUMNS
#: Side of the fitted window, centred on the crop.
PSF_WINDOW: Final = 11
#: A good fit's centroid is at most this far (px, per axis) from the crop centre.
MAX_CENTRE_OFFSET: Final = 2.0
#: A good fit's FWHM, in px (the smallest PSF in the data is 1.19 px; ADR 0006).
FWHM_RANGE: Final = (1.0, 6.0)
#: A good fit's S/N: the Gaussian's peak over the residual RMS in the window.
MIN_FIT_SNR: Final = 5.0
#: The generator stream of C7 (``class_augment.augment_rng``).
FLUX_STREAM: Final = 4

_FWHM_PER_SIGMA: Final = 2.0 * float(np.sqrt(2.0 * np.log(2.0)))
_ITERATIONS: Final = 60
_FLOAT32_SAFE: Final = 1e30
_CHUNK: Final = 2048
_BOGUS: Final = LABEL_NAMES.index("Bogus")
_MAGPSF: Final = KEPT_FEATURE_NAMES.index("magpsf")
_SIGMAPSF: Final = KEPT_FEATURE_NAMES.index("sigmapsf")
_DIFFMAGLIM: Final = KEPT_FEATURE_NAMES.index("diffmaglim")
_PSF: Final = len(FIT_COLUMNS)  # first PSF column in an AUGMENT_COLUMNS table


def _profile(grid: torch.Tensor, centre: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """A unit-flux Gaussian integrated over each pixel of ``grid`` (pixel centres)."""
    scale = sigma[..., None] * np.sqrt(2.0)
    high = (grid + 0.5 - centre[..., None]) / scale
    low = (grid - 0.5 - centre[..., None]) / scale
    return 0.5 * (torch.erf(high) - torch.erf(low))


def psf_image(params: torch.Tensor, side: int) -> torch.Tensor:
    """The fitted Gaussians (B, side, side), without the constant, from (B, >= 4) parameters.

    ``params`` columns are :data:`PSF_COLUMNS` (flux, x0, y0, sigma; offsets from the centre
    pixel ``side // 2``). Pixel-integrated, so the image sums to the flux on an infinite grid.
    """
    grid = torch.arange(side, dtype=params.dtype, device=params.device) - side // 2
    flux, x0, y0, sigma = (params[:, k] for k in range(4))
    rows = _profile(grid, y0, sigma)
    cols = _profile(grid, x0, sigma)
    return flux[:, None, None] * rows[:, :, None] * cols[:, None, :]


def _window_model(p: torch.Tensor) -> torch.Tensor:
    """One stamp's model on the window: ``exp(log flux)`` Gaussian plus a constant, flat."""
    grid = torch.arange(PSF_WINDOW, dtype=p.dtype) - PSF_WINDOW // 2
    sigma = torch.exp(p[3])
    image = torch.outer(_profile(grid, p[2], sigma), _profile(grid, p[1], sigma))
    return (torch.exp(p[0]) * image + p[4]).flatten()


def _fit_chunk(cuts: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Levenberg-Marquardt on (N, W*W) windows; (N, 6): log flux, x0, y0, log sigma, c, cost."""
    n, half = cuts.shape[0], PSF_WINDOW // 2
    window = cuts.view(n, PSF_WINDOW, PSF_WINDOW)
    background = cuts.median(dim=1).values
    peak = window[:, half - 1 : half + 2, half - 1 : half + 2].amax(dim=(1, 2)) - background
    params = torch.stack(
        [
            torch.log(peak.clamp_min(1e-6) * 2 * np.pi * 0.8),  # a sigma-0.9 Gaussian's flux
            torch.zeros(n, dtype=cuts.dtype),
            torch.zeros(n, dtype=cuts.dtype),
            torch.full((n,), float(np.log(0.9)), dtype=cuts.dtype),
            background,
        ],
        dim=1,
    )

    def residual(p: torch.Tensor, data: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return (_window_model(p) - data) * w

    residuals = torch.func.vmap(residual)
    jacobians = torch.func.vmap(torch.func.jacrev(residual))
    damping = torch.full((n,), 1e-3, dtype=cuts.dtype)
    cost = (residuals(params, cuts, weights) ** 2).sum(dim=1)
    eye = torch.eye(5, dtype=cuts.dtype)
    for _ in range(_ITERATIONS):
        r, jac = residuals(params, cuts, weights), jacobians(params, cuts, weights)
        jtj = jac.transpose(1, 2) @ jac
        diagonal = torch.diag_embed(torch.diagonal(jtj, dim1=1, dim2=2))
        step = torch.linalg.solve(
            jtj + damping[:, None, None] * diagonal + 1e-12 * eye,
            -(jac.transpose(1, 2) @ r[..., None])[..., 0],
        )
        trial = params + step
        trial_cost = (residuals(trial, cuts, weights) ** 2).sum(dim=1)
        better = torch.isfinite(trial_cost) & (trial_cost < cost)
        params = torch.where(better[:, None], trial, params)
        cost = torch.where(better, trial_cost, cost)
        damping = torch.where(better, damping / 3, damping * 4).clamp(1e-9, 1e9)
    return torch.cat([params, cost[:, None]], dim=1)


def fit_central_psf(images: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float32]:
    """:data:`PSF_COLUMNS` of the central source of each difference crop, float32 (N, 5).

    ``images`` is (N, C >= 3, S, S) crops (channel 2, the difference, is fitted), normally
    a ``TensorSplit``'s. The fit and the checks are in the module docstring; the masked
    pixels (``class_augment.masked_pixels``) have weight 0. A failed check sets ``psf_ok``
    to 0 (the other columns are still the fit's). Runs on the CPU in chunks of 2048 stamps,
    about 3 ms per stamp.

    Raises:
        ValueError: if ``images`` is not (N, C >= 3, S, S) with ``S >= 11``.
    """
    if images.ndim != 4 or images.shape[1] < 3 or images.shape[2] != images.shape[3]:
        raise ValueError(f"images must be (N, C >= 3, S, S), got {images.shape}")
    side = images.shape[-1]
    if side < PSF_WINDOW:
        raise ValueError(f"the crops must be at least {PSF_WINDOW} px, got {side}")
    start = side // 2 - PSF_WINDOW // 2
    window = slice(start, start + PSF_WINDOW)
    out = np.empty((images.shape[0], len(PSF_COLUMNS)), dtype=np.float32)
    for first in range(0, images.shape[0], _CHUNK):
        crops = torch.as_tensor(np.asarray(images[first : first + _CHUNK, 2], dtype=np.float64))
        masked = masked_pixels(crops[:, None])[:, 0]
        cuts = crops[:, window, window].flatten(1)
        weights = (~masked[:, window, window]).flatten(1).to(crops.dtype)
        with torch.no_grad():
            fit = _fit_chunk(cuts, weights)
        flux, x0, y0 = torch.exp(fit[:, 0]), fit[:, 1], fit[:, 2]
        sigma = torch.exp(fit[:, 3])
        used = weights.sum(dim=1).clamp_min(1)
        rms = (fit[:, 5] / used).sqrt()
        snr = flux / (2 * np.pi * sigma**2) / rms
        fwhm = _FWHM_PER_SIGMA * sigma
        ok = (
            torch.isfinite(fit).all(dim=1)
            & (flux > 0)
            & (x0.abs() <= MAX_CENTRE_OFFSET)
            & (y0.abs() <= MAX_CENTRE_OFFSET)
            & (fwhm >= FWHM_RANGE[0])
            & (fwhm <= FWHM_RANGE[1])
            & (snr >= MIN_FIT_SNR)
        )
        table = torch.stack([flux, x0, y0, sigma, ok.to(flux.dtype)], dim=1)
        # A failed fit can diverge (huge or non-finite values): keep float32-safe numbers.
        values = np.nan_to_num(table.numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        out[first : first + len(crops)] = np.clip(values, -_FLOAT32_SAFE, _FLOAT32_SAFE)
    return out


def psf_exclusions(fits: npt.NDArray[np.float32], labels: npt.NDArray[np.int64]) -> dict[str, Any]:
    """The share of each class's stamps whose PSF fit failed (never rescaled by C7).

    ``fits`` is an (N, 11) :data:`AUGMENT_COLUMNS` table. Bogus is reported too, although C7
    never rescales it. Returns ``{"skipped": {class name: fraction}}``, recorded in
    ``metrics.json["paper_data"]["flux_rescale"]``.
    """
    ok = fits[:, _PSF + 4] > 0
    skipped = {
        name: float((~ok[labels == label]).mean()) if (labels == label).any() else 0.0
        for label, name in enumerate(LABEL_NAMES)
    }
    return {"skipped": skipped}


def _stats(index: int, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean, std = M1_FEATURE_NORM_STATS[KEPT_FEATURE_NAMES[index]]
    return (
        torch.tensor(mean, dtype=like.dtype, device=like.device),
        torch.tensor(std, dtype=like.dtype, device=like.device),
    )


def _raw(features: torch.Tensor, index: int) -> torch.Tensor:
    mean, std = _stats(index, features)
    return features[:, index] * std + mean


def k_floor(features: torch.Tensor) -> torch.Tensor:
    """(B,) the smallest ``k`` that keeps each stamp an alert: ``10^(-0.4 (diffmaglim - magpsf))``.

    ``features`` (B, 23) standardised with M1's statistics. A stamp already fainter than its
    limit gets a floor above 1.
    """
    headroom = _raw(features, _DIFFMAGLIM) - _raw(features, _MAGPSF)
    floor: torch.Tensor = torch.pow(10.0, -0.4 * headroom)
    return floor


def rescaled_features(features: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """``features`` (B, 23) of transients ``k`` (B,) times brighter: magpsf and sigmapsf.

    ``magpsf - 2.5 log10 k`` and ``sigmapsf / k`` (the module docstring), in raw units, then
    re-standardised with M1's statistics. The other features are unchanged; ``features`` is
    not modified.
    """
    out = features.clone()
    mean, std = _stats(_MAGPSF, features)
    out[:, _MAGPSF] = (_raw(features, _MAGPSF) - 2.5 * torch.log10(k) - mean) / std
    mean, std = _stats(_SIGMAPSF, features)
    out[:, _SIGMAPSF] = (_raw(features, _SIGMAPSF) / k - mean) / std
    return out


def rescale_transient(
    crops: torch.Tensor, params: torch.Tensor, a: torch.Tensor, k: torch.Tensor
) -> torch.Tensor:
    """The crops (B, 3, S, S) with their central transient ``k`` (B,) times, before the min-max.

    ``params`` (B, >= 4) are the :data:`PSF_COLUMNS` of each crop, ``a`` (B,) the science
    coefficients of the cross-channel fit. The difference gains ``(k - 1) * PSF``, the
    science ``(k - 1) * PSF / a``, the template nothing; everything else (noise, host,
    neighbours) is unchanged. ``crops`` is not modified.
    """
    extra = (k - 1)[:, None, None] * psf_image(params, crops.shape[-1])
    out = crops.clone()
    out[:, 2] += extra
    out[:, 0] += extra / a[:, None, None]
    return out


def flux_rescale(
    images: torch.Tensor,
    features: torch.Tensor,
    targets: torch.Tensor,
    fits: torch.Tensor,
    probability: float,
    k_range: tuple[float, float],
    rng: np.random.Generator,
    *,
    skip: npt.NDArray[np.bool_] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, npt.NDArray[np.float64]]:
    """C7: rescale the central transient of each row with ``probability``, never for Bogus.

    ``images`` (B, C >= 3, S, S), ``features`` (B, 23), ``targets`` (B, 5) one-hot, ``fits``
    (B, 11) :data:`AUGMENT_COLUMNS`. A row is rescaled when it is drawn (``U < probability``),
    is not Bogus, not in ``skip``, its PSF fit is ``ok``, ``a != 0`` and its alert floor
    (:func:`k_floor`) is at most ``k_range[1]``. Its ``k`` is log-uniform in
    ``[max(k_range[0], floor), k_range[1]]``; the image and the features change as the module
    docstring says (channels beyond the first 3 are untouched). Draws, always: B uniforms
    (which rows) and B uniforms (``k``). Returns the new images and features and ``k`` per
    row (1 where not rescaled). The inputs are not modified.

    Raises:
        ValueError: if ``fits`` does not have the :data:`AUGMENT_COLUMNS` columns, or
            ``k_range`` is not ``0 < low <= high``.
    """
    plan = plan_flux(
        images.detach().cpu(),
        features.detach().cpu(),
        targets.detach().cpu(),
        fits.detach().cpu().numpy(),
        probability,
        k_range,
        rng,
        skip=skip,
    )
    k = plan.pop("k")
    if "rows" not in plan:
        return images, features, k
    device_plan = {name: value.to(images.device) for name, value in plan.items()}
    return (*apply_flux(images, features, device_plan), k)


def plan_flux(
    images: torch.Tensor,
    features: torch.Tensor,
    targets: torch.Tensor,
    fits: npt.NDArray[np.floating[Any]],
    probability: float,
    k_range: tuple[float, float],
    rng: np.random.Generator,
    *,
    skip: npt.NDArray[np.bool_] | None = None,
) -> dict[str, Any]:
    """The host part of :func:`flux_rescale`, on CPU tensors and the host ``fits``.

    Returns ``{"k": float64 ndarray (B,)}`` plus, when a row is rescaled, ``"rows"`` (int64),
    ``"built"`` (float32 (r, 3, S, S): the rescaled crops after the min-max, computed in
    float64 on the host as :func:`flux_rescale` always did) and ``"scale"`` (float32 (r,),
    their ``k``). The draws are the same, in the same order, whatever the result.

    Raises:
        ValueError: as :func:`flux_rescale`.
    """
    if fits.ndim != 2 or fits.shape[1] != len(AUGMENT_COLUMNS):
        raise ValueError(f"fits must be (B, {len(AUGMENT_COLUMNS)}), got {tuple(fits.shape)}")
    low, high = k_range
    if not 0 < low <= high:
        raise ValueError(f"k_range must be 0 < low <= high, got {k_range}")
    n = images.shape[0]
    chosen = rng.random(n) < probability
    u = rng.random(n)
    k = np.ones(n)
    if probability == 0:
        return {"k": k}

    table = np.asarray(fits, dtype=np.float64)
    a = table[:, 0]
    labels = targets.argmax(dim=1).numpy()
    floor = k_floor(features.double()).numpy()
    lower = np.maximum(low, floor)
    rows = chosen & (labels != _BOGUS) & (table[:, _PSF + 4] > 0) & (a != 0) & (lower <= high)
    if skip is not None:
        rows &= ~skip
    if not rows.any():
        return {"k": k}
    k[rows] = np.exp(np.log(lower[rows]) + u[rows] * (np.log(high) - np.log(lower[rows])))

    index = np.flatnonzero(rows)
    crops = images[index, :3].double()
    changed = rescale_transient(
        crops,
        torch.from_numpy(table[index, _PSF : _PSF + 4]),
        torch.from_numpy(a[index]),
        torch.from_numpy(k[index]),
    )
    return {
        "k": k,
        "rows": torch.from_numpy(index),
        "built": renormalise(changed, masked_pixels(crops)).to(torch.float32),
        "scale": torch.from_numpy(k[index]).to(torch.float32),
    }


def apply_flux(
    images: torch.Tensor, features: torch.Tensor, plan: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """The device part of :func:`flux_rescale`, from a :func:`plan_flux` with rows.

    ``plan``'s tensors are on ``images``' device; the feature update runs there, as before.
    """
    index = plan["rows"]
    out_images = images.clone()
    out_images[index, :3] = plan["built"].to(images.dtype)
    out_features = features.clone()
    scale = plan["scale"].to(features.dtype)
    out_features[index] = rescaled_features(features[index], scale)
    return out_images, out_features
