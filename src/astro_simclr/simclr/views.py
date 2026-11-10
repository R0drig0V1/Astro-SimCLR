"""The SimCLR views of phase 5 (task 5.2's family, written for the 5.0b pilot; ADR 0025).

A view of a 63x63 stamp (``(B, C, 63, 63)`` float32 on the training device, channels
science, template, difference first, per-channel min-max with NaN -> 0 as stored) is, in
this order (steps 2, 3, 6 and 7 are off by default; ADR 0025, amendment "stronger views"):

1. **noise** (optional, ``ViewConfig.noise_scale > 0``): C5's background noise on the
   whole 63-pixel stamp (:mod:`astro_simclr.stamp_relation`, ADR 0006): the science and
   the template get Gaussian noise of ``u * noise_scale * sigma_bg``, the difference moves
   through the stamp's fit ``diff ~ a*sci + b*tmpl + c`` (it is recomputed, never noised
   on its own), then the per-channel min-max again, so the view keeps the [0, 1], minimum
   0, maximum 1 statistics of the stored stamps. Stamps whose fit has ``R^2 < min_r2`` are
   left unchanged. Only the first three channels are touched. First, because the fit
   describes the stored stamp.
2. **blur** (optional, ``blur_sigma > 0``; an ablation only, plan 5.2): a Gaussian of
   ``sigma = u * blur_sigma`` pixels, the same kernel on the three channels, as a
   normalised convolution that leaves the former-NaN pixels out (and at 0), reflected at
   the stamp's edges, then the per-channel min-max again (:func:`gaussian_blur`). It is
   linear and the same on every channel, so the fit relation of the difference survives
   it (up to the min-max's affine change); after the noise, so it also smooths the noise
   (a worse seeing of the observed stamp), and the noise's ``sigma_bg`` stays the stored
   stamp's.
3. **intensity** (optional, ``intensity_jitter > 0``): a gamma ``x -> x ** gamma`` with
   ``ln gamma`` uniform in ``[-intensity_jitter, intensity_jitter]``, **the same gamma on
   the three channels** (:func:`gamma_remap`). A per-channel change would be the hue or
   saturation the plan forbids. The remap must fix 0 and 1, so that the view keeps the
   min-max statistics: an affine contrast would either be undone by a min-max or move the
   minimum and maximum off 0 and 1, which no evaluation stamp has; the gamma is the
   simplest one-parameter family of monotone maps that fix them. It is not linear, so
   (unlike the blur) it bends the fit relation of the difference: the admission test
   measures what that costs. Last of the photometric views, because it needs [0, 1] input.
4. **dihedral** element ``k`` in 0-7: ``rot90`` by ``k % 4`` quarter turns (counter-
   clockwise, ``torch.rot90`` on the last two axes), then a left-right flip if ``k >= 4``.
   All eight elements of the square's symmetry group, exact (no interpolation, no fill).
5. **shift**: a 57x57 window at an integer offset ``(dy, dx)`` in 0-6, the same for every
   channel. Every pixel of the view is a stored pixel: no fill, no resampling, and the
   source cannot leave the view (a shift of at most 3 pixels from the centre).
6. **scale jitter** (optional, ``scale_jitter > 0``; an ablation only, plan 5.2): a zoom
   by ``s`` about the window's centre, ``s`` log-uniform in ``[1, 1 + scale_jitter]``.
   Magnification only, as SimCLR's resized crop: a zoom out would need pixels beyond the
   stamp (a fill). The source stays at the centre.
7. **rotation** (optional, ``rotation_degrees > 0``): a rotation by ``theta`` uniform in
   ``[-rotation_degrees, rotation_degrees]`` about the window's centre, at most
   :func:`rotation_bound` (6.52 degrees for 57 of 63 px).

With scale or rotation on, steps 5-7 are **one bilinear resampling** of the 63-pixel stamp
(:func:`window_transform`, :func:`resample`; one interpolation, not two): view pixel ``q``
(from the window's centre) reads the stamp at ``c + t + R(theta) q / s``, ``c`` the
stamp's centre and ``t`` the shift. **No fill:** the window's corners reach ``e = h (|cos
theta| + |sin theta|) / s`` from its centre (``h = 28``), which leaves a margin of ``31 -
e`` pixels on each side; the shift gives way to the rotation, ``t = (offset - 3) * min(1,
(31 - e) / 3)``, so every view pixel reads inside the stamp. Without rotation (``s >= 1``)
the shift is the integer one, unchanged; at ``rotation_bound`` it is 0. Off (both 0), the
view is the integer crop of step 5 itself, bit for bit.

**Evaluation** uses :func:`centre_crop`: the 57x57 window at offset (3, 3), no noise, no
dihedral element, so the evaluation view is the centre of the training distribution
(scale 1, rotation 0, gamma 1 and sigma 0 are the centre or the edge of their ranges).

**Never here** (plan 5.2): hue, saturation, greyscale, colour jitter (they mix or blend the
science, template and difference channels, whose relation defines the classes) and crops
that can drop the source. Scale jitter and blur are ablations only: off by default.

**Randomness** (ADR 0014, ``engine.randomness``): the per-sample parameters of both views
of step ``step`` come from one host generator ``default_rng([seed, step, stream])`` with
``stream = RESERVED_STREAMS["simclr_views"]`` (:func:`draw_view_params`): the same numbers
on every device. They are always drawn in full (``k`` (2, B), the offsets (2, B, 2) and the
noise strengths ``u`` (2, B, 2)), so turning the noise on or off never changes the
dihedral elements or the offsets. The noise's per-pixel fields are drawn on the device from
``engine.randomness.device_generator`` with ``RESERVED_STREAMS["simclr_view_noise"]``:
device-dependent values, recorded as such (only a run with noise has them). Each optional
view draws its ``u`` (2, B) from a host generator of its own stream
(:func:`draw_extra_params`: ``simclr_view_scale`` 14, ``simclr_view_blur`` 15,
``simclr_view_intensity`` 16, ``simclr_view_rotation`` 17), and only when it is on: turning
one on changes no other view's numbers, and with all of them off nothing new is drawn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import torch

from astro_simclr.engine.randomness import RESERVED_STREAMS, device_generator, host_generator
from astro_simclr.stamp_relation import DEFAULT_MIN_R2, masked_pixels, perturb, renormalise

__all__ = [
    "CENTRE_OFFSET",
    "MAX_OFFSET",
    "N_DIHEDRAL",
    "STAMP_SIDE",
    "VIEW_SIDE",
    "ExtraViewParams",
    "ViewConfig",
    "ViewParams",
    "background_noise",
    "centre_crop",
    "crop",
    "dihedral",
    "draw_extra_params",
    "draw_view_params",
    "gamma_remap",
    "gaussian_blur",
    "make_views",
    "noise_generator",
    "noise_with_fields",
    "resample",
    "rotation_bound",
    "window_transform",
]

#: The stored stamp's side, in pixels.
STAMP_SIDE: Final = 63
#: A view's side: the 57x57 window of a shift of at most 3 pixels from the centre.
VIEW_SIDE: Final = 57
#: The largest offset of the window (0-6 on each axis).
MAX_OFFSET: Final = STAMP_SIDE - VIEW_SIDE
#: The evaluation window's offset on both axes: the centre.
CENTRE_OFFSET: Final = MAX_OFFSET // 2
#: The order of the dihedral group of the square (4 rotations x 2 reflections).
N_DIHEDRAL: Final = 8
#: Views drawn per stamp per step.
_N_VIEWS: Final = 2
#: The largest ``blur_sigma``: its kernel radius ``ceil(3 sigma)`` must stay well inside the
#: stamp (the reflected padding needs a radius below the side).
MAX_BLUR_SIGMA: Final = 10.0


def rotation_bound(stamp_side: int = STAMP_SIDE, view_side: int = VIEW_SIDE) -> float:
    """The largest rotation (degrees) of the centred ``view_side`` window that reads no fill.

    The window's corner ``(h, h)``, ``h = (view_side - 1) / 2``, rotated by ``theta``,
    reaches ``h (cos theta + sin theta)`` from the centre on one axis; it must stay within
    ``H = (stamp_side - 1) / 2`` (pixel centres, as bilinear sampling reads them). So
    ``sqrt(2) h sin(theta + 45 deg) <= H``: ``theta* = asin(H / (sqrt(2) h)) - 45 deg``,
    6.52 degrees for 57 of 63 pixels (45 when every angle fits). With the window shifted
    by ``t`` the margin is ``H - |t|``: :func:`window_transform` shrinks the shift instead.

    Raises:
        ValueError: if the window does not fit in the stamp.
    """
    if not 0 < view_side <= stamp_side:
        raise ValueError(f"no {view_side}-pixel window in a {stamp_side}-pixel stamp")
    half, room = (view_side - 1) / 2, (stamp_side - 1) / 2
    if half == 0 or room >= half * math.sqrt(2):
        return 45.0
    return math.degrees(math.asin(room / (half * math.sqrt(2))) - math.pi / 4)


@dataclass(frozen=True, slots=True)
class ViewConfig:
    """What a training view does besides the dihedral element and the shift.

    Every strength is 0 (off) by default (module docstring, steps 1-3, 6 and 7).

    Attributes:
        noise_scale: C5's ``aug_noise``: the noise's standard deviation is at most this
            many background sigmas (0 turns the noise off).
        noise_min_r2: stamps whose fit has ``R^2`` below this are not noised.
        scale_jitter: the largest zoom is ``1 + scale_jitter`` (an ablation only).
        blur_sigma: the largest Gaussian blur, in pixels (an ablation only), at most
            :data:`MAX_BLUR_SIGMA`.
        intensity_jitter: the largest ``|ln gamma|`` of the gamma remap.
        rotation_degrees: the largest ``|theta|``, at most :func:`rotation_bound`.

    Raises:
        ValueError: for a negative or non-finite strength, a blur above
            :data:`MAX_BLUR_SIGMA` or a rotation above :func:`rotation_bound`.
    """

    noise_scale: float = 0.0
    noise_min_r2: float = DEFAULT_MIN_R2
    scale_jitter: float = 0.0
    blur_sigma: float = 0.0
    intensity_jitter: float = 0.0
    rotation_degrees: float = 0.0

    def __post_init__(self) -> None:
        if self.noise_scale < 0:
            raise ValueError(f"noise_scale must be >= 0, got {self.noise_scale}")
        for name in ("scale_jitter", "blur_sigma", "intensity_jitter", "rotation_degrees"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be a finite number >= 0, got {value}")
        if self.blur_sigma > MAX_BLUR_SIGMA:
            raise ValueError(f"blur_sigma must be <= {MAX_BLUR_SIGMA}, got {self.blur_sigma}")
        bound = rotation_bound()
        if self.rotation_degrees > bound:
            raise ValueError(
                f"rotation_degrees must be <= {bound:.4f} (the 57-px window of the 63-px "
                f"stamp reads no fill up to it), got {self.rotation_degrees}"
            )

    @property
    def noise(self) -> bool:
        """Whether the views are noised."""
        return self.noise_scale > 0

    @property
    def scale(self) -> bool:
        """Whether the views are zoomed (scale jitter)."""
        return self.scale_jitter > 0

    @property
    def blur(self) -> bool:
        """Whether the views are blurred."""
        return self.blur_sigma > 0

    @property
    def intensity(self) -> bool:
        """Whether the views get the gamma remap."""
        return self.intensity_jitter > 0

    @property
    def rotation(self) -> bool:
        """Whether the views are rotated by a small angle."""
        return self.rotation_degrees > 0

    @property
    def geometric(self) -> bool:
        """Whether the window is resampled (scale or rotation) instead of cropped."""
        return self.scale or self.rotation

    @property
    def extra(self) -> bool:
        """Whether any optional view of :func:`draw_extra_params` is on."""
        return self.geometric or self.blur or self.intensity

    @property
    def blur_radius(self) -> int:
        """The blur kernel's radius, ``ceil(3 * blur_sigma)`` pixels (0 when off)."""
        return math.ceil(3 * self.blur_sigma)


@dataclass(frozen=True, slots=True)
class ViewParams:
    """The per-sample parameters of the two views of one step (host arrays).

    Attributes:
        dihedral: int64 (2, B), the group element of each view (0-7).
        offsets: int64 (2, B, 2), the window's ``(dy, dx)`` of each view (0-6).
        noise_u: float64 (2, B, 2), the noise strengths of the science and the template,
            ``U(0, 1)`` (used only when the views are noised).
    """

    dihedral: npt.NDArray[np.int64]
    offsets: npt.NDArray[np.int64]
    noise_u: npt.NDArray[np.float64]

    @property
    def batch_size(self) -> int:
        """B, the stamps of the step."""
        return int(self.dihedral.shape[1])


def draw_view_params(
    seed: int, step: int, batch_size: int, *, stream: int = RESERVED_STREAMS["simclr_views"]
) -> ViewParams:
    """The parameters of both views of step ``step`` (module docstring, "Randomness").

    Draws, in this order from ``default_rng([seed, step, stream])``: ``k`` (2, B)
    integers in 0-7, the offsets (2, B, 2) integers in 0-6, ``u`` (2, B, 2) uniforms.
    ``stream`` is ``simclr_views`` for training; the pilot's fixed diagnostic views use
    ``simclr_diagnostics``.

    Raises:
        ValueError: for a negative ``seed`` or ``step``, or ``batch_size < 1``.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    rng = host_generator(seed, step, stream)
    k = rng.integers(0, N_DIHEDRAL, size=(_N_VIEWS, batch_size), dtype=np.int64)
    offsets = rng.integers(0, MAX_OFFSET + 1, size=(_N_VIEWS, batch_size, 2), dtype=np.int64)
    u = rng.random((_N_VIEWS, batch_size, 2))
    return ViewParams(k, offsets, u)


def noise_generator(seed: int, step: int, device: torch.device) -> torch.Generator:
    """The device generator of the noise fields of step ``step`` (``simclr_view_noise``)."""
    return device_generator(seed, step, RESERVED_STREAMS["simclr_view_noise"], device)


@dataclass(frozen=True, slots=True)
class ExtraViewParams:
    """The per-sample parameters of the optional views of one step (host float64 arrays).

    Each is ``(2, B)`` when its view is on and ``None`` when it is off.

    Attributes:
        scale: the zoom factors ``s`` in ``[1, 1 + scale_jitter]``.
        blur_sigma: the Gaussian sigmas, pixels, in ``[0, blur_sigma]``.
        gamma: the gammas, ``ln gamma`` in ``[-intensity_jitter, intensity_jitter]``.
        angle: the rotations, degrees, in ``[-rotation_degrees, rotation_degrees]``.
    """

    scale: npt.NDArray[np.float64] | None = None
    blur_sigma: npt.NDArray[np.float64] | None = None
    gamma: npt.NDArray[np.float64] | None = None
    angle: npt.NDArray[np.float64] | None = None


def _uniforms(seed: int, step: int, stream: str, batch_size: int) -> npt.NDArray[np.float64]:
    return host_generator(seed, step, RESERVED_STREAMS[stream]).random((_N_VIEWS, batch_size))


def draw_extra_params(seed: int, step: int, batch_size: int, config: ViewConfig) -> ExtraViewParams:
    """The parameters of the optional views of step ``step`` (module docstring, "Randomness").

    Each view that ``config`` turns on draws ``u`` (2, B) uniforms in ``[0, 1)`` from
    ``default_rng([seed, step, stream])`` of its own stream, and maps them: the scale
    ``(1 + scale_jitter) ** u`` (log-uniform), the blur ``u * blur_sigma``, the gamma
    ``exp(intensity_jitter * (2u - 1))`` (``ln gamma`` uniform), the angle
    ``rotation_degrees * (2u - 1)``. A view that is off draws nothing.

    Raises:
        ValueError: for a negative ``seed`` or ``step``, or ``batch_size < 1``.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    def draw(on: bool, stream: str) -> npt.NDArray[np.float64] | None:
        return _uniforms(seed, step, stream, batch_size) if on else None

    scale = draw(config.scale, "simclr_view_scale")
    blur = draw(config.blur, "simclr_view_blur")
    gamma = draw(config.intensity, "simclr_view_intensity")
    angle = draw(config.rotation, "simclr_view_rotation")
    return ExtraViewParams(
        scale=None if scale is None else (1.0 + config.scale_jitter) ** scale,
        blur_sigma=None if blur is None else blur * config.blur_sigma,
        gamma=None if gamma is None else np.exp(config.intensity_jitter * (2.0 * gamma - 1.0)),
        angle=None if angle is None else config.rotation_degrees * (2.0 * angle - 1.0),
    )


def _as_index(values: npt.ArrayLike | torch.Tensor, n: int, what: str) -> torch.Tensor:
    index = torch.as_tensor(np.asarray(values), dtype=torch.int64)
    if index.shape[0] != n:
        raise ValueError(f"{what} has {index.shape[0]} rows for a batch of {n}")
    return index


def dihedral(images: torch.Tensor, k: npt.ArrayLike | torch.Tensor) -> torch.Tensor:
    """Apply dihedral element ``k[b]`` to ``images[b]`` (module docstring, step 2).

    ``images`` is (B, C, S, S); ``k`` (B,) integers in 0-7. Returns a new tensor; the
    input is not modified. The group action is exact, so applying ``k`` and then its
    inverse gives the input back bit for bit.

    Raises:
        ValueError: if ``images`` is not square in its last two axes, or ``k`` has the
            wrong length or a value outside 0-7.
    """
    if images.ndim != 4 or images.shape[-1] != images.shape[-2]:
        raise ValueError(f"images must be (B, C, S, S), got {tuple(images.shape)}")
    index = _as_index(k, images.shape[0], "k")
    if index.numel() and (index.min() < 0 or index.max() >= N_DIHEDRAL):
        raise ValueError(f"dihedral elements must be in 0-{N_DIHEDRAL - 1}")
    out = torch.empty_like(images)
    for element in range(N_DIHEDRAL):
        rows = torch.nonzero(index == element).flatten().to(images.device)
        if rows.numel() == 0:
            continue
        view = torch.rot90(images[rows], element % 4, dims=(-2, -1))
        if element >= 4:  # the reflections are elements 4-7
            view = torch.flip(view, dims=(-1,))
        out[rows] = view
    return out


def crop(
    images: torch.Tensor, offsets: npt.ArrayLike | torch.Tensor, side: int = VIEW_SIDE
) -> torch.Tensor:
    """The ``side`` x ``side`` window of each stamp at its integer ``(dy, dx)`` offset.

    ``images`` is (B, C, S, S), ``offsets`` (B, 2) integers in ``0 .. S - side``; every
    channel of a stamp gets the same window. Returns (B, C, side, side), a gather of stored
    pixels (no fill, no interpolation).

    Raises:
        ValueError: if ``side`` exceeds the stamp, or an offset is out of range.
    """
    if images.ndim != 4 or not 0 < side <= images.shape[-1]:
        raise ValueError(f"cannot take a {side}-pixel window of {tuple(images.shape)}")
    index = _as_index(offsets, images.shape[0], "offsets")
    largest = images.shape[-1] - side
    if index.shape[1:] != (2,) or (index.numel() and (index.min() < 0 or index.max() > largest)):
        raise ValueError(f"offsets must be (B, 2) integers in 0-{largest}")
    index = index.to(images.device)
    span = torch.arange(side, device=images.device)
    rows = (index[:, 0, None] + span)[:, None, :, None]  # (B, 1, side, 1)
    cols = (index[:, 1, None] + span)[:, None, None, :]  # (B, 1, 1, side)
    batch = torch.arange(images.shape[0], device=images.device)[:, None, None, None]
    channel = torch.arange(images.shape[1], device=images.device)[None, :, None, None]
    return images[batch, channel, rows, cols]


def centre_crop(images: torch.Tensor, side: int = VIEW_SIDE) -> torch.Tensor:
    """The evaluation view: the centred ``side`` x ``side`` window (offset 3 at 57 of 63).

    Raises:
        ValueError: if ``side`` exceeds the stamp or does not centre exactly (the margin
            ``S - side`` must be even).
    """
    margin = images.shape[-1] - side
    if images.ndim != 4 or margin < 0 or margin % 2:
        raise ValueError(f"no centred {side}-pixel window in {tuple(images.shape)}")
    start = margin // 2
    return images[..., start : start + side, start : start + side]


def background_noise(
    images: torch.Tensor,
    fits: torch.Tensor,
    u: npt.NDArray[np.float64],
    generator: torch.Generator,
    config: ViewConfig,
) -> torch.Tensor:
    """C5's noise on (B, C, S, S) stamps, before the min-max again (module docstring, step 1).

    ``fits`` is (B, 6) of :func:`astro_simclr.stamp_relation.stamp_fits` of the same
    stamps, ``u`` (B, 2) the noise strengths of :class:`ViewParams`. Draws (B, 2, S, S)
    standard normals from ``generator`` always (a fixed draw per call), then for the rows
    with ``R^2 >= config.noise_min_r2``: ``delta = u * noise_scale * sigma_bg * z`` on the
    science and the template, the difference through the fit, the masked pixels kept out
    and 0, and the per-channel min-max. Other rows and channels beyond the third are
    returned unchanged. ``images`` is not modified.
    """
    n, side = images.shape[0], images.shape[-1]
    z = torch.randn(
        (n, 2, side, side), generator=generator, device=images.device, dtype=images.dtype
    )
    return noise_with_fields(images, fits, u, z, config)


def noise_with_fields(
    images: torch.Tensor,
    fits: torch.Tensor,
    u: npt.NDArray[np.float64],
    z: torch.Tensor,
    config: ViewConfig,
) -> torch.Tensor:
    """:func:`background_noise` with its standard-normal fields ``z`` (B, 2, S, S) given.

    The same arithmetic: :func:`background_noise` draws ``z`` on the device and calls this;
    the scratch baseline (:mod:`astro_simclr.simclr.scratch`) draws ``z`` on the host from
    its own stream, so its views do not depend on the device.

    Raises:
        ValueError: if ``z`` is not (B, 2, S, S) for (B, C, S, S) ``images``.
    """
    n, side = images.shape[0], images.shape[-1]
    if tuple(z.shape) != (n, 2, side, side):
        raise ValueError(f"z must be ({n}, 2, {side}, {side}), got {tuple(z.shape)}")
    z = z.to(images.device, images.dtype)
    strength = torch.from_numpy(u * config.noise_scale).to(images.device, images.dtype)
    sigma = strength * fits[:, 4:6].to(images.dtype)
    deltas = sigma[:, :, None, None] * z
    crops = images[:, :3]
    noisy = renormalise(perturb(crops, fits, deltas[:, 0], deltas[:, 1]), masked_pixels(crops))
    keep = (fits[:, 3] >= config.noise_min_r2)[:, None, None, None]
    out = images.clone()
    out[:, :3] = torch.where(keep, noisy, crops)
    return out


def _per_row(values: npt.ArrayLike, n: int, what: str) -> npt.NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (n,):
        raise ValueError(f"{what} must be ({n},), got {array.shape}")
    return array


def _filter_axis(padded: torch.Tensor, weights: torch.Tensor, side: int, axis: int) -> torch.Tensor:
    """``sum_j weights[:, j] * padded`` shifted by ``j`` on ``axis`` (-1 or -2): a 1-D filter.

    A sum of shifted slices (no convolution kernel), so it is deterministic on every device
    and a delta filter returns its input exactly.
    """
    out = torch.zeros_like(padded.narrow(axis, 0, side))
    for j in range(weights.shape[1]):
        out = out + weights[:, j, None, None, None] * padded.narrow(axis, j, side)
    return out


def gaussian_blur(images: torch.Tensor, sigma: npt.ArrayLike, *, radius: int) -> torch.Tensor:
    """The blur of (B, C, S, S) stamps by a Gaussian of ``sigma[b]`` pixels (step 2).

    The first three channels of stamp ``b`` get the same separable kernel ``exp(-i^2 / (2
    sigma^2))``, ``i`` in ``-radius .. radius``, normalised to sum 1, as a **normalised
    convolution**: the former-NaN pixels (:func:`~astro_simclr.stamp_relation.
    masked_pixels`) have weight 0, so they neither spread their 0 nor get a value, and each
    blurred pixel is the weighted mean of its valid neighbours. The stamp is reflected at
    its edges (``torch.nn.functional.pad``, ``reflect``). Then the per-channel min-max
    again (masked pixels out and at 0, :func:`~astro_simclr.stamp_relation.renormalise`).
    A stamp with ``sigma == 0`` is returned unchanged; so are the channels beyond the
    third. ``images`` is not modified.

    Raises:
        ValueError: if ``images`` is not (B, C >= 3, S, S), ``sigma`` is not (B,) and
            ``>= 0``, or ``radius`` is not in ``1 .. S - 1``.
    """
    if images.ndim != 4 or images.shape[1] < 3 or images.shape[-1] != images.shape[-2]:
        raise ValueError(f"images must be (B, C >= 3, S, S), got {tuple(images.shape)}")
    n, side = images.shape[0], images.shape[-1]
    sigmas = _per_row(sigma, n, "sigma")
    if not 0 < radius < side:
        raise ValueError(f"radius must be in 1-{side - 1}, got {radius}")
    if (sigmas < 0).any() or not np.isfinite(sigmas).all():
        raise ValueError("sigma must be finite and >= 0")
    taps = np.arange(-radius, radius + 1, dtype=np.float64)
    safe = np.where(sigmas > 0, sigmas, 1.0)[:, None]
    kernel = np.exp(-(taps[None, :] ** 2) / (2.0 * safe**2))
    kernel = np.where((sigmas > 0)[:, None], kernel, (taps == 0)[None, :].astype(np.float64))
    kernel /= kernel.sum(axis=1, keepdims=True)
    weights = torch.from_numpy(kernel).to(images.device, images.dtype)
    crops = images[:, :3]
    masked = masked_pixels(crops)
    valid = (~masked).to(images.dtype)
    pad = (radius, radius, radius, radius)
    both = torch.nn.functional.pad(torch.cat([crops * valid, valid]), pad, mode="reflect")
    rows = _filter_axis(both, torch.cat([weights, weights]), side, -1)
    blurred = _filter_axis(rows, torch.cat([weights, weights]), side, -2)
    numerator, weight = blurred[:n], blurred[n:]
    mean = torch.where(weight > 0, numerator / weight.clamp_min(1e-30), torch.zeros_like(weight))
    out = images.clone()
    changed = torch.from_numpy(sigmas > 0).to(images.device)[:, None, None, None]
    out[:, :3] = torch.where(changed, renormalise(mean, masked), crops)
    return out


def gamma_remap(images: torch.Tensor, gamma: npt.ArrayLike) -> torch.Tensor:
    """``x ** gamma[b]`` on the first three channels of stamp ``b`` (step 3).

    The same gamma on the three channels (never a per-channel change: plan 5.2). The
    stamps are in [0, 1] with minimum 0 and maximum 1 (stored, noised or blurred), and the
    remap fixes 0 and 1, so the view keeps those statistics; the input is clamped to [0, 1]
    first, so a rounding error below 0 cannot become a NaN. A stamp with ``gamma == 1`` is
    returned unchanged, bit for bit; so are the channels beyond the third. ``images`` is
    not modified.

    Raises:
        ValueError: if ``images`` is not (B, C >= 3, S, S) or ``gamma`` is not (B,) and
            positive.
    """
    if images.ndim != 4 or images.shape[1] < 3:
        raise ValueError(f"images must be (B, C >= 3, S, S), got {tuple(images.shape)}")
    gammas = _per_row(gamma, images.shape[0], "gamma")
    if not (np.isfinite(gammas).all() and (gammas > 0).all()):
        raise ValueError("gamma must be finite and > 0")
    exponent = torch.from_numpy(gammas).to(images.device, images.dtype)[:, None, None, None]
    crops = images[:, :3]
    out = images.clone()
    changed = torch.from_numpy(gammas != 1.0).to(images.device)[:, None, None, None]
    out[:, :3] = torch.where(changed, crops.clamp(0.0, 1.0) ** exponent, crops)
    return out


def window_transform(
    offsets: npt.ArrayLike,
    scale: npt.ArrayLike | None = None,
    angle: npt.ArrayLike | None = None,
    *,
    stamp_side: int = STAMP_SIDE,
    view_side: int = VIEW_SIDE,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The affine map of each training window: ``(A (B, 2, 2), t (B, 2))``, float64.

    View pixel ``q`` (row, column; from the window's centre) reads the stamp at ``c + t + A
    q``, ``c`` the stamp's centre, ``A = R(theta) / s`` (``R`` the rotation by ``angle``
    degrees, counter-clockwise in (row, column); ``s`` the zoom, 1 without ``scale``) and
    ``t`` the window's shift ``offset - 3`` scaled by ``min(1, (H - e) / 3)``: ``e = h
    (|cos theta| + |sin theta|) / s`` is how far the window's corners reach from its
    centre, ``h = (view_side - 1) / 2``, ``H = (stamp_side - 1) / 2`` (module docstring,
    "No fill"). Without rotation the shift is the integer offset, unchanged.

    Raises:
        ValueError: if ``offsets`` is not (B, 2) in ``0 .. stamp_side - view_side``, a
            scale is below 1, or an angle leaves no margin (above :func:`rotation_bound`).
    """
    shifts = np.asarray(offsets, dtype=np.float64)
    largest = stamp_side - view_side
    if shifts.ndim != 2 or shifts.shape[1] != 2 or (shifts < 0).any() or (shifts > largest).any():
        raise ValueError(f"offsets must be (B, 2) in 0-{largest}")
    n = shifts.shape[0]
    zoom = np.ones(n) if scale is None else _per_row(scale, n, "scale")
    theta = np.zeros(n) if angle is None else np.radians(_per_row(angle, n, "angle"))
    if (zoom < 1).any() or not np.isfinite(zoom).all():
        raise ValueError("scale must be finite and >= 1 (a zoom out would read a fill)")
    cos, sin = np.cos(theta), np.sin(theta)
    half, room, centre = (view_side - 1) / 2, (stamp_side - 1) / 2, largest // 2
    reach = half * (np.abs(cos) + np.abs(sin)) / zoom
    margin = room - reach
    if (margin < -1e-9).any():
        raise ValueError(f"an angle above rotation_bound ({rotation_bound():.4f} degrees)")
    factor = np.minimum(1.0, np.maximum(margin, 0.0) / centre) if centre else np.ones(n)
    shift = (shifts - centre) * factor[:, None]
    matrix = (
        np.stack([np.stack([cos, -sin], -1), np.stack([sin, cos], -1)], -2) / zoom[:, None, None]
    )
    return matrix, shift


def resample(
    images: torch.Tensor,
    matrix: npt.ArrayLike,
    shift: npt.ArrayLike,
    *,
    side: int = VIEW_SIDE,
) -> torch.Tensor:
    """The (B, C, side, side) bilinear resampling of (B, C, S, S) stamps by ``c + t + A q``.

    ``matrix`` (B, 2, 2) and ``shift`` (B, 2) as :func:`window_transform` gives them; ``q``
    runs over the ``side`` x ``side`` pixel centres around the output's centre. The
    coordinates are computed on the stamps' device in their dtype, clamped to the stamp
    (a training window never needs it, :func:`window_transform`; the 63-pixel output of the
    admission test does, on its outer ring), and every channel is sampled with the same
    weights by gathers: an integer coordinate reads its pixel exactly, so the identity map
    with an integer shift is :func:`crop` bit for bit.

    Raises:
        ValueError: if ``images`` is not (B, C, S, S), or ``matrix`` / ``shift`` do not
            have B rows.
    """
    if images.ndim != 4 or images.shape[-1] != images.shape[-2]:
        raise ValueError(f"images must be (B, C, S, S), got {tuple(images.shape)}")
    n, channels, stamp = images.shape[0], images.shape[1], images.shape[-1]
    a = np.array(matrix, dtype=np.float64)  # copies: a broadcast view is read-only
    t = np.array(shift, dtype=np.float64)
    if a.shape != (n, 2, 2) or t.shape != (n, 2):
        raise ValueError(f"matrix must be ({n}, 2, 2) and shift ({n}, 2)")
    device, dtype = images.device, images.dtype
    a_t = torch.from_numpy(a).to(device, dtype)
    t_t = torch.from_numpy(t).to(device, dtype)
    q = torch.arange(side, device=device, dtype=dtype) - (side - 1) / 2
    q_row, q_col = q[None, :, None], q[None, None, :]
    centre = (stamp - 1) / 2

    def coordinate(axis: int) -> torch.Tensor:
        value = (
            centre
            + t_t[:, axis, None, None]
            + a_t[:, axis, 0, None, None] * q_row
            + a_t[:, axis, 1, None, None] * q_col
        )
        return value.clamp(0.0, stamp - 1.0)

    rows, cols = coordinate(0), coordinate(1)
    row0 = rows.floor().clamp(max=stamp - 2)
    col0 = cols.floor().clamp(max=stamp - 2)
    w_row = (rows - row0)[:, None]
    w_col = (cols - col0)[:, None]
    flat = images.reshape(n, channels, stamp * stamp)
    base = (row0 * stamp + col0).to(torch.int64).reshape(n, 1, side * side)

    def pixels(step: int) -> torch.Tensor:
        index = (base + step).expand(n, channels, side * side)
        return torch.gather(flat, 2, index).reshape(n, channels, side, side)

    top = (1 - w_col) * pixels(0) + w_col * pixels(1)
    bottom = (1 - w_col) * pixels(stamp) + w_col * pixels(stamp + 1)
    return (1 - w_row) * top + w_row * bottom


def make_views(
    images: torch.Tensor,
    params: ViewParams,
    config: ViewConfig,
    *,
    fits: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    extra: ExtraViewParams | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The two training views (B, C, 57, 57) of a batch of (B, C, 63, 63) stamps.

    Noise (when ``config.noise``; it needs ``fits`` and ``generator``, the first view's
    field drawn first), the blur and the gamma (``extra``'s parameters), then the dihedral
    element, then the 57-pixel window: the integer crop, or with scale or rotation the
    resampling of :func:`window_transform` (module docstring), each with the parameters of
    its view. With every optional view off this is the code of before them, unchanged.

    Raises:
        ValueError: if the batch is not (B, C >= 3, 63, 63) with B equal to the
            parameters' batch, the noise lacks ``fits`` or ``generator``, or an optional
            view lacks its ``extra`` parameters.
    """
    if images.ndim != 4 or images.shape[1] < 3 or images.shape[-2:] != (STAMP_SIDE, STAMP_SIDE):
        raise ValueError(f"images must be (B, C >= 3, 63, 63), got {tuple(images.shape)}")
    if images.shape[0] != params.batch_size:
        raise ValueError(f"{images.shape[0]} stamps for view parameters of {params.batch_size}")
    if config.noise and (fits is None or generator is None):
        raise ValueError("noised views need the stamp fits and a device generator")
    missing = [
        name
        for name, on in (
            ("scale", config.scale),
            ("blur_sigma", config.blur),
            ("gamma", config.intensity),
            ("angle", config.rotation),
        )
        if on and (extra is None or getattr(extra, name) is None)
    ]
    if missing:
        raise ValueError(f"the optional views need their parameters: {', '.join(missing)}")
    views = []
    for v in range(_N_VIEWS):
        x = images
        if config.noise:
            assert fits is not None and generator is not None  # noqa: S101 - checked above
            x = background_noise(x, fits, params.noise_u[v], generator, config)
        if config.blur:
            assert extra is not None and extra.blur_sigma is not None  # noqa: S101 - checked
            x = gaussian_blur(x, extra.blur_sigma[v], radius=config.blur_radius)
        if config.intensity:
            assert extra is not None and extra.gamma is not None  # noqa: S101 - checked above
            x = gamma_remap(x, extra.gamma[v])
        x = dihedral(x, params.dihedral[v])
        if config.geometric:
            assert extra is not None  # noqa: S101 - checked above
            matrix, shift = window_transform(
                params.offsets[v],
                None if extra.scale is None else extra.scale[v],
                None if extra.angle is None else extra.angle[v],
            )
            views.append(resample(x, matrix, shift))
        else:
            views.append(crop(x, params.offsets[v]))
    return views[0], views[1]
