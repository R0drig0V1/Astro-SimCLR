"""The SimCLR views of phase 5 (task 5.2's family, written for the 5.0b pilot; ADR 0025).

A view of a 63x63 stamp (``(B, C, 63, 63)`` float32 on the training device, channels
science, template, difference first, per-channel min-max with NaN -> 0 as stored) is, in
this order:

1. **noise** (optional, ``ViewConfig.noise_scale > 0``): C5's background noise on the
   whole 63-pixel stamp (:mod:`astro_simclr.stamp_relation`, ADR 0006): the science and
   the template get Gaussian noise of ``u * noise_scale * sigma_bg``, the difference moves
   through the stamp's fit ``diff ~ a*sci + b*tmpl + c`` (it is recomputed, never noised
   on its own), then the per-channel min-max again, so the view keeps the [0, 1], minimum
   0, maximum 1 statistics of the stored stamps. Stamps whose fit has ``R^2 < min_r2`` are
   left unchanged. Only the first three channels are touched.
2. **dihedral** element ``k`` in 0-7: ``rot90`` by ``k % 4`` quarter turns (counter-
   clockwise, ``torch.rot90`` on the last two axes), then a left-right flip if ``k >= 4``.
   All eight elements of the square's symmetry group, exact (no interpolation, no fill).
3. **shift**: a 57x57 window at an integer offset ``(dy, dx)`` in 0-6, the same for every
   channel. Every pixel of the view is a stored pixel: no fill, no resampling, and the
   source cannot leave the view (a shift of at most 3 pixels from the centre).

**Evaluation** uses :func:`centre_crop`: the 57x57 window at offset (3, 3), no noise, no
dihedral element, so the evaluation view is the centre of the training distribution.

**Never here** (plan 5.2): hue, saturation, greyscale, colour jitter (they mix or blend the
science, template and difference channels, whose relation defines the classes) and crops
that can drop the source. Scale jitter and blur are ablations only, not in this module.

**Randomness** (ADR 0014, ``engine.randomness``): the per-sample parameters of both views
of step ``step`` come from one host generator ``default_rng([seed, step, stream])`` with
``stream = RESERVED_STREAMS["simclr_views"]`` (:func:`draw_view_params`): the same numbers
on every device. They are always drawn in full (``k`` (2, B), the offsets (2, B, 2) and the
noise strengths ``u`` (2, B, 2)), so turning the noise on or off never changes the
dihedral elements or the offsets. The noise's per-pixel fields are drawn on the device from
``engine.randomness.device_generator`` with ``RESERVED_STREAMS["simclr_view_noise"]``:
device-dependent values, recorded as such (only a run with noise has them).
"""

from __future__ import annotations

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
    "ViewConfig",
    "ViewParams",
    "background_noise",
    "centre_crop",
    "crop",
    "dihedral",
    "draw_view_params",
    "make_views",
    "noise_generator",
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


@dataclass(frozen=True, slots=True)
class ViewConfig:
    """What a training view does besides the dihedral element and the shift.

    Attributes:
        noise_scale: C5's ``aug_noise``: the noise's standard deviation is at most this
            many background sigmas (0 turns the noise off).
        noise_min_r2: stamps whose fit has ``R^2`` below this are not noised.

    Raises:
        ValueError: for a negative ``noise_scale``.
    """

    noise_scale: float = 0.0
    noise_min_r2: float = DEFAULT_MIN_R2

    def __post_init__(self) -> None:
        if self.noise_scale < 0:
            raise ValueError(f"noise_scale must be >= 0, got {self.noise_scale}")

    @property
    def noise(self) -> bool:
        """Whether the views are noised."""
        return self.noise_scale > 0


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
    strength = torch.from_numpy(u * config.noise_scale).to(images.device, images.dtype)
    sigma = strength * fits[:, 4:6].to(images.dtype)
    deltas = sigma[:, :, None, None] * z
    crops = images[:, :3]
    noisy = renormalise(perturb(crops, fits, deltas[:, 0], deltas[:, 1]), masked_pixels(crops))
    keep = (fits[:, 3] >= config.noise_min_r2)[:, None, None, None]
    out = images.clone()
    out[:, :3] = torch.where(keep, noisy, crops)
    return out


def make_views(
    images: torch.Tensor,
    params: ViewParams,
    config: ViewConfig,
    *,
    fits: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The two training views (B, C, 57, 57) of a batch of (B, C, 63, 63) stamps.

    Noise (when ``config.noise``; it needs ``fits`` and ``generator``, the first view's
    field drawn first), then the dihedral element, then the 57-pixel window, each with
    the parameters of its view (module docstring).

    Raises:
        ValueError: if the batch is not (B, C >= 3, 63, 63) with B equal to the
            parameters' batch, or the noise lacks ``fits`` or ``generator``.
    """
    if images.ndim != 4 or images.shape[1] < 3 or images.shape[-2:] != (STAMP_SIDE, STAMP_SIDE):
        raise ValueError(f"images must be (B, C >= 3, 63, 63), got {tuple(images.shape)}")
    if images.shape[0] != params.batch_size:
        raise ValueError(f"{images.shape[0]} stamps for view parameters of {params.batch_size}")
    if config.noise and (fits is None or generator is None):
        raise ValueError("noised views need the stamp fits and a device generator")
    views = []
    for v in range(_N_VIEWS):
        x = images
        if config.noise:
            assert fits is not None and generator is not None  # noqa: S101 - checked above
            x = background_noise(x, fits, params.noise_u[v], generator, config)
        views.append(crop(dihedral(x, params.dihedral[v]), params.offsets[v]))
    return views[0], views[1]
