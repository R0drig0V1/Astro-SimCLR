"""Stamp augmentations: the ``Augmentation`` protocol (task 1.1) and the registry (task 1.5).

A dataset holds uint8 stamps (HxWx3, the output of the legacy ``img_float2int``) and asks
its augmentation for either one view (classifier datasets) or two independent views of the
same stamp (contrastive datasets). The datasets (task 1.2) depend only on the protocol,
never on a concrete transform.

The registry holds exact ports of the six legacy transforms that phase 1 uses
(``utils/transformations.py``, selected by name in ``utils/training.py``): ``simclr``,
``simclr2``, ``astro7``, ``astro8``, ``rotation_v3`` and ``without_aug``. Each one converts
the stamp to a PIL image and runs the same torchvision pipeline, in the same order, as the
legacy class (the legacy datasets ran ``img_float2int`` → ``ToPILImage`` → transform).
Every other legacy name is deferred and raises :class:`DeferredAugmentationError`.

Randomness is not passed in: the transforms draw from the global generators (torchvision
from torch, :class:`RandomResizedCenterCrop` from Python ``random``, :class:`GaussianBlur`
from ``np.random``), which ``lightning.seed_everything(seed, workers=True)`` seeds in the
main process and, through Lightning's worker init function, in every DataLoader worker.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as nnf
from PIL.Image import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf

__all__ = [
    "AUGMENTATIONS",
    "DEFERRED_AUGMENTATIONS",
    "Augmentation",
    "DeferredAugmentationError",
    "GaussianBlur",
    "PipelineAugmentation",
    "RandomResizedCenterCrop",
    "RandomRot90",
    "Stamp",
    "UnknownAugmentationError",
    "get_augmentation",
]

#: A stamp as stored by the datasets: uint8, shape (H, W, 3) (science, template, difference).
type Stamp = npt.NDArray[np.uint8]


@runtime_checkable
class Augmentation(Protocol):
    """Turns a stamp into one augmented view, or into a pair of views.

    Both methods return float tensors of shape (3, H', W'), where H'xW' is the configured
    crop size.
    """

    def view(self, img: Stamp) -> torch.Tensor:
        """Return one augmented view of ``img``."""

    def pair(self, img: Stamp) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two independently augmented views of ``img`` (the SimCLR positive pair)."""


# ------------------------------------------------------------------------------------------
# Building blocks ported from the legacy tree.


class RandomResizedCenterCrop:
    """``RandomResizedCrop`` that always takes its crop from the centre of the image.

    Port of ``utils/random_resized_center_crop.py``. The crop area and aspect ratio are drawn
    from Python ``random`` (not torch), exactly as in the legacy code, then the centred crop
    is resized to ``size`` with ``interpolation``. After 10 rejected draws it falls back to
    the largest centred crop within ``ratio``, like torchvision.

    Raises:
        ValueError: ``scale`` or ``ratio`` is not an increasing ``(min, max)`` pair (the
            legacy class only warned).
    """

    def __init__(
        self,
        size: int,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        if scale[0] > scale[1] or ratio[0] > ratio[1]:
            msg = f"scale and ratio must be (min, max) pairs, got {scale} and {ratio}"
            raise ValueError(msg)
        self.size = (size, size)
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    @staticmethod
    def get_params(
        img: Image, scale: tuple[float, float], ratio: tuple[float, float]
    ) -> tuple[int, int, int, int]:
        """Draw ``(top, left, height, width)`` of a centred crop of ``img``."""
        width, height = img.size
        area = height * width
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        for _ in range(10):
            # S311: augmentation sampling, not cryptography; Python `random` is the legacy
            # generator, and replacing it would change the seeded sequence.
            target_area = random.uniform(*scale) * area  # noqa: S311
            aspect_ratio = math.exp(random.uniform(*log_ratio))  # noqa: S311
            w = round(math.sqrt(target_area * aspect_ratio))
            h = round(math.sqrt(target_area / aspect_ratio))
            if 0 < w <= width and 0 < h <= height:
                return (height - h) // 2, (width - w) // 2, h, w
        in_ratio = width / height
        if in_ratio < min(ratio):
            w, h = width, round(width / min(ratio))
        elif in_ratio > max(ratio):
            w, h = round(height * max(ratio)), height
        else:
            w, h = width, height
        return (height - h) // 2, (width - w) // 2, h, w

    def __call__(self, img: Image) -> Image:
        """Return the resized centred crop of ``img``."""
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        out: Image = tvf.resized_crop(img, i, j, h, w, list(self.size), self.interpolation)
        return out


class GaussianBlur:
    """Separable Gaussian blur of a PIL image, with sigma ~ U(0.1, 2.0) from ``np.random``.

    Port of ``utils/gaussian_blur.py`` (itself from sthalles/SimCLR). The kernel has
    ``2 * (kernel_size // 2) + 1`` taps (so an even size rounds up), the image is
    reflection-padded, and the result goes back through ``ToPILImage``, which truncates to
    uint8. The legacy class held two ``nn.Conv2d`` modules whose weights it overwrote; this
    port calls ``conv2d`` with the same float32 weights, which gives the same output.

    The legacy module also ran ``np.random.seed(0)`` at import time; that side effect is
    dropped, because seeding is ``seed_everything``'s job.
    """

    def __init__(self, kernel_size: int) -> None:
        self.r = kernel_size // 2
        self.k = 2 * self.r + 1

    def __call__(self, img: Image) -> Image:
        """Return ``img`` blurred with a freshly drawn sigma."""
        x = tvf.to_tensor(img).unsqueeze(0)
        # The legacy generator is the global np.random; a local Generator would change the
        # seeded sequence.
        sigma = np.random.uniform(0.1, 2.0)
        taps = np.arange(-self.r, self.r + 1)
        kernel = np.exp(-np.power(taps, 2) / (2 * sigma * sigma))
        kernel = kernel / kernel.sum()
        weight = torch.from_numpy(kernel).to(torch.float32).view(1, -1).repeat(3, 1)
        with torch.no_grad():
            x = nnf.pad(x, (self.r, self.r, self.r, self.r), mode="reflect")
            x = nnf.conv2d(x, weight.view(3, 1, self.k, 1), groups=3)
            x = nnf.conv2d(x, weight.view(3, 1, 1, self.k), groups=3)
        out: Image = tvf.to_pil_image(x.squeeze(0))
        return out


class RandomRot90:
    """Rotate a (C, H, W) tensor by a uniformly drawn multiple of 90 degrees (torch RNG).

    Port of the legacy ``Random_rotation``.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` rotated by 0, 90, 180 or 270 degrees in the (H, W) plane."""
        times = int(torch.randint(4, (1,))[0])
        return torch.rot90(x, times, [1, 2])


# ------------------------------------------------------------------------------------------
# The protocol implementation and the registry.


class PipelineAugmentation:
    """An :class:`Augmentation` that runs one PIL → tensor pipeline per view.

    ``view`` converts the uint8 stamp to an RGB PIL image and applies ``pipeline``; ``pair``
    calls ``view`` twice, first view first, as the legacy ``__call__`` did.
    """

    def __init__(self, name: str, pipeline: Callable[[Image], torch.Tensor]) -> None:
        self.name = name
        self.pipeline = pipeline

    def view(self, img: Stamp) -> torch.Tensor:
        """Return one augmented view of ``img``: float32 (3, size, size) in [0, 1]."""
        return self.pipeline(tvf.to_pil_image(img))

    def pair(self, img: Stamp) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two independently augmented views of ``img``."""
        first = self.view(img)
        return first, self.view(img)

    def __repr__(self) -> str:
        return f"PipelineAugmentation(name={self.name!r})"


def _color_jitter(hue: float) -> transforms.ColorJitter:
    # The legacy strength constant s = 1: brightness, contrast and saturation 0.8.
    return transforms.ColorJitter(0.8, 0.8, 0.8, hue)


def _blur(size: int) -> transforms.RandomApply:
    return transforms.RandomApply([GaussianBlur(kernel_size=int(0.1 * size))], p=0.5)


def _simclr(size: int) -> list[Callable[..., object]]:
    # Legacy SimCLR_augmentation.
    return [
        transforms.RandomResizedCrop(scale=(0.08, 1.0), size=size),
        transforms.RandomApply([_color_jitter(0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        _blur(size),
        transforms.ToTensor(),
    ]


def _simclr2(size: int) -> list[Callable[..., object]]:
    # Legacy SimCLR_augmentation_v2.
    return [
        RandomResizedCenterCrop(scale=(0.08, 1.0), size=size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([_color_jitter(0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        _blur(size),
        transforms.ToTensor(),
        RandomRot90(),
    ]


def _astro7(size: int) -> list[Callable[..., object]]:
    # Legacy Astro_augmentation_v7.
    return [
        transforms.RandomRotation(180, expand=False),
        RandomResizedCenterCrop(scale=(0.18, 0.5), size=size),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([_color_jitter(0.2)], p=0.8),
        _blur(size),
        transforms.ToTensor(),
    ]


def _astro8(size: int) -> list[Callable[..., object]]:
    # Legacy Astro_augmentation_v8: astro7 plus RandomGrayscale before the blur.
    return [
        transforms.RandomRotation(180, expand=False),
        RandomResizedCenterCrop(scale=(0.18, 0.5), size=size),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([_color_jitter(0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        _blur(size),
        transforms.ToTensor(),
    ]


def _rotation_v3(size: int) -> list[Callable[..., object]]:
    # Legacy Rotation_v3.
    return [
        transforms.RandomRotation(180, expand=False),
        transforms.CenterCrop(size=size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]


def _without_aug(size: int) -> list[Callable[..., object]]:
    # Legacy Resize_img (the classifiers' `without_aug`, and every validation/test set).
    return [transforms.CenterCrop(size=size), transforms.ToTensor()]


#: Legacy name → pipeline builder (crop size → list of transforms) of the ported transforms.
AUGMENTATIONS: dict[str, Callable[[int], list[Callable[..., object]]]] = {
    "simclr": _simclr,
    "simclr2": _simclr2,
    "astro7": _astro7,
    "astro8": _astro8,
    "rotation_v3": _rotation_v3,
    "without_aug": _without_aug,
}

#: Legacy names (``utils/training.py``) that are not ported in phase 1. Most need
#: albumentations or OpenCV, and no phase-1 configuration uses them.
DEFERRED_AUGMENTATIONS = frozenset(
    {
        "astro",
        "astro0",
        "astro2",
        "astro3",
        "astro4",
        "astro5",
        "astro6",
        "astro9",
        "blur",
        "crop_astro",
        "crop_rotation",
        "crop_simclr",
        "elastic_grid",
        "elastic_prespective",
        "elastic_transform",
        "gray_scale",
        "grid_distortion",
        "grid_perspective",
        "jitter_astro",
        "jitter_astro_v2",
        "jitter_astro_v3",
        "jitter_simclr",
        "perspective",
        "rot_elastic",
        "rot_elastic_blur",
        "rot_elastic_grid_perspective",
        "rot_grid",
        "rot_grid_blur",
        "rot_perspective",
        "rot_perspective_blur",
        "rotation",
        "rotation_v2",
        "simclr3",
    }
)


class UnknownAugmentationError(ValueError):
    """The augmentation name is neither ported nor a known legacy name."""


class DeferredAugmentationError(UnknownAugmentationError):
    """The name is a legacy augmentation that phase 1 does not port."""


def get_augmentation(name: str, size: int) -> PipelineAugmentation:
    """Build the augmentation registered under the legacy ``name`` for ``size`` px crops.

    Raises:
        DeferredAugmentationError: ``name`` is a legacy augmentation not ported yet.
        UnknownAugmentationError: ``name`` is not a legacy augmentation at all; the message
            lists the valid names.
        ValueError: ``size`` is not a positive integer.
    """
    if name in DEFERRED_AUGMENTATIONS:
        msg = (
            f"augmentation {name!r} is a legacy variant deferred to a later phase; "
            f"phase 1 ports only {sorted(AUGMENTATIONS)}"
        )
        raise DeferredAugmentationError(msg)
    if name not in AUGMENTATIONS:
        msg = f"unknown augmentation {name!r}; valid names: {sorted(AUGMENTATIONS)}"
        raise UnknownAugmentationError(msg)
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        msg = f"size must be a positive integer, got {size!r}"
        raise ValueError(msg)
    return PipelineAugmentation(name, transforms.Compose(AUGMENTATIONS[name](size)))
