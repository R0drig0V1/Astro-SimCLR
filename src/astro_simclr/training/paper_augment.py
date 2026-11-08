"""The paper model's training augmentation as one engine-free policy object (task 4.6).

:class:`PaperAugment` holds the 2.5 augmentation flags of the paper classifier and does
what ``PaperClassifierModule.augment_plan`` / ``apply_plan`` did (they now delegate here),
so the Lightning module and the plain engine's task (``tasks.paper``) share one
implementation, as CLAUDE.md's rule for training flags asks (ADR 0014, section 7):

- :meth:`PaperAugment.plan` is the **host** part: every draw and decision of a step, from
  ``[seed, step, stream]`` generators (``class_augment.augment_rng``; the rotation keeps
  ``[seed, step]``), on a CPU copy of the batch, in this order: C6 (synthetic Bogus), C7
  (the transient flux rescaling), C5's noise, the rotation, C5's feature jitter (ADR 0006,
  0007). The class-aware augmentation runs before the rotation because C7's PSF fit
  describes the unrotated crop.
- :meth:`PaperAugment.apply` is the **device** part: the plan applied with the same torch
  operations, so the result is bit for bit the in-step one on every device.

``step`` is the number of optimizer steps done before the step (Lightning's ``global_step``
inside ``training_step``; the plain loop's ``steps_done``). Nothing here imports Lightning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from astro_simclr.models import N_CLASSES
from astro_simclr.training.class_augment import (
    DEFAULT_MIN_R2,
    JITTER_STREAM,
    NOISE_STREAM,
    SYNTHETIC_STREAM,
    apply_noise,
    apply_synthetic,
    augment_rng,
    plan_jitter,
    plan_noise,
    plan_synthetic,
)
from astro_simclr.training.flux_rescale import FLUX_STREAM, apply_flux, plan_flux

__all__ = ["AugmentBatch", "ClassifierBatch", "PaperAugment", "check_augment_context"]

#: A classifier batch: images (B, C, S, S), features (B, 23), one-hot labels (B, n_classes).
ClassifierBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
#: A training batch with the stamp fits (B, 6) or (B, 11) as a fourth element.
AugmentBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def check_augment_context(
    synthetic_bogus: float, flux_rescale: float, context: str, n_classes: int
) -> None:
    """Refuse C6 / C7 where they cannot apply: with context channels, or not on CD21's classes.

    Raises:
        ValueError: if ``flux_rescale`` or ``synthetic_bogus`` is set with ``context`` other
            than ``none``, or with ``n_classes`` other than 5.
    """
    if flux_rescale and context != "none":
        raise ValueError("flux_rescale changes no context channels: needs context='none'")
    if synthetic_bogus and context != "none":
        raise ValueError("synthetic_bogus builds no context channels: needs context='none'")
    if (synthetic_bogus or flux_rescale) and n_classes != N_CLASSES:
        # C6 and C7 use CD21's label indices (Asteroid and Bogus sources; never Bogus).
        raise ValueError("synthetic_bogus and flux_rescale need the 5 CD21 classes")


@dataclass(frozen=True, slots=True)
class PaperAugment:
    """The paper classifier's training augmentation (module docstring).

    Attributes:
        seed: the run's seed (``augment_seed``): every draw is keyed by it and the step.
        rot90: rotate every training stamp by ``k * 90`` degrees, ``k`` uniform in 0..3.
        noise: C5's background-noise strength (0: off).
        feature_jitter: C5's feature-jitter strength (0: off).
        synthetic_bogus: C6's fraction of the batch replaced by synthetic Bogus, in [0, 1).
        min_r2: the cross-channel fit quality below which C5 / C6 skip a stamp.
        flux_rescale: C7's probability per non-Bogus row, in [0, 1].
        flux_k_min, flux_k_max: C7's log-uniform factor range, ``0 < min <= max``.

    Raises:
        ValueError: for a negative seed or strength, ``synthetic_bogus`` outside [0, 1),
            ``min_r2`` outside [0, 1], ``flux_rescale`` outside [0, 1], or not
            ``0 < flux_k_min <= flux_k_max`` (``PaperClassifierModule``'s messages).
    """

    seed: int = 0
    rot90: bool = False
    noise: float = 0.0
    feature_jitter: float = 0.0
    synthetic_bogus: float = 0.0
    min_r2: float = DEFAULT_MIN_R2
    flux_rescale: float = 0.0
    flux_k_min: float = 0.5
    flux_k_max: float = 1.5

    def __post_init__(self) -> None:
        """Check the values (module docstring of the class)."""
        if self.seed < 0:
            raise ValueError(f"augment_seed must be >= 0, got {self.seed}")
        if not (self.noise >= 0 and self.feature_jitter >= 0):
            raise ValueError("aug_noise and aug_feature_jitter must be >= 0")
        if not 0 <= self.synthetic_bogus < 1:
            raise ValueError(f"synthetic_bogus must be in [0, 1), got {self.synthetic_bogus}")
        if not 0 <= self.min_r2 <= 1:
            raise ValueError(f"aug_min_r2 must be in [0, 1], got {self.min_r2}")
        if not 0 <= self.flux_rescale <= 1:
            raise ValueError(f"flux_rescale must be in [0, 1], got {self.flux_rescale}")
        if not 0 < self.flux_k_min <= self.flux_k_max:
            raise ValueError("flux_k_min and flux_k_max must satisfy 0 < min <= max")

    @property
    def active(self) -> bool:
        """Whether any training augmentation is on (the run then plans it on the host)."""
        return bool(
            self.rot90
            or self.noise
            or self.feature_jitter
            or self.synthetic_bogus
            or self.flux_rescale
        )

    @property
    def needs_fits(self) -> bool:
        """Whether the training batches need the stamp fits (C5's noise, C6, C7)."""
        return bool(self.noise or self.synthetic_bogus or self.flux_rescale)

    def rotation_steps(self, n: int, step: int) -> torch.Tensor:
        """The ``k`` (int64, CPU, values 0..3) of ``n`` training stamps at step ``step``.

        Drawn from a NumPy generator seeded with ``[seed, step]`` only (its ``SeedSequence``
        mixes both into the whole state), so the same run on another device, or resumed,
        rotates the same stamps the same way.
        """
        rng = np.random.default_rng([self.seed, step])
        return torch.from_numpy(rng.integers(0, 4, size=n, dtype=np.int64))

    def plan(self, batch: ClassifierBatch | AugmentBatch, step: int) -> dict[str, Any]:
        """The host part of step ``step``'s augmentation (module docstring).

        ``batch`` is on the CPU (crops, features, one-hot targets and, for C5 / C6 / C7,
        the stamp fits). Returns a dict of CPU tensors (keys ``c6``, ``c7``, ``c5``,
        ``rot``, ``jitter`` for the augmentations that change the batch).

        Raises:
            ValueError: if C5's noise, C6 or C7 is on and the batch has no stamp fits, or C7
                is on and the fits have no PSF columns.
        """
        images, features = batch[0], batch[1]
        plan: dict[str, Any] = {}
        if self.needs_fits:
            if len(batch) != 4:
                msg = (
                    "aug_noise, synthetic_bogus and flux_rescale need the stamp fits in the "
                    "training batches: build the Train TensorSplit with fits="
                    "class_augment.stamp_fits (plus flux_rescale.fit_central_psf for C7)"
                )
                raise ValueError(msg)
            plan.update(self._class_plan(batch, step))
        if self.rot90:
            plan["rot"] = self.rotation_steps(images.shape[0], step)
        if self.feature_jitter:
            rng = augment_rng(self.seed, step, JITTER_STREAM)
            plan["jitter"] = plan_jitter(tuple(features.shape), self.feature_jitter, rng)
        return plan

    def _class_plan(self, batch: AugmentBatch, step: int) -> dict[str, Any]:
        """C6, C7 and C5's noise of :meth:`plan`, in that order (host tensors)."""
        images, features, targets, fits_tensor = batch
        fits = fits_tensor.numpy()
        plan: dict[str, Any] = {}
        replaced = None
        if self.synthetic_bogus:
            rng = augment_rng(self.seed, step, SYNTHETIC_STREAM)
            c6 = plan_synthetic(
                images, features, targets, fits, self.synthetic_bogus, rng, min_r2=self.min_r2
            )
            replaced = c6.pop("replaced")
            if replaced.any():
                plan["c6"] = c6
        if self.flux_rescale:
            rng = augment_rng(self.seed, step, FLUX_STREAM)
            c7 = plan_flux(
                images,
                features,
                targets,
                fits,
                self.flux_rescale,
                (self.flux_k_min, self.flux_k_max),
                rng,
                skip=replaced,
            )
            c7.pop("k")
            if c7:
                plan["c7"] = c7
        if self.noise:
            rng = augment_rng(self.seed, step, NOISE_STREAM)
            c5 = plan_noise(
                fits, images.shape[-1], self.noise, rng, min_r2=self.min_r2, skip=replaced
            )
            if c5 is not None:
                plan["c5"] = c5
        return plan

    def apply(self, batch: ClassifierBatch | AugmentBatch, plan: dict[str, Any]) -> ClassifierBatch:
        """The device part: ``plan`` (on ``batch``'s device) applied to ``batch``.

        Returns ``(images, features, targets)``; ``batch`` is not modified.
        """
        images, features, targets = batch[0], batch[1], batch[2]
        if "c6" in plan:
            images, features, targets = apply_synthetic(images, features, targets, plan["c6"])
        if "c7" in plan:
            images, features = apply_flux(images, features, plan["c7"])
        if "c5" in plan and len(batch) == 4:
            images = apply_noise(images, batch[3], plan["c5"])
        if "rot" in plan:
            rotations = torch.stack([torch.rot90(images, j, dims=(2, 3)) for j in range(4)])
            images = rotations[plan["rot"], torch.arange(images.shape[0], device=images.device)]
        if "jitter" in plan:
            features = features + plan["jitter"].to(features.dtype)
        return images, features, targets
