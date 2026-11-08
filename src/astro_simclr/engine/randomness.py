"""Seeded per-step generators: the engine's only sources of training randomness (ADR 0014).

Two kinds, by what the draw is for:

- **host draws** (:func:`host_generator`): per-sample parameters (a rotation ``k``, a noise
  scale, a flux factor ``k``, a crop offset, a SimCLR view's jitter strengths). A NumPy
  generator from ``SeedSequence([seed, step, stream])``, the rule of
  ``class_augment.augment_rng``: the same numbers on every device, so these draws are
  device-independent and reproducible across machines.
- **device fields** (:func:`device_generator`): per-pixel fields (SimCLR noise images)
  whose host draw and upload would cost more than the step (plan finding F12). A
  ``torch.Generator`` on the training device, seeded with 64 bits of the same
  ``SeedSequence``. The seed does not depend on the device, the values do (CPU and CUDA
  generators are different algorithms), so a run that uses one records the device type
  and its results are compared only within it.

``step`` is always the number of optimizer steps done before the step, and ``stream`` a
number from :data:`RESERVED_STREAMS` (a new augmentation takes a new number; reusing one
would correlate two augmentations).
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

import numpy as np
import torch

__all__ = ["RESERVED_STREAMS", "device_generator", "host_generator"]

#: Stream numbers in use (the third ``SeedSequence`` word), by augmentation. ``rot90``
#: keeps the two-word ``[seed, step]`` of task 2.5-models and has no number. Phase 5's
#: SimCLR views take new numbers here.
RESERVED_STREAMS: Final = MappingProxyType(
    {
        "aug_noise": 1,
        "aug_feature_jitter": 2,
        "synthetic_bogus": 3,
        "flux_rescale": 4,
        # rj23_regime.ClassUniformStream (task 3.4b): the classes and items of each batch.
        "rj23_class_uniform": 5,
    }
)


def _check(seed: int, step: int, stream: int) -> None:
    if seed < 0 or step < 0 or stream < 0:
        raise ValueError(f"seed, step and stream must be >= 0, got {seed}, {step}, {stream}")


def host_generator(seed: int, step: int, stream: int) -> np.random.Generator:
    """The host generator of ``stream`` at ``step``: ``default_rng([seed, step, stream])``.

    Raises:
        ValueError: if a word is negative.
    """
    _check(seed, step, stream)
    return np.random.default_rng([seed, step, stream])


def device_generator(seed: int, step: int, stream: int, device: torch.device) -> torch.Generator:
    """A ``torch.Generator`` on ``device`` for the per-pixel fields of ``stream`` at ``step``.

    Seeded with the first 64-bit word of ``SeedSequence([seed, step, stream])``: every bit
    of the three words matters, unlike a sum or a 32-bit seed.

    Raises:
        ValueError: if a word is negative.
    """
    _check(seed, step, stream)
    word = int(np.random.SeedSequence([seed, step, stream]).generate_state(1, np.uint64)[0])
    generator = torch.Generator(device=device)
    generator.manual_seed(word)
    return generator
