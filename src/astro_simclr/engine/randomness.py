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

:func:`seed_everything` seeds the **global** streams at the start of a run (ADR 0014, C1),
as ``lightning.seed_everything`` does, so both engines build the same initial weights.
"""

from __future__ import annotations

import random
from types import MappingProxyType
from typing import Final

import numpy as np
import torch

__all__ = ["RESERVED_STREAMS", "device_generator", "host_generator", "seed_everything"]

#: Stream numbers in use (the third ``SeedSequence`` word), by augmentation. ``rot90``
#: keeps the two-word ``[seed, step]`` of task 2.5-models and has no number (the same two
#: words as the ``row`` training stream's passes, ``engine.stream``; ADR 0018).
#: ``block_shuffle`` is the ``block`` training stream's pass word (``[seed, pass, 5]``),
#: task 4.3. Phase 5 (task 5.0b, :mod:`astro_simclr.simclr`; ADR 0025): ``simclr_views``
#: the per-sample view parameters (host), ``simclr_view_noise`` the views' per-pixel noise
#: fields (device), ``simclr_batch_order`` the permutation of each pass over the stamps (its
#: second word is the pass, not the step) and ``simclr_diagnostics`` the fixed diagnostic
#: sample and views (second word 0). ``rj23_class_uniform`` is the batch draw of
#: ``rj23_regime.ClassUniformStream`` (``[seed, step, 10]``, task 3.4b). Task 5.1:
#: ``corpus_duplicates``, the draw of ``build-corpus-index --duplicates random`` among the
#: alerts of one ``oid`` (``[seed, 0, 11]``; data building, not training, but a number of
#: its own all the same).
RESERVED_STREAMS: Final = MappingProxyType(
    {
        "aug_noise": 1,
        "aug_feature_jitter": 2,
        "synthetic_bogus": 3,
        "flux_rescale": 4,
        "block_shuffle": 5,
        "simclr_views": 6,
        "simclr_view_noise": 7,
        "simclr_batch_order": 8,
        "simclr_diagnostics": 9,
        # rj23_regime.ClassUniformStream (task 3.4b): the classes and items of each batch.
        # Two streams on one number would draw correlated numbers.
        "rj23_class_uniform": 10,
        # archive.corpus_index (task 5.1): the one alert kept per oid with --duplicates random.
        "corpus_duplicates": 11,
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


#: NumPy's legacy seeding accepts 32-bit unsigned seeds only (Lightning's bounds).
_MAX_SEED: Final = 2**32 - 1


def seed_everything(seed: int) -> int:
    """Seed Python's ``random``, NumPy's global generator and torch (CPU and every CUDA device).

    The three calls of ``lightning.seed_everything(seed)``, in its order (``random.seed``,
    ``np.random.seed``, ``torch.manual_seed``, which also seeds every CUDA device), without
    its environment variables (``PL_GLOBAL_SEED``, ``PL_SEED_WORKERS``: the plain engine has
    no worker processes). Returns ``seed``.

    Raises:
        ValueError: if ``seed`` is not in ``[0, 2**32 - 1]``.
    """
    if isinstance(seed, bool) or not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"seed must be in [0, {_MAX_SEED}], got {seed!r}")
    random.seed(seed)
    np.random.seed(seed)  # the global stream is what is being seeded
    torch.manual_seed(seed)
    return seed
