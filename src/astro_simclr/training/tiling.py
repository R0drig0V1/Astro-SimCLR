"""Class balance by deterministic tiling and ALeRCE's shuffled batch stream (task 2.2a).

The implementation moved to :mod:`astro_simclr.engine.stream` (task 4.3), which imports no
Lightning, so the plain engine's data tiers and today's ``DeviceTensorDataModule`` share
one definition of the stream; its module docstring documents ALeRCE's tiling and shuffle.
This module re-exports the two names for the Lightning path and its tests.
"""

from __future__ import annotations

from astro_simclr.engine.stream import PermutationStream, tile_indices

__all__ = ["PermutationStream", "tile_indices"]
