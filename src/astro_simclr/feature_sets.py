"""The alert features the paper model's head reads: ``Settings.feature_set`` (ADR 0016).

A diagnostic of task 2.5 (``docs/phases/02-stamp-classifier.md``): how much does a model lose
with only the images, or with the images and the sky position? A feature-importance analysis
of the replication found 0.943 Validation accuracy with the 23 features and 0.886 retrained
without them, and that the position (``gal_lat``, ``ecl_lat``) carries the VS-vs-AGN and
Asteroid-vs-SN cues that the pixels lack.

The sets, as columns of the 23 kept features (:data:`KEPT_FEATURES`: the stored 26 without
``rb``, ``distnr`` and ``magnr``):

- ``all``: the 23 (the faithful model);
- ``position``: the 6 of :data:`POSITION_FEATURES`, ``ra``, ``dec``, ``ecl_lat``,
  ``ecl_long``, ``gal_lat`` and ``gal_long`` (kept columns 10, 11, 16, 17, 18, 19);
- ``none``: no column, the images only (the ``with_features: false`` model).

The selection is part of the model, not of the data: every batch still carries the 23
columns (so the data path, the split runs, the C5 jitter and the C7 ``magpsf`` update are
unchanged), and the model picks its columns before the features BatchNorm, which (with fc2)
is sized by the selected count. ``all`` reads the 23 columns as before, with no indexing, so
the faithful model's draws and outputs are unchanged. ``none`` builds exactly the
``with_features: false`` model.

These are pure functions with no randomness and no engine import (ADR 0014): the models of
:mod:`astro_simclr.paper_model` and :mod:`astro_simclr.paper_variants` call them.
"""

from __future__ import annotations

from typing import Final

from astro_simclr.config import FEATURE_SETS
from astro_simclr.data.schema import DISCARDED_FEATURES, FEATURE_NAMES

__all__ = [
    "FEATURE_SETS",
    "KEPT_FEATURES",
    "POSITION_FEATURES",
    "feature_columns",
    "feature_names",
]

#: The 23 features the paper model reads, in order (the stored 26 without columns 13-15).
#: The same tuple as ``training.class_augment.KEPT_FEATURE_NAMES`` (a test checks it); kept
#: here so that the models need not import the training package.
KEPT_FEATURES: Final = tuple(
    name for i, name in enumerate(FEATURE_NAMES) if i not in DISCARDED_FEATURES
)
#: ``feature_set: position``: the sky coordinates among the kept features, in stored order.
POSITION_FEATURES: Final = ("ra", "dec", "ecl_lat", "ecl_long", "gal_lat", "gal_long")

_COLUMNS: Final[dict[str, tuple[int, ...]]] = {
    "all": tuple(range(len(KEPT_FEATURES))),
    "position": tuple(KEPT_FEATURES.index(name) for name in POSITION_FEATURES),
    "none": (),
}


def feature_columns(feature_set: str, *, with_features: bool = True) -> tuple[int, ...]:
    """The columns of the (N, 23) features that the head reads, in increasing order.

    ``all`` gives 0..22, ``position`` the 6 coordinates (:data:`POSITION_FEATURES`), ``none``
    nothing. ``with_features=False`` reads nothing whatever the set, except that it
    contradicts ``position``.

    Raises:
        ValueError: if ``feature_set`` is not in :data:`FEATURE_SETS`, or it is ``position``
            with ``with_features=False``.
    """
    if feature_set not in FEATURE_SETS:
        valid = ", ".join(FEATURE_SETS)
        raise ValueError(f"feature_set must be one of {valid}, got {feature_set!r}")
    if not with_features:
        if feature_set == "position":
            raise ValueError("feature_set 'position' needs with_features=True")
        return ()
    return _COLUMNS[feature_set]


def feature_names(feature_set: str, *, with_features: bool = True) -> tuple[str, ...]:
    """The names of :func:`feature_columns`, in the same order.

    Raises:
        ValueError: as :func:`feature_columns`.
    """
    columns = feature_columns(feature_set, with_features=with_features)
    return tuple(KEPT_FEATURES[i] for i in columns)
