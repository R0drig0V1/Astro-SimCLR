"""Synthetic stand-ins for the four ZTF pickles, with the real per-file schemas.

``make-synthetic`` lets the whole pipeline (prepare-data, training, the preflight) run
without the real data, which is not public. The files copy the containers and dtypes of
the real ones (checked on 2026-11-07, ``docs/data.md``), so the loader is exercised on the
same structure:

- classifier files: per split, ``images`` a list of float32 (63, 63, 3) arrays in [0, 1],
  ``features`` a list of object arrays of 26 values (numpy float64 and Python float),
  ``labels`` a list of numpy int64; keys in the real order;
- the pretraining file: ``Train`` holds only ``images``, a list of float64 arrays;
  ``Validation`` and ``Test`` are the classifier's.

Each stamp carries a strong, rotation-invariant class pattern inside the central 27x27
crop (the stamp encoders' input), so that a classifier can learn it in one epoch. The
generator is deterministic: the same seed gives byte-identical files.

:func:`synthetic_raw_alerts` is the stand-in of a fetched ``raw-alerts`` dataset (task
3.1b, :mod:`astro_simclr.data.raw_alerts`): raw float32 cutouts in counts, with NaN and inf
pixels, any number of classes, and a per-row table for ``write_raw_alerts``;
:func:`synthetic_cd21_features` the optional 23-feature table of that layout.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from astro_simclr.data.files import atomic_write_bytes
from astro_simclr.data.preprocessing import preprocess_stamps
from astro_simclr.data.schema import (
    DISCARDED_FEATURES,
    FEATURE_NAMES,
    LABEL_NAMES,
    SPLITS,
    STAMP_SHAPE,
)

__all__ = [
    "CLASSIFIER_FILES",
    "PRETRAINING_FILE",
    "class_pattern",
    "make_synthetic",
    "synthetic_cd21_features",
    "synthetic_raw_alerts",
]

#: Classifier file names: the 100 %, 10 % and 1 % label sets, named as the received files
#: (the 100 % one is ``red-002``, the deduplicated original that stands in for the legacy
#: ``td_ztf_stamp_17_06_20.pkl``, never received), so configs work on both.
CLASSIFIER_FILES: tuple[str, ...] = (
    "td_ztf_stamp_17_06_20_red-002.pkl",
    "td_ztf_stamp_17_06_20_sup_10.pkl",
    "td_ztf_stamp_17_06_20_sup_1.pkl",
)
#: Pretraining file name (legacy ``utils/simclr_dataset.py``).
PRETRAINING_FILE = "td_ztf_stamp_simclr_300.pkl"

#: Pickle protocol of the real files.
_PROTOCOL = 4
#: Standard deviation of the pixel noise, relative to a pattern amplitude of 1.
_NOISE = 0.05
#: Feature columns stored as Python floats in the real files (the rest are np.float64).
_PYTHON_FLOAT_COLUMNS = (17, 18, 23)

#: Per class: Gaussian width of the science, template and difference channels (``None``:
#: noise only), the sign of the difference, and a ring radius for the difference (0: none).
_PATTERNS: dict[int, tuple[float | None, float | None, float | None, float, float]] = {
    0: (2.0, 2.0, 2.0, 1.0, 0.0),  # AGN: point source in all three, brighter
    1: (2.0, 6.0, 2.0, 1.0, 0.0),  # SN: point source on a wide host
    2: (2.0, 2.0, 2.0, -1.0, 0.0),  # VS: point source that faded (negative difference)
    3: (1.2, None, 1.2, 1.0, 0.0),  # Asteroid: sharp, absent from the template
    4: (None, None, 1.5, 1.0, 5.0),  # Bogus: a ring artefact in the difference only
}


def class_pattern(label: int) -> npt.NDArray[np.float64]:
    """The noise-free (63, 63, 3) pattern of class ``label``, before normalization.

    Raises:
        KeyError: if ``label`` is not in 0-4.
    """
    sigma_sci, sigma_tmp, sigma_diff, sign, ring = _PATTERNS[label]
    yy, xx = np.mgrid[: STAMP_SHAPE[0], : STAMP_SHAPE[1]] - STAMP_SHAPE[0] // 2
    radius = np.hypot(xx, yy)

    def blob(sigma: float | None, centre: float = 0.0) -> npt.NDArray[np.float64]:
        if sigma is None:
            return np.zeros(STAMP_SHAPE[:2])
        profile: npt.NDArray[np.float64] = np.exp(-((radius - centre) ** 2) / (2 * sigma**2))
        return profile

    return np.stack([blob(sigma_sci), blob(sigma_tmp), sign * blob(sigma_diff, ring)], axis=-1)


def _stamp(rng: np.random.Generator, label: int, dtype: type[np.floating[Any]]) -> Any:
    noisy = class_pattern(label) + rng.normal(0.0, _NOISE, STAMP_SHAPE)
    # The real stamps went through the same normalization (nan_val=0 for the 300k set).
    return preprocess_stamps(noisy, nan_val=0).astype(dtype)


def _features(rng: np.random.Generator, label: int) -> npt.NDArray[np.object_]:
    values = rng.normal(0.0, 1.0, len(FEATURE_NAMES))
    values[label] += 2.0  # a weaker signal than the stamps, in a kept column
    row = np.empty(len(FEATURE_NAMES), dtype=object)
    for column, value in enumerate(values):
        row[column] = float(value) if column in _PYTHON_FLOAT_COLUMNS else np.float64(value)
    return row


def _balanced_labels(rng: np.random.Generator, n: int) -> npt.NDArray[np.int64]:
    return rng.permutation(np.arange(n) % len(LABEL_NAMES)).astype(np.int64)


def _labelled_split(rng: np.random.Generator, n: int) -> dict[str, list[Any]]:
    labels = _balanced_labels(rng, n)
    return {
        "images": [_stamp(rng, int(label), np.float32) for label in labels],
        "features": [_features(rng, int(label)) for label in labels],
        "labels": [np.int64(label) for label in labels],
    }


def _subsample(
    rng: np.random.Generator, train: dict[str, list[Any]], n: int
) -> dict[str, list[Any]]:
    # The real 1 % / 10 % files are a random subset of the full Train split, balanced by
    # duplicating stamps; here: n stamps drawn with replacement, an equal share per class.
    labels = np.asarray(train["labels"])
    target = _balanced_labels(rng, n)
    picks = [int(rng.choice(np.flatnonzero(labels == label))) for label in target]
    return {key: [values[i] for i in picks] for key, values in train.items()}


def make_synthetic(out_dir: Path, n_per_split: int, seed: int) -> list[Path]:
    """Write the four synthetic pickles into ``out_dir`` and return their paths.

    Every split of every file has ``n_per_split`` stamps, with the classes as balanced as
    ``n_per_split`` allows. The 10 % and 1 % files share the full file's Validation and
    Test splits and draw their Train split from its Train split (with duplicates, like the
    real ones); the pretraining file holds ``n_per_split`` fresh unlabelled stamps.

    Raises:
        ValueError: if ``n_per_split`` is smaller than the number of classes (5).
        OSError: if ``out_dir`` cannot be written.
    """
    if n_per_split < len(LABEL_NAMES):
        raise ValueError(f"n_per_split must be >= {len(LABEL_NAMES)}, got {n_per_split}")
    rng = np.random.default_rng(seed)
    full = {split: _labelled_split(rng, n_per_split) for split in SPLITS}
    files: dict[str, dict[str, Any]] = {CLASSIFIER_FILES[0]: full}
    for name in CLASSIFIER_FILES[1:]:
        files[name] = {
            "Train": _subsample(rng, full["Train"], n_per_split),
            "Validation": full["Validation"],
            "Test": full["Test"],
        }
    unlabelled = _balanced_labels(rng, n_per_split)
    files[PRETRAINING_FILE] = {
        "Train": {"images": [_stamp(rng, int(label), np.float64) for label in unlabelled]},
        "Validation": full["Validation"],
        "Test": full["Test"],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, content in files.items():
        path = out_dir / name
        atomic_write_bytes(path, pickle.dumps(content, protocol=_PROTOCOL))
        paths.append(path)
    return paths


#: Raw cutouts: sky level and difference amplitude, in counts, and the pixel noise sigma.
_RAW_SKY, _RAW_AMPLITUDE, _RAW_NOISE = 500.0, 2000.0, 20.0
#: The first synthetic ``candid`` (ZTF candids are positive 64-bit integers near 1e18).
_FIRST_CANDID = 1_200_000_000_000_000_000


def synthetic_raw_alerts(
    label_names: tuple[str, ...], n_per_class: int, seed: int
) -> tuple[npt.NDArray[np.float32], dict[str, npt.NDArray[Any]]]:
    """Synthetic raw cutouts and their per-row columns, for ``raw_alerts.write_raw_alerts``.

    ``n_per_class`` rows per class, shuffled. Class ``k`` has a Gaussian point source of
    width ``1 + 0.6 k`` pixels in the science and difference channels, on a sky of 500
    counts with noise; every 7th row has a NaN corner (a stamp on a CCD edge) and every 11th
    a ``+inf`` pixel, as raw alerts can. The columns are those of ``raw_alerts.INDEX_DTYPE``,
    with unique ``candid`` and ``oid`` values, ``label_source`` and ``source_dataset``
    ``"synthetic"``, and ``source_row`` the row number. Deterministic in ``seed``.

    Raises:
        ValueError: if there are fewer than 2 classes or ``n_per_class < 1``.
    """
    if len(label_names) < 2 or n_per_class < 1:
        raise ValueError(
            f"need >= 2 classes and n_per_class >= 1, got {label_names}, {n_per_class}"
        )
    rng = np.random.default_rng(seed)
    labels = rng.permutation(np.repeat(np.arange(len(label_names)), n_per_class)).astype(np.int64)
    n = len(labels)
    yy, xx = np.mgrid[: STAMP_SHAPE[0], : STAMP_SHAPE[1]] - STAMP_SHAPE[0] // 2
    radius2 = (xx**2 + yy**2).astype(np.float64)
    cutouts = _RAW_SKY + rng.normal(0.0, _RAW_NOISE, (n, *STAMP_SHAPE))
    for row, label in enumerate(labels):
        blob = _RAW_AMPLITUDE * np.exp(-radius2 / (2 * (1.0 + 0.6 * float(label)) ** 2))
        cutouts[row, :, :, 0] += blob
        cutouts[row, :, :, 2] += blob - _RAW_SKY  # a difference is centred on 0
    cutouts[::7, :5, :5, :] = np.nan
    cutouts[::11, 0, -1, 1] = np.inf
    rows = np.arange(n)
    columns: dict[str, npt.NDArray[Any]] = {
        "oid": np.array([f"ZTF{18 + i % 4}syn{i:05d}" for i in rows]),
        "candid": _FIRST_CANDID + rows * 1_000,
        "jd": 2458300.5 + rng.uniform(0.0, 1200.0, n),
        "fid": 1 + rows % 2,
        "ra": rng.uniform(0.0, 360.0, n),
        "dec": rng.uniform(-30.0, 90.0, n),
        "label": labels,
        "label_source": np.full(n, "synthetic"),
        "source_dataset": np.full(n, "synthetic"),
        "source_row": rows,
    }
    return cutouts.astype(np.float32), columns


def synthetic_cd21_features(n: int, seed: int) -> npt.NDArray[np.float64]:
    """Synthetic float64 (n, 23) CD21 features for ``write_raw_alerts(features=...)``.

    Normal values around 0, with a NaN (a missing alert field) in every 5th row's first
    column. Deterministic in ``seed``.

    Raises:
        ValueError: if ``n < 1``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    kept = len(FEATURE_NAMES) - len(DISCARDED_FEATURES)
    features = np.random.default_rng(seed).normal(0.0, 1.0, (n, kept))
    features[::5, 0] = np.nan
    return features
