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
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from astro_simclr.data.files import atomic_write_bytes
from astro_simclr.data.preprocessing import preprocess_stamps
from astro_simclr.data.schema import FEATURE_NAMES, LABEL_NAMES, SPLITS, STAMP_SHAPE

__all__ = ["CLASSIFIER_FILES", "PRETRAINING_FILE", "class_pattern", "make_synthetic"]

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
