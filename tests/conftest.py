"""Shared pytest fixtures: the synthetic dataset in the prepared layout.

``classifier_f32_dir`` is the 100 % file prepared with its float32 stamps (the paper's input,
task 2.2c), with one Train stamp made a copy of a Validation stamp, like ``red``'s leaked
Train 2995.

The constants the tests import (``N_PER_SPLIT``, ``ACCELERATORS``) live in
``_training_support.py``: test modules should not import ``conftest`` directly.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest
from _training_support import LEAKED_TRAIN, LEAKED_VALIDATION, N_PER_SPLIT

from astro_simclr.data import prepare
from astro_simclr.data.synthetic import CLASSIFIER_FILES, PRETRAINING_FILE, make_synthetic


@pytest.fixture(scope="session")
def synthetic_prepared(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """``make-synthetic`` then ``prepare-data`` for the four files: file stem -> layout dir."""
    raw = tmp_path_factory.mktemp("synthetic_raw")
    out = tmp_path_factory.mktemp("synthetic_prepared")
    make_synthetic(raw, n_per_split=N_PER_SPLIT, seed=0)
    return {
        Path(name).stem: prepare(raw / name, out) for name in (*CLASSIFIER_FILES, PRETRAINING_FILE)
    }


@pytest.fixture(scope="session")
def classifier_dir(synthetic_prepared: dict[str, Path]) -> Path:
    """The prepared synthetic 100 % classifier file."""
    return synthetic_prepared[Path(CLASSIFIER_FILES[0]).stem]


@pytest.fixture(scope="session")
def pretraining_dir(synthetic_prepared: dict[str, Path]) -> Path:
    """The prepared synthetic pretraining file (unlabelled Train split)."""
    return synthetic_prepared[Path(PRETRAINING_FILE).stem]


@pytest.fixture(scope="session")
def classifier_f32_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The synthetic 100 % classifier file with float32 stamps and one leaked Train stamp.

    The same pickle as ``classifier_dir`` (seed 0), except that Train stamp
    ``LEAKED_TRAIN`` gets the image and features of Validation stamp ``LEAKED_VALIDATION``
    (its label is kept, as in ``red``), prepared with ``images="float32"``.
    """
    raw = tmp_path_factory.mktemp("synthetic_raw_f32")
    out = tmp_path_factory.mktemp("synthetic_prepared_f32")
    make_synthetic(raw, n_per_split=N_PER_SPLIT, seed=0)
    source = raw / CLASSIFIER_FILES[0]
    # Our own synthetic file, written just above: unpickling it is safe.
    content = pickle.loads(source.read_bytes())  # noqa: S301
    for key in ("images", "features"):
        content["Train"][key][LEAKED_TRAIN] = content["Validation"][key][LEAKED_VALIDATION]
    source.write_bytes(pickle.dumps(content))
    return prepare(source, out, images="float32")
