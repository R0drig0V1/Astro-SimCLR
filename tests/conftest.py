"""Shared pytest fixtures: the synthetic dataset in the prepared layout.

The constants the tests import (``N_PER_SPLIT``, ``ACCELERATORS``) live in
``_training_support.py``: test modules should not import ``conftest`` directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _training_support import N_PER_SPLIT

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
