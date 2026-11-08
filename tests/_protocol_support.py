"""Helpers of the protocol tests: small prepared layouts and fake run directories."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from astro_simclr.data.prepared import write_prepared
from astro_simclr.data.schema import FEATURE_NAMES, LABEL_NAMES, STAMP_SHAPE, SplitData, StampData
from astro_simclr.protocol.predictions import Predictions, predictions_file, save_predictions

N_CLASSES = len(LABEL_NAMES)
#: The machine block of a fake run (what ``experiments._machine`` writes).
MACHINE: dict[str, Any] = {"device": "cpu", "gpu": None, "torch": "2.14.0", "cuda": None}


def _split(rng: np.random.Generator, per_class: int) -> SplitData:
    n = per_class * N_CLASSES
    images = rng.integers(0, 256, size=(n, *STAMP_SHAPE), dtype=np.uint8)
    labels = np.repeat(np.arange(N_CLASSES, dtype=np.int64), per_class)
    features = rng.normal(size=(n, len(FEATURE_NAMES)))
    return SplitData(images, labels, features)


def make_layout(
    dest: Path,
    *,
    train: int,
    validation: int,
    test: int,
    duplicate: tuple[int, str, int] | None = None,
    seed: int = 0,
) -> Path:
    """Write a prepared classifier layout with ``train``/``validation``/``test`` per class.

    ``duplicate = (i, split, j)`` copies Train stamp ``i`` (image and features) over stamp
    ``j`` of ``split``, as in ``red`` (the label stays the split's own).
    """
    rng = np.random.default_rng(seed)
    splits = {
        "Train": _split(rng, train),
        "Validation": _split(rng, validation),
        "Test": _split(rng, test),
    }
    if duplicate is not None:
        i, name, j = duplicate
        other = splits[name]
        assert other.features is not None and splits["Train"].features is not None
        other.images[j] = splits["Train"].images[i]
        other.features[j] = splits["Train"].features[i]
    source = {"name": f"{dest.name}.pkl", "size_bytes": 0, "sha256": f"{seed:064d}"}
    return write_prepared(StampData("classifier", splits), dest, source)


def predictions(
    labels: Sequence[int], predicted: Sequence[int], index: Sequence[int] | None = None
) -> Predictions:
    """Predictions with one-hot-ish probabilities that argmax to ``predicted``."""
    n = len(labels)
    probabilities = np.full((n, N_CLASSES), 0.1, dtype=np.float32)
    probabilities[np.arange(n), np.asarray(predicted)] = 0.6
    return Predictions(
        probabilities,
        np.asarray(labels, dtype=np.int64),
        np.arange(n, dtype=np.int64) if index is None else np.asarray(index, dtype=np.int64),
    )


def fake_run(
    root: Path,
    name: str,
    preds: dict[str, Predictions],
    *,
    seed: int = 0,
    machine: dict[str, Any] | None = None,
    **extra: Any,
) -> Path:
    """A run directory with a ``metrics.json`` and one predictions file per stage."""
    out = root / name
    out.mkdir()
    entries = {}
    for stage, pred in preds.items():
        save_predictions(pred, out / predictions_file(stage))
        entries[stage] = {
            "file": predictions_file(stage),
            "split": {"val": "Validation", "test": "Test"}[stage],
            "n": int(pred.labels.size),
        }
    metrics = {
        "command": "train-supervised",
        "seed": seed,
        "machine": MACHINE if machine is None else machine,
        "predictions": entries,
        **extra,
    }
    (out / "metrics.json").write_text(json.dumps(metrics))
    return out


def fake_split_run(
    root: Path,
    name: str,
    pred: Predictions,
    *,
    sha256: str,
    seed: int = 0,
    fold: int | None = None,
    machine: dict[str, Any] | None = None,
    source: str = "0" * 64,
) -> Path:
    """A ``train-supervised --splits`` run directory: ``protocol`` and evaluation predictions.

    ``fold=None`` makes a screen run. ``sha256`` is the splits file's, as ``experiments``
    records it.
    """
    out = root / name
    out.mkdir(parents=True)
    save_predictions(pred, out / predictions_file("evaluation"))
    metrics = {
        "command": "train-supervised",
        "seed": seed,
        "machine": MACHINE if machine is None else machine,
        "data": {"source": {"sha256": source}},
        "protocol": {
            "mode": "screen" if fold is None else "fold",
            "fold": fold,
            "splits_sha256": sha256,
        },
        "predictions": {
            "evaluation": {
                "file": predictions_file("evaluation"),
                "split": "evaluation",
                "n": int(pred.labels.size),
            }
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics))
    return out
