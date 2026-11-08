"""Helpers of the phase-3 benchmark tests: synthetic alert tables and fake benchmark runs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from astro_simclr.protocol.alerts import BENCHMARK_LABEL_NAMES, AlertTable
from astro_simclr.protocol.predictions import Predictions, predictions_file, save_predictions

#: RJ23's class sizes (AGN, SN, VS, Asteroid, Bogus, Satellite), as relative weights.
RJ23_WEIGHTS = (9774, 3615, 10211, 9180, 15950, 580)
MACHINE: dict[str, Any] = {"device": "cpu", "gpu": None, "torch": "2.14.1", "cuda": None}


def synthetic_table(
    n_objects: int,
    *,
    seed: int = 0,
    weights: Sequence[float] = RJ23_WEIGHTS,
    mean_alerts: float = 2.0,
    max_alerts: int = 30,
    label_names: Sequence[str] = BENCHMARK_LABEL_NAMES,
) -> AlertTable:
    """A random labelled alert table: objects with 1..max_alerts alerts (geometric).

    The classes are R5's five by default; ``weights`` are RJ23's class sizes (the first
    ``len(label_names)``).

    Every class gets at least 6 objects, so that a 3-way split always has room.
    """
    rng = np.random.default_rng(seed)
    n_classes = len(label_names)
    p = np.asarray(weights, dtype=np.float64)[:n_classes]
    labels = np.concatenate(
        [np.repeat(np.arange(n_classes), 6), rng.choice(n_classes, n_objects, p=p / p.sum())]
    )
    sizes = np.minimum(rng.geometric(1 / mean_alerts, size=labels.size), max_alerts)
    oids = [f"ZTF{18 + i % 3}{i:07d}" for i in range(labels.size)]
    total = int(sizes.sum())
    candids = 10**18 + rng.choice(10 * total, size=total, replace=False)
    jd = 2458270.5 + rng.uniform(0, 1000, size=total)
    rows_oid = np.repeat(oids, sizes).tolist()
    rows_label = np.repeat(labels, sizes).tolist()
    return AlertTable.from_rows(rows_oid, candids.tolist(), jd.tolist(), rows_label, label_names)


def noisy_predictions(
    labels: npt.NDArray[np.int64], accuracy: float, n_classes: int, seed: int
) -> npt.NDArray[np.int64]:
    """Predicted labels: right with probability ``accuracy``, else a random class."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, n_classes, size=labels.size)
    return np.where(rng.random(labels.size) < accuracy, labels, noise).astype(np.int64)


def as_predictions(
    labels: npt.NDArray[np.int64],
    predicted: npt.NDArray[np.int64],
    candids: npt.NDArray[np.int64],
    n_classes: int,
) -> Predictions:
    """Predictions whose argmax is ``predicted``, indexed by ``candid``."""
    probabilities = np.full((labels.size, n_classes), 0.1, dtype=np.float32)
    probabilities[np.arange(labels.size), predicted] = 0.9
    return Predictions(probabilities, labels, candids, n_classes)


def bench_run(
    root: Path,
    name: str,
    predictions: Predictions,
    *,
    splits_sha256: str | None,
    seed: object = 0,
    stage: str = "test",
    machine: dict[str, Any] | None = None,
) -> Path:
    """A benchmark run directory: ``metrics.json`` and one predictions file (the contract)."""
    out = root / name
    out.mkdir(parents=True)
    save_predictions(predictions, out / predictions_file(stage))
    metrics: dict[str, Any] = {
        "seed": seed,
        "machine": MACHINE if machine is None else machine,
        "predictions": {stage: {"file": predictions_file(stage), "split": "Test"}},
    }
    if splits_sha256 is not None:
        metrics["benchmark"] = {"splits_sha256": splits_sha256}
    (out / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return out
