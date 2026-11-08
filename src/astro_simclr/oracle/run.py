"""The M1 oracle: run the published model through our input path and model (task 2.2b).

``docs/phases/02-stamp-classifier.md``, task 2.2b. With M1's weights (:mod:`.m1`) and the
prepared ``red`` data (float32 stamps, ``prepare-data --images float32``),
:func:`run_m1_oracle` makes two **separate** checks, both on the 500 Validation stamps:

1. **Port equivalence.** The torch model (:class:`~astro_simclr.paper_model.
   PaperStampClassifier` with the mapped weights) against the independent NumPy NHWC pass
   on the raw TF-layout weights (:func:`.m1.numpy_forward`), on the same inputs. It passes
   when ``max |delta logit| <= 1e-4`` and the argmax agrees on every stamp; a miss is a
   mapping or architecture bug.
2. **Data path.** M1 through ``crop_split`` (task 2.1, NaN approximation on) on ``red``
   Validation. It passes when the number of correct stamps is within ±2 of M1's 475/500
   **and** the selection loss is within ±0.005 of M1's -0.0616. The loss is M1's: the
   ``CE - 0.5 H(softmax)`` of :class:`~astro_simclr.losses.P_stamps_loss`, as the mean of
   batch-64 means in ``red`` file order (TF's evaluation, and the definition of task 2.2a).

The report's top-level ``passed`` (both checks) is the contract that
``compare-reference --ref CD21 --oracle-report`` reads (:func:`astro_simclr.reference.
read_oracle_report`). The report also has, without gating anything:

- the stamps the NaN approximation touches (:func:`~astro_simclr.data.float_input.
  approximated_stamps`), with their accuracy, separately from the others;
- M1's **test** accuracy and loss on ``red`` Test (computed and archived; test is never a
  decision input, ``docs/evaluation-protocol.md``);
- the diagnostics priced without training, **on Validation only**: the accuracy change
  with the NaN approximation off, with the phase-1 uint8 stamps cropped at 27 real pixels
  (no per-crop re-normalisation, no padding; evaluated with the NumPy pass, since the torch
  model takes 21x21 crops), and with the uint8 21x21 crops padded to 27 (the interim input
  of task 2.2a);
- M1's validation curve from its event log (3,000 evaluations), written as CSV by
  :func:`write_curve`, and its best point.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from astro_simclr.data.float_input import PAPER_CROP, approximated_stamps, crop_split
from astro_simclr.data.prepared import load_prepared, read_meta
from astro_simclr.data.schema import DISCARDED_FEATURES, LABEL_NAMES, DataError, SplitData
from astro_simclr.losses import P_stamps_loss
from astro_simclr.oracle.m1 import M1_COMMIT, load_m1_model, numpy_forward
from astro_simclr.oracle.tf_files import read_scalar_events

__all__ = [
    "M1_BETA",
    "M1_EVAL_BATCH",
    "M1_VAL_CORRECT",
    "M1_VAL_LOSS",
    "CurvePoint",
    "read_val_curve",
    "run_m1_oracle",
    "selection_loss",
    "write_curve",
    "write_report",
]

#: M1's validation result at its selected checkpoint (iteration 4,780; plan "Facts").
M1_VAL_CORRECT: Final = 475
M1_VAL_LOSS: Final = -0.0616
#: Acceptance of check (2): stamps and loss.
CORRECT_TOLERANCE: Final = 2
LOSS_TOLERANCE: Final = 0.005
#: Acceptance of check (1).
LOGIT_TOLERANCE: Final = 1e-4
#: M1's entropy weight (``EntropyRegBeta0.5000``) and evaluation batch size.
M1_BETA: Final = 0.5
M1_EVAL_BATCH: Final = 64
#: Side of the phase-1 uint8 crop (torchvision ``CenterCrop(27)``).
_UINT8_SIDE: Final = 27


@dataclass(frozen=True, slots=True)
class CurvePoint:
    """One evaluation of M1's validation log: iteration, loss and accuracy."""

    step: int
    loss: float
    accuracy: float


def read_val_curve(path: Path) -> list[CurvePoint]:
    """M1's validation curve from its TF event file (tags ``loss`` and ``accuracy``).

    Raises:
        OSError: if the file cannot be read.
        TFFormatError: if it is corrupt.
        ValueError: if the two tags are missing or their steps differ.
    """
    scalars = read_scalar_events(path.read_bytes())
    loss, accuracy = scalars.get("loss"), scalars.get("accuracy")
    if not loss or not accuracy:
        raise ValueError(f"{path}: no 'loss' and 'accuracy' scalars")
    if [s for s, _ in loss] != [s for s, _ in accuracy]:
        raise ValueError(f"{path}: the 'loss' and 'accuracy' steps differ")
    return [CurvePoint(s, v, a) for (s, v), (_, a) in zip(loss, accuracy, strict=True)]


def write_curve(points: Sequence[CurvePoint], path: Path) -> None:
    """Write the curve as CSV: ``step,val_loss,val_accuracy``, one row per evaluation.

    Values are written in the shortest form that reads back as the same float32.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["step", "val_loss", "val_accuracy"])
        for point in points:
            # The log stores float32: print its shortest round-trip form.
            writer.writerow([point.step, np.float32(point.loss), np.float32(point.accuracy)])


def selection_loss(
    logits: npt.NDArray[np.floating[Any]],
    labels: npt.NDArray[np.int64],
    *,
    beta: float = M1_BETA,
    batch: int = M1_EVAL_BATCH,
) -> float:
    """M1's selection loss: the mean over batches of ``batch`` (file order) of the batch loss.

    The batch loss is :class:`~astro_simclr.losses.P_stamps_loss` (``CE - beta * H``, both
    batch means); the last batch may be shorter, and counts as one batch, as in TF.

    Raises:
        ValueError: if ``logits`` is not ``(N, 5)`` with ``N == len(labels) > 0``.
    """
    if logits.ndim != 2 or logits.shape != (len(labels), len(LABEL_NAMES)) or not len(labels):
        raise ValueError(f"logits must be (N, 5) with N = len(labels) > 0, got {logits.shape}")
    loss = P_stamps_loss(beta)
    values = torch.from_numpy(np.asarray(logits, dtype=np.float64))
    targets = F.one_hot(torch.from_numpy(np.array(labels, dtype=np.int64)), len(LABEL_NAMES))
    targets = targets.double()
    means = [
        float(loss(values[i : i + batch], targets[i : i + batch]))
        for i in range(0, len(labels), batch)
    ]
    return float(np.mean(means))


@dataclass(frozen=True, slots=True)
class _Split:
    """A labelled split's inputs: stored float32 stamps, uint8 stamps, 23 features, labels."""

    stamps: npt.NDArray[np.float32]
    images: npt.NDArray[np.uint8]
    features: npt.NDArray[np.float32]
    labels: npt.NDArray[np.int64]


def _split(data_dir: Path, name: str, split: SplitData) -> _Split:
    if split.labels is None or split.features is None:
        raise DataError(f"{data_dir}: {name} is unlabelled; the oracle needs labels")
    if split.images_f32 is None:
        raise DataError(
            f"{data_dir}: no images_f32.npy for {name}; re-run prepare-data --images float32"
        )
    kept = np.delete(np.asarray(split.features), list(DISCARDED_FEATURES), axis=1)
    return _Split(
        stamps=np.asarray(split.images_f32),
        images=np.asarray(split.images),
        features=kept.astype(np.float32),
        labels=np.asarray(split.labels, dtype=np.int64),
    )


def _torch_logits(
    model: torch.nn.Module, crops_nhwc: npt.NDArray[np.float32], features: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """The model's eval-mode logits for NHWC crops (moved to NCHW), on the CPU."""
    x_img = torch.from_numpy(np.ascontiguousarray(crops_nhwc.transpose(0, 3, 1, 2)))
    x_feat = torch.from_numpy(np.ascontiguousarray(features))
    with torch.no_grad():
        logits: npt.NDArray[np.float32] = model.eval()(x_img, x_feat).numpy()
    return logits


def _uint8_crop(images: npt.NDArray[np.uint8], side: int) -> npt.NDArray[np.float32]:
    """The centre ``side`` x ``side`` of uint8 stamps, divided by 255 (no re-normalisation).

    ``(63 - side) // 2`` is torchvision's ``CenterCrop`` offset for odd sides; for 21 it
    is ALeRCE's ``[21, 42)``.
    """
    start = (images.shape[1] - side) // 2
    window = images[:, start : start + side, start : start + side, :]
    return np.asarray(window, dtype=np.float32) / np.float32(255.0)


def _scores(logits: npt.NDArray[np.floating[Any]], labels: npt.NDArray[np.int64]) -> dict[str, Any]:
    """Correct count, accuracy, selection loss and per-class recall of ``logits``."""
    hits = logits.argmax(axis=1) == labels
    recall = {
        name: float(hits[labels == k].mean()) if (labels == k).any() else None
        for k, name in enumerate(LABEL_NAMES)
    }
    return {
        "n": len(labels),
        "correct": int(hits.sum()),
        "accuracy": float(hits.mean()),
        "loss": selection_loss(logits, labels),
        "recall": recall,
    }


def _subset(hits: npt.NDArray[np.bool_], mask: npt.NDArray[np.bool_]) -> dict[str, Any]:
    n = int(mask.sum())
    return {
        "n": n,
        "correct": int(hits[mask].sum()),
        "accuracy": float(hits[mask].mean()) if n else None,
    }


def _port_check(
    weights: Mapping[str, npt.NDArray[Any]],
    torch_logits: npt.NDArray[np.float32],
    crops: npt.NDArray[np.float32],
    features: npt.NDArray[np.float32],
) -> dict[str, Any]:
    reference = numpy_forward(weights, crops, features)
    diff = float(np.abs(torch_logits.astype(np.float64) - reference).max())
    agreement = float((torch_logits.argmax(axis=1) == reference.argmax(axis=1)).mean())
    return {
        "passed": diff <= LOGIT_TOLERANCE and agreement == 1.0,
        "n": len(crops),
        "max_abs_logit_diff": diff,
        "tolerance": LOGIT_TOLERANCE,
        "argmax_agreement": agreement,
    }


def _data_check(scores: Mapping[str, Any]) -> dict[str, Any]:
    correct_ok = abs(scores["correct"] - M1_VAL_CORRECT) <= CORRECT_TOLERANCE
    loss_ok = abs(scores["loss"] - M1_VAL_LOSS) <= LOSS_TOLERANCE
    return {
        "passed": correct_ok and loss_ok,
        "correct": scores["correct"],
        "n": scores["n"],
        "accuracy": scores["accuracy"],
        "expected_correct": M1_VAL_CORRECT,
        "correct_tolerance": CORRECT_TOLERANCE,
        "correct_passed": correct_ok,
        "exact": scores["correct"] == M1_VAL_CORRECT,
        "loss": scores["loss"],
        "expected_loss": M1_VAL_LOSS,
        "loss_tolerance": LOSS_TOLERANCE,
        "loss_passed": loss_ok,
        "recall": scores["recall"],
    }


def _diagnostics(
    model: torch.nn.Module,
    weights: Mapping[str, npt.NDArray[Any]],
    val: _Split,
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    """The Validation accuracy of the input variants, and their change from the float path."""
    variants: dict[str, npt.NDArray[np.floating[Any]]] = {
        "no_nan_approximation": _torch_logits(
            model, crop_split(val.stamps, nan_approximation=False), val.features
        ),
        "uint8_21_padded": _torch_logits(model, _uint8_crop(val.images, PAPER_CROP), val.features),
        "uint8_27": numpy_forward(
            weights, _uint8_crop(val.images, _UINT8_SIDE), val.features, pad=0
        ),
    }
    out: dict[str, Any] = {}
    for name, logits in variants.items():
        scores = _scores(logits, val.labels)
        out[name] = {
            "correct": scores["correct"],
            "accuracy": scores["accuracy"],
            # Rounded: a difference of two k/500 accuracies, without float noise.
            "delta_accuracy": round(scores["accuracy"] - baseline["accuracy"], 6),
            "loss": scores["loss"],
        }
    return out


def _curve_summary(points: Sequence[CurvePoint]) -> dict[str, Any]:
    best = min(points, key=lambda p: p.loss)
    peak = max(points, key=lambda p: p.accuracy)
    return {
        "n_evaluations": len(points),
        "last_step": points[-1].step,
        "best_step": best.step,
        "best_val_loss": best.loss,
        "best_val_accuracy": best.accuracy,
        "peak_val_accuracy": peak.accuracy,
        "peak_val_accuracy_step": peak.step,
    }


def run_m1_oracle(
    data_dir: Path,
    weights: Mapping[str, npt.NDArray[Any]],
    *,
    curve: Sequence[CurvePoint] | None = None,
) -> dict[str, Any]:
    """Run both oracle checks and the reported numbers; return the report (module docstring).

    ``data_dir`` is a prepared ``red`` layout with the float32 stamps; ``weights`` are M1's
    TF-layout weights (:func:`.m1.load_weights`); ``curve`` is M1's validation curve
    (:func:`read_val_curve`), summarised in the report when given. Everything runs on the
    CPU in eval mode. The report is JSON-serialisable, with top-level ``passed``.

    Raises:
        DataError: if the layout is invalid, a split is unlabelled or has no float32 stamps.
        ValueError: if ``weights`` are not M1's variables.
    """
    model = load_m1_model(weights)
    data = load_prepared(data_dir)
    val = _split(data_dir, "Validation", data.splits["Validation"])
    test = _split(data_dir, "Test", data.splits["Test"])

    val_crops = crop_split(val.stamps)
    val_logits = _torch_logits(model, val_crops, val.features)
    port = _port_check(weights, val_logits, val_crops, val.features)
    val_scores = _scores(val_logits, val.labels)
    data_path = _data_check(val_scores)

    hits = val_logits.argmax(axis=1) == val.labels
    flagged = approximated_stamps(val.stamps)
    test_scores = _scores(_torch_logits(model, crop_split(test.stamps), test.features), test.labels)
    source = read_meta(data_dir).get("source", {})
    report: dict[str, Any] = {
        "passed": bool(port["passed"] and data_path["passed"]),
        "checks": {"port_equivalence": port, "data_path": data_path},
        "nan_approximation": {
            "approximated": {
                **_subset(hits, flagged),
                "indices": [int(i) for i in np.flatnonzero(flagged)],
            },
            "others": _subset(hits, ~flagged),
            "test_approximated": int(approximated_stamps(test.stamps).sum()),
        },
        "test": test_scores,
        "diagnostics_validation": _diagnostics(model, weights, val, val_scores),
        "m1": {"repository": "alercebroker/pipeline", "commit": M1_COMMIT},
        "data": {"name": source.get("name"), "sha256": source.get("sha256")},
        "versions": {
            "astro_simclr": version("astro-simclr"),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    if curve:
        report["m1"]["val_curve"] = _curve_summary(curve)
    return report


def write_report(report: Mapping[str, Any], path: Path) -> None:
    """Write ``report`` as indented JSON (UTF-8, trailing newline)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
