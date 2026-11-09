"""The classifier results of ``metrics.json``, shared by both engines (task 4.6; Lightning-free).

Moved from ``experiments`` so that a plain run (``tasks.plain``) writes exactly the results a
Lightning run writes. Each evaluated stage is a :class:`StageOutput` (the probabilities,
the predicted and true labels, and the row-normalised confusion matrix, as
``ClassifierModule`` keeps them after ``validate`` / ``test``):

- :func:`classifier_results`: a normal run's Validation / Test results (``val_acc_best``,
  ``test_acc_at_best``, the macro-F1s, ``evaluation``, ``predictions_{val,test}.npz``, the
  confusion-matrix figures), or a split run's selection / evaluation results
  (``selection_acc_best``, ``evaluation_*``, ``predictions_evaluation.npz``);
- :func:`evaluation_entry` (with the ECE of plan row D3, a secondary metric),
  :func:`write_predictions`, :func:`plot_confusion_matrix`, :func:`selection_entry` (which
  weights were scored: ``metrics.json["selection"]``), and the JSON helpers :func:`finite`
  and :func:`data_info`.

The keys ``val_acc_best`` / ``test_acc_at_best`` (and a split run's ``selection_acc_best``)
keep their names under ``checkpoint_selection: final``, where they are the final weights'
values: ``metrics.json["selection"]`` says which weights they are.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from astro_simclr.config import Settings
from astro_simclr.data.prepared import read_meta
from astro_simclr.data.schema import LABEL_NAMES, DataError
from astro_simclr.protocol.calibration import ECE_BINS, expected_calibration_error
from astro_simclr.protocol.predictions import Predictions, predictions_file, save_predictions
from astro_simclr.protocol.split_runs import SplitRun
from astro_simclr.training.metrics import confusion_counts, confusion_matrix, macro_f1

__all__ = [
    "StageOutput",
    "classifier_results",
    "data_info",
    "evaluation_entry",
    "finite",
    "plot_confusion_matrix",
    "recalls_of",
    "selection_entry",
    "write_predictions",
]


@dataclass(frozen=True, slots=True)
class StageOutput:
    """One evaluated stage, on the CPU, in loader (file) order.

    Attributes:
        probabilities: (N, C) softmax outputs.
        y_pred: (N,) int64 argmax predictions.
        y_true: (N,) int64 true labels.
        confusion: (C, C) row-normalised confusion matrix (``training.metrics``).
    """

    probabilities: torch.Tensor
    y_pred: torch.Tensor
    y_true: torch.Tensor
    confusion: torch.Tensor

    @classmethod
    def of(
        cls, probabilities: torch.Tensor, targets: torch.Tensor, device: torch.device
    ) -> StageOutput:
        """From probabilities and labels; the confusion matrix is computed on ``device``.

        ``ClassifierModule`` computes it on the training device; doing the same keeps the
        recorded matrix bit for bit the Lightning run's on every device.
        """
        y_pred = probabilities.argmax(dim=1)
        matrix = confusion_matrix(
            y_pred.to(device), targets.to(device), num_classes=probabilities.shape[1]
        )
        return cls(probabilities.cpu(), y_pred.cpu(), targets.cpu(), matrix.cpu())


def finite(value: object) -> object:
    """Recursively replace non-finite floats with None (strict JSON has no NaN)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [finite(item) for item in value]
    return value


def data_info(path: Path) -> dict[str, Any]:
    """``metrics.json["data"]``: the prepared layout's path, kind, source and class counts."""
    meta = read_meta(path)
    return {
        "path": str(path),
        "kind": meta["kind"],
        "source": meta["source"],
        "class_counts": {split: info["class_counts"] for split, info in meta["splits"].items()},
    }


def recalls_of(counts: list[list[int]]) -> dict[str, float | None]:
    """Per class (by name), the recall: diagonal over row sum; ``None`` for an absent class."""
    return {
        name: (row[i] / sum(row) if sum(row) else None)
        for i, (name, row) in enumerate(zip(LABEL_NAMES, counts, strict=True))
    }


def selection_entry(settings: Settings, *, best_step: int | None, last_step: int) -> dict[str, Any]:
    """``metrics.json["selection"]``: which weights the run's results are of (plan row D3).

    ``checkpoint_selection``; ``scored_step``, the steps done when the scored weights were
    reached (the best validation's, or the last step's for ``final``); ``ema``, whether
    they are the weight EMA's; and ``ece_bins``, the bins of every ``ece`` entry.
    """
    final = settings.checkpoint_selection == "final"
    return {
        "checkpoint_selection": settings.checkpoint_selection,
        "scored_step": last_step if final else best_step,
        "ema": settings.ema_decay is not None,
        "ece_bins": ECE_BINS,
    }


def evaluation_entry(stage: StageOutput) -> dict[str, Any]:
    """Accuracy, balanced accuracy, macro-F1, per-class recall, both confusion matrices and
    the top-label ECE (15 bins; ``protocol.calibration``, a secondary metric)."""
    y_pred, y_true = stage.y_pred, stage.y_true
    counts = confusion_counts(y_pred, y_true).tolist()
    recalls = recalls_of(counts)
    present = [value for value in recalls.values() if value is not None]
    return {
        "accuracy": float((y_pred == y_true).double().mean()),
        "balanced_accuracy": sum(present) / len(present),
        "macro_f1": macro_f1(y_pred, y_true),
        "recall_per_class": recalls,
        "confusion_matrix": stage.confusion.tolist(),
        "confusion_counts": counts,
        "ece": expected_calibration_error(stage.probabilities.double().numpy(), y_true.numpy()),
    }


def write_predictions(
    stage: StageOutput,
    split: str,
    out: Path,
    *,
    name: str,
    index: npt.NDArray[np.int64] | None = None,
) -> dict[str, Any]:
    """Write the stage's per-stamp probabilities; return its ``metrics["predictions"]`` entry.

    The evaluation loaders are unshuffled, so the loader order is the split's file order
    and ``index`` is ``arange(N)`` by default; a split run passes its evaluation part's
    pool (or Validation) indices, in the same order. ``name`` is the file's stage name.

    Raises:
        DataError: if ``index`` does not have one entry per predicted stamp.
    """
    labels = stage.y_true.numpy().astype(np.int64)
    if index is None:
        index = np.arange(labels.size, dtype=np.int64)
    if index.shape != labels.shape:
        raise DataError(f"{index.size} indices for {labels.size} predicted stamps")
    predictions = Predictions(stage.probabilities.float().numpy(), labels, index)
    file = predictions_file(name)
    save_predictions(predictions, out / file)
    return {"file": file, "split": split, "n": int(labels.size)}


def plot_confusion_matrix(matrix: list[list[float]], title: str, path: Path) -> None:
    """Save ``matrix`` (row-normalised) as a labelled heat map at ``path`` (PNG)."""
    import matplotlib as mpl

    mpl.use("Agg")  # no display on servers; only files are written
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.5))
    image = ax.imshow([[100 * v for v in row] for row in matrix], vmin=0, vmax=100, cmap="Blues")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(
                j,
                i,
                f"{100 * value:.1f}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
                fontsize=8,
            )
    ax.set_xticks(range(len(LABEL_NAMES)), LABEL_NAMES, rotation=45)
    ax.set_yticks(range(len(LABEL_NAMES)), LABEL_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, fontsize=9)
    fig.colorbar(image, ax=ax, label="% of the true class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _split_results(
    command: str, val: StageOutput, test: StageOutput, out: Path, split_run: SplitRun
) -> dict[str, Any]:
    """A split run's results: the selection part (``val``) and the evaluation part (``test``).

    The ``Test`` role of a split run is its evaluation part (``protocol_splits``), so the
    real Test split is never predicted.
    """
    parts = split_run.parts()
    selection, evaluation = evaluation_entry(val), evaluation_entry(test)
    predictions = {
        "evaluation": write_predictions(
            test, parts.evaluation_name, out, index=parts.evaluation_index, name="evaluation"
        )
    }
    figures = []
    for part, result in (("selection", selection), ("evaluation", evaluation)):
        title = (
            f"{command}: {split_run.mode} {part}: balanced accuracy "
            f"{100 * result['balanced_accuracy']:.1f} %"
        )
        figure = f"confusion_matrix_{part}.png"
        plot_confusion_matrix(result["confusion_matrix"], title, out / figure)
        figures.append(figure)
    return {
        "selection_acc_best": selection["accuracy"],
        "evaluation_accuracy": evaluation["accuracy"],
        "evaluation_balanced_accuracy": evaluation["balanced_accuracy"],
        "labels": list(LABEL_NAMES),
        "evaluation": {"selection": selection, "evaluation": evaluation},
        "predictions": predictions,
        "figures": figures,
    }


def classifier_results(
    command: str,
    val: StageOutput,
    test: StageOutput,
    out: Path,
    split_run: SplitRun | None = None,
) -> dict[str, Any]:
    """The results of the evaluated best weights (module docstring); writes the files in ``out``.

    Raises:
        DataError: if a split run's evaluation index does not match its predictions.
    """
    if split_run is not None:
        return _split_results(command, val, test, out, split_run)
    val_entry, test_entry = evaluation_entry(val), evaluation_entry(test)
    predictions = {
        "val": write_predictions(val, "Validation", out, name="val"),
        "test": write_predictions(test, "Test", out, name="test"),
    }
    for stage, result in (("val", val_entry), ("test", test_entry)):
        title = f"{command}: {stage} accuracy {100 * result['accuracy']:.1f} %"
        plot_confusion_matrix(
            result["confusion_matrix"], title, out / f"confusion_matrix_{stage}.png"
        )
    return {
        "val_acc_best": val_entry["accuracy"],
        "test_acc_at_best": test_entry["accuracy"],
        "val_macro_f1": val_entry["macro_f1"],
        "test_macro_f1": test_entry["macro_f1"],
        "labels": list(LABEL_NAMES),
        "evaluation": {"val": val_entry, "test": test_entry},
        "predictions": predictions,
        "figures": ["confusion_matrix_val.png", "confusion_matrix_test.png"],
    }
