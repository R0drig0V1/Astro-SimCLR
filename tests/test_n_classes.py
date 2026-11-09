"""N classes from the dataset (task 3.1b): the phase-2 path unchanged, a 6-class run end to end.

1. **The 5-class path is byte-identical.** The digests below were computed with the code
   *before* ``n_classes`` existed (``main`` at 3b1191d): seeded initial weights of the phase-2
   networks (the paper task's ``PaperNet`` has the state-dict keys and draws of the former
   Lightning ``PaperClassifierModule``, so its digest is that module's) and the bytes of a
   predictions file. The golden fixtures (``tests/fixtures/``) check the forward passes.
2. **A 6-class classifier trains end to end on a synthetic ``raw-alerts`` layout**, on the
   CPU, for a few steps, through the plain engine's runner: layout -> object splits ->
   ``run_plain`` (``schedule: rj23``, whose task takes ``n_classes`` from the data) ->
   metrics, predictions and checkpoints with 6 logits. The paper task stays on CD21's five
   classes; its class-aware augmentations (C6, C7) refuse any other count.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from astro_simclr.config import Settings, settings_from_mapping
from astro_simclr.data.raw_alerts import label_names_of, load_raw_alerts, write_raw_alerts
from astro_simclr.data.schema import DataError
from astro_simclr.data.synthetic import synthetic_raw_alerts
from astro_simclr.engine import Task, Validation, load_best, load_checkpoint
from astro_simclr.models import N_CLASSES, N_FEATURES, StampClassifier, check_n_classes
from astro_simclr.paper_model import PaperStampClassifier
from astro_simclr.paper_variants import build_paper_model
from astro_simclr.presets import PRESETS
from astro_simclr.protocol.alerts import RJ23_LABEL_NAMES, alert_table_from_raw
from astro_simclr.protocol.object_splits import make_object_splits, write_object_splits
from astro_simclr.protocol.predictions import Predictions, load_predictions, save_predictions
from astro_simclr.tasks.paper import build_paper_task
from astro_simclr.tasks.plain import run_plain
from astro_simclr.tasks.rj23 import RJ23Request
from astro_simclr.training.paper_augment import check_augment_context
from astro_simclr.training.tensor_splits import TensorSplit

#: The CD21 optimizer settings of the phase-2 digests.
_OPTIM: dict[str, object] = {"beta_loss": 0.5, "lr": 1e-3, "optimizer": "Adam"}

# --- 1. The 5-class path ---------------------------------------------------------------------

#: SHA-256 of the seeded (torch.manual_seed(0)) initial state dicts, before n_classes existed.
_STATE_DIGESTS = {
    "stamp_classifier": "196696f616f1b8ee0f630e77cd804b4acf571e735e55c33e73788704b2423095",
    "paper": "9e5b0620d766b03ec60c5e631718acb1a6cf8787b2654bddeca122045ad39228",
    "paper_no_features": "acf7623045d1bcf3336e59abe69bd9c62fe8bf0bf25403f8eab8241df89ecad6",
    "p4m": "7b0f708f494c702c81b69d7632b1ab0a32b2597ca370b9b2540a5cacbad5f8b6",
    # Recorded on Lightning's PaperClassifierModule; the plain task's net is the same.
    "paper_module": "c8c2754cfc4f91bfe00b92103520634994cf18e579d41e0493214f8729d7705f",
}
_BUILDERS: dict[str, Callable[[], nn.Module]] = {
    "stamp_classifier": lambda: StampClassifier(0.5, True),
    "paper": lambda: PaperStampClassifier(),
    "paper_no_features": lambda: PaperStampClassifier(with_features=False),
    "p4m": lambda: build_paper_model("p4m"),
    "paper_module": lambda: build_paper_task(Settings().replace(**_OPTIM), {}).net,
}
#: SHA-256 of save_predictions on the 7x5 input of the test, before n_classes existed.
_PREDICTIONS_DIGEST = "10991e302cdb9fd74de32af01830ab74932aa9a624ad85e517e51674c3ade4fd"


def _last_linear(module: nn.Module) -> nn.Linear:
    """The last ``nn.Linear`` of ``module``: the layer that gives the logits."""
    return [m for m in module.modules() if isinstance(m, nn.Linear)][-1]


def _state_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in module.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize("name", sorted(_STATE_DIGESTS))
def test_phase2_initial_weights_are_unchanged(name: str) -> None:
    torch.manual_seed(0)
    assert _state_digest(_BUILDERS[name]()) == _STATE_DIGESTS[name]


def test_explicit_five_classes_is_the_default() -> None:
    torch.manual_seed(0)
    explicit = build_paper_model("p4m", n_classes=N_CLASSES)
    assert _state_digest(explicit) == _STATE_DIGESTS["p4m"]
    torch.manual_seed(0)
    assert _state_digest(PaperStampClassifier(n_classes=N_CLASSES)) == _STATE_DIGESTS["paper"]
    assert N_CLASSES == 5


def test_phase2_predictions_file_is_unchanged(tmp_path: Path) -> None:
    probabilities = np.random.default_rng(0).random((7, 5)).astype(np.float32)
    file = tmp_path / "p.npz"
    save_predictions(Predictions(probabilities, np.arange(7) % 5, np.arange(7)), file)
    assert hashlib.sha256(file.read_bytes()).hexdigest() == _PREDICTIONS_DIGEST
    assert load_predictions(file).n_classes == 5


# --- 2. A 6-class layout trains end to end --------------------------------------------------

#: The dataset name of the 6-class layout under its data directory.
_DATASET = "rj23-six"


@pytest.fixture(scope="module")
def six_class_layout(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """(data directory, object splits) of a synthetic 6-class layout (RJ23's classes)."""
    root = tmp_path_factory.mktemp("raw")
    cutouts, columns = synthetic_raw_alerts(RJ23_LABEL_NAMES, n_per_class=20, seed=0)
    meta = {"name": "synthetic"}
    path = write_raw_alerts(root / "data" / _DATASET, cutouts, columns, RJ23_LABEL_NAMES, meta)
    table = alert_table_from_raw(load_raw_alerts(path))
    splits = root / "splits.json"
    write_object_splits(make_object_splits(table, seed=0, name="synthetic"), splits)
    return root / "data", splits


def test_six_class_classifier_trains_end_to_end(
    six_class_layout: tuple[Path, Path], tmp_path: Path
) -> None:
    data_dir, splits = six_class_layout
    n_classes = len(label_names_of(data_dir / _DATASET))  # from meta.json, before the data
    assert n_classes == 6
    settings = settings_from_mapping(
        {
            **PRESETS["rj23-lowres"],
            "dataset": _DATASET,
            "data_dir": data_dir,
            "output_dir": tmp_path,
            "accelerator": "cpu",
            "val_every_n_steps": 2,
            "max_steps": 4,
            "patience": 50,
            "batch_size": 12,
            "eval_batch_size": 8,
            "checkpoint_every_n_steps": 2,
        }
    )
    initial: dict[str, torch.Tensor] = {}

    def at_validation(record: Validation, task: Task) -> None:
        if record.steps_done == 0:  # the rj23 schedule validates the initial weights
            initial.update({k: v.detach().clone() for k, v in task.module.state_dict().items()})

    metrics = run_plain(
        "train-supervised", settings, rj23=RJ23Request(splits), on_validation=at_validation
    )
    assert metrics["last_step"] == 4
    assert metrics["labels"] == list(RJ23_LABEL_NAMES)
    for part in metrics["benchmark"]["parts"].values():
        assert len(part["class_counts"]) == 6
    test = metrics["evaluation"]["test"]
    assert np.asarray(test["confusion_counts"]).shape == (6, 6)
    assert set(test["recall_per_class"]) == set(RJ23_LABEL_NAMES)

    # The predictions files hold 6 probabilities per alert, and need n_classes to load.
    file = tmp_path / metrics["predictions"]["test"]["file"]
    predictions = load_predictions(file, n_classes=6)
    assert predictions.probabilities.shape == (test["n"], 6)
    np.testing.assert_allclose(predictions.probabilities.sum(axis=1), 1.0, rtol=1e-5)
    with pytest.raises(DataError, match=r"probabilities \(N, 5\)"):
        load_predictions(file)

    # Both checkpoints hold the 6-logit head, and the weights trained.
    best = load_best(tmp_path / "ckpt" / "best.pt")["model"]
    last = load_checkpoint(tmp_path / "ckpt" / "last.pt")["model"]
    heads = [key for key, value in last.items() if value.ndim == 2]
    assert last[heads[-1]].shape[0] == best[heads[-1]].shape[0] == 6
    assert initial.keys() == last.keys()
    assert any(not torch.equal(initial[key], value) for key, value in last.items())


@pytest.mark.parametrize(
    "build",
    [
        lambda n: StampClassifier(0.5, True, n),
        lambda n: build_paper_model("rot4", n_classes=n),
        lambda n: build_paper_model("p4", n_classes=n),
    ],
)
def test_every_classifier_takes_n_classes(build: Callable[[int], nn.Module]) -> None:
    assert _last_linear(build(6)).out_features == 6
    with pytest.raises(ValueError, match="n_classes must be an integer >= 2"):
        build(1)


def test_check_n_classes() -> None:
    assert check_n_classes(6) == 6
    for bad in (1, 0, True, 2.0):
        with pytest.raises(ValueError, match="n_classes"):
            check_n_classes(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize(("synthetic_bogus", "flux_rescale"), [(0.1, 0.0), (0.0, 0.5)])
def test_c6_and_c7_need_the_cd21_classes(synthetic_bogus: float, flux_rescale: float) -> None:
    check_augment_context(synthetic_bogus, flux_rescale, "none", N_CLASSES)
    with pytest.raises(ValueError, match="need the 5 CD21 classes"):
        check_augment_context(synthetic_bogus, flux_rescale, "none", 6)


def test_tensor_split_checks_labels_against_n_classes() -> None:
    images = np.zeros((2, 3, 21, 21), dtype=np.float32)
    features = np.zeros((2, N_FEATURES), dtype=np.float32)
    labels = np.array([0, 5], dtype=np.int64)
    with pytest.raises(ValueError, match="labels must be in range\\(5\\)"):
        TensorSplit(images, features, labels)
    assert TensorSplit(images, features, labels, n_classes=6).n_classes == 6
    with pytest.raises(ValueError, match="n_classes"):
        TensorSplit(images, features, labels, n_classes=1)
