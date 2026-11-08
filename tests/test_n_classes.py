"""N classes from the dataset (task 3.1b): the phase-2 path unchanged, a 6-class run end to end.

1. **The 5-class path is byte-identical.** The digests below were computed with the code
   *before* ``n_classes`` existed (``main`` at 3b1191d): seeded initial weights of the phase-2
   networks, the hyperparameter names of the modules (so ``hparams.yaml`` and checkpoints
   are unchanged) and the bytes of a predictions file. The golden fixtures
   (``tests/fixtures/``) check the forward passes.
2. **A 6-class classifier trains end to end on a synthetic ``raw-alerts`` layout**, on the
   CPU, for a few steps: layout -> loader -> crops -> ``TensorSplit`` -> the device data
   module -> ``PaperClassifierModule(n_classes=6)`` -> metrics, predictions, checkpoint.
   The crops use the paper's float input (``crop_then_normalise``) with inf treated as
   NaN: a stand-in, since RJ23's input path is task 3.3.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

import lightning as L
import numpy as np
import numpy.typing as npt
import pytest
import torch
from torch import nn

from astro_simclr.data.float_input import crop_then_normalise
from astro_simclr.data.raw_alerts import (
    RawAlerts,
    label_names_of,
    load_raw_alerts,
    write_raw_alerts,
)
from astro_simclr.data.schema import DataError
from astro_simclr.data.synthetic import synthetic_raw_alerts
from astro_simclr.models import N_CLASSES, N_FEATURES, StampClassifier, check_n_classes
from astro_simclr.paper_model import PaperStampClassifier
from astro_simclr.paper_variants import build_paper_model
from astro_simclr.protocol.alerts import RJ23_LABEL_NAMES
from astro_simclr.protocol.predictions import (
    Predictions,
    load_predictions,
    save_predictions,
)
from astro_simclr.training.device_data import DeviceTensorDataModule, TensorSplit
from astro_simclr.training.modules import (
    FineTuneModule,
    LinearEvalModule,
    PaperClassifierModule,
    SimCLRModule,
    SupervisedCEModule,
    SupervisedCEResNetModule,
)

RJ23_NAMES = RJ23_LABEL_NAMES


class _Optim(TypedDict):
    beta_loss: float
    lr: float
    optimizer: str


_OPTIM: _Optim = {"beta_loss": 0.5, "lr": 1e-3, "optimizer": "Adam"}

# --- 1. The 5-class path ---------------------------------------------------------------------

#: SHA-256 of the seeded (torch.manual_seed(0)) initial state dicts, before n_classes existed.
_STATE_DIGESTS = {
    "stamp_classifier": "196696f616f1b8ee0f630e77cd804b4acf571e735e55c33e73788704b2423095",
    "paper": "9e5b0620d766b03ec60c5e631718acb1a6cf8787b2654bddeca122045ad39228",
    "paper_no_features": "acf7623045d1bcf3336e59abe69bd9c62fe8bf0bf25403f8eab8241df89ecad6",
    "p4m": "7b0f708f494c702c81b69d7632b1ab0a32b2597ca370b9b2540a5cacbad5f8b6",
    "paper_module": "c8c2754cfc4f91bfe00b92103520634994cf18e579d41e0493214f8729d7705f",
}
_BUILDERS: dict[str, Callable[[], nn.Module]] = {
    "stamp_classifier": lambda: StampClassifier(0.5, True),
    "paper": lambda: PaperStampClassifier(),
    "paper_no_features": lambda: PaperStampClassifier(with_features=False),
    "p4m": lambda: build_paper_model("p4m"),
    "paper_module": lambda: PaperClassifierModule(**_OPTIM),
}
#: The hyperparameter names before n_classes existed.
_PAPER_HPARAMS = [
    "adam_betas",
    "aug_feature_jitter",
    "aug_min_r2",
    "aug_noise",
    "augment_seed",
    "beta_loss",
    "context",
    "drop_rate",
    "dropout_active",
    "flux_k_max",
    "flux_k_min",
    "flux_rescale",
    "group_fields",
    "lr",
    "optimizer",
    "paper_arch",
    "rot90_augment",
    "synthetic_bogus",
    "with_features",
]
_SUPERVISED_HPARAMS = ["beta_loss", "drop_rate", "lr", "optimizer", "with_features"]
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
    explicit = PaperClassifierModule(**_OPTIM, n_classes=N_CLASSES)
    assert _state_digest(explicit) == _STATE_DIGESTS["paper_module"]
    assert N_CLASSES == 5


def test_phase2_hyperparameters_are_unchanged() -> None:
    assert sorted(PaperClassifierModule(**_OPTIM).hparams) == _PAPER_HPARAMS
    assert sorted(PaperClassifierModule(**_OPTIM, n_classes=5).hparams) == _PAPER_HPARAMS
    supervised = SupervisedCEModule(drop_rate=0.5, **_OPTIM)
    assert sorted(supervised.hparams) == _SUPERVISED_HPARAMS
    # An N-class module records its count, so that it reloads with it.
    assert PaperClassifierModule(**_OPTIM, n_classes=6).hparams["n_classes"] == 6


def test_phase2_predictions_file_is_unchanged(tmp_path: Path) -> None:
    probabilities = np.random.default_rng(0).random((7, 5)).astype(np.float32)
    file = tmp_path / "p.npz"
    save_predictions(Predictions(probabilities, np.arange(7) % 5, np.arange(7)), file)
    assert hashlib.sha256(file.read_bytes()).hexdigest() == _PREDICTIONS_DIGEST
    assert load_predictions(file).n_classes == 5


# --- 2. A 6-class layout trains end to end --------------------------------------------------


def _crops(cutouts: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Raw cutouts -> (N, 3, 21, 21) paper crops; inf counts as NaN (stand-in for 3.3)."""
    finite = np.where(np.isfinite(cutouts), cutouts, np.float32(np.nan))
    crops = crop_then_normalise(finite, nan_approximation=False)
    return np.ascontiguousarray(crops.transpose(0, 3, 1, 2))


def _tensor_split(data: RawAlerts, rows: npt.NDArray[np.int64]) -> TensorSplit:
    return TensorSplit(
        images=_crops(np.asarray(data.cutouts[rows])),
        features=np.zeros((len(rows), N_FEATURES), dtype=np.float32),
        labels=data.labels[rows],
        n_classes=data.n_classes,
    )


@pytest.fixture(scope="module")
def six_class_layout(tmp_path_factory: pytest.TempPathFactory) -> Path:
    cutouts, columns = synthetic_raw_alerts(RJ23_NAMES, n_per_class=9, seed=0)
    dest = tmp_path_factory.mktemp("raw") / "rj23"
    return write_raw_alerts(dest, cutouts, columns, RJ23_NAMES, {"name": "synthetic"})


def test_six_class_classifier_trains_end_to_end(six_class_layout: Path, tmp_path: Path) -> None:
    n_classes = len(label_names_of(six_class_layout))  # from meta.json, before the data
    data = load_raw_alerts(six_class_layout)
    # Stratified thirds: the k-th row of each class goes to part k % 3 (3 per class each).
    labels = data.labels
    part = np.empty(len(data), dtype=np.int64)
    for label in range(n_classes):
        rows_of_class = np.flatnonzero(labels == label)
        part[rows_of_class] = np.arange(len(rows_of_class)) % 3
    splits = {
        name: _tensor_split(data, np.flatnonzero(part == k))
        for k, name in enumerate(("Train", "Validation", "Test"))
    }
    torch.manual_seed(0)
    module = PaperClassifierModule(**_OPTIM, with_features=False, n_classes=n_classes)
    assert _last_linear(module).out_features == 6
    before = {k: v.clone() for k, v in module.state_dict().items()}
    datamodule = DeviceTensorDataModule(splits, batch_size=8, eval_batch_size=8, seed=0)
    trainer = L.Trainer(
        accelerator="cpu",
        max_steps=4,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, datamodule=datamodule)
    assert trainer.global_step == 4
    assert any(not torch.equal(before[k], v) for k, v in module.state_dict().items())

    trainer.test(module, datamodule=datamodule, verbose=False)
    n_test = len(splits["Test"].labels)
    probabilities = module.probabilities["test"]
    assert probabilities.shape == (n_test, 6)
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(n_test))
    assert module.confusion_matrices["test"].shape == (6, 6)
    assert set(module.predictions["test"][1].tolist()) == set(range(6))

    predictions = Predictions(
        probabilities.numpy(), splits["Test"].labels, np.arange(n_test, dtype=np.int64), 6
    )
    file = tmp_path / "predictions_test.npz"
    save_predictions(predictions, file)
    assert load_predictions(file, n_classes=6).n_classes == 6
    with pytest.raises(DataError, match=r"probabilities \(N, 5\)"):
        load_predictions(file)

    checkpoint = tmp_path / "six.ckpt"
    trainer.save_checkpoint(checkpoint)
    reloaded = PaperClassifierModule.load_from_checkpoint(checkpoint, weights_only=False)
    assert reloaded.n_classes == 6 and _last_linear(reloaded).out_features == 6


@pytest.mark.parametrize(
    "build",
    [
        lambda n: StampClassifier(0.5, True, n),
        lambda n: build_paper_model("rot4", n_classes=n),
        lambda n: build_paper_model("p4", n_classes=n),
        lambda n: SupervisedCEModule(drop_rate=0.5, **_OPTIM, n_classes=n),
        lambda n: SupervisedCEResNetModule(**_OPTIM, n_classes=n),
    ],
)
def test_every_classifier_takes_n_classes(build: Callable[[int], nn.Module]) -> None:
    assert _last_linear(build(6)).out_features == 6
    with pytest.raises(ValueError, match="n_classes must be an integer >= 2"):
        build(1)


def test_downstream_modules_take_n_classes() -> None:
    pretrained = SimCLRModule(
        encoder_name="stamp1", image_size=27, projection_dim=8, temperature=0.5, lr_encoder=1e-3
    )
    linear = LinearEvalModule(pretrained, **_OPTIM, with_features=False, n_classes=6)
    assert linear.classifier.model.out_features == 6
    assert linear.hparams["n_classes"] == 6
    assert "n_classes" not in LinearEvalModule(pretrained, **_OPTIM, with_features=False).hparams
    tuned = FineTuneModule(pretrained, drop_rate=0.5, **_OPTIM, n_classes=6)
    assert _last_linear(tuned).out_features == 6


def test_check_n_classes() -> None:
    assert check_n_classes(6) == 6
    for bad in (1, 0, True, 2.0):
        with pytest.raises(ValueError, match="n_classes"):
            check_n_classes(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("augment", [{"synthetic_bogus": 0.1}, {"flux_rescale": 0.5}])
def test_c6_and_c7_need_the_cd21_classes(augment: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="need the 5 CD21 classes"):
        PaperClassifierModule(**_OPTIM, **augment, n_classes=6)  # type: ignore[arg-type]


def test_tensor_split_checks_labels_against_n_classes() -> None:
    images = np.zeros((2, 3, 21, 21), dtype=np.float32)
    features = np.zeros((2, N_FEATURES), dtype=np.float32)
    labels = np.array([0, 5], dtype=np.int64)
    with pytest.raises(ValueError, match="labels must be in range\\(5\\)"):
        TensorSplit(images, features, labels)
    assert TensorSplit(images, features, labels, n_classes=6).n_classes == 6
    with pytest.raises(ValueError, match="n_classes"):
        TensorSplit(images, features, labels, n_classes=1)


def test_data_module_refuses_splits_that_disagree() -> None:
    images = np.zeros((8, 3, 21, 21), dtype=np.float32)
    features = np.zeros((8, N_FEATURES), dtype=np.float32)
    labels = np.arange(8, dtype=np.int64) % 5
    five = TensorSplit(images, features, labels)
    six = TensorSplit(images, features, labels, n_classes=6)
    datamodule = DeviceTensorDataModule({"Train": five, "Validation": six, "Test": five})
    with pytest.raises(DataError, match="disagree on n_classes"):
        datamodule.setup("fit")
