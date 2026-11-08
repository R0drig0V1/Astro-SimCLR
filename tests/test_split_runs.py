"""``train-supervised --splits FILE --fold K | --screen`` on synthetic prepared data (2.5-infra).

The data: the synthetic 100 % file with 25 stamps per split (5 per class) and float32
stamps, with Train stamp 3 a copy of Validation stamp 7 (the leaked stamp, excluded by
the splits). Its splits file is made with ``make_splits`` (1 stamp per class per screen half
and per selection set, so that the small classes fit 5 folds). Runs use the ``carrasco2021``
schedule with a few steps of batch 4 on the CPU (MPS/CUDA where available for the fold run).
"""

from __future__ import annotations

import dataclasses
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pytest
from _training_support import ACCELERATORS
from lightning.pytorch.callbacks import Callback

from astro_simclr import experiments
from astro_simclr.cli import main
from astro_simclr.config import ConfigError, Settings
from astro_simclr.data import prepare
from astro_simclr.data.schema import LABEL_NAMES
from astro_simclr.data.synthetic import CLASSIFIER_FILES, make_synthetic
from astro_simclr.models import N_FEATURES
from astro_simclr.paper_model import PAPER_CROP
from astro_simclr.protocol.predictions import load_predictions
from astro_simclr.protocol.split_runs import SplitRequest, SplitRun, file_sha256, open_split_run
from astro_simclr.protocol.splits import (
    CVFold,
    ProtocolSplits,
    SplitsError,
    make_splits,
    read_splits,
    write_splits,
)
from astro_simclr.protocol.stats import balanced_accuracy
from astro_simclr.training.class_augment import stamp_fits
from astro_simclr.training.device_data import (
    DeviceTensorDataModule,
    TensorSplit,
    float_crops,
    protocol_splits,
)
from astro_simclr.training.flux_rescale import fit_central_psf

N_PER_SPLIT = 25
LEAKED_TRAIN, LEAKED_VALIDATION = 3, 7

_SCHEDULE: dict[str, Any] = {
    "schedule": "carrasco2021",
    "supervised_model": "paper_stamp_classifier",
    "image_size": 21,
    "eval_batch_size": 64,
    "early_stopping": False,
    "monitor": "loss_val",
    "optimizer": "Adam",
    "lr": 1e-3,
    "beta_loss": 0.5,
    "batch_size": 4,
    "max_steps": 4,
    "val_every_n_steps": 2,
    "horizon_increment": 0,
    "checkpoint_every_n_steps": 2,
}


@pytest.fixture(scope="module")
def cv_data(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, ProtocolSplits]:
    """(prepared dir, splits file, splits) of the module docstring's data."""
    raw = tmp_path_factory.mktemp("cv_raw")
    out = tmp_path_factory.mktemp("cv_prepared")
    make_synthetic(raw, n_per_split=N_PER_SPLIT, seed=1)
    source = raw / CLASSIFIER_FILES[0]
    # Our own synthetic file, written just above: unpickling it is safe.
    content = pickle.loads(source.read_bytes())  # noqa: S301
    for key in ("images", "features"):
        content["Train"][key][LEAKED_TRAIN] = content["Validation"][key][LEAKED_VALIDATION]
    source.write_bytes(pickle.dumps(content))
    prepared = prepare(source, out, images="float32")
    splits = make_splits(prepared, selection_per_class=1, screen_per_class=1)
    path = tmp_path_factory.mktemp("cv_splits") / "splits.json"
    write_splits(splits, path)
    return prepared, path, read_splits(path)


def _settings(prepared: Path, out: Path, **changes: Any) -> Settings:
    base = Settings(
        accelerator="cpu",
        deterministic=True,
        data_dir=prepared.parent,
        dataset=prepared.name,
        output_dir=out,
        **_SCHEDULE,
    )
    return base.replace(**changes)


def _pool_features(prepared: Path) -> np.ndarray:
    """The pool's features, in pool order; the leaked Train stamp's row never matches."""
    crops = float_crops(prepared, names=("Train", "Validation"))
    table = np.concatenate([crops["Train"].features, crops["Validation"].features])
    table[LEAKED_TRAIN] = np.nan  # it duplicates Validation 7, and is in no part
    return table


def _rows(features: np.ndarray, table: np.ndarray) -> np.ndarray:
    """For each row of ``features``, its row in ``table`` (every stamp is unique)."""
    matches = (features[:, None, :] == table[None, :, :]).all(axis=2)
    assert (matches.sum(axis=1) == 1).all()
    return np.argmax(matches, axis=1)


class _Spy(Callback):
    """Record the features of the data module's three roles at the start of training."""

    def __init__(self) -> None:
        self.features: dict[str, np.ndarray] = {}

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        datamodule = getattr(trainer, "datamodule", None)  # untyped in Lightning
        assert isinstance(datamodule, DeviceTensorDataModule)
        for name in ("Train", "Validation", "Test"):
            split = datamodule._split(name)
            self.features[name] = split.features.cpu().numpy()


def test_parts_are_disjoint_and_cover_the_pool(cv_data: tuple[Path, Path, ProtocolSplits]) -> None:
    _, path, splits = cv_data
    n_pool = splits.n_train + splits.n_validation
    assert splits.excluded_train == (LEAKED_TRAIN,)
    for fold in splits.folds:
        parts = SplitRun(path, "x", splits, fold.fold).parts()
        everything = np.concatenate([parts.training, parts.selection, parts.evaluation])
        assert np.unique(everything).size == everything.size  # disjoint
        assert sorted([*everything.tolist(), LEAKED_TRAIN]) == list(range(n_pool))
        assert np.array_equal(parts.evaluation_index, fold.evaluation)
    screen = SplitRun(path, "x", splits, None).parts()
    assert screen.training.tolist() == [i for i in range(splits.n_train) if i != LEAKED_TRAIN]
    assert np.array_equal(screen.selection, splits.n_train + splits.screen.selection)
    assert np.array_equal(screen.evaluation_index, splits.screen.decision)
    # A malformed splits object (a stamp in two parts) is refused.
    fold0 = splits.folds[0]
    leaky = CVFold(0, fold0.evaluation, fold0.selection, fold0.evaluation)
    broken = dataclasses.replace(splits, folds=(leaky, *splits.folds[1:]))
    with pytest.raises(ValueError, match="training and evaluation parts share stamps"):
        SplitRun(path, "x", broken, 0).parts()


def test_protocol_splits_gathers_by_pool_index(cv_data: tuple[Path, Path, ProtocolSplits]) -> None:
    _, path, splits = cv_data
    n_train, n_val = splits.n_train, splits.n_validation

    def split(start: int, n: int) -> TensorSplit:
        features = np.zeros((n, N_FEATURES), dtype=np.float32)
        features[:, 0] = np.arange(start, start + n)  # the pool index
        return TensorSplit(
            images=np.zeros((n, 3, PAPER_CROP, PAPER_CROP), dtype=np.float32),
            features=features,
            labels=np.zeros(n, dtype=np.int64),
        )

    pool = {"Train": split(0, n_train), "Validation": split(n_train, n_val)}
    for fold in (0, 3, None):
        parts = SplitRun(path, "x", splits, fold).parts()
        roles = protocol_splits(pool, parts, n_train)
        for role, part in (
            ("Train", parts.training),
            ("Validation", parts.selection),
            ("Test", parts.evaluation),
        ):
            assert roles[role].features[:, 0].astype(np.int64).tolist() == part.tolist()
    with pytest.raises(ValueError, match="Train stamps"):
        protocol_splits(pool, parts, n_train + 1)


@pytest.mark.parametrize("accelerator", ACCELERATORS)
def test_fold_run_trains_selects_and_predicts_its_parts(
    cv_data: tuple[Path, Path, ProtocolSplits],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accelerator: str,
) -> None:
    prepared, path, splits = cv_data
    fold = splits.folds[2]
    loaded: list[tuple[str, ...]] = []

    def recording(
        path: Path, *, names: tuple[str, ...] = ("Train", "Validation", "Test"), **kw: Any
    ) -> dict[str, TensorSplit]:
        loaded.append(tuple(names))
        return float_crops(path, names=names, **kw)

    monkeypatch.setattr(experiments, "float_crops", recording)
    spy = _Spy()
    settings = _settings(prepared, tmp_path, accelerator=accelerator)
    metrics = experiments.run_command(
        "train-supervised", settings, split=SplitRequest(path, 2), callbacks=[spy]
    )
    # Test is never loaded.
    assert loaded == [("Train", "Validation")]
    # Isolation: the stamps trained on are exactly the training part, the selection and
    # evaluation roles exactly their parts, and no evaluation stamp is trained on.
    table = _pool_features(prepared)
    assert _rows(spy.features["Train"], table).tolist() == fold.training.tolist()
    assert _rows(spy.features["Validation"], table).tolist() == fold.selection.tolist()
    assert _rows(spy.features["Test"], table).tolist() == fold.evaluation.tolist()
    assert not set(fold.evaluation.tolist()) & set(_rows(spy.features["Train"], table).tolist())

    protocol = metrics["protocol"]
    assert protocol["mode"] == "fold" and protocol["fold"] == 2
    assert protocol["splits_sha256"] == file_sha256(path)
    assert protocol["n_training"] == fold.training.size
    assert protocol["n_evaluation"] == fold.evaluation.size
    assert metrics["paper_data"]["n_train"] == fold.training.size
    assert metrics["paper_data"]["excluded_train"] == [LEAKED_TRAIN]
    assert metrics["preset"]["name"] == "carrasco2021"
    for key in ("val_acc_best", "test_acc_at_best"):
        assert key not in metrics
    assert set(metrics["predictions"]) == {"evaluation"}
    predictions = load_predictions(tmp_path / metrics["predictions"]["evaluation"]["file"])
    assert np.array_equal(predictions.index, fold.evaluation)  # keyed by pool index
    labels = np.concatenate([np.load(prepared / s / "labels.npy") for s in ("Train", "Validation")])
    assert np.array_equal(predictions.labels, labels[fold.evaluation])
    assert metrics["evaluation_balanced_accuracy"] == pytest.approx(
        balanced_accuracy(predictions.correct, predictions.labels)
    )
    assert not (tmp_path / "predictions_test.npz").exists()
    assert (tmp_path / "confusion_matrix_evaluation.png").is_file()
    on_disk = json.loads((tmp_path / "metrics.json").read_text())
    assert on_disk["protocol"]["splits_sha256"] == file_sha256(path)


def test_screen_run_through_the_cli(
    cv_data: tuple[Path, Path, ProtocolSplits], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prepared, path, splits = cv_data
    sets = [f"{key}={value}" for key, value in _SCHEDULE.items()]
    argv = [
        "train-supervised",
        "--preset",
        "carrasco2021",
        "--data-dir",
        str(prepared.parent),
        "--output-dir",
        str(tmp_path),
        "--accelerator",
        "cpu",
        "--splits",
        str(path),
        "--screen",
        "--set",
        f"dataset={prepared.name}",
        *sets,
    ]
    assert main(argv) == 0
    assert "evaluation_balanced_accuracy" in capsys.readouterr().out
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["protocol"]["mode"] == "screen" and metrics["protocol"]["fold"] is None
    assert metrics["protocol"]["n_training"] == splits.n_train - 1
    assert metrics["protocol"]["n_selection"] == splits.screen.selection.size
    predictions = load_predictions(tmp_path / "predictions_evaluation.npz")
    assert np.array_equal(predictions.index, splits.screen.decision)  # Validation indices


def _edited(path: Path, dest: Path, edit: Any) -> Path:
    raw = json.loads(path.read_text())
    edit(raw)
    dest.write_text(json.dumps(raw))
    return dest


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(lambda raw: raw["dataset"].update(source_sha256="f" * 64), id="source"),
        pytest.param(
            lambda raw: raw["dataset"]["labels_sha256"].update(Train="0" * 64), id="labels"
        ),
    ],
)
def test_wrong_fingerprint_is_refused(
    cv_data: tuple[Path, Path, ProtocolSplits],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    edit: Any,
) -> None:
    prepared, path, _ = cv_data
    other = _edited(path, tmp_path / "other.json", edit)
    with pytest.raises(SplitsError, match="regenerate them with `astro-simclr make-splits`"):
        open_split_run(SplitRequest(other, 0), prepared)
    out = tmp_path / "run"
    argv = [
        "train-supervised",
        "--preset",
        "carrasco2021",
        "--data-dir",
        str(prepared.parent),
        "--output-dir",
        str(out),
        "--splits",
        str(other),
        "--fold",
        "0",
        "--set",
        f"dataset={prepared.name}",
    ]
    assert main(argv) == 1  # a data error, before anything is built or trained
    assert "make-splits" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--fold", "0"], "--fold and --screen need --splits FILE"),
        (["--screen"], "--fold and --screen need --splits FILE"),
        (["--splits", "SPLITS"], "--splits needs --fold K or --screen"),
        (["--splits", "SPLITS", "--fold", "5"], "has folds 0-4"),
        (["--splits", "SPLITS", "--fold", "0", "--set", "schedule=epochs"], "carrasco2021"),
        (
            ["--splits", "SPLITS", "--fold", "0", "--set", "exclude_leaked_train=false"],
            "exclude_leaked_train must be true",
        ),
    ],
)
def test_split_flag_errors(
    cv_data: tuple[Path, Path, ProtocolSplits],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra: list[str],
    message: str,
) -> None:
    prepared, path, _ = cv_data
    argv = [
        "train-supervised",
        "--preset",
        "carrasco2021",
        "--data-dir",
        str(prepared.parent),
        "--output-dir",
        str(tmp_path / "run"),
        *[str(path) if item == "SPLITS" else item for item in extra],
    ]
    if "--set" not in extra:
        argv += ["--set"]
    argv += [f"dataset={prepared.name}"]
    assert main(argv) == 2
    assert message in capsys.readouterr().err


def test_fold_and_screen_are_exclusive(cv_data: tuple[Path, Path, ProtocolSplits]) -> None:
    _, path, _ = cv_data
    with pytest.raises(SystemExit):
        main(["train-supervised", "--splits", str(path), "--fold", "0", "--screen"])


def test_epochs_schedule_is_refused_in_run_command(
    cv_data: tuple[Path, Path, ProtocolSplits], tmp_path: Path
) -> None:
    prepared, path, _ = cv_data
    settings = Settings(data_dir=prepared.parent, dataset=prepared.name, output_dir=tmp_path)
    with pytest.raises(ConfigError, match="--splits trains only train-supervised"):
        experiments.run_command("pretrain", settings, split=SplitRequest(path, 0))


def test_split_request_and_excluded_stamps(cv_data: tuple[Path, Path, ProtocolSplits]) -> None:
    _, path, splits = cv_data
    assert SplitRequest(path, None).mode == "screen"
    assert SplitRequest(path, 1).mode == "fold"
    # A selection set holding the excluded Train stamp is refused (malformed object).
    fold0 = splits.folds[0]
    selection = np.sort(np.append(fold0.selection, LEAKED_TRAIN))
    leaky = CVFold(0, fold0.evaluation, selection, fold0.training)
    broken = dataclasses.replace(splits, folds=(leaky, *splits.folds[1:]))
    with pytest.raises(ValueError, match="selection part holds an excluded Train stamp"):
        SplitRun(path, "x", broken, 0).parts()


def test_uint8_diagnostic_split_run(
    cv_data: tuple[Path, Path, ProtocolSplits], tmp_path: Path
) -> None:
    prepared, path, splits = cv_data
    settings = _settings(prepared, tmp_path, paper_input="uint8")
    metrics = experiments.run_command("train-supervised", settings, split=SplitRequest(path, 4))
    assert metrics["paper_data"]["input"] == "uint8"
    assert metrics["protocol"]["fold"] == 4
    predictions = load_predictions(tmp_path / "predictions_evaluation.npz")
    assert np.array_equal(predictions.index, splits.folds[4].evaluation)


def test_screen_run_roles(cv_data: tuple[Path, Path, ProtocolSplits], tmp_path: Path) -> None:
    prepared, path, splits = cv_data
    spy = _Spy()
    experiments.run_command(
        "train-supervised",
        _settings(prepared, tmp_path),
        split=SplitRequest(path, None),
        callbacks=[spy],
    )
    table = _pool_features(prepared)
    trained = _rows(spy.features["Train"], table).tolist()
    assert trained == [i for i in range(splits.n_train) if i != LEAKED_TRAIN]
    selection = _rows(spy.features["Validation"], table)
    assert selection.tolist() == (splits.n_train + splits.screen.selection).tolist()
    decision = _rows(spy.features["Test"], table)
    assert decision.tolist() == (splits.n_train + splits.screen.decision).tolist()


#: C5, C6 (ADR 0006) and C7 (ADR 0007) on a p4 model; R^2 0 so that the random synthetic
#: stamps qualify.
_AUGMENT: dict[str, Any] = {
    "paper_arch": "p4",
    "aug_noise": 1.0,
    "aug_feature_jitter": 0.1,
    "synthetic_bogus": 0.25,
    "aug_min_r2": 0.0,
    "flux_rescale": 0.5,
}


class _CopyLast(Callback):
    """At step ``step``, copy ``last.ckpt`` (written at an earlier step) to ``dst``."""

    def __init__(self, step: int, src: Path, dst: Path) -> None:
        self.step, self.src, self.dst = step, src, dst

    def on_train_batch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule, *args: Any
    ) -> None:
        if trainer.global_step == self.step:
            self.dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.src, self.dst)


class _Steps(Callback):
    """Count the training batches that start."""

    def __init__(self) -> None:
        self.batches = 0

    def on_train_batch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule, *args: Any
    ) -> None:
        self.batches += 1


@pytest.fixture(scope="module")
def interrupted(
    cv_data: tuple[Path, Path, ProtocolSplits], tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, dict[str, Any], Path]:
    """Fold 2 run for 6 steps, with a copy of its ``last.ckpt`` from step 2 (mid-run).

    Returns (the mid-run checkpoint, the uninterrupted run's metrics, its output dir).
    """
    prepared, path, _ = cv_data
    root = tmp_path_factory.mktemp("interrupted")
    whole, mid = root / "whole", root / "mid.ckpt"
    copy = _CopyLast(3, whole / "checkpoints" / "last.ckpt", mid)
    metrics = experiments.run_command(
        "train-supervised",
        _settings(prepared, whole, max_steps=6),
        split=SplitRequest(path, 2),
        callbacks=[copy],
        progress_bar=False,
    )
    return mid, metrics, whole


@pytest.mark.parametrize(
    ("fold", "screen"),
    [pytest.param(3, False, id="other-fold"), pytest.param(None, True, id="screen")],
)
def test_resume_from_another_split_run_is_refused(
    cv_data: tuple[Path, Path, ProtocolSplits],
    interrupted: tuple[Path, dict[str, Any], Path],
    tmp_path: Path,
    fold: int | None,
    screen: bool,
) -> None:
    prepared, path, _ = cv_data
    mid, _, _ = interrupted
    steps = _Steps()
    expected = "the screen" if screen else "fold 3"
    with pytest.raises(ConfigError, match=f"from fold 2 of .* but this run is {expected}"):
        experiments.run_command(
            "train-supervised",
            _settings(prepared, tmp_path, max_steps=6),
            resume=str(mid),
            split=SplitRequest(path, fold),
            callbacks=[steps],
        )
    assert steps.batches == 0  # refused before any step
    assert not (tmp_path / "metrics.json").exists()


def test_resume_across_split_and_normal_runs_is_refused(
    cv_data: tuple[Path, Path, ProtocolSplits],
    interrupted: tuple[Path, dict[str, Any], Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared, path, _ = cv_data
    mid, _, _ = interrupted
    steps = _Steps()
    # A split run's checkpoint resumed as a normal run ...
    with pytest.raises(ConfigError, match=r"from fold 2 of .* but this run is a normal run"):
        experiments.run_command(
            "train-supervised",
            _settings(prepared, tmp_path / "normal", max_steps=6),
            resume=str(mid),
            callbacks=[steps],
        )
    assert steps.batches == 0
    # ... and a normal run's checkpoint resumed as a split run (through the CLI: exit 2).
    normal = tmp_path / "plain"
    experiments.run_command(
        "train-supervised", _settings(prepared, normal, max_steps=2), progress_bar=False
    )
    argv = [
        "train-supervised",
        "--preset",
        "carrasco2021",
        "--data-dir",
        str(prepared.parent),
        "--output-dir",
        str(tmp_path / "split"),
        "--accelerator",
        "cpu",
        "--splits",
        str(path),
        "--fold",
        "2",
        "--resume",
        str(normal / "checkpoints" / "last.ckpt"),
        "--set",
        f"dataset={prepared.name}",
        *[f"{key}={value}" for key, value in _SCHEDULE.items()],
    ]
    assert main(argv) == 2
    assert "from a normal run (no --splits), but this run is fold 2" in capsys.readouterr().err


def test_resume_of_the_same_fold_continues(
    cv_data: tuple[Path, Path, ProtocolSplits],
    interrupted: tuple[Path, dict[str, Any], Path],
    tmp_path: Path,
) -> None:
    prepared, path, splits = cv_data
    mid, whole_metrics, whole = interrupted
    steps = _Steps()
    metrics = experiments.run_command(
        "train-supervised",
        _settings(prepared, tmp_path, max_steps=6),
        resume=str(mid),
        split=SplitRequest(path, 2),
        callbacks=[steps],
        progress_bar=False,
    )
    assert steps.batches == 4  # steps 3-6 after the checkpoint of step 2
    assert metrics["last_step"] == whole_metrics["last_step"] == 6
    assert metrics["resumed_from"] == str(mid)
    resumed = load_predictions(tmp_path / "predictions_evaluation.npz")
    uninterrupted = load_predictions(whole / "predictions_evaluation.npz")
    assert np.array_equal(resumed.index, splits.folds[2].evaluation)
    # The resume is exact (the stream is a function of the step).
    np.testing.assert_allclose(resumed.probabilities, uninterrupted.probabilities, atol=1e-6)


@pytest.mark.parametrize(
    ("changes", "split"),
    [
        ({"paper_arch": "p4", "context": "avgpool63"}, {"fold": 1}),
        ({"paper_arch": "p4m", "group_fields": 2, "rot90_augment": True}, {}),
        (_AUGMENT, {"fold": 3}),
    ],
    ids=["p4-context-fold", "p4m-screen", "p4-augment-fold"],
)
def test_variant_split_runs(
    cv_data: tuple[Path, Path, ProtocolSplits],
    tmp_path: Path,
    changes: dict[str, Any],
    split: dict[str, Any],
) -> None:
    # The 2.5-models variants (task 2.5-models) on the 2.5-infra split runs: the roles
    # carry the 6-channel context through protocol_splits, gathered by pool index.
    prepared, path, _ = cv_data
    images: dict[str, np.ndarray] = {}
    fits: dict[str, Any] = {}

    class _Images(Callback):
        def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
            datamodule = getattr(trainer, "datamodule", None)
            assert isinstance(datamodule, DeviceTensorDataModule)
            for name in ("Train", "Validation", "Test"):
                images[name] = datamodule._split(name).images.cpu().numpy()
                fits[name] = datamodule._split(name).fits

    request = SplitRequest(path, split.get("fold"))  # fold None: the screen
    settings = _settings(prepared, tmp_path, **changes)
    metrics = experiments.run_command(
        "train-supervised", settings, split=request, callbacks=[_Images()]
    )
    for key, value in changes.items():
        assert metrics["config"][key] == value
    assert set(metrics["preset"]["deviations"]) >= set(changes)
    channels = 6 if changes.get("context") == "avgpool63" else 3
    pool = float_crops(prepared, names=("Train", "Validation"), context=channels == 6)
    table = np.concatenate([pool["Train"].images, pool["Validation"].images])
    parts = open_split_run(request, prepared).parts()
    for name, part in (
        ("Train", parts.training),
        ("Validation", parts.selection),
        ("Test", parts.evaluation),
    ):
        assert images[name].shape[1:] == (channels, PAPER_CROP, PAPER_CROP)
        np.testing.assert_array_equal(images[name], table[part])
    predictions = load_predictions(tmp_path / metrics["predictions"]["evaluation"]["file"])
    assert np.array_equal(predictions.index, parts.evaluation_index)
    # C5 / C6 / C7 (ADR 0006, 0007): the fits are those of the fold's own training part.
    assert fits["Validation"] is None and fits["Test"] is None
    if changes is _AUGMENT:
        training = table[parts.training]
        expected = np.concatenate([stamp_fits(training), fit_central_psf(training)], axis=1)
        np.testing.assert_array_equal(fits["Train"].cpu().numpy(), expected)
        assert set(metrics["paper_data"]["class_augment"]["excluded"]) == set(LABEL_NAMES)
        assert set(metrics["paper_data"]["flux_rescale"]["skipped"]) == set(LABEL_NAMES)
    else:
        assert fits["Train"] is None


def test_augmented_resume_is_exact(
    cv_data: tuple[Path, Path, ProtocolSplits], tmp_path: Path
) -> None:
    # The C5 / C6 / C7 draws are a function of [seed, global_step]: a resumed fold run repeats
    # them, so its predictions match the uninterrupted run's.
    prepared, path, _ = cv_data
    whole, mid = tmp_path / "whole", tmp_path / "mid.ckpt"
    whole_metrics = experiments.run_command(
        "train-supervised",
        _settings(prepared, whole, max_steps=6, **_AUGMENT),
        split=SplitRequest(path, 2),
        callbacks=[_CopyLast(3, whole / "checkpoints" / "last.ckpt", mid)],
        progress_bar=False,
    )
    resumed = tmp_path / "resumed"
    metrics = experiments.run_command(
        "train-supervised",
        _settings(prepared, resumed, max_steps=6, **_AUGMENT),
        resume=str(mid),
        split=SplitRequest(path, 2),
        progress_bar=False,
    )
    assert metrics["last_step"] == whole_metrics["last_step"] == 6
    first = load_predictions(resumed / "predictions_evaluation.npz")
    second = load_predictions(whole / "predictions_evaluation.npz")
    np.testing.assert_allclose(first.probabilities, second.probabilities, atol=1e-6)
