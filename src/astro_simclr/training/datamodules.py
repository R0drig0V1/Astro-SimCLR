"""LightningDataModules over the prepared dataset layout (task 1.6).

The legacy training modules built their own datasets in ``prepare_data()`` (unpickling the
whole file there) and their own loaders. Here two data modules do it, and build their
datasets in ``setup()``: Lightning runs ``prepare_data()`` on one process only, so datasets
assigned there do not exist on the other processes of a multi-device run.

Both read a prepared layout directory (``astro_simclr.data.prepared``, written by
``astro-simclr prepare-data``), memory-mapped, and take their augmentations from
``get_augmentation(name, image_size)``; names may be descriptive registry names or legacy
aliases. Loaders keep the legacy settings: shuffled training batches with
``drop_last=True`` and ``pin_memory=False``; unshuffled evaluation batches of
``eval_batch_size`` (legacy: 100), always with the ``center_crop`` transform (legacy
``Resize_img``).

With ``num_workers > 0`` under the ``spawn`` start method (the macOS default) each worker
receives a pickled copy of the dataset, and a pickled memory map is a full in-memory copy
of the split. Under ``fork`` (Linux) workers share the map.
"""

from __future__ import annotations

from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from astro_simclr.augmentations import get_augmentation, resolve_augmentation_name
from astro_simclr.data.datasets import ContrastiveDataset, StampDataset
from astro_simclr.data.prepared import load_prepared
from astro_simclr.data.schema import DataError

__all__ = ["EVAL_AUGMENTATION", "ClassifierDataModule", "ContrastiveDataModule"]

#: Transform of every validation and test split (the legacy ``without_aug``).
EVAL_AUGMENTATION = "center_crop"


def _check_positive(**values: int) -> None:
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            msg = f"{name} must be a positive integer, got {value!r}"
            raise ValueError(msg)


def _save_hyperparameters(datamodule: L.LightningDataModule) -> None:
    # Lightning stores these in every checkpoint. The path goes in as a string: a
    # pathlib.Path would make the checkpoint unloadable with torch.load(weights_only=True),
    # the safe default of torch >= 2.6.
    datamodule.save_hyperparameters(ignore=["data_path"])
    datamodule.hparams["data_path"] = str(datamodule.data_path)  # type: ignore[attr-defined]


def _check_batches(n_train: int, batch_size: int, path: Path) -> None:
    # With drop_last=True a split smaller than one batch yields no batch at all, and
    # Lightning would "train" silently on nothing.
    if n_train < batch_size:
        msg = (
            f"{path}: the Train split has {n_train} stamps, fewer than batch_size "
            f"{batch_size}; with drop_last no batch would be trained"
        )
        raise DataError(msg)


class ClassifierDataModule(L.LightningDataModule):
    """Labelled stamps of a classifier file: ``(image, features, one-hot label)`` batches.

    Train uses ``augmentation``; Validation and Test use :data:`EVAL_AUGMENTATION`. The
    features are the 23 kept columns (13-15 dropped), as in the legacy loaders.

    Raises:
        UnknownAugmentationError: at construction, for an unknown augmentation name.
        ValueError: at construction, for a non-positive size or batch size, or a negative
            ``num_workers``.
        DataError: in :meth:`setup`, if ``data_path`` is not a valid prepared layout of a
            classifier file, or (``stage="fit"``) its Train split is smaller than one batch.
    """

    def __init__(
        self,
        data_path: Path | str,
        *,
        image_size: int,
        batch_size: int,
        eval_batch_size: int = 100,
        augmentation: str = EVAL_AUGMENTATION,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        resolve_augmentation_name(augmentation)
        _check_positive(
            image_size=image_size, batch_size=batch_size, eval_batch_size=eval_batch_size
        )
        if num_workers < 0:
            msg = f"num_workers must be >= 0, got {num_workers}"
            raise ValueError(msg)
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.augmentation = augmentation
        self.num_workers = num_workers
        self.datasets: dict[str, StampDataset] = {}
        _save_hyperparameters(self)

    def setup(self, stage: str) -> None:
        """Open the prepared layout and build the three datasets (once; later calls reuse them).

        Only ``stage="fit"`` requires the Train split to hold at least one batch: a
        ``validate`` or ``test`` run trains nothing.
        """
        if not self.datasets:
            self._build()
        if stage == "fit":
            _check_batches(len(self.datasets["Train"]), self.batch_size, self.data_path)

    def _build(self) -> None:
        data = load_prepared(self.data_path)
        if data.kind != "classifier":
            msg = f"{self.data_path}: expected a classifier file, got a {data.kind} file"
            raise DataError(msg)
        evaluation = get_augmentation(EVAL_AUGMENTATION, self.image_size)
        self.datasets = {
            "Train": StampDataset(
                data.splits["Train"], get_augmentation(self.augmentation, self.image_size)
            ),
            "Validation": StampDataset(data.splits["Validation"], evaluation),
            "Test": StampDataset(data.splits["Test"], evaluation),
        }

    def train_dataloader(self) -> DataLoader[tuple[object, ...]]:
        """Shuffled training batches of ``batch_size``, the last incomplete one dropped."""
        return DataLoader(
            self.datasets["Train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader[tuple[object, ...]]:
        """Validation batches of ``eval_batch_size``, in file order."""
        return self._eval_loader("Validation")

    def test_dataloader(self) -> DataLoader[tuple[object, ...]]:
        """Test batches of ``eval_batch_size``, in file order."""
        return self._eval_loader("Test")

    def _eval_loader(self, split: str) -> DataLoader[tuple[object, ...]]:
        return DataLoader(
            self.datasets[split], batch_size=self.eval_batch_size, num_workers=self.num_workers
        )


class ContrastiveDataModule(L.LightningDataModule):
    """Pairs of augmented views of the Train stamps, for SimCLR pretraining.

    Any prepared file works (labels are ignored); the legacy runs used the pretraining file
    (``td_ztf_stamp_simclr_300``). There is no validation or test loader: SimCLR selects on
    its training loss.

    Raises:
        UnknownAugmentationError: at construction, for an unknown augmentation name.
        ValueError: at construction, for a non-positive size or batch size, or a negative
            ``num_workers``.
        DataError: in :meth:`setup`, if ``data_path`` is not a valid prepared layout or
            (``stage="fit"``) its Train split is smaller than one batch.
    """

    def __init__(
        self,
        data_path: Path | str,
        *,
        image_size: int,
        batch_size: int,
        augmentation: str,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        resolve_augmentation_name(augmentation)
        _check_positive(image_size=image_size, batch_size=batch_size)
        if num_workers < 0:
            msg = f"num_workers must be >= 0, got {num_workers}"
            raise ValueError(msg)
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.num_workers = num_workers
        self.train_set: ContrastiveDataset | None = None
        _save_hyperparameters(self)

    def setup(self, stage: str) -> None:
        """Open the prepared layout and build the pair dataset (once; later calls reuse it).

        With ``stage="fit"``, the Train split must hold at least one batch.
        """
        if self.train_set is None:
            train = load_prepared(self.data_path).splits["Train"]
            self.train_set = ContrastiveDataset(
                train, get_augmentation(self.augmentation, self.image_size)
            )
        if stage == "fit":
            _check_batches(len(self.train_set), self.batch_size, self.data_path)

    def train_dataloader(self) -> DataLoader[tuple[object, ...]]:
        """Shuffled batches of view pairs, the last incomplete one dropped.

        Raises:
            RuntimeError: if :meth:`setup` has not run.
        """
        if self.train_set is None:
            msg = "ContrastiveDataModule.setup() must run before train_dataloader()"
            raise RuntimeError(msg)
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
        )
