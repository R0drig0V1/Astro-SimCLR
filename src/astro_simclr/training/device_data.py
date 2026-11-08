"""Device-resident batches for the ``carrasco2021`` schedule (task 2.2a).

The paper's schedule validates every 10 steps for 30,000+ steps; a per-item ``DataLoader``
would dominate the run on the GTX 1050. So the 21x21 float crops of every split are built
once per run (about 270 MB for ``red``), moved to the training device once, and batches are
gathered there by index: no worker processes, no host-to-device copy per step.

- :class:`TensorSplit`: one split as float32 crops (N, 3, 21, 21), the 23 kept features and
  the int64 labels.
- :class:`DeviceTensorDataModule`: takes the splits (a mapping, or a callable that builds
  them, called once in ``setup``) and yields ``(crops, features, one-hot labels)`` batches,
  the batch format of ``ClassifierModule``:

  - **Train:** :func:`tiling.tile_indices` (ALeRCE's class balance), then the unbroken
    shuffled stream of :class:`tiling.PermutationStream`. An epoch is one pass length,
    ``len(tiled) // batch_size`` batches (1,136 for ``red`` at batch 64), so Lightning's
    epoch-end hooks run. The batch of global step ``s`` is batch ``s`` of the stream: the
    loader starts each epoch at ``trainer.global_step``, so a run resumed from a mid-epoch
    ``last.ckpt`` sees exactly the batches of an uninterrupted run. The loader implements
    ``state_dict`` / ``load_state_dict`` (Lightning's resumable-loader interface), which
    record the stream and refuse a resume with other data, batch size or seed.
  - **Validation / Test:** consecutive batches of ``eval_batch_size`` in file order, the last
    one partial: with 64, the epoch loss of ``ClassifierModule`` (the mean of the batch
    losses) is TF's "mean of batch-64 means in file order", the selection loss of M1.

The device is the Trainer's (``trainer.strategy.root_device``), never named here (the
portability rule). Without a Trainer (unit tests) the tensors stay on the CPU.

:func:`interim_uint8_crops` builds the splits from the existing prepared layout (uint8
stamps): a centre crop of 21 scaled to [0, 1]. It is a **stand-in** so that the schedule
runs end to end on the current data path; task 2.2c replaces it with the float path of task
2.1 (``crop_then_normalise`` on the stored float32 stamps), which is what the paper used.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import numpy.typing as npt
import torch

from astro_simclr.data.prepared import load_prepared
from astro_simclr.data.schema import DISCARDED_FEATURES, LABEL_NAMES, SPLITS, DataError
from astro_simclr.models import N_FEATURES
from astro_simclr.paper_model import PAPER_CROP
from astro_simclr.training.tiling import PermutationStream, tile_indices

__all__ = [
    "DeviceTensorDataModule",
    "SplitSource",
    "TensorSplit",
    "interim_uint8_crops",
]

#: (crops, features, one-hot labels), as ``ClassifierModule`` expects.
Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
_TRAIN, _VALIDATION, _TEST = SPLITS


@dataclass(frozen=True, slots=True)
class TensorSplit:
    """One labelled split, ready for the paper model.

    Attributes:
        images: float32 (N, 3, 21, 21) crops, channels first.
        features: float32 (N, 23), the kept alert features.
        labels: int64 (N,), values in ``range(5)``.

    Raises:
        ValueError: if a dtype or shape is wrong, the lengths differ, N is 0 or a label is
            out of range.
    """

    images: npt.NDArray[np.float32]
    features: npt.NDArray[np.float32]
    labels: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        """Check dtypes, shapes, lengths and label values."""
        n = len(self.labels)
        expected = {
            "images": (np.float32, (n, 3, PAPER_CROP, PAPER_CROP)),
            "features": (np.float32, (n, N_FEATURES)),
            "labels": (np.int64, (n,)),
        }
        for name, (dtype, shape) in expected.items():
            array = getattr(self, name)
            if array.dtype != dtype or array.shape != shape:
                msg = (
                    f"TensorSplit.{name} must be {np.dtype(dtype)} {shape}, "
                    f"got {array.dtype} {array.shape}"
                )
                raise ValueError(msg)
        if n == 0:
            raise ValueError("TensorSplit is empty")
        if self.labels.min() < 0 or self.labels.max() >= len(LABEL_NAMES):
            raise ValueError(f"labels must be in range({len(LABEL_NAMES)})")


#: The splits of a run (``Train``, ``Validation``, ``Test``), or a callable that builds them.
SplitSource = Mapping[str, TensorSplit] | Callable[[], Mapping[str, TensorSplit]]


@dataclass(frozen=True, slots=True)
class _DeviceSplit:
    images: torch.Tensor
    features: torch.Tensor
    one_hot: torch.Tensor

    @classmethod
    def of(cls, split: TensorSplit, device: torch.device) -> _DeviceSplit:
        labels = torch.from_numpy(np.ascontiguousarray(split.labels))
        return cls(
            images=torch.from_numpy(np.ascontiguousarray(split.images)).to(device),
            features=torch.from_numpy(np.ascontiguousarray(split.features)).to(device),
            one_hot=torch.nn.functional.one_hot(labels, num_classes=len(LABEL_NAMES)).to(device),
        )

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def gather(self, index: torch.Tensor) -> Batch:
        return self.images[index], self.features[index], self.one_hot[index]


class _StreamLoader:
    """The training batches of one epoch, starting at the Trainer's global step.

    Iterating yields the stream batches ``start ... end - 1``, where ``start`` is the current
    global step and ``end`` the next multiple of ``len(self)``. Each pass's permutation is
    moved to the device once; a batch is a slice of it (or of two, across a pass boundary)
    and a gather on the device.
    """

    def __init__(
        self,
        split: _DeviceSplit,
        tiled: npt.NDArray[np.int64],
        stream: PermutationStream,
        position: Callable[[], int],
    ) -> None:
        self._split = split
        self._tiled = tiled
        self._stream = stream
        self._position = position
        self._passes: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return self._stream.batches_per_pass

    def _pass(self, index: int) -> torch.Tensor:
        if index not in self._passes:
            # At most the current and the next pass are kept on the device.
            self._passes = {k: v for k, v in self._passes.items() if k >= index - 1}
            order = self._tiled[self._stream.permutation(index)]
            self._passes[index] = torch.from_numpy(order).to(self._split.images.device)
        return self._passes[index]

    def indices(self, batch: int) -> torch.Tensor:
        """The (device) indices into the split of stream batch ``batch``."""
        parts = [self._pass(index)[start:stop] for index, start, stop in self._stream.span(batch)]
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    def __iter__(self) -> Iterator[Batch]:
        start = self._position()
        end = (start // len(self) + 1) * len(self)
        for batch in range(start, end):
            yield self._split.gather(self.indices(batch))

    def state_dict(self) -> dict[str, Any]:
        """The stream definition (Lightning saves it in every checkpoint)."""
        return self._stream.state()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Check that a resumed run streams the same data; the position is the global step.

        Raises:
            DataError: if the checkpoint's stream (item count, batch size, seed) differs.
        """
        if state != self._stream.state():
            msg = (
                f"the checkpoint was trained on the stream {state}, but this run's is "
                f"{self._stream.state()}: resume with the same data, batch_size and seed"
            )
            raise DataError(msg)


class _SequentialLoader:
    """Consecutive batches of ``batch_size`` in file order; the last one may be partial."""

    def __init__(self, split: _DeviceSplit, batch_size: int) -> None:
        self._split = split
        self._batch_size = batch_size

    def __len__(self) -> int:
        return -(-len(self._split) // self._batch_size)

    def __iter__(self) -> Iterator[Batch]:
        for start in range(0, len(self._split), self._batch_size):
            index = slice(start, start + self._batch_size)
            yield (
                self._split.images[index],
                self._split.features[index],
                self._split.one_hot[index],
            )


class DeviceTensorDataModule(L.LightningDataModule):
    """Device-resident batches of 21x21 crops for the ``carrasco2021`` schedule.

    ``source`` is a mapping of the three splits (``Train``, ``Validation``, ``Test``) or a
    callable returning one; it is resolved once, at the first ``setup``. ``balance=True``
    (the paper) tiles the Train split (:func:`tiling.tile_indices`); False streams it as
    it is. ``seed`` defines the training stream (normally ``Settings.seed``). There are no
    worker processes: everything happens on the training device.

    Raises:
        ValueError: at construction, for a non-positive batch size or a negative seed.
        DataError: in ``setup``, if a split is missing or not a :class:`TensorSplit`, or the
            (tiled) Train split is smaller than one batch; when resuming, if the checkpoint's
            stream differs (see :class:`_StreamLoader`).
    """

    def __init__(
        self,
        source: SplitSource,
        *,
        batch_size: int = 64,
        eval_batch_size: int = 64,
        seed: int = 0,
        balance: bool = True,
    ) -> None:
        super().__init__()
        for name, value in (("batch_size", batch_size), ("eval_batch_size", eval_batch_size)):
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if seed < 0:
            raise ValueError(f"seed must be >= 0, got {seed}")
        self._source = source
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.seed = seed
        self.balance = balance
        # Plain values only, so checkpoints load with torch.load(weights_only=True).
        self.save_hyperparameters(ignore=["source"])
        self._splits: dict[str, _DeviceSplit] | None = None
        self._train: _StreamLoader | None = None
        #: The tiled Train order (indices into the Train split), set by ``setup``.
        self.tiled_indices: npt.NDArray[np.int64] | None = None

    def _device(self) -> torch.device:
        trainer = self.trainer
        return torch.device("cpu") if trainer is None else trainer.strategy.root_device

    def _resolve(self) -> Mapping[str, TensorSplit]:
        splits = self._source() if callable(self._source) else self._source
        for name in SPLITS:
            if not isinstance(splits.get(name), TensorSplit):
                raise DataError(f"DeviceTensorDataModule needs a TensorSplit for {name!r}")
        return splits

    def setup(self, stage: str) -> None:
        """Build the splits (once) and move them to the Trainer's device (once per device).

        Raises:
            DataError: see the class docstring; the Train size is checked for ``fit`` only.
        """
        device = self._device()
        if self._splits is None or self._splits[_TRAIN].images.device != device:
            splits = self._resolve()
            self._splits = {name: _DeviceSplit.of(splits[name], device) for name in SPLITS}
            labels = splits[_TRAIN].labels
            balanced = tile_indices(labels) if self.balance else None
            tiled = np.arange(len(labels), dtype=np.int64) if balanced is None else balanced
            self.tiled_indices = tiled
            self._train = None
            if len(tiled) >= self.batch_size:
                stream = PermutationStream(len(tiled), self.batch_size, self.seed)
                self._train = _StreamLoader(self._splits[_TRAIN], tiled, stream, self._global_step)
        if stage == "fit" and self._train is None:
            n = 0 if self.tiled_indices is None else len(self.tiled_indices)
            msg = (
                f"the Train split has {n} entries after tiling, fewer than "
                f"batch_size {self.batch_size}"
            )
            raise DataError(msg)

    def _global_step(self) -> int:
        trainer = self.trainer
        return 0 if trainer is None else trainer.global_step

    def _split(self, name: str) -> _DeviceSplit:
        if self._splits is None:
            raise DataError("DeviceTensorDataModule.setup() has not run")
        return self._splits[name]

    def train_dataloader(self) -> _StreamLoader:
        """The training stream of the current epoch (see the module docstring)."""
        if self._train is None:
            raise DataError("DeviceTensorDataModule.setup() has not run")
        return self._train

    def val_dataloader(self) -> _SequentialLoader:
        """Validation in file order, ``eval_batch_size`` per batch."""
        return _SequentialLoader(self._split(_VALIDATION), self.eval_batch_size)

    def test_dataloader(self) -> _SequentialLoader:
        """Test in file order, ``eval_batch_size`` per batch."""
        return _SequentialLoader(self._split(_TEST), self.eval_batch_size)


def _centre_crop_unit(images: npt.NDArray[np.uint8], size: int) -> npt.NDArray[np.float32]:
    """(N, H, W, 3) uint8 -> (N, 3, size, size) float32 in [0, 1], the centre crop."""
    side = images.shape[1]
    start = (side - size) // 2
    crop = np.asarray(images[:, start : start + size, start : start + size, :], dtype=np.float32)
    return np.ascontiguousarray(crop.transpose(0, 3, 1, 2) / np.float32(255.0))


def interim_uint8_crops(path: Path) -> dict[str, TensorSplit]:
    """The splits of the prepared layout ``path``, as 21x21 crops of its uint8 stamps.

    A **stand-in** until task 2.2c wires the float path of 2.1: the centre 21x21 of the
    stored uint8 stamps, divided by 255 (no re-normalisation on the crop), with the 23 kept
    features (columns 13-15 dropped, as in ``StampDataset``) as float32.

    Raises:
        DataError: if ``path`` is not a valid prepared layout of a classifier file.
    """
    data = load_prepared(path)
    splits: dict[str, TensorSplit] = {}
    for name in SPLITS:
        split = data.splits[name]
        if split.labels is None or split.features is None:
            raise DataError(f"{path}: {name} is unlabelled; the paper model needs labels")
        kept = np.delete(np.asarray(split.features), list(DISCARDED_FEATURES), axis=1)
        splits[name] = TensorSplit(
            images=_centre_crop_unit(split.images, PAPER_CROP),
            features=kept.astype(np.float32),
            labels=np.array(split.labels, dtype=np.int64, copy=True),
        )
    return splits
