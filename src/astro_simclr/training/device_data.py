"""Device-resident batches for the ``carrasco2021`` schedule (task 2.2a).

The paper's schedule validates every 10 steps for 30,000+ steps; a per-item ``DataLoader``
would dominate the run on the GTX 1050. So the 21x21 float crops of every split are built
once per run (about 270 MB for ``red``), moved to the training device once, and batches are
gathered there by index: no worker processes, no host-to-device copy per step.

- :class:`TensorSplit`: one split as float32 crops (N, 3, 21, 21), or (N, 6, 21, 21) with
  the 63x63 context channels (task 2.5-models), or the whole stamps (N, 3, 63, 63) for the
  ConvNeXt rows (plan row B8, ADR 0022), the 23 kept features and the int64 labels.
- :class:`DeviceTensorDataModule`: takes the splits (a mapping, or a callable that builds
  them, called once in ``setup``) and yields ``(crops, features, one-hot labels)`` batches,
  the batch format of ``ClassifierModule``; when the Train split carries ``fits`` (the
  ``class_augment.stamp_fits`` that C5 / C6 need, plan 2.5-input/training), its batches
  have them as a fourth element, gathered with the rest:

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

The splits come from a prepared layout through one of two builders:

- :func:`float_crops` (task 2.2c), **the paper's input**: 2.1's float path,
  ``data.float_input.crop_split`` (centre 21x21, per-channel min-max on the crop without
  epsilon, NaN -> 0, the NaN approximation on by default) on the stored float32 stamps
  (``images_f32.npy``, written by ``prepare-data --images float32``); with
  ``context=True`` (``Settings.context: avgpool63``, plan row B1'), the 3 channels of
  ``data.float_input.context_split`` (the whole stamp average-pooled to 21x21, normalised
  like the crop, the same NaN approximation) follow the crop's. With ``side=63`` (the
  ConvNeXt rows) the "crop" is the whole stamp, re-normalised the same way on its own
  pixels: about 2.4 GB for ``red`` instead of 270 MB, which on the 4 GB GTX 1050 leaves
  little room for the model;
- :func:`interim_uint8_crops` (task 2.2a), the stand-in that ran the schedule before the
  float path existed: the centre 21x21 of the uint8 stamps divided by 255, no
  re-normalisation. It stays as the ``paper_input: uint8`` diagnostic.

:func:`without_train_indices` drops Train stamps (the leaked one, ``protocol.leakage``)
from Train only, before the tiling.

**Split runs** (task 2.5-infra, ``train-supervised --splits FILE --fold K | --screen``;
:mod:`astro_simclr.protocol.split_runs`): both builders take ``names`` to read only some
splits (Train and Validation: Test is never loaded), and :func:`protocol_splits` gathers
the run's parts from them by pool index into the three roles of
:class:`DeviceTensorDataModule`: ``Train`` is the part trained on (and tiled),
``Validation`` the checkpoint-selection part, ``Test`` the held-out evaluation part that
the run predicts. The data module takes the run's identity (``run_identity``:
``SplitRun.identity()``, the splits SHA-256, mode and fold) and the training loader adds
it to its checkpoint state as ``split_run``, so that a resume can never continue a run of
another fold or mode: two folds can have the same tiled size, and the stream state alone
would accept it. A normal run has no identity and its state has no ``split_run`` key (its
checkpoints are unchanged; a missing key reads as a normal run).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import lightning as L
import numpy as np
import numpy.typing as npt
import torch

from astro_simclr.data.schema import SPLITS, DataError
from astro_simclr.training.class_augment import upload_plan
from astro_simclr.training.tensor_splits import (
    IMAGE_CHANNELS,
    IMAGE_SIDES,
    SPLIT_RUN_STATE,
    Batch,
    PlannedTrainBatch,
    Planner,
    SplitSource,
    TensorSplit,
    TrainBatch,
    check_split_identity,
    float_crops,
    interim_uint8_crops,
    protocol_splits,
    without_train_indices,
)
from astro_simclr.training.tiling import PermutationStream, tile_indices

#: The non-Lightning names live in ``training.tensor_splits`` (task 4.6), re-exported here.
_TRAIN, _VALIDATION, _TEST = SPLITS

__all__ = [
    "IMAGE_CHANNELS",
    "IMAGE_SIDES",
    "SPLIT_RUN_STATE",
    "Batch",
    "DeviceTensorDataModule",
    "PlannedTrainBatch",
    "Planner",
    "SplitSource",
    "TensorSplit",
    "TrainBatch",
    "check_split_identity",
    "float_crops",
    "interim_uint8_crops",
    "protocol_splits",
    "without_train_indices",
]

_TRAIN, _VALIDATION, _TEST = SPLITS


@dataclass(frozen=True, slots=True)
class _DeviceSplit:
    images: torch.Tensor
    features: torch.Tensor
    one_hot: torch.Tensor
    fits: torch.Tensor | None = None

    @classmethod
    def of(cls, split: TensorSplit, device: torch.device) -> _DeviceSplit:
        labels = torch.from_numpy(np.ascontiguousarray(split.labels))
        fits = None if split.fits is None else torch.from_numpy(split.fits).to(device)
        return cls(
            images=torch.from_numpy(np.ascontiguousarray(split.images)).to(device),
            features=torch.from_numpy(np.ascontiguousarray(split.features)).to(device),
            one_hot=torch.nn.functional.one_hot(labels, num_classes=split.n_classes).to(device),
            fits=fits,
        )

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def gather(self, index: torch.Tensor) -> TrainBatch:
        batch = (self.images[index], self.features[index], self.one_hot[index])
        return batch if self.fits is None else (*batch, self.fits[index])


class _StreamLoader:
    """The training batches of one epoch, starting at the Trainer's global step.

    Iterating yields the stream batches ``start ... end - 1``, where ``start`` is the current
    global step and ``end`` the next multiple of ``len(self)``. Each pass's permutation is
    moved to the device once; a batch is a slice of it (or of two, across a pass boundary)
    and a gather on the device.

    With a ``planner`` (and ``host``, the Train split on the CPU), each batch also gets its
    augmentation plan: the same rows are gathered on the host, ``planner(host_batch, b)``
    makes every draw and decision of step ``b`` there, and the plan is uploaded
    (``class_augment.upload_plan``: pinned, ``non_blocking`` on CUDA) and yielded as a last
    element. The step itself then makes no host round trip (ADR 0006, "Prefetch").
    """

    def __init__(
        self,
        split: _DeviceSplit,
        tiled: npt.NDArray[np.int64],
        stream: PermutationStream,
        position: Callable[[], int],
        identity: Mapping[str, object] | None = None,
        *,
        host: _DeviceSplit | None = None,
        planner: Planner | None = None,
    ) -> None:
        self._split = split
        self._tiled = tiled
        self._stream = stream
        self._position = position
        self._identity = None if identity is None else dict(identity)
        self._passes: dict[int, torch.Tensor] = {}
        self._host_passes: dict[int, torch.Tensor] = {}
        self._host = host
        self._planner = planner

    def __len__(self) -> int:
        return self._stream.batches_per_pass

    def _pass(self, index: int) -> torch.Tensor:
        if index not in self._passes:
            # At most the current and the next pass are kept on the device.
            self._passes = {k: v for k, v in self._passes.items() if k >= index - 1}
            order = self._tiled[self._stream.permutation(index)]
            self._passes[index] = torch.from_numpy(order).to(self._split.images.device)
        return self._passes[index]

    def _host_pass(self, index: int) -> torch.Tensor:
        if index not in self._host_passes:
            self._host_passes = {k: v for k, v in self._host_passes.items() if k >= index - 1}
            order = self._tiled[self._stream.permutation(index)]
            self._host_passes[index] = torch.from_numpy(order)
        return self._host_passes[index]

    def indices(self, batch: int) -> torch.Tensor:
        """The (device) indices into the split of stream batch ``batch``."""
        parts = [self._pass(index)[start:stop] for index, start, stop in self._stream.span(batch)]
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    def host_indices(self, batch: int) -> torch.Tensor:
        """The same indices as :meth:`indices`, on the CPU."""
        spans = self._stream.span(batch)
        parts = [self._host_pass(index)[start:stop] for index, start, stop in spans]
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    def __iter__(self) -> Iterator[TrainBatch | PlannedTrainBatch]:
        start = self._position()
        end = (start // len(self) + 1) * len(self)
        device = self._split.images.device
        for batch in range(start, end):
            gathered = self._split.gather(self.indices(batch))
            if self._planner is None or self._host is None:
                yield gathered
                continue
            plan = self._planner(self._host.gather(self.host_indices(batch)), batch)
            yield (*gathered, upload_plan(plan, device))

    def state_dict(self) -> dict[str, Any]:
        """The stream definition, plus a split run's identity (saved in every checkpoint)."""
        state = self._stream.state()
        if self._identity is not None:
            state[SPLIT_RUN_STATE] = dict(self._identity)
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Check that a resumed run is the same run; the position is the global step.

        Raises:
            ConfigError: if the checkpoint belongs to another split run (splits file, mode
                or fold) or to a normal run when this is a split run, or the reverse.
            DataError: if the checkpoint's stream (item count, batch size, seed) differs.
        """
        check_split_identity(state.get(SPLIT_RUN_STATE), self._identity)
        state = {key: value for key, value in state.items() if key != SPLIT_RUN_STATE}
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
    it is. ``seed`` defines the training stream (normally ``Settings.seed``).
    ``run_identity`` is a split run's ``SplitRun.identity()`` (``None`` for a normal run),
    kept in the training loader's checkpoint state and checked on resume. ``planner``
    (``PaperClassifierModule.augment_plan``, when the run augments) makes the training
    loader plan each step's augmentation on the host, from a CPU copy of the Train split
    kept for that, and yield the uploaded plan with the batch (:class:`_StreamLoader`).
    There are no worker processes: everything else happens on the training device.

    Raises:
        ValueError: at construction, for a non-positive batch size or a negative seed.
        DataError: in ``setup``, if a split is missing or not a :class:`TensorSplit`, the
            splits have different ``n_classes``, or the (tiled) Train split is smaller than
            one batch; when resuming, if the checkpoint's
            stream differs (see :class:`_StreamLoader`).
        ConfigError: when resuming, if the checkpoint is from another split run or mode.
    """

    def __init__(
        self,
        source: SplitSource,
        *,
        batch_size: int = 64,
        eval_batch_size: int = 64,
        seed: int = 0,
        balance: bool = True,
        run_identity: Mapping[str, object] | None = None,
        planner: Planner | None = None,
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
        self._identity = run_identity
        self._planner = planner
        # Plain values only, so checkpoints load with torch.load(weights_only=True). The
        # identity is not a hyperparameter: it lives in the loader state (module docstring).
        # The planner is a bound method of the module: runtime wiring, not a setting.
        self.save_hyperparameters(ignore=["source", "run_identity", "planner"])
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
        counts = {name: splits[name].n_classes for name in SPLITS}
        if len(set(counts.values())) != 1:
            raise DataError(f"the splits disagree on n_classes: {counts}")
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
                host = None
                if self._planner is not None:
                    host = _DeviceSplit.of(splits[_TRAIN], torch.device("cpu"))
                self._train = _StreamLoader(
                    self._splits[_TRAIN],
                    tiled,
                    stream,
                    self._global_step,
                    self._identity,
                    host=host,
                    planner=self._planner,
                )
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
