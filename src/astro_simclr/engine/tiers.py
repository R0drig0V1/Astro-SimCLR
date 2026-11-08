"""The three data tiers: where a dataset lives while a run trains (task 4.3, ADR 0014, 0018).

Every tier serves the same thing: the training stream of :class:`TrainSource` (the batch of
step ``s`` is a pure function of the stream definition and ``s``) and the evaluation passes
of :class:`EvalBatches` (consecutive batches in row order, the last one partial). Only where
the bytes are, and how they reach the device, changes:

1. ``device`` (:class:`DataTier.DEVICE`): the whole dataset on the training device; a batch
   is a gather there, by indices uploaded once per pass (each pass's row order is moved to
   the device once and sliced, so a step does no host-to-device copy at all). This is
   ``device_data.DeviceTensorDataModule``'s design, and its batches are ``torch.equal`` to
   that module's (``tests/test_engine_tiers.py``).
2. ``pinned``: the dataset in host memory; per step the host gathers the batch's rows into a
   **pinned staging buffer** and issues a ``non_blocking`` upload, one step ahead (the next
   step's batch is gathered and enqueued as soon as the current one is handed out). Two
   buffers alternate; a buffer is refilled only after the event recorded behind its last
   upload has completed. The copy runs on the device's current stream, ahead of the
   compute that reads it, so the values and their order are those of tier 1: the batches
   are ``torch.equal``. The dataset itself stays in pageable memory: the gather is a CPU
   copy, which pinning would not speed up, and pinning gigabytes would lock them out of the
   page cache (ADR 0018).
3. ``memmap``: the images stay in their file (a read-only ``np.memmap``, data larger than
   host memory, phase 5's corpus). Per step the batch's rows are read **sorted by file
   offset** (one forward sweep through the shards the batch touches) and scattered into the
   pinned staging buffer at their batch positions, then uploaded as in tier 2. The small
   columns (features, labels, fits) are in memory. Same values, same batches. The **shuffle
   granularity** is the stream's, not the tier's: ``row`` (default) makes every batch
   ``batch_size`` scattered reads; ``block`` (``engine.stream``) bounds a batch to two
   windows of contiguous blocks, at the price of correlated batches.

The tier never changes the numbers, so it is not part of the stream definition: a run may
resume on another tier exactly. The **storage dtype** does change them: ``float16`` or
``uint8`` images (``TierChoice.storage_dtype``, a recorded preset deviation, never an
automatic fallback) are quantised once when the tier is built and dequantised on the device
after each gather (``float16`` -> ``float32``; ``uint8``, which needs images in [0, 1], ->
``k / 255`` correctly rounded, from a 256-entry table so the bits do not depend on the
device); the definition records the dtype, so a resume under another one is refused. The
``memmap`` tier serves the file's float32 only.

On a CPU training device nothing is staged or pinned: the host gather is the batch.
:meth:`TrainSource.describe` is the record for the manifest (tier, storage dtype, whether it
is a deviation, pinned staging, prefetch depth, shuffle).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import torch

from astro_simclr.config import ConfigError
from astro_simclr.data.schema import DataError
from astro_simclr.engine.data import Batch, DataTier, StorageDtype, TierChoice, TrainBatch
from astro_simclr.engine.dataset import Dataset
from astro_simclr.engine.stream import SHUFFLES, Shuffle, TrainOrder

__all__ = [
    "STAGING_SLOTS",
    "Columns",
    "EvalBatches",
    "StreamSpec",
    "TrainSource",
    "check_stream",
    "open_eval",
    "open_train",
]

#: Named batch tensors on the training device (``images``, ``features``, ``labels``, ``fits``).
Columns = dict[str, torch.Tensor]
#: Pinned staging buffers per stream: the one being uploaded and the one being filled.
STAGING_SLOTS: Final = 2

_TORCH_DTYPES: Final[dict[StorageDtype, torch.dtype]] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "uint8": torch.uint8,
}


@dataclass(frozen=True, slots=True)
class StreamSpec:
    """The training stream's definition, as a config gives it (``engine.stream.TrainOrder``).

    ``balance`` tiles the rows by class first (the paper; needs labels). ``shuffle`` is the
    granularity (``row`` or ``block``); ``block_rows`` and ``window_blocks`` are read by
    ``block`` only.

    Raises:
        ValueError: for a non-positive batch size or block parameter, a negative seed or an
            unknown shuffle.
    """

    batch_size: int = 64
    seed: int = 0
    balance: bool = True
    shuffle: Shuffle = "row"
    block_rows: int = 4096
    window_blocks: int = 16

    def __post_init__(self) -> None:
        """Check the values (the rest is checked by ``TrainOrder``)."""
        for name in ("batch_size", "block_rows", "window_blocks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
        if self.shuffle not in SHUFFLES:
            raise ValueError(f"shuffle must be one of {SHUFFLES}, got {self.shuffle!r}")

    def order(self, dataset: Dataset) -> TrainOrder:
        """The :class:`TrainOrder` of ``dataset`` under this spec.

        Raises:
            ValueError: if ``balance`` is set and the dataset has no labels, or the stream
                is smaller than one batch.
            ConfigError: for ``block`` shuffle with ``balance``.
        """
        return TrainOrder(
            len(dataset),
            batch_size=self.batch_size,
            seed=self.seed,
            labels=dataset.labels_of_rows(),
            balance=self.balance,
            shuffle=self.shuffle,
            block_rows=self.block_rows,
            window_blocks=self.window_blocks,
        )


def _quantised(images: npt.NDArray[np.float32], dtype: StorageDtype) -> npt.NDArray[Any]:
    """``images`` as stored under ``dtype`` (module docstring).

    Raises:
        DataError: for ``uint8`` when an image value is not finite or outside [0, 1].
    """
    if dtype == "float32":
        return images
    if dtype == "float16":
        return images.astype(np.float16)
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        msg = (
            "storage_dtype uint8 needs finite images in [0, 1] (it stores round(255 x)); "
            "these are not: keep float32"
        )
        raise DataError(msg)
    return np.rint(images * np.float32(255.0)).astype(np.uint8)


#: ``k / 255`` in float32 for every uint8 ``k``, correctly rounded (numpy's IEEE division).
#: The dequantisation is a gather of this table, never a division on the device: CUDA
#: divides a tensor by a Python scalar as a multiplication by its reciprocal, which differs
#: from ``k / 255`` by one ulp for 126 of the 256 values (measured on torch 2.14.1+cu130,
#: RTX 5090), so the same uint8 batch would give other float32 images on CUDA than on CPU.
_UINT8_TO_FLOAT32: Final = torch.from_numpy(np.arange(256, dtype=np.float32) / np.float32(255.0))
#: The table on each device it was used on (a 1 KiB upload once, not once per batch).
_UINT8_TABLES: dict[torch.device, torch.Tensor] = {}


def _uint8_table(device: torch.device) -> torch.Tensor:
    table = _UINT8_TABLES.get(device)
    if table is None:
        table = _UINT8_TABLES[device] = _UINT8_TO_FLOAT32.to(device)
    return table


def _dequantised(images: torch.Tensor, dtype: StorageDtype) -> torch.Tensor:
    """The float32 images of a gathered batch stored under ``dtype`` (on its device).

    ``uint8`` gives ``k / 255`` correctly rounded, the same bits on every device
    (:data:`_UINT8_TO_FLOAT32`).
    """
    if dtype == "float32":
        return images
    if dtype == "float16":
        return images.to(torch.float32)
    # index_select reads int32 indices on every backend; a uint8 index would be a mask.
    index = images.reshape(-1).to(torch.int32)
    return _uint8_table(images.device).index_select(0, index).view(images.shape)


class _Staging:
    """Host buffers for one stream of uploads (train or eval) to ``device``.

    On a CPU device, :meth:`upload` fills fresh tensors and returns them. Otherwise it fills
    one of :data:`STAGING_SLOTS` pinned buffers, after waiting for the event of that
    buffer's previous upload, enqueues ``non_blocking`` copies on the current stream and
    records a new event behind them.
    """

    def __init__(
        self, specs: dict[str, tuple[tuple[int, ...], torch.dtype]], rows: int, device: torch.device
    ) -> None:
        self._specs = specs
        self._rows = rows
        self._device = device
        self.pinned = device.type != "cpu"
        self._slots: list[Columns] = []
        self._events: list[torch.Event | None] = [None] * STAGING_SLOTS
        self._next = 0

    def _empty(self, rows: int, *, pin: bool) -> Columns:
        return {
            name: torch.empty((rows, *shape), dtype=dtype, pin_memory=pin)
            for name, (shape, dtype) in self._specs.items()
        }

    def upload(self, count: int, fill: Callable[[Columns], None]) -> Columns:
        """Fill the first ``count`` rows of a buffer with ``fill`` and return them on the device."""
        if not self.pinned:
            fresh = self._empty(count, pin=False)
            fill(fresh)
            return fresh
        if not self._slots:
            self._slots = [self._empty(self._rows, pin=True) for _ in range(STAGING_SLOTS)]
        slot = self._next
        self._next = (slot + 1) % STAGING_SLOTS
        event = self._events[slot]
        if event is not None:
            event.synchronize()  # the buffer's previous upload has been read
        staged = {name: tensor[:count] for name, tensor in self._slots[slot].items()}
        fill(staged)
        uploaded = {
            name: tensor.to(self._device, non_blocking=True) for name, tensor in staged.items()
        }
        done = torch.Event(device=self._device)
        done.record()
        self._events[slot] = done
        return uploaded


class _Fetch(ABC):
    """How one tier turns dataset rows into device columns (the tier-specific part)."""

    tier: DataTier
    prefetch: int = 0
    pinned: bool = False

    @abstractmethod
    def train(self, order: TrainOrder, step: int) -> Columns:
        """The columns of stream batch ``step`` (gathered, uploaded; not dequantised)."""

    @abstractmethod
    def span(self, start: int, stop: int) -> Columns:
        """The columns of dataset rows ``start ... stop - 1`` (not dequantised)."""


class _DeviceFetch(_Fetch):
    """Tier 1: the materialised dataset on the device; gathers by device indices."""

    tier = DataTier.DEVICE

    def __init__(self, dataset: Dataset, dtype: StorageDtype, device: torch.device) -> None:
        held = dataset.materialised()
        self._device = device
        self._columns: Columns = {}
        self._columns["images"] = torch.from_numpy(_quantised(held.images, dtype)).to(device)
        for name, array in held.columns().items():
            if name != "images":
                self._columns[name] = torch.from_numpy(array).to(device)
        self._passes: dict[int, torch.Tensor] = {}

    def _pass(self, order: TrainOrder, index: int) -> torch.Tensor:
        if index not in self._passes:
            # The current and the next pass are kept on the device.
            self._passes = {k: v for k, v in self._passes.items() if k >= index - 1}
            self._passes[index] = torch.from_numpy(order.pass_rows(index).copy()).to(self._device)
        return self._passes[index]

    def train(self, order: TrainOrder, step: int) -> Columns:
        """A gather of the batch's rows, by slices of the cached device pass orders."""
        parts = [self._pass(order, i)[a:b] for i, a, b in order.stream.span(step)]
        index = parts[0] if len(parts) == 1 else torch.cat(parts)
        return {name: tensor[index] for name, tensor in self._columns.items()}

    def span(self, start: int, stop: int) -> Columns:
        """Views of the rows ``start ... stop - 1`` (as ``device_data``'s evaluation batches)."""
        return {name: tensor[start:stop] for name, tensor in self._columns.items()}


class _HostFetch(_Fetch):
    """Tiers 2 and 3: a host gather into staging, then an upload (see the module docstring)."""

    def __init__(
        self,
        dataset: Dataset,
        dtype: StorageDtype,
        device: torch.device,
        *,
        memmap: bool,
        train_rows: int,
        eval_rows: int,
    ) -> None:
        self.tier = DataTier.MEMMAP if memmap else DataTier.PINNED
        self._dataset = dataset
        if memmap:
            # The images stay where they are; the small columns come into memory once,
            # indexed by stored row.
            self._images: npt.NDArray[np.float32] | torch.Tensor = dataset.images
            self._small = {
                name: torch.from_numpy(np.array(array, copy=True))
                for name, array in dataset.columns().items()
                if name != "images"
            }
        else:
            held = dataset.materialised()
            self._images = torch.from_numpy(_quantised(held.images, dtype))
            self._small = {
                name: torch.from_numpy(array)
                for name, array in held.columns().items()
                if name != "images"
            }
        image_dtype = _TORCH_DTYPES[dtype]
        specs = {"images": (tuple(dataset.images.shape[1:]), image_dtype)}
        specs |= {name: (tuple(t.shape[1:]), t.dtype) for name, t in self._small.items()}
        self._train = _Staging(specs, train_rows, device)
        self._eval = _Staging(specs, eval_rows, device)
        self.pinned = self._train.pinned
        # Prefetching pays only when the upload is asynchronous; on a CPU device the host
        # gather is the batch, so it is done when asked for.
        self.prefetch = 1 if self.pinned else 0

    def _fill(self, rows: npt.NDArray[np.int64]) -> Callable[[Columns], None]:
        """A filler of a staging buffer with dataset rows ``rows``, in order."""

        def fill(staged: Columns) -> None:
            source = self._images
            if isinstance(source, torch.Tensor):  # pinned: the materialised images
                index = torch.from_numpy(np.array(rows, dtype=np.int64))  # writable copy
                torch.index_select(source, 0, index, out=staged["images"])
            else:  # memmap: the file, read through the dataset's selection
                stored = np.array(self._dataset.stored_rows(rows), dtype=np.int64)  # writable
                order = np.argsort(stored, kind="stable")
                # One forward sweep through the file, then each row to its batch position.
                images = staged["images"].numpy()
                images[order] = source[stored[order]]
                index = torch.from_numpy(stored)
            for name, tensor in self._small.items():
                torch.index_select(tensor, 0, index, out=staged[name])

        return fill

    def train(self, order: TrainOrder, step: int) -> Columns:
        """The batch's rows, gathered on the host and uploaded through the train staging."""
        rows = order.rows(step)
        return self._train.upload(len(rows), self._fill(rows))

    def span(self, start: int, stop: int) -> Columns:
        """Rows ``start ... stop - 1`` (``stop`` clipped to the end), via the eval staging."""
        rows = np.arange(start, min(stop, len(self._dataset)), dtype=np.int64)
        return self._eval.upload(len(rows), self._fill(rows))


def _fetch(
    dataset: Dataset,
    choice: TierChoice,
    device: torch.device,
    *,
    train_rows: int,
    eval_rows: int,
) -> _Fetch:
    """The fetcher of ``choice.tier``.

    Raises:
        ConfigError: for the ``memmap`` tier with a storage dtype other than float32.
        DataError: for ``uint8`` storage of images outside [0, 1].
    """
    if choice.tier is DataTier.DEVICE:
        return _DeviceFetch(dataset, choice.storage_dtype, device)
    if choice.tier is DataTier.MEMMAP and choice.storage_dtype != "float32":
        msg = (
            f"the memmap tier reads the file's float32 images; storage_dtype "
            f"{choice.storage_dtype} would need a converted file (use the device or pinned tier)"
        )
        raise ConfigError(msg)
    return _HostFetch(
        dataset,
        choice.storage_dtype,
        device,
        memmap=choice.tier is DataTier.MEMMAP,
        train_rows=train_rows,
        eval_rows=eval_rows,
    )


def _classifier_batch(columns: Columns, n_classes: int | None) -> Batch:
    """``(images, features, one-hot labels)``; the one-hot is int64, as ``device_data``'s.

    Raises:
        ConfigError: if the dataset has no labels or features (an unlabelled kind).
    """
    if "labels" not in columns or "features" not in columns or n_classes is None:
        msg = (
            "this dataset has no labels or features (an unlabelled kind): read its "
            "columns() instead of classifier batches"
        )
        raise ConfigError(msg)
    one_hot = torch.nn.functional.one_hot(columns["labels"], num_classes=n_classes)
    return columns["images"], columns["features"], one_hot


def check_stream(saved: dict[str, Any], current: dict[str, Any]) -> None:
    """Refuse a resume on another stream: ``saved`` (a checkpoint's) must equal ``current``.

    Raises:
        DataError: if they differ (the message names both).
    """
    if saved != current:
        msg = (
            f"the checkpoint was trained on the stream {saved}, but this run's is {current}: "
            "resume with the same data, batch size, seed, shuffle and storage dtype"
        )
        raise DataError(msg)


class TrainSource:
    """The training stream on one tier: an ``engine.data.BatchSource``.

    :meth:`batch` gives step ``s``'s classifier batch (``TrainBatch``: the fits are a fourth
    element when the dataset has them); :meth:`columns` gives every column by name (an
    unlabelled kind has images only). Both are pure functions of the definition and ``s``.
    The source remembers one thing, its **position** (the step it expects next); a call for
    that step returns the batch prefetched by the previous call, any other step is fetched
    synchronously (still exact), and every call prefetches the step after it on the host
    tiers. Build it with :func:`open_train`.
    """

    def __init__(self, fetch: _Fetch, order: TrainOrder, dataset: Dataset, dtype: StorageDtype):
        self._fetch = fetch
        self._order = order
        self._n_classes = dataset.n_classes
        self._dtype = dtype
        self._choice = TierChoice(fetch.tier, dtype)
        self._ahead: tuple[int, Columns] | None = None
        #: The step the source expects next (``steps_done`` of the engine).
        self.position = 0

    @property
    def batches_per_pass(self) -> int:
        """Whole batches in one pass of the (tiled) rows: the logging "epoch"."""
        return self._order.batches_per_pass

    @property
    def order(self) -> TrainOrder:
        """The stream (tiling and shuffle) this source serves."""
        return self._order

    def columns(self, step: int) -> Columns:
        """Every column of the batch of the step after ``step`` optimizer steps.

        Raises:
            ValueError: if ``step`` is negative.
        """
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")
        if self._ahead is not None and self._ahead[0] == step:
            raw = self._ahead[1]
        else:
            raw = self._fetch.train(self._order, step)
        self._ahead = None
        if self._fetch.prefetch:
            self._ahead = (step + 1, self._fetch.train(self._order, step + 1))
        self.position = step + 1
        return {**raw, "images": _dequantised(raw["images"], self._dtype)}

    def batch(self, step: int) -> TrainBatch:
        """The classifier batch of step ``step`` (``engine.data.BatchSource.batch``).

        Raises:
            ConfigError: if the dataset has no labels or features.
            ValueError: if ``step`` is negative.
        """
        columns = self.columns(step)
        batch = _classifier_batch(columns, self._n_classes)
        return batch if "fits" not in columns else (*batch, columns["fits"])

    def state(self) -> dict[str, Any]:
        """The stream definition (``TrainOrder.state()`` plus the storage dtype).

        Not the tier: the tiers serve the same values, so a run may resume on another one.
        """
        return {**self._order.state(), "storage_dtype": self._dtype}

    def resume(self, state: dict[str, Any], steps_done: int) -> None:
        """Continue a checkpointed run: check its stream, then expect step ``steps_done``.

        Raises:
            DataError: if ``state`` is another stream (:func:`check_stream`).
            ValueError: if ``steps_done`` is negative.
        """
        if steps_done < 0:
            raise ValueError(f"steps_done must be >= 0, got {steps_done}")
        check_stream(state, self.state())
        self._ahead = None
        self.position = steps_done

    def describe(self) -> dict[str, Any]:
        """The data record for the manifest: tier, storage dtype, deviation, staging, shuffle."""
        return {
            "tier": str(self._choice.tier),
            "storage_dtype": self._dtype,
            "deviation": self._choice.is_deviation,
            "pinned_staging": self._fetch.pinned,
            "prefetch": self._fetch.prefetch,
            "shuffle": self._order.state().get("shuffle", "row"),
        }


class EvalBatches:
    """An evaluation split on one tier: an ``engine.data.EvalSource``.

    Consecutive batches of ``batch_size`` rows in dataset order, the last one partial (with
    64, ``mean_of_batch_losses`` is TF's "mean of batch-64 means in file order"). On the
    host tiers each batch is prefetched while the previous one is in use. Build it with
    :func:`open_eval`.
    """

    def __init__(self, fetch: _Fetch, dataset: Dataset, dtype: StorageDtype, batch_size: int):
        self._fetch = fetch
        self._n = len(dataset)
        self._n_classes = dataset.n_classes
        self._dtype = dtype
        self.batch_size = batch_size

    def __len__(self) -> int:
        """The number of batches."""
        return -(-self._n // self.batch_size)

    def iter_columns(self) -> Iterator[Columns]:
        """Every column of each batch of one pass, in order (an unlabelled kind too)."""
        size = self.batch_size
        pending: Columns | None = None
        for start in range(0, self._n, size):
            raw = self._fetch.span(start, start + size) if pending is None else pending
            pending = None
            if self._fetch.prefetch and start + size < self._n:
                pending = self._fetch.span(start + size, start + 2 * size)
            yield {**raw, "images": _dequantised(raw["images"], self._dtype)}

    def __iter__(self) -> Iterator[Batch]:
        """The classifier batches of one pass.

        Raises:
            ConfigError: if the dataset has no labels or features.
        """
        for columns in self.iter_columns():
            yield _classifier_batch(columns, self._n_classes)


def open_train(
    dataset: Dataset, choice: TierChoice, spec: StreamSpec, device: torch.device | str
) -> TrainSource:
    """The training stream of ``dataset`` on tier ``choice`` and ``device``.

    Raises:
        ValueError: see :meth:`StreamSpec.order` (balance without labels, a stream smaller
            than one batch).
        ConfigError: for ``block`` shuffle with ``balance``, or the ``memmap`` tier with a
            non-float32 storage dtype.
        DataError: for ``uint8`` storage of images outside [0, 1].
    """
    order = spec.order(dataset)
    target = torch.device(device)
    fetch = _fetch(dataset, choice, target, train_rows=spec.batch_size, eval_rows=1)
    return TrainSource(fetch, order, dataset, choice.storage_dtype)


def open_eval(
    dataset: Dataset, choice: TierChoice, device: torch.device | str, *, batch_size: int = 64
) -> EvalBatches:
    """The evaluation batches of ``dataset`` on tier ``choice`` and ``device``.

    Raises:
        ValueError: for a non-positive batch size.
        ConfigError: for the ``memmap`` tier with a non-float32 storage dtype.
        DataError: for ``uint8`` storage of images outside [0, 1].
    """
    if isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    target = torch.device(device)
    fetch = _fetch(dataset, choice, target, train_rows=1, eval_rows=batch_size)
    return EvalBatches(fetch, dataset, choice.storage_dtype, batch_size)
