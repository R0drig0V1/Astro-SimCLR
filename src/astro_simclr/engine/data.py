"""The batch format and the three data tiers of the plain engine (ADR 0012, task 4.3).

**Batch format** (frozen in B0): :data:`Batch` is ``(images, features, one-hot labels)``
and :data:`TrainBatch` adds the per-stamp fits that C5-C7 need, exactly
``training.device_data.Batch`` / ``TrainBatch`` (``tests/test_engine_contracts.py`` checks
that they stay equal; they are restated here because ``device_data`` imports Lightning).

**Tiers** (:class:`DataTier`), each a :class:`BatchSource` whose batch of step ``s`` is a
pure function of the stream definition and ``s``:

1. ``device``: every split resident on the training device, batches gathered there by
   index (today's ``DeviceTensorDataModule``). Exact.
2. ``pinned``: float32 splits in pinned host memory, a per-step gather on the host and a
   non-blocking upload, prefetched one step ahead. Exact: the same float32 values reach the
   same ops, only their location changes.
3. ``memmap``: float32 splits in a disk memmap read in shards (phase 5, data larger than
   host memory). Exact for the same reason.

Quantised storage (``uint8``, ``float16``) is never chosen by the engine to make data fit:
it is a :class:`TierChoice` with a non-float32 ``storage_dtype``, which is a recorded preset
deviation (:attr:`TierChoice.is_deviation`), never an automatic fallback.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, Protocol

import torch

from astro_simclr.config import ConfigError

__all__ = [
    "Batch",
    "BatchSource",
    "DataTier",
    "EvalSource",
    "StorageDtype",
    "TierChoice",
    "TrainBatch",
]

#: (crops, features, one-hot labels): ``training.device_data.Batch``.
Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
#: A training batch, plus the stamp fits when the Train split has them:
#: ``training.device_data.TrainBatch``.
TrainBatch = Batch | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
#: How the stored images are held; anything but ``float32`` is a recorded deviation.
StorageDtype = Literal["float32", "float16", "uint8"]


class DataTier(StrEnum):
    """Where the splits live while a run trains (module docstring)."""

    DEVICE = "device"
    PINNED = "pinned"
    MEMMAP = "memmap"


@dataclass(frozen=True, slots=True)
class TierChoice:
    """The tier and storage dtype of a run, chosen explicitly by its config.

    Raises:
        ConfigError: for an unknown tier or storage dtype.
    """

    tier: DataTier = DataTier.DEVICE
    storage_dtype: StorageDtype = "float32"

    def __post_init__(self) -> None:
        if not isinstance(self.tier, DataTier):
            raise ConfigError(f"unknown data tier {self.tier!r}")
        if self.storage_dtype not in ("float32", "float16", "uint8"):
            raise ConfigError(f"unknown storage dtype {self.storage_dtype!r}")

    @property
    def is_deviation(self) -> bool:
        """Whether the stored values are quantised (a preset deviation to record)."""
        return self.storage_dtype != "float32"


class BatchSource(Protocol):
    """The training stream: the batch of every optimizer step, on the training device."""

    @property
    def batches_per_pass(self) -> int:
        """Whole batches in one pass of the (tiled) Train split: the logging "epoch"."""

    def batch(self, step: int) -> TrainBatch:
        """The batch of the step after ``step`` optimizer steps (stream batch ``step``).

        A pure function of the stream definition and ``step``, so a resume at step ``s``
        continues with exactly the batches of an uninterrupted run.
        """

    def state(self) -> dict[str, Any]:
        """The stream definition (item count, batch size, seed, run identity) to check on resume."""


class EvalSource(Protocol):
    """An evaluation split: consecutive batches in file order, the last one partial."""

    def __iter__(self) -> Iterator[Batch]:
        """The batches of one pass, on the training device."""

    def __len__(self) -> int:
        """The number of batches."""
