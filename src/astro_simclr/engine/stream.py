"""The seeded, resumable training stream: class balance, the shuffle and its granularity.

Engine-free (no Lightning), so both engines and every data tier (:mod:`.tiers`) share it;
``training.tiling`` re-exports :func:`tile_indices` and :class:`PermutationStream` from here.

**Class balance** (task 2.2a, Carrasco-Davis et al. 2021). In ALeRCE's code
(``alercebroker/pipeline@a3705ac``, ``stamp_classifier_step/model/``; never copied here),
``Dataset.balance_data_by_replication`` (``modules/data_set_generic.py``) visits the classes
in sorted label order; a class smaller than the largest one is replaced by
``_oversampling_array``: ``max_count // n`` full copies of its stamps, in file order, then
the first ``max_count - copies * n`` of them again (the prefix). The largest class is kept
as it is. :func:`tile_indices` reproduces it exactly, on indices.

**The shuffle.** ALeRCE's training iterator is ``tf.data``
``shuffle(buffer_size=100000).repeat().batch(64)`` (``modules/iterators/train_iterator.py``,
``drop_remainder=False``). The buffer holds the whole tiled set (72,710 < 100,000 for
``red``), so each pass is a uniform random permutation, reshuffled on every repetition, and
batches are cut from the concatenated passes: a batch may end one pass and start the next,
and no stamp is dropped. :class:`PermutationStream` reproduces it in distribution (not TF's
random numbers): batch ``b`` holds the stream positions ``[b * batch_size, (b + 1) *
batch_size)``, and pass ``j`` is the permutation drawn by ``numpy.random.default_rng([seed,
j])``. A batch is therefore a pure function of ``(seed, b)``: a run resumed at any step
sees the same batches as an uninterrupted one. This is a documented exception to
"randomness comes from ``seed_everything``" (``docs/architecture.md``): the stream takes the
run seed explicitly.

**Shuffle granularity** (ADR 0016), the one choice that changes which items share a batch:

- ``row`` (:class:`PermutationStream`, the default and the paper's): every pass is a uniform
  permutation of all positions, so any two items are equally likely to share a batch. On
  the ``memmap`` tier a batch is then ``batch_size`` scattered reads (sorted by file offset
  before reading, :mod:`.tiers`).
- ``block`` (:class:`BlockShuffleStream`, for data read from a disk too slow for scattered
  reads): the positions are cut into blocks of ``block_rows`` consecutive positions (the
  last block may be short); each pass permutes the blocks, then shuffles the positions
  inside each run of ``window_blocks`` consecutive blocks of that order. A batch then draws
  from at most two windows, each ``window_blocks`` contiguous file ranges. The price: items
  of one block always land in the same window of a pass, so neighbours in the file (for an
  alert corpus, alerts of one night) share batches far more often than under ``row``.
  ``block`` is refused on a tiled (class-balanced) stream (:class:`TrainOrder`): the tiled
  order is sorted by class, so its blocks would be single-class.

Both are pure functions of the stream definition and the batch number. The definition
(:meth:`PermutationStream.state`) is what a checkpoint stores and a resume checks; the only
mutable state of a stream is its position, which is the engine's ``steps_done``.
"""

from __future__ import annotations

import hashlib
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt

from astro_simclr.config import ConfigError
from astro_simclr.engine.randomness import RESERVED_STREAMS

__all__ = [
    "BLOCK_STREAM_WORD",
    "SHUFFLES",
    "BlockShuffleStream",
    "PermutationStream",
    "Shuffle",
    "TrainOrder",
    "tile_indices",
]

#: The shuffle granularities (module docstring).
type Shuffle = Literal["row", "block"]
SHUFFLES: Final[tuple[Shuffle, ...]] = ("row", "block")
#: The third ``SeedSequence`` word of a block stream's passes
#: (``RESERVED_STREAMS["block_shuffle"]``), so that it shares no generator with a row
#: stream or an augmentation of the same seed.
BLOCK_STREAM_WORD: Final = RESERVED_STREAMS["block_shuffle"]


def tile_indices(labels: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Indices of a class-balanced Train split, exactly as ALeRCE's tiling.

    For each class, in sorted label order: its indices in file order if it is the largest
    class; otherwise ``max_count // n`` full copies of them followed by their first
    ``max_count % n`` (ALeRCE ``_oversampling_array``). The result has
    ``n_classes * max_count`` entries, ``max_count`` per class.

    Example: labels ``[0, 0, 0, 0, 0, 1, 1, 2]`` give
    ``[0, 1, 2, 3, 4, 5, 6, 5, 6, 5, 7, 7, 7, 7, 7]``.

    Raises:
        ValueError: if ``labels`` is empty or not one-dimensional.
    """
    array = np.asarray(labels)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"labels must be a non-empty 1-D array, got shape {array.shape}")
    classes, counts = np.unique(array, return_counts=True)
    max_count = int(counts.max())
    parts: list[npt.NDArray[np.int64]] = []
    for label in classes:
        members = np.flatnonzero(array == label).astype(np.int64)
        copies, extra = divmod(max_count, members.size)
        parts.append(np.concatenate([np.tile(members, copies), members[:extra]]))
    return np.concatenate(parts)


class PermutationStream:
    """The batches of a shuffled, repeated, unbroken stream over ``n`` items (``row`` shuffle).

    ``batch(b)`` gives the item indices of batch ``b`` (0-based, counted over the whole
    run): the stream positions ``b * batch_size ... (b + 1) * batch_size - 1``, where
    position ``p`` is item ``permutation(p // n)[p % n]``. ``batches_per_pass`` (``n //
    batch_size``) is the length of one "epoch" for Lightning; because the stream is unbroken,
    epochs after the first start part-way into a pass, as in TF.

    Raises:
        ValueError: if ``n`` or ``batch_size`` is not positive, ``batch_size > n``, or
            ``seed`` is negative.
    """

    def __init__(self, n: int, batch_size: int, seed: int) -> None:
        if n < 1 or batch_size < 1:
            raise ValueError(f"n and batch_size must be positive, got {n} and {batch_size}")
        if batch_size > n:
            raise ValueError(f"batch_size {batch_size} is larger than the {n} items")
        if seed < 0:
            raise ValueError(f"seed must be >= 0, got {seed}")
        self.n = n
        self.batch_size = batch_size
        self.seed = seed

    @property
    def batches_per_pass(self) -> int:
        """Whole batches in one pass (``n // batch_size``): the Lightning epoch length."""
        return self.n // self.batch_size

    def permutation(self, index: int) -> npt.NDArray[np.int64]:
        """Pass ``index``: a permutation of ``range(n)`` from ``default_rng([seed, index])``."""
        if index < 0:
            raise ValueError(f"pass index must be >= 0, got {index}")
        return np.random.default_rng([self.seed, index]).permutation(self.n).astype(np.int64)

    def span(self, batch: int) -> list[tuple[int, int, int]]:
        """Where batch ``batch`` lies in the stream: ``(pass, start, stop)`` slices, in order.

        One slice, or two when the batch crosses the end of a pass.

        Raises:
            ValueError: if ``batch`` is negative.
        """
        if batch < 0:
            raise ValueError(f"batch index must be >= 0, got {batch}")
        first = batch * self.batch_size
        last = first + self.batch_size  # exclusive
        slices = []
        position = first
        while position < last:
            index, start = divmod(position, self.n)
            stop = min(self.n, start + last - position)
            slices.append((index, start, stop))
            position += stop - start
        return slices

    def batch(self, batch: int) -> npt.NDArray[np.int64]:
        """The ``batch_size`` item indices of batch ``batch``."""
        return np.concatenate(
            [self.permutation(index)[start:stop] for index, start, stop in self.span(batch)]
        )

    def state(self) -> dict[str, Any]:
        """What defines the stream (``n``, ``batch_size``, ``seed``), for checkpoints.

        The keys are those of task 2.2a's Lightning checkpoints (``device_data``), so a
        ``row`` stream's definition is unchanged; :class:`BlockShuffleStream` adds its own.
        """
        return {"n": self.n, "batch_size": self.batch_size, "seed": self.seed}


class BlockShuffleStream(PermutationStream):
    """A stream with ``block`` shuffle granularity (module docstring).

    Pass ``j`` draws from ``default_rng([seed, j, BLOCK_STREAM_WORD])``: first the
    permutation of the ``ceil(n / block_rows)`` blocks, then, in order, one in-place shuffle
    of each window of ``window_blocks * block_rows`` consecutive positions of the block
    order (the last window may be short).

    Raises:
        ValueError: as :class:`PermutationStream`, or if ``block_rows`` or
            ``window_blocks`` is not positive.
    """

    def __init__(
        self, n: int, batch_size: int, seed: int, *, block_rows: int, window_blocks: int
    ) -> None:
        super().__init__(n, batch_size, seed)
        if block_rows < 1 or window_blocks < 1:
            msg = (
                f"block_rows and window_blocks must be positive, got {block_rows} "
                f"and {window_blocks}"
            )
            raise ValueError(msg)
        self.block_rows = block_rows
        self.window_blocks = window_blocks

    def permutation(self, index: int) -> npt.NDArray[np.int64]:
        """Pass ``index``: blocks permuted, then positions shuffled within each window."""
        if index < 0:
            raise ValueError(f"pass index must be >= 0, got {index}")
        rng = np.random.default_rng([self.seed, index, BLOCK_STREAM_WORD])
        n_blocks = -(-self.n // self.block_rows)
        starts = rng.permutation(n_blocks).astype(np.int64) * self.block_rows
        order = np.concatenate([np.arange(s, min(s + self.block_rows, self.n)) for s in starts])
        window = self.window_blocks * self.block_rows
        for start in range(0, self.n, window):
            rng.shuffle(order[start : start + window])
        return order.astype(np.int64)

    def state(self) -> dict[str, Any]:
        """The row definition plus ``shuffle``, ``block_rows`` and ``window_blocks``."""
        return {
            **super().state(),
            "shuffle": "block",
            "block_rows": self.block_rows,
            "window_blocks": self.window_blocks,
        }


def _rows_digest(rows: npt.NDArray[np.int64]) -> str:
    """SHA-256 of the little-endian int64 bytes of ``rows`` (the position -> row map)."""
    return hashlib.sha256(np.ascontiguousarray(rows, dtype="<i8").tobytes()).hexdigest()


class TrainOrder:
    """The rows of every training batch: an optional tiling, then a stream over its positions.

    ``n_rows`` items (the rows of a dataset, :mod:`.dataset`) are first mapped to positions:
    ``tile_indices(labels)`` with ``balance=True`` (the paper), or the identity. The stream
    (``row`` or ``block``, :data:`SHUFFLES`) runs over the positions, and :meth:`rows` maps
    the positions of stream batch ``b`` back to dataset rows. Each pass's row order is
    computed once and cached (the current and the next pass are kept), because a
    permutation of a large corpus costs more than a step.

    :meth:`state` is the definition a checkpoint stores: the stream's keys plus
    ``rows_sha256``, the digest of the position -> row map, which catches a resume on other
    labels with the same tiled size (the device-resident Lightning loader of 2.2a checked
    the size only).

    Raises:
        ValueError: for a non-positive batch size, a negative seed, an unknown shuffle,
            labels of another length than ``n_rows``, ``balance`` without labels, or a
            stream smaller than one batch.
        ConfigError: for ``block`` shuffle on a balanced (tiled) stream.
    """

    def __init__(
        self,
        n_rows: int,
        *,
        batch_size: int,
        seed: int,
        labels: npt.ArrayLike | None = None,
        balance: bool = False,
        shuffle: Shuffle = "row",
        block_rows: int = 4096,
        window_blocks: int = 16,
    ) -> None:
        if shuffle not in SHUFFLES:
            raise ValueError(f"shuffle must be one of {SHUFFLES}, got {shuffle!r}")
        if balance:
            if labels is None:
                raise ValueError("balance=True needs the labels")
            label_array = np.asarray(labels)
            if label_array.shape != (n_rows,):
                raise ValueError(f"labels must have shape ({n_rows},), got {label_array.shape}")
            if shuffle == "block":
                msg = (
                    "shuffle='block' on a class-balanced stream would make single-class "
                    "batches (the tiled order is sorted by class): use shuffle='row'"
                )
                raise ConfigError(msg)
            self.positions = tile_indices(label_array)
        else:
            self.positions = np.arange(n_rows, dtype=np.int64)
        n = len(self.positions)
        if shuffle == "block":
            self.stream: PermutationStream = BlockShuffleStream(
                n, batch_size, seed, block_rows=block_rows, window_blocks=window_blocks
            )
        else:
            self.stream = PermutationStream(n, batch_size, seed)
        self._digest = _rows_digest(self.positions)
        self._passes: dict[int, npt.NDArray[np.int64]] = {}

    @property
    def batch_size(self) -> int:
        """Rows per training batch."""
        return self.stream.batch_size

    @property
    def batches_per_pass(self) -> int:
        """Whole batches in one pass of the (tiled) positions: the logging "epoch"."""
        return self.stream.batches_per_pass

    def pass_rows(self, index: int) -> npt.NDArray[np.int64]:
        """The dataset rows of pass ``index``, in stream order (cached; read-only)."""
        if index not in self._passes:
            self._passes = {k: v for k, v in self._passes.items() if k >= index - 1}
            rows = self.positions[self.stream.permutation(index)]
            rows.setflags(write=False)
            self._passes[index] = rows
        return self._passes[index]

    def rows(self, batch: int) -> npt.NDArray[np.int64]:
        """The ``batch_size`` dataset rows of stream batch ``batch`` (the step's batch)."""
        parts = [self.pass_rows(i)[start:stop] for i, start, stop in self.stream.span(batch)]
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    def state(self) -> dict[str, Any]:
        """The stream definition plus ``rows_sha256`` (class docstring)."""
        return {**self.stream.state(), "rows_sha256": self._digest}
