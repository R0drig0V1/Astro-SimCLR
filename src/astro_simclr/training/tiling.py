"""Class balance by deterministic tiling and ALeRCE's shuffled batch stream (task 2.2a).

Carrasco-Davis et al. 2021 balance the Train split before training, and draw batches from a
shuffled, repeated stream. In ALeRCE's code (``alercebroker/pipeline@a3705ac``,
``stamp_classifier_step/model/``; never copied here):

- ``Dataset.balance_data_by_replication`` (``modules/data_set_generic.py``) visits the
  classes in sorted label order; a class smaller than the largest one is replaced by
  ``_oversampling_array``: ``max_count // n`` full copies of its stamps, in file order,
  then the first ``max_count - copies * n`` of them again (the prefix). The largest class
  is kept as it is. Every class then has exactly ``max_count`` entries.
- The training iterator is ``tf.data`` ``shuffle(buffer_size=100000).repeat().batch(64)``
  (``modules/iterators/train_iterator.py``, ``drop_remainder=False``). The buffer holds the
  whole tiled set (72,710 < 100,000 for ``red``), so each pass is a uniform random
  permutation, reshuffled on every repetition, and batches are cut from the concatenated
  passes: a batch may end one pass and start the next, and no stamp is dropped.

:func:`tile_indices` reproduces the first exactly, on indices. :class:`PermutationStream`
reproduces the second in distribution (not TF's random numbers): batch ``b`` holds the
stream positions ``[b * batch_size, (b + 1) * batch_size)``, and pass ``j`` is the
permutation drawn by ``numpy.random.default_rng([seed, j])``. A batch is therefore a pure
function of ``(seed, b)``: a run resumed at any step sees the same batches as an
uninterrupted one. This is a documented exception to "randomness comes from
``seed_everything``" (``docs/architecture.md``): the stream takes the run seed explicitly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["PermutationStream", "tile_indices"]


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
    """The batches of a shuffled, repeated, unbroken stream over ``n`` items.

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
        """What defines the stream (``n``, ``batch_size``, ``seed``), for checkpoints."""
        return {"n": self.n, "batch_size": self.batch_size, "seed": self.seed}
