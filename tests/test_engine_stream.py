"""Tests of ``engine.stream``: the training order, its definition and the shuffle granularity."""

from __future__ import annotations

import numpy as np
import pytest

from astro_simclr.config import ConfigError
from astro_simclr.engine import RESERVED_STREAMS
from astro_simclr.engine.stream import (
    BLOCK_STREAM_WORD,
    BlockShuffleStream,
    PermutationStream,
    TrainOrder,
    tile_indices,
)
from astro_simclr.training import tiling

_LABELS = np.array([0, 0, 0, 0, 0, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4])  # tiled: 25 positions


def test_training_tiling_re_exports_the_engine_s_stream() -> None:
    assert tiling.PermutationStream is PermutationStream
    assert tiling.tile_indices is tile_indices


def test_balanced_order_is_the_tiled_permutation_stream() -> None:
    order = TrainOrder(len(_LABELS), batch_size=4, seed=3, labels=_LABELS, balance=True)
    tiled = tile_indices(_LABELS)
    stream = PermutationStream(25, 4, 3)
    assert order.batches_per_pass == 6
    assert order.batch_size == 4
    for batch in range(20):  # several passes, batches across pass boundaries
        assert np.array_equal(order.rows(batch), tiled[stream.batch(batch)])


def test_unbalanced_order_streams_the_rows() -> None:
    order = TrainOrder(10, batch_size=3, seed=1)
    stream = PermutationStream(10, 3, 1)
    for batch in range(12):
        assert np.array_equal(order.rows(batch), stream.batch(batch))
    assert order.state() == {**stream.state(), "rows_sha256": order.state()["rows_sha256"]}


def test_pass_rows_are_cached_read_only_and_bounded() -> None:
    order = TrainOrder(10, batch_size=3, seed=1)
    first = order.pass_rows(0)
    assert order.pass_rows(0) is first
    assert not first.flags.writeable
    for index in range(1, 6):
        order.pass_rows(index)
    assert len(order._passes) <= 2  # the current and the previous pass only


def test_definition_names_the_position_to_row_map() -> None:
    order = TrainOrder(len(_LABELS), batch_size=4, seed=0, labels=_LABELS, balance=True)
    # Another labelling with the same tiled size (25) is another stream.
    other_labels = _LABELS[::-1].copy()
    other = TrainOrder(len(_LABELS), batch_size=4, seed=0, labels=other_labels, balance=True)
    assert len(other.positions) == len(order.positions)
    assert order.state()["rows_sha256"] != other.state()["rows_sha256"]
    assert set(order.state()) == {"n", "batch_size", "seed", "rows_sha256"}


def test_order_validates() -> None:
    with pytest.raises(ValueError, match="needs the labels"):
        TrainOrder(5, batch_size=2, seed=0, balance=True)
    with pytest.raises(ValueError, match="shape"):
        TrainOrder(5, batch_size=2, seed=0, labels=[0, 1], balance=True)
    with pytest.raises(ValueError, match="shuffle"):
        TrainOrder(5, batch_size=2, seed=0, shuffle="pixel")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="larger"):
        TrainOrder(5, batch_size=6, seed=0)
    with pytest.raises(ConfigError, match="single-class"):
        TrainOrder(
            len(_LABELS), batch_size=2, seed=0, labels=_LABELS, balance=True, shuffle="block"
        )


def test_block_stream_permutes_blocks_then_shuffles_windows() -> None:
    n, block, window = 103, 10, 3
    stream = BlockShuffleStream(n, 8, 5, block_rows=block, window_blocks=window)
    for index in range(3):
        order = stream.permutation(index)
        assert sorted(order.tolist()) == list(range(n))
        # Each window of 30 positions holds whole blocks (the last block, 100..102, is short).
        cursor = 0
        while cursor < n:
            rows = order[cursor : cursor + window * block]
            blocks = set((rows // block).tolist())
            assert len(blocks) <= window + 1  # a short block shifts the cut by < one block
            cursor += window * block
    assert not np.array_equal(stream.permutation(0), stream.permutation(1))
    assert np.array_equal(stream.permutation(2), stream.permutation(2))


def test_block_windows_hold_whole_blocks_when_blocks_divide_n() -> None:
    stream = BlockShuffleStream(120, 8, 0, block_rows=10, window_blocks=4)
    order = stream.permutation(0)
    for start in range(0, 120, 40):
        blocks = order[start : start + 40] // 10
        assert len(set(blocks.tolist())) == 4
        assert np.all(np.bincount(blocks, minlength=12)[np.unique(blocks)] == 10)
        # Inside a window the rows are shuffled, not left in block order.
    assert not np.array_equal(order[:40], np.sort(order[:40]))


def test_block_stream_definition_and_generator() -> None:
    stream = BlockShuffleStream(50, 4, 2, block_rows=8, window_blocks=2)
    assert stream.state() == {
        "n": 50,
        "batch_size": 4,
        "seed": 2,
        "shuffle": "block",
        "block_rows": 8,
        "window_blocks": 2,
    }
    assert RESERVED_STREAMS["block_shuffle"] == BLOCK_STREAM_WORD
    # The pass generator is not the row stream's [seed, pass]: the orders differ.
    assert not np.array_equal(stream.permutation(0), PermutationStream(50, 4, 2).permutation(0))
    order = TrainOrder(50, batch_size=4, seed=2, shuffle="block", block_rows=8, window_blocks=2)
    assert order.state()["shuffle"] == "block"
    for batch in range(30):
        assert np.array_equal(order.rows(batch), stream.batch(batch))


def test_block_stream_validates() -> None:
    with pytest.raises(ValueError, match="block_rows"):
        BlockShuffleStream(10, 2, 0, block_rows=0, window_blocks=1)
    with pytest.raises(ValueError, match="pass index"):
        BlockShuffleStream(10, 2, 0, block_rows=2, window_blocks=1).permutation(-1)
