"""Tests of ``training.tiling``: ALeRCE's class balance and the shuffled batch stream (2.2a)."""

from __future__ import annotations

import numpy as np
import pytest

from astro_simclr.training.tiling import PermutationStream, tile_indices


def test_tiling_documented_example() -> None:
    # Class 1 (2 stamps): 2 full copies and a prefix of 1; class 2 (1 stamp): 5 copies.
    labels = [0, 0, 0, 0, 0, 1, 1, 2]
    expected = [0, 1, 2, 3, 4, 5, 6, 5, 6, 5, 7, 7, 7, 7, 7]
    assert tile_indices(labels).tolist() == expected


def test_tiling_visits_classes_in_label_order_with_file_order_inside() -> None:
    labels = np.array([2, 0, 2, 1, 0, 2, 2, 1, 0])  # counts 3, 2, 4; interleaved in the file
    tiled = tile_indices(labels)
    assert tiled.dtype == np.int64
    assert tiled.tolist() == [1, 4, 8, 1, 3, 7, 3, 7, 0, 2, 5, 6]


def test_tiling_balances_exactly() -> None:
    rng = np.random.default_rng(0)
    labels = rng.choice(5, size=997, p=[0.4, 0.03, 0.3, 0.17, 0.1])
    tiled = tile_indices(labels)
    counts = np.bincount(labels[tiled], minlength=5)
    assert np.all(counts == np.bincount(labels).max())
    # Each class block is full copies of its members, then a prefix of them.
    for label in range(5):
        members = np.flatnonzero(labels == label)
        block = tiled[labels[tiled] == label]
        repeated = np.resize(members, block.size)  # numpy repeats from the start: copies + prefix
        assert np.array_equal(block, repeated)


@pytest.mark.parametrize("labels", [[], [[0, 1]]])
def test_tiling_rejects_bad_labels(labels: list[int] | list[list[int]]) -> None:
    with pytest.raises(ValueError, match="1-D"):
        tile_indices(labels)


def test_stream_passes_are_seeded_permutations() -> None:
    stream = PermutationStream(n=11, batch_size=4, seed=5)
    first = stream.permutation(0)
    assert sorted(first.tolist()) == list(range(11))
    assert np.array_equal(first, PermutationStream(11, 4, 5).permutation(0))
    assert not np.array_equal(first, stream.permutation(1))
    assert not np.array_equal(first, PermutationStream(11, 4, 6).permutation(0))
    assert stream.batches_per_pass == 2
    assert stream.state() == {"n": 11, "batch_size": 4, "seed": 5}


def test_stream_batches_cut_the_unbroken_stream() -> None:
    stream = PermutationStream(n=11, batch_size=4, seed=1)
    unbroken = np.concatenate([stream.permutation(j) for j in range(4)])
    for batch in range(10):
        assert np.array_equal(stream.batch(batch), unbroken[4 * batch : 4 * batch + 4])
    # Batch 2 covers positions 8-11: the end of pass 0 and the start of pass 1.
    assert stream.span(2) == [(0, 8, 11), (1, 0, 1)]
    assert stream.span(0) == [(0, 0, 4)]


def test_stream_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="positive"):
        PermutationStream(0, 1, 0)
    with pytest.raises(ValueError, match="larger"):
        PermutationStream(3, 4, 0)
    with pytest.raises(ValueError, match="seed"):
        PermutationStream(3, 1, -1)
    stream = PermutationStream(3, 1, 0)
    with pytest.raises(ValueError, match="pass index"):
        stream.permutation(-1)
    with pytest.raises(ValueError, match="batch index"):
        stream.span(-1)
