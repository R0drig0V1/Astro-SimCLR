"""Throughput microbenchmark of the data tiers (task 4.3): batches per second, per tier.

``python -m astro_simclr.engine.bench --device cpu --device mps`` builds a synthetic labelled
dataset (random float32 images of ``--shape``, 23 features, 5 classes), writes its images to
a ``.npy`` file for the ``memmap`` tier, and times ``--steps`` training batches of each tier
(``TrainSource.columns``), after ``--warmup`` untimed ones. Each batch is consumed by one
reduction on the device, so the host cannot run ahead of the uploads unboundedly, and the
device is synchronised before the clock stops.

**Interleaved rounds.** The tiers run in turn inside each of ``--rounds`` rounds, the order
rotated every round, and the median over rounds is reported: on a machine shared with other
jobs (the development Mac runs several agents), sequential before/after timings measure the
load, not the tier.

**What it does not measure.** The ``memmap`` file was just written, so it is read from the
page cache: the number is the tier's own overhead (sorted reads, scatter, staging), not the
disk's. A cold-disk figure needs a file larger than the RAM, on the GPU machine (phase 5).
The model's step is not included: the question is whether the data path keeps up with it.

The result is printed as a table and, with ``--out``, written as JSON with the machine.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from astro_simclr.engine.data import DataTier, TierChoice
from astro_simclr.engine.dataset import Dataset
from astro_simclr.engine.tiers import StreamSpec, TrainSource, open_train

__all__ = ["TierTiming", "bench_tiers", "main"]


@dataclass(frozen=True, slots=True)
class TierTiming:
    """One tier on one device: the median batches per second over the rounds.

    Attributes:
        tier: the tier's name.
        device: the training device.
        image_shape: the per-row image shape.
        batch_size: rows per batch.
        batches_per_s: the median over rounds.
        mb_per_s: the image megabytes per second at that median (10^6 bytes).
        rounds: batches per second of every round, in order.
        build_s: seconds to build the source (materialise, upload, pin).
        describe: ``TrainSource.describe()`` of the source.
    """

    tier: str
    device: str
    image_shape: tuple[int, ...]
    batch_size: int
    batches_per_s: float
    mb_per_s: float
    rounds: tuple[float, ...]
    build_s: float
    describe: dict[str, object]


def _synchronize(device: torch.device) -> None:
    if device.type != "cpu":
        torch.accelerator.synchronize(device)


def _time(source: TrainSource, device: torch.device, start: int, steps: int) -> float:
    """Batches per second of ``steps`` batches from stream batch ``start``."""
    total = torch.zeros((), device=device)
    _synchronize(device)
    began = time.perf_counter()
    for step in range(start, start + steps):
        total += source.columns(step)["images"].sum()
    _synchronize(device)
    return steps / (time.perf_counter() - began)


def bench_tiers(
    devices: Sequence[str],
    *,
    n: int = 20_000,
    image_shape: tuple[int, ...] = (3, 21, 21),
    batch_size: int = 64,
    steps: int = 200,
    warmup: int = 20,
    rounds: int = 5,
    workdir: Path,
    tiers: Sequence[DataTier] = tuple(DataTier),
    seed: int = 0,
) -> list[TierTiming]:
    """Time every tier of ``tiers`` on every device of ``devices`` (module docstring).

    ``workdir`` holds the ``memmap`` tier's image file.

    Raises:
        ValueError: for a non-positive count, or ``n`` smaller than one batch.
    """
    for name, value in (("n", n), ("steps", steps), ("rounds", rounds), ("batch", batch_size)):
        if value < 1:
            raise ValueError(f"{name} must be positive, got {value}")
    rng = np.random.default_rng(seed)
    images = rng.random((n, *image_shape), dtype=np.float32)
    path = workdir / "bench_images.npy"
    np.save(path, images)
    memmap = np.load(path, mmap_mode="r")
    features = rng.random((n, 23), dtype=np.float32)
    labels = rng.integers(0, 5, n).astype(np.int64)
    spec = StreamSpec(batch_size=batch_size, seed=seed, balance=False)
    row_mb = images[0].nbytes * batch_size / 1e6
    results: list[TierTiming] = []
    for name in devices:
        device = torch.device(name)
        sources: dict[DataTier, TrainSource] = {}
        build: dict[DataTier, float] = {}
        for tier in tiers:
            stored = memmap if tier is DataTier.MEMMAP else images
            dataset = Dataset(images=stored, features=features, labels=labels, n_classes=5)
            began = time.perf_counter()
            sources[tier] = open_train(dataset, TierChoice(tier), spec, device)
            _synchronize(device)
            build[tier] = time.perf_counter() - began
        timings: dict[DataTier, list[float]] = {tier: [] for tier in tiers}
        position = dict.fromkeys(tiers, 0)
        for index in range(rounds):
            turn = list(tiers[index % len(tiers) :]) + list(tiers[: index % len(tiers)])
            for tier in turn:
                source = sources[tier]
                _time(source, device, position[tier], warmup)
                position[tier] += warmup
                timings[tier].append(_time(source, device, position[tier], steps))
                position[tier] += steps
        for tier in tiers:
            median = statistics.median(timings[tier])
            results.append(
                TierTiming(
                    tier=str(tier),
                    device=name,
                    image_shape=tuple(image_shape),
                    batch_size=batch_size,
                    batches_per_s=median,
                    mb_per_s=median * row_mb,
                    rounds=tuple(timings[tier]),
                    build_s=build[tier],
                    describe=sources[tier].describe(),
                )
            )
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--device", action="append", help="repeatable; default: cpu (+ mps)")
    parser.add_argument("--n", type=int, default=20_000, help="rows of the synthetic dataset")
    parser.add_argument("--shape", default="3,21,21", help="per-row image shape, e.g. 3,63,63")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200, help="timed batches per round")
    parser.add_argument("--warmup", type=int, default=20, help="untimed batches per round")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--out", type=Path, help="write the results as JSON here")
    return parser


def main(argv: Sequence[str] | None = None) -> list[TierTiming]:
    """Run the benchmark from the command line (module docstring) and return its results."""
    args = _parser().parse_args(argv)
    devices = args.device or ["cpu", *(["mps"] if torch.backends.mps.is_available() else [])]
    shape = tuple(int(part) for part in args.shape.split(","))
    with tempfile.TemporaryDirectory() as workdir:
        results = bench_tiers(
            devices,
            n=args.n,
            image_shape=shape,
            batch_size=args.batch_size,
            steps=args.steps,
            warmup=args.warmup,
            rounds=args.rounds,
            workdir=Path(workdir),
        )
    print(f"{'device':<7}{'tier':<8}{'batches/s':>11}{'MB/s':>9}{'build s':>9}  rounds")
    for r in results:
        spread = " ".join(f"{value:.0f}" for value in r.rounds)
        line = f"{r.device:<7}{r.tier:<8}{r.batches_per_s:>11.0f}{r.mb_per_s:>9.0f}"
        print(f"{line}{r.build_s:>9.2f}  {spread}")
    if args.out is not None:
        record = {
            "machine": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "torch": torch.__version__,
            },
            "args": {key: str(value) for key, value in vars(args).items()},
            "results": [asdict(r) for r in results],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n")
    return results


if __name__ == "__main__":  # pragma: no cover - the command-line entry point
    main()
