"""Throughput of the paper model and its 2.5-models variants (plan task 2.5, B7 / B2' / B1').

For every variant of ``astro_simclr.paper_variants.VARIANTS`` (the replication ``rot4``,
the G-CNNs and the control arms) this prints, and with ``--out`` writes as JSON:

- the trainable parameters and the MACs per stamp (``count_parameters``, ``count_macs``);
- on the CPU, with the default thread count and with 1 thread: the median latency of a
  batch of 1 (ms) and the throughput at batch 512 (stamps/s);
- on the accelerator torch reports (``torch.accelerator``: CUDA or MPS), if any and unless
  ``--cpu-only``: the batch-1 latency and the throughput at batch 512 and 4096.

Every measurement is a median over ``--reps`` timed calls after warm-up, in eval mode under
``torch.inference_mode``, float32, on random weights and inputs (the speed does not depend
on them). The group layers expand their kernels on every call (no cache), so the batch-1
numbers include that cost; an exported (ONNX, constant-folded) model would not pay it. The
deployment numbers of the plan (x86 CPU with ONNX Runtime) are not measured here.

Run from the repository root, on an idle machine (do not run it on a GPU that is training)::

    uv run python scripts/bench_paper_models.py [--quick] [--cpu-only] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from astro_simclr.config import settings_from_mapping
from astro_simclr.paper_variants import (
    VARIANTS,
    build_paper_model,
    count_macs,
    count_parameters,
    paper_in_channels,
)
from astro_simclr.presets import CARRASCO2021_PRESET, PRESETS


def _inputs(n: int, channels: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    images = torch.rand(n, channels, 21, 21, generator=generator)
    features = torch.randn(n, 23, generator=generator)
    return images.to(device), features.to(device)


def _median_seconds(
    call: Callable[[], object], reps: int, warmup: int, sync: Callable[[], None]
) -> float:
    for _ in range(warmup):
        call()
    sync()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        call()
        sync()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def _speed(
    model: torch.nn.Module,
    channels: int,
    device: torch.device,
    batches: tuple[int, ...],
    reps: int,
    sync: Callable[[], None],
) -> dict[str, float]:
    out: dict[str, float] = {}
    model.to(device)
    with torch.inference_mode():
        for n in batches:
            images, features = _inputs(n, channels, device)
            seconds = _median_seconds(
                lambda x=images, f=features: model(x, f),
                reps if n == 1 else max(3, reps // 10),
                5,
                sync,
            )
            if n == 1:
                out["b1_ms"] = 1e3 * seconds
            else:
                out[f"b{n}_stamps_per_s"] = n / seconds
    model.to("cpu")
    return out


def _no_sync() -> None:
    return None


def _cpu(model: torch.nn.Module, channels: int, threads: int, reps: int) -> dict[str, float]:
    default = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        return _speed(model, channels, torch.device("cpu"), (1, 512), reps, _no_sync)
    finally:
        torch.set_num_threads(default)


def _accelerator() -> torch.device | None:
    if not torch.accelerator.is_available():
        return None
    return torch.accelerator.current_accelerator()


def bench(*, reps: int, cpu_only: bool) -> dict[str, Any]:
    """Measure every variant; returns ``{"meta": ..., "rows": {name: ...}}``."""
    threads = torch.get_num_threads()
    device = None if cpu_only else _accelerator()
    rows: dict[str, Any] = {}
    for name, variant in VARIANTS.items():
        settings = settings_from_mapping({**PRESETS[CARRASCO2021_PRESET], **variant.overrides})
        channels = paper_in_channels(settings.context)
        torch.manual_seed(0)
        model = build_paper_model(
            settings.paper_arch,
            group_fields=settings.group_fields,
            context=settings.context,
            with_features=settings.with_features,
        ).eval()
        row: dict[str, Any] = {
            "row": variant.row,
            "overrides": dict(variant.overrides),
            "parameters": count_parameters(model),
            "macs": count_macs(model, in_channels=channels),
        }
        row.update(
            {f"cpu{threads}t_{k}": v for k, v in _cpu(model, channels, threads, reps).items()}
        )
        row.update({f"cpu1t_{k}": v for k, v in _cpu(model, channels, 1, reps).items()})
        if device is not None:
            speed = _speed(
                model, channels, device, (1, 512, 4096), reps, torch.accelerator.synchronize
            )
            row.update({f"{device.type}_{k}": v for k, v in speed.items()})
        rows[name] = row
        print(name, json.dumps(row), flush=True)
    meta = {
        "torch": torch.__version__,
        "machine": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_threads_default": threads,
        "accelerator": None if device is None else device.type,
        "reps": reps,
    }
    return {"meta": meta, "rows": rows}


def main() -> None:
    """Parse the arguments, run :func:`bench` and write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="30 timed batch-1 calls, not 200")
    parser.add_argument("--cpu-only", action="store_true", help="skip the accelerator")
    parser.add_argument("--out", type=Path, help="write the results as JSON to this file")
    args = parser.parse_args()
    result = bench(reps=30 if args.quick else 200, cpu_only=args.cpu_only)
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
