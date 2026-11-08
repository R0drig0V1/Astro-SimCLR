"""The plain engine's run record: its machine entry and the manifest's lifecycle (ADR 0015).

A plain run writes the same ``manifest.json`` as a Lightning run (``runs.record.start_run``),
with ``engine: plain``, which is part of its run id: a plain run never passes for an
attempt of a Lightning run, and comparisons refuse mixed engines (task A1).

- :func:`machine_record` is ``experiments._machine`` without the Lightning ``Trainer``: the
  device and build of a run, with ADR 0010's precision record and ``engine: plain``. It is
  what ``start_run(machine=...)`` and ``metrics.json["machine"]`` take.
- :func:`recorded` ends a started record with ``finished`` when its block completes,
  ``cancelled`` on ``KeyboardInterrupt`` or ``SystemExit``, and ``failed`` on any other
  exception (``run_command``'s rule, task A1), and re-raises.

Wiring the CLI to the plain engine is task 4.6; this is what it will call.
"""

from __future__ import annotations

import platform
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Final, cast

import torch

from astro_simclr.engine.precision import fp32_precision_record
from astro_simclr.runs.record import RunRecord

__all__ = ["ENGINE", "machine_record", "recorded"]

#: The ``engine`` of every plain run (``runs.identity.Engine``).
ENGINE: Final = "plain"


def machine_record(device: torch.device) -> dict[str, Any]:
    """The machine and build a run on ``device`` used (``experiments._machine``'s keys).

    ``gpu`` is the CUDA device name (``None`` elsewhere); ``fp32_precision`` (effective),
    ``fp32_backends`` and ``compute_capability`` come from ADR 0010's record, so call it
    after the precision is applied.
    """
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    # torch.backends.cudnn is untyped; the casts state the documented signatures.
    cudnn = cast(Any, torch.backends.cudnn)
    return {
        "device": device.type,
        "gpu": gpu,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": cudnn.version() if cudnn.is_available() else None,
        "platform": platform.platform(),
        "processor": platform.machine(),
        "python": platform.python_version(),
        "engine": ENGINE,
        **fp32_precision_record(device),
    }


@contextmanager
def recorded(record: RunRecord) -> Iterator[RunRecord]:
    """End ``record`` by how the block ends (module docstring); yields the record.

    Raises:
        ManifestError, OSError: if the final manifest cannot be written; the block's own
            exception otherwise, unchanged.
    """
    try:
        yield record
    except (KeyboardInterrupt, SystemExit):
        record.end("cancelled")
        raise
    except BaseException:
        record.end("failed")
        raise
    record.end("finished")
