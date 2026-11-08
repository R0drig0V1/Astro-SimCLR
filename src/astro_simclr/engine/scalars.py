"""Local scalars: ``scalars.jsonl``, one JSON object per line (ADR 0015's run layout).

The plain engine's replacement for Lightning's TensorBoard logger and progress-bar metrics
(phase 4: experiment tracking is local only). The loop writes one line per validation:
the step, the derived epoch, the monitored value, the mean training loss since the
previous validation, whether it was a new best, the horizon and early stopping's wait
count, and the wall-clock time. Each line is flushed as it is written, so a killed run
keeps every validation it finished.

**Resume.** A run killed after its last checkpoint has logged validations the resumed run
will repeat. :meth:`ScalarLog.open` with ``resume_from=s`` keeps only the lines of steps
``<= s`` (the checkpoint's), so the file never holds a step twice; a fresh run truncates
the file (a new attempt in the same directory replaces the old one's record, as its
manifest does).

Non-finite floats are written as ``null``: strict JSON has no NaN (a diverged run's loss).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Self

__all__ = ["ScalarLog", "read_scalars"]


def _finite(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_scalars(path: Path) -> list[dict[str, Any]]:
    """Every record of ``path``, in order (``[]`` if the file does not exist).

    Raises:
        ValueError: on a line that is not a JSON object.
    """
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{number} is not a JSON object")
        records.append(record)
    return records


class ScalarLog:
    """An append-only JSONL file of scalar records (module docstring). A context manager."""

    def __init__(self, path: Path, handle: IO[str]) -> None:
        self.path = path
        self._handle = handle

    @classmethod
    def open(cls, path: Path, *, resume_from: int | None = None) -> ScalarLog:
        """Open ``path`` for a fresh run (truncated) or a resume at step ``resume_from``.

        Raises:
            ValueError: if an existing line is not a JSON object, or has no integer ``step``.
            OSError: if the file cannot be written.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        kept: list[dict[str, Any]] = []
        if resume_from is not None:
            for record in read_scalars(path):
                step = record.get("step")
                if not isinstance(step, int):
                    raise ValueError(f"{path}: a record without an integer step: {record}")
                if step <= resume_from:
                    kept.append(record)
        handle = path.open("w", encoding="utf-8")
        log = cls(path, handle)
        for record in kept:
            log.write(record)
        return log

    def write(self, record: dict[str, Any]) -> None:
        """Append ``record`` as one line and flush it."""
        clean = {key: _finite(value) for key, value in record.items()}
        self._handle.write(json.dumps(clean, allow_nan=False, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        """Close the file (idempotent)."""
        self._handle.close()

    def __enter__(self) -> Self:
        """The log itself."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the file."""
        self.close()
