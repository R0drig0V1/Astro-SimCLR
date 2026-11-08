"""Which slot, if any, may start a job now: concurrency and measured peak memory (task A3).

A **slot** is one device a worker runs jobs on: a GPU (by index) or the CPU. Rules, in
order (:func:`admits`):

1. a slot runs at most ``max_jobs`` jobs (default 1: the GTX 1050 is compute-bound, so two
   jobs only share its time; plan, task A3);
2. an empty slot always admits: a job alone either fits or fails on its own;
3. a job joins running ones only if every one of them, and the job itself, has a measured
   peak memory for this machine class (``store.peak_memory``: the
   ``max_memory_reserved`` of an earlier finished run of the same config hash), the slot's
   capacity is known, and the sum stays under ``capacity * (1 - margin)``. An unmeasured
   job runs alone: the first run of a config is the measurement.

Pure functions, so the rules are tested without a GPU.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

__all__ = ["DEFAULT_MARGIN", "Slot", "admits", "choose_slot"]

#: Share of a GPU's memory kept free when jobs share it (allocator fragmentation, the
#: CUDA context of each process).
DEFAULT_MARGIN: Final = 0.1


@dataclass(frozen=True, slots=True)
class Slot:
    """One device of a worker: ``cuda:<gpu>`` or ``cpu``.

    ``capacity_bytes`` is the device's total memory (``None``: unknown, so jobs never share
    it); ``max_jobs`` the most jobs it runs at once.

    Raises:
        ValueError: if ``max_jobs`` is not positive.
    """

    name: str
    gpu: int | None
    capacity_bytes: int | None
    max_jobs: int = 1

    def __post_init__(self) -> None:
        if self.max_jobs < 1:
            raise ValueError(f"slot {self.name}: max_jobs must be >= 1, got {self.max_jobs}")


def admits(
    slot: Slot,
    job_peak: int | None,
    running_peaks: Sequence[int | None],
    *,
    margin: float = DEFAULT_MARGIN,
) -> bool:
    """Whether ``slot``, running jobs of ``running_peaks``, may start a job of ``job_peak``.

    Peaks are bytes, ``None`` when not measured (module docstring for the rules).
    """
    if len(running_peaks) >= slot.max_jobs:
        return False
    if not running_peaks:
        return True
    if slot.capacity_bytes is None or job_peak is None:
        return False
    measured = [peak for peak in running_peaks if peak is not None]
    if len(measured) != len(running_peaks):
        return False
    return job_peak + sum(measured) <= slot.capacity_bytes * (1 - margin)


def choose_slot(
    slots: Sequence[Slot],
    job_peak: int | None,
    running: Mapping[str, Sequence[int | None]],
    *,
    margin: float = DEFAULT_MARGIN,
) -> Slot | None:
    """The least loaded slot that admits the job (ties: the first), or ``None``.

    ``running`` maps a slot name to the peaks of the jobs it runs.
    """
    candidates = [
        slot for slot in slots if admits(slot, job_peak, running.get(slot.name, ()), margin=margin)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda slot: len(running.get(slot.name, ())))
