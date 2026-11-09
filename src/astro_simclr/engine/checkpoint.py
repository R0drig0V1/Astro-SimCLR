"""The plain engine's checkpoint: its key set, RNG streams, stream position and identity.

A checkpoint (``ckpt/last.pt``, ``ckpt/best.pt`` in the run layout of task A0) is a
:class:`Checkpoint` of plain values, so ``torch.load(weights_only=True)`` reads it: tensors,
numbers, strings, lists, tuples and dicts only (NumPy RNG keys are stored as tensors).
Unlike Lightning's (``trainer.save_checkpoint``), it holds **every global RNG stream**, so a
resume is exact even for a task that draws from them (``dropout_active=True``); the
per-step generators of :mod:`.randomness` need no state, being functions of the step.

The **stream position** is ``steps_done``: the next batch is stream batch ``steps_done``
(:class:`~astro_simclr.engine.data.BatchSource`), so no loader position is stored, only the
stream definition to check. ``last`` is written after the validation of its step and the
policy update, so a resume owes no validation (the reason ``HorizonCallback`` writes it
there).

The resume order (ADR 0014, "Resume") is: :func:`check_resumable`; seed; build the task on
the CPU; load ``model``; move to the device; build the optimizer and load ``optimizer``;
``task``; ``policy``; check ``stream``; restore ``rng`` last, just before the first step.

Task 4.1 writes and reads them: :func:`capture_rng` / :func:`restore_rng`,
:func:`numerics_record`, :func:`save_checkpoint` (atomic: a ``kill -9`` mid-write leaves the
previous file), :func:`load_checkpoint` (``weights_only``, then :func:`check_resumable`),
and the resume checks :func:`check_identity` and :func:`check_numerics`. ``best.pt`` is not
a resumable checkpoint but :class:`BestWeights`, the weights of the best validation (what
Lightning's weights-only ``best.ckpt`` is), so :func:`check_resumable` refuses it.
"""

from __future__ import annotations

import os
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal, NotRequired, TypedDict, cast

import numpy as np
import torch

from astro_simclr.config import LIGHTNING_TAG, ConfigError
from astro_simclr.engine.device import DevicePolicy
from astro_simclr.engine.precision import fp32_precision_record

__all__ = [
    "BEST_KEYS",
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_KEYS",
    "BestWeights",
    "Checkpoint",
    "IdentityState",
    "PolicyState",
    "RngState",
    "capture_rng",
    "check_identity",
    "check_numerics",
    "check_resumable",
    "load_best",
    "load_checkpoint",
    "numerics_record",
    "restore_rng",
    "save_checkpoint",
]

#: The checkpoint format version; a change of the key set or of a key's meaning bumps it.
CHECKPOINT_FORMAT: Final = 1


class RngState(TypedDict):
    """Every global random stream a step could draw from."""

    #: ``random.getstate()``.
    python: tuple[Any, ...]
    #: NumPy's legacy global generator, ``np.random.get_state(legacy=False)`` with the key
    #: array as a tensor.
    numpy: dict[str, Any]
    #: ``torch.get_rng_state()``.
    torch_cpu: torch.Tensor
    #: ``torch.cuda.get_rng_state_all()`` on CUDA, else ``None``.
    torch_cuda: list[torch.Tensor] | None
    #: ``torch.mps.get_rng_state()`` on MPS, else ``None``.
    torch_mps: torch.Tensor | None


class IdentityState(TypedDict):
    """Which run a checkpoint belongs to; a resume of another run is refused."""

    #: Task A0's run id (the hash of the canonical config, seed, fold, data, splits,
    #: machine class and ``numerics_epoch``); ``None`` only before A0 lands.
    run_id: str | None
    #: A0's attempt counter, or ``None`` with ``run_id``.
    attempt: int | None
    #: ``SplitRun.identity()`` of a split run (splits SHA-256, mode, fold), else ``None``.
    split_run: dict[str, Any] | None


class PolicyState(TypedDict):
    """The schedule policies' state (``policies``)."""

    monitor: str
    mode: Literal["min", "max"]
    #: The current horizon (optimizer steps).
    horizon: int
    #: The best monitored value (a Python float of the float32 value), or ``None``.
    best_score: float | None
    #: Steps done at the best validation, or ``None``.
    best_step: int | None
    #: Validations since the last improvement (early stopping; 0 without it).
    wait_count: int
    #: Early stopping's own best value (Lightning's ``EarlyStopping.best_score``, which
    #: differs from ``best_score`` when ``min_delta > 0``); ``None`` without early stopping
    #: or before the first validation. Added in task 4.1 (ADR 0014 amendment).
    stop_score: float | None
    #: The best weights on the CPU (``keep`` includes ``best``), else ``None``.
    best_model: dict[str, torch.Tensor] | None
    #: Every fit validation's monitored value, in order, for a schedule with a
    #: ``stop_rule`` (the ``rj23`` schedule, task 3.4b); absent otherwise.
    history: NotRequired[list[float]]


class Checkpoint(TypedDict):
    """Everything a resume needs, and nothing that depends on the engine's internals."""

    format_version: int
    #: Always ``"plain"``: a Lightning checkpoint is refused (no cross-engine resume).
    engine: Literal["plain"]
    identity: IdentityState
    #: Optimizer steps done; also the stream position.
    steps_done: int
    #: ``module.state_dict()`` on the CPU.
    model: dict[str, torch.Tensor]
    #: ``optimizer.state_dict()``.
    optimizer: dict[str, Any]
    #: ``Task.state()``.
    task: dict[str, Any]
    policy: PolicyState
    #: ``BatchSource.state()``: the stream definition to check, not a position.
    stream: dict[str, Any]
    rng: RngState
    #: ``training.runs.fp32_precision_record`` and the determinism flags at save time; a
    #: resume under another precision policy is refused (ADR 0010).
    numerics: dict[str, Any]
    #: ``astro_simclr`` and ``torch`` versions and the git SHA, for the record, as plain
    #: ``str`` (``torch.__version__`` is a ``TorchVersion``, which ``weights_only`` refuses).
    versions: dict[str, str | None]


#: The frozen key set of :class:`Checkpoint`.
CHECKPOINT_KEYS: Final[frozenset[str]] = frozenset(Checkpoint.__required_keys__)

#: A key only Lightning checkpoints have.
_LIGHTNING_KEY: Final = "pytorch-lightning_version"


class BestWeights(TypedDict):
    """``ckpt/best.pt``: the weights of the best validation (not resumable).

    Lightning's ``best.ckpt`` is a weights-only checkpoint stamped with the best's epoch and
    step; this is the same record in plain values. ``kind`` is ``"best"`` so that
    :func:`check_resumable` cannot mistake it for a training state.
    """

    format_version: int
    engine: Literal["plain"]
    kind: Literal["best"]
    identity: IdentityState
    #: Steps done at the best validation (``best_step``).
    steps_done: int
    monitor: str
    #: The best monitored value (a Python float of the float32 value).
    score: float
    model: dict[str, torch.Tensor]
    numerics: dict[str, Any]
    versions: dict[str, str | None]


#: The key set of :class:`BestWeights`.
BEST_KEYS: Final[frozenset[str]] = frozenset(BestWeights.__required_keys__)


def check_resumable(checkpoint: Mapping[str, Any]) -> None:
    """Refuse, before anything is built, a checkpoint the plain engine cannot continue.

    Raises:
        ConfigError: for a Lightning checkpoint (a Lightning run resumes only from a
            checkout of the git tag ``lightning-final``), another engine, another format
            version, or missing keys.
    """
    if _LIGHTNING_KEY in checkpoint:
        msg = (
            "this is a Lightning checkpoint: the plain engine does not resume Lightning runs "
            f"(ADR 0014; resume it from a checkout of the git tag {LIGHTNING_TAG})"
        )
        raise ConfigError(msg)
    engine = checkpoint.get("engine")
    if engine != "plain":
        raise ConfigError(f"checkpoint of engine {engine!r}, not 'plain'")
    version = checkpoint.get("format_version")
    if version != CHECKPOINT_FORMAT:
        raise ConfigError(f"checkpoint format {version!r}, this engine reads {CHECKPOINT_FORMAT}")
    missing = CHECKPOINT_KEYS - set(checkpoint)
    if missing:
        raise ConfigError(f"checkpoint lacks {', '.join(sorted(missing))}")


def capture_rng(device: DevicePolicy) -> RngState:
    """Every global random stream, as plain values (NumPy's key array becomes a tensor).

    CUDA streams are captured on a ``cuda`` run and the MPS one on an ``mps`` run; the other
    entries are ``None`` (capturing CUDA state would initialise CUDA on a CPU run).
    """
    numpy_state = np.random.get_state(legacy=False)
    inner = dict(numpy_state["state"])
    inner["key"] = torch.from_numpy(np.asarray(inner["key"]).astype(np.int64))
    return {
        "python": random.getstate(),
        "numpy": {**numpy_state, "state": inner},
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if device.accelerator == "cuda" else None,
        "torch_mps": torch.mps.get_rng_state() if device.accelerator == "mps" else None,
    }


def restore_rng(state: RngState) -> None:
    """Restore :func:`capture_rng`'s streams (the device ones only where they were saved).

    Raises:
        ConfigError: if CUDA streams were saved for another number of CUDA devices.
    """
    version, internal, gauss = state["python"]
    random.setstate((version, tuple(internal), gauss))
    numpy_state = dict(state["numpy"])
    inner = dict(numpy_state["state"])
    inner["key"] = np.asarray(inner["key"].numpy(), dtype=np.uint32)
    np.random.set_state({**numpy_state, "state": inner})
    torch.set_rng_state(state["torch_cpu"])
    cuda = state["torch_cuda"]
    if cuda is not None:
        if len(cuda) != torch.cuda.device_count():
            msg = (
                f"the checkpoint holds {len(cuda)} CUDA generator states, this machine has "
                f"{torch.cuda.device_count()} CUDA devices"
            )
            raise ConfigError(msg)
        torch.cuda.set_rng_state_all(cuda)
    if state["torch_mps"] is not None:
        torch.mps.set_rng_state(state["torch_mps"])


def numerics_record(device: DevicePolicy) -> dict[str, Any]:
    """The ``numerics`` entry: the precision setting, determinism and what torch reports.

    ``fp32_precision`` and ``deterministic`` are the policy (compared on resume);
    ``effective_fp32_precision``, ``fp32_backends`` and ``compute_capability`` are ADR 0010's
    record of what the device computed, for the record only.
    """
    record = fp32_precision_record(device.device)
    return {
        "accelerator": device.accelerator,
        "fp32_precision": device.fp32_precision,
        "deterministic": device.deterministic,
        "effective_fp32_precision": record["fp32_precision"],
        "fp32_backends": record["fp32_backends"],
        "compute_capability": record["compute_capability"],
    }


#: The ``numerics`` entries a resume must not change (ADR 0010: one run, one policy).
_POLICY_KEYS: Final = ("fp32_precision", "deterministic")


def check_numerics(saved: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """Refuse a resume under another precision or determinism policy.

    Raises:
        ConfigError: if ``fp32_precision`` or ``deterministic`` differ (both named).
    """
    old = {key: saved.get(key) for key in _POLICY_KEYS}
    new = {key: current.get(key) for key in _POLICY_KEYS}
    if old != new:
        msg = (
            f"the checkpoint was trained with {old}, this run is {new}: a resumed run keeps "
            "its precision and determinism policy (ADR 0010)"
        )
        raise ConfigError(msg)


def check_identity(saved: Mapping[str, Any], current: IdentityState) -> None:
    """Refuse a resume of another run: the run id and the split-run identity must match.

    The attempt may differ (a resume after a failure is the next attempt, ADR 0015).

    Raises:
        ConfigError: on another run id or another split run (or a split run resumed as a
            normal run, or the reverse: that would leak evaluation stamps into training).
    """
    for key in ("run_id", "split_run"):
        if saved.get(key) != current[key]:
            msg = (
                f"the checkpoint is from {key} {saved.get(key)!r}, this run has "
                f"{current[key]!r}: resume the same run"
            )
            raise ConfigError(msg)


def save_checkpoint(path: Path, checkpoint: Checkpoint | BestWeights) -> Path:
    """Write ``checkpoint`` to ``path`` atomically (a temporary file, then a rename).

    A process killed mid-write leaves the previous file intact, which the durable queue's
    resume relies on (task A3). Returns ``path``.

    Raises:
        OSError: if the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    torch.save(checkpoint, partial)
    os.replace(partial, path)
    return path


def load_checkpoint(path: Path) -> Checkpoint:
    """Read a resumable checkpoint (``weights_only``, on the CPU) and check it.

    Raises:
        ConfigError: as :func:`check_resumable` (also for ``best.pt``, which is weights only).
        OSError: if the file cannot be read.
    """
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} is not a checkpoint")
    if loaded.get("kind") == "best":
        raise ConfigError(f"{path} holds the best weights only: resume from ckpt/last.pt")
    check_resumable(loaded)
    return cast(Checkpoint, loaded)


def load_best(path: Path) -> BestWeights:
    """Read ``best.pt`` (``weights_only``, on the CPU).

    Raises:
        ConfigError: if it is not a plain engine's best-weights file.
        OSError: if the file cannot be read.
    """
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict) or loaded.get("kind") != "best":
        raise ConfigError(f"{path} is not a plain engine's best weights")
    if loaded.get("format_version") != CHECKPOINT_FORMAT or set(loaded) != BEST_KEYS:
        raise ConfigError(f"{path}: best weights of another format")
    return cast(BestWeights, loaded)
