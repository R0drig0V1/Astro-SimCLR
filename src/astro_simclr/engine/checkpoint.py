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

The resume order (ADR 0012, "Resume") is: :func:`check_resumable`; seed; build the task on
the CPU; load ``model``; move to the device; build the optimizer and load ``optimizer``;
``task``; ``policy``; check ``stream``; restore ``rng`` last, just before the first step.

Contracts only (task B0): writing and reading checkpoints arrives in 4.1.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, Literal, TypedDict

import torch

from astro_simclr.config import ConfigError

__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_KEYS",
    "Checkpoint",
    "IdentityState",
    "PolicyState",
    "RngState",
    "check_resumable",
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
    #: The best weights on the CPU (``keep`` includes ``best``), else ``None``.
    best_model: dict[str, torch.Tensor] | None


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


def check_resumable(checkpoint: Mapping[str, Any]) -> None:
    """Refuse, before anything is built, a checkpoint the plain engine cannot continue.

    Raises:
        ConfigError: for a Lightning checkpoint (in-flight Lightning runs finish on
            Lightning), another engine, another format version, or missing keys.
    """
    if _LIGHTNING_KEY in checkpoint:
        msg = (
            "this is a Lightning checkpoint: the plain engine does not resume Lightning runs "
            "(finish them on Lightning, ADR 0012)"
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
