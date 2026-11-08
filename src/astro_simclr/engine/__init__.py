"""The plain-PyTorch training engine (phase 4, ADR 0014): the contracts and the data layer.

Task B0 of ``docs/phases/04-training-platform.md`` freezes the interfaces that the engine
core (4.1), the typed config (4.2) and the data layer (4.3) are built against, so that those
tasks can proceed in parallel and the 4.7 equivalence gate has a fixed target:

- :mod:`.task`: the :class:`~.task.Task` protocol (``train_step``, ``eval_step``, ``state``)
  and :func:`~.task.mean_of_batch_losses`, the monitored value;
- :mod:`.policies`: the monitor rule, the validation grid, the horizon, early stopping, the
  ``keep`` choice and the stop-rule protocol, as frozen dataclasses with the pure rules
  that define them;
- :mod:`.checkpoint`: the checkpoint key set, every RNG stream, the stream position and the
  run identity, and the refusal of a cross-engine resume;
- :mod:`.device`: the device policy (one device only, the determinism flags, ADR 0010's
  float32 precision);
- :mod:`.data`: the batch tuple format and the three data tiers;
- :mod:`.randomness`: the seeded per-step generators (host and device).

Task 4.3 (ADR 0018) implements the data layer against them:

- :mod:`.dataset`: :class:`~.dataset.Dataset`, one API over the prepared kinds
  (classifier, unlabelled, raw-alerts) and in-memory arrays;
- :mod:`.stream`: the class tiling, the seeded resumable stream and its shuffle
  granularity (``row`` or ``block``);
- :mod:`.tiers`: the three tiers (``open_train`` -> :class:`~.tiers.TrainSource`,
  ``open_eval`` -> :class:`~.tiers.EvalBatches`), exact with each other;
- :mod:`.bench`: the tiers' throughput microbenchmark (``python -m astro_simclr.engine.bench``).

Task 4.1 implements the engine core:

- :mod:`.loop`: :func:`~.loop.fit` (the loop, its validations, checkpoints and exact
  resume), :func:`~.loop.evaluate` and :func:`~.loop.evaluate_kept` (the final evaluation
  of the kept weights);
- :mod:`.precision`: ADR 0010's float32 policy (shared with the Lightning path);
- :mod:`.scalars`: the local scalars, ``scalars.jsonl``;
- :mod:`.record`: the machine record and the manifest's lifecycle (``engine: plain``).

**Engine-free rule** (ADR 0014, CLAUDE.md): nothing in this package imports Lightning, so a
flag written against these contracts works on both engines until Lightning is removed (4.8).
``tests/test_engine_contracts.py`` checks it.
"""

from __future__ import annotations

from astro_simclr.engine.checkpoint import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_KEYS,
    BestWeights,
    Checkpoint,
    IdentityState,
    PolicyState,
    RngState,
    check_resumable,
    load_best,
    load_checkpoint,
)
from astro_simclr.engine.data import (
    Batch,
    BatchSource,
    DataTier,
    EvalSource,
    TierChoice,
    TrainBatch,
)
from astro_simclr.engine.dataset import Dataset, prepared_split, raw_alerts_dataset
from astro_simclr.engine.device import CUBLAS_WORKSPACE_CONFIG, DevicePolicy
from astro_simclr.engine.loop import (
    Evaluation,
    FitResult,
    Score,
    TrainStream,
    Validation,
    evaluate,
    evaluate_kept,
    fit,
)
from astro_simclr.engine.policies import (
    EarlyStop,
    Horizon,
    Keep,
    Monitor,
    Schedule,
    StopDecision,
    StopRule,
    ValidationGrid,
    ValidationTrigger,
)
from astro_simclr.engine.randomness import (
    RESERVED_STREAMS,
    device_generator,
    host_generator,
    seed_everything,
)
from astro_simclr.engine.record import machine_record, recorded
from astro_simclr.engine.scalars import ScalarLog, read_scalars
from astro_simclr.engine.stream import TrainOrder
from astro_simclr.engine.task import EvalOutput, Task, mean_of_batch_losses
from astro_simclr.engine.tiers import EvalBatches, StreamSpec, TrainSource, open_eval, open_train

__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_KEYS",
    "CUBLAS_WORKSPACE_CONFIG",
    "RESERVED_STREAMS",
    "Batch",
    "BatchSource",
    "BestWeights",
    "Checkpoint",
    "DataTier",
    "Dataset",
    "DevicePolicy",
    "EarlyStop",
    "EvalBatches",
    "EvalOutput",
    "EvalSource",
    "Evaluation",
    "FitResult",
    "Horizon",
    "IdentityState",
    "Keep",
    "Monitor",
    "PolicyState",
    "RngState",
    "ScalarLog",
    "Schedule",
    "Score",
    "StopDecision",
    "StopRule",
    "StreamSpec",
    "Task",
    "TierChoice",
    "TrainBatch",
    "TrainOrder",
    "TrainSource",
    "TrainStream",
    "Validation",
    "ValidationGrid",
    "ValidationTrigger",
    "check_resumable",
    "device_generator",
    "evaluate",
    "evaluate_kept",
    "fit",
    "host_generator",
    "load_best",
    "load_checkpoint",
    "machine_record",
    "mean_of_batch_losses",
    "open_eval",
    "open_train",
    "prepared_split",
    "raw_alerts_dataset",
    "read_scalars",
    "recorded",
    "seed_everything",
]
