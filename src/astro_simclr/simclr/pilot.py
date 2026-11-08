"""The 5.0b pilot: a minimal plain-PyTorch SimCLR pretraining loop on CD21 Train (ADR 0025).

Phase 5's task 5.0b runs SimCLR on the 50.6k CD21 Train stamps before any download, to
decide whether the corpus download is worth asking for. This module is that run, written
against the engine contracts (ADR 0014: per-step generators, steps done as the only clock)
but not on the engine loop, which does not exist yet (task 4.1); 5.3 moves the same pure
functions onto it. Nothing here imports Lightning.

**Data** (:func:`load_pilot_data`, tier 1 or 2 of ``engine.data``): a prepared classifier
layout's Train stamps, float32 (``images_f32.npy``; a uint8-only layout is read as
``images / 255`` and recorded as a deviation), as (N, 3, 63, 63). With ``splits`` (a
stored splits file, verified against the layout), the Train stamps a screened run may not
use (``screen_training_excluded``: the leaked stamp and the stamps grouped with
Validation) are left out of pretraining, and the kNN monitor queries the screen's
**selection** half. The **decision** half is never read here. ``tier="device"`` keeps the
stamps on the training device (2.4 GB for CD21 Train); ``"host"`` keeps them in host memory
and uploads each batch.

**Stream** (:class:`BatchOrder`): pass ``p`` over the pretraining stamps is the permutation
``default_rng([seed, p, simclr_batch_order])``; step ``s`` takes batch ``s mod
steps_per_pass`` of pass ``s // steps_per_pass`` (the last partial batch of a pass is
dropped, so each batch has B distinct stamps). A pure function of ``(seed, s)``.

**Step** ``s`` (:meth:`PilotRun.step`): two views of the batch (:mod:`.views`, parameters
from ``(seed, s)``), one forward of both views concatenated (BatchNorm statistics over the
2B views), NT-Xent in paper mode in float32 (:mod:`.objective`), LARS at the rate of
:func:`~.optim.warmup_cosine_lr` for ``s`` (:mod:`.optim`), ``zero_grad(set_to_none=True)``,
backward, step.

**Run length.** ``epochs`` (whole passes) or ``steps``: the schedule's ``total_steps`` is
exactly the run's length, and the warmup is ``warmup_epochs`` passes (at most
``total_steps - 1`` steps).

**Outputs** under ``out_dir``:

- ``pilot.json``: the config, the data record, the schedule, the device and versions;
- ``metrics.jsonl``: every ``log_every`` steps, the step's loss, mean positive and negative
  cosine, and rate;
- ``diagnostics.jsonl``: every ``diagnostics_every`` steps and at the end
  (:func:`diagnose`): the effective rank of ``cov(h)``, the per-dimension std of ``z``,
  alignment and uniformity on a fixed sample with fixed views, and the kNN monitor;
- ``ckpt/last.pt`` every ``checkpoint_every`` steps and at the end, ``ckpt/step-NNNNNNN.pt``
  every ``keep_every_epochs`` passes, ``ckpt/final.pt`` at the end (:data:`CKPT_FORMAT`:
  plain values, read with ``weights_only=True``);
- ``embeddings/`` (:func:`dump_embeddings`): ``h`` of every Train and Validation stamp on
  the centred 57-pixel view, in eval mode, with the labels, for the probe.

**Resume** (``resume=True``): ``ckpt/last.pt`` restores the weights (BatchNorm statistics
included), the optimizer and the steps done; the config must be the same. Every draw is a
function of the step, so the continued run takes the same batches and views as an
uninterrupted one; the log lines past the checkpoint are dropped. The resume is exact
whenever the kernels are deterministic (``deterministic=True`` on CUDA; always on CPU).
"""

from __future__ import annotations

import json
import os
import platform
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final, Literal, cast

import numpy as np
import numpy.typing as npt
import torch

from astro_simclr.config import ConfigError
from astro_simclr.data.files import sha256_file
from astro_simclr.data.prepared import load_prepared, read_meta
from astro_simclr.data.schema import LABEL_NAMES, DataError
from astro_simclr.engine.device import DevicePolicy
from astro_simclr.engine.randomness import RESERVED_STREAMS, host_generator
from astro_simclr.protocol.splits import ScreenSplit, read_splits, verify_splits
from astro_simclr.simclr.diagnostics import (
    alignment,
    dimension_std,
    effective_rank,
    knn_balanced_accuracy,
    uniformity,
)
from astro_simclr.simclr.network import PROJECTION_DIM, REPRESENTATION_DIM, SimCLRNetwork
from astro_simclr.simclr.objective import SimCLRLoss
from astro_simclr.simclr.optim import (
    LR_PER_256,
    WEIGHT_DECAY,
    base_learning_rate,
    build_lars,
    set_learning_rate,
    warmup_cosine_lr,
)
from astro_simclr.simclr.probe import draw_label_subsets
from astro_simclr.simclr.views import (
    ViewConfig,
    centre_crop,
    draw_view_params,
    make_views,
    noise_generator,
)
from astro_simclr.stamp_relation import DEFAULT_MIN_R2, stamp_fits

__all__ = [
    "CKPT_FORMAT",
    "EMBEDDINGS_FORMAT",
    "BatchOrder",
    "PilotConfig",
    "PilotData",
    "PilotRun",
    "diagnose",
    "dump_embeddings",
    "embed",
    "load_pilot_data",
    "resolve_device",
    "train_pilot",
]

#: ``format`` of a pilot checkpoint; its version is :data:`_CKPT_VERSION`.
CKPT_FORMAT: Final = "astro-simclr-pilot-checkpoint"
_CKPT_VERSION: Final = 1
#: ``format`` of ``embeddings/meta.json``.
EMBEDDINGS_FORMAT: Final = "astro-simclr-embeddings"
_EMBEDDINGS_VERSION: Final = 1
_MATMUL_PRECISION: Final = {"ieee": "highest", "tf32": "high"}

Tier = Literal["device", "host"]


@dataclass(frozen=True, slots=True)
class PilotConfig:
    """Everything that defines a pilot run (module docstring); ``to_plain`` is its record.

    Raises:
        ConfigError: unless exactly one of ``epochs`` and ``steps`` is set and positive,
            and every size, interval and rate is valid.
    """

    data: str
    splits: str | None = None
    epochs: int | None = 100
    steps: int | None = None
    batch_size: int = 256
    temperature: float = 0.5
    lr_per_256: float = LR_PER_256
    weight_decay: float = WEIGHT_DECAY
    warmup_epochs: float = 10.0
    lr_min: float = 0.0
    noise_scale: float = 0.0
    noise_min_r2: float = DEFAULT_MIN_R2
    seed: int = 0
    device: str = "auto"
    fp32_precision: Literal["ieee", "tf32"] = "ieee"
    deterministic: bool = False
    tier: Tier = "device"
    log_every: int = 50
    checkpoint_every: int = 1000
    diagnostics_every: int = 1000
    keep_every_epochs: int | None = 25
    diag_samples: int = 512
    knn_bank_per_class: int = 1000
    embed_batch: int = 1024

    def __post_init__(self) -> None:
        if (self.epochs is None) == (self.steps is None):
            raise ConfigError("set exactly one of epochs and steps")
        minimum = {
            "epochs": (self.epochs, 1),
            "steps": (self.steps, 1),
            "batch_size": (self.batch_size, 2),  # NT-Xent needs a negative
            "log_every": (self.log_every, 1),
            "checkpoint_every": (self.checkpoint_every, 1),
            "diagnostics_every": (self.diagnostics_every, 1),
            "keep_every_epochs": (self.keep_every_epochs, 1),
            "diag_samples": (self.diag_samples, 2),  # a covariance needs two
            "knn_bank_per_class": (self.knn_bank_per_class, 1),
            "embed_batch": (self.embed_batch, 1),
        }
        bad = [f"{k} >= {low}" for k, (v, low) in minimum.items() if v is not None and v < low]
        if bad:
            raise ConfigError(f"need {', '.join(bad)}")
        if not (self.temperature > 0 and self.lr_per_256 > 0 and self.weight_decay >= 0):
            raise ConfigError("need temperature > 0, lr_per_256 > 0 and weight_decay >= 0")
        if self.warmup_epochs < 0 or self.lr_min < 0 or self.noise_scale < 0 or self.seed < 0:
            raise ConfigError("warmup_epochs, lr_min, noise_scale and seed must be >= 0")
        if self.fp32_precision not in ("ieee", "tf32") or self.tier not in ("device", "host"):
            raise ConfigError("fp32_precision must be ieee|tf32 and tier device|host")

    def to_plain(self) -> dict[str, Any]:
        """The config as a JSON-ready dict (every field)."""
        return asdict(self)

    @property
    def views(self) -> ViewConfig:
        """The training views' configuration."""
        return ViewConfig(self.noise_scale, self.noise_min_r2)


@dataclass(frozen=True, slots=True)
class PilotData:
    """The stamps of a pilot run, on the host (module docstring, "Data").

    Attributes:
        train: float32 (N_train, 3, 63, 63), every Train stamp in file order.
        validation: float32 (N_val, 3, 63, 63), every Validation stamp.
        rows: int64, sorted, the Train indices pretraining uses.
        train_labels: int64 (N_train,), or ``None`` for an unlabelled layout.
        validation_labels: int64 (N_val,), or ``None``.
        screen: the screen halves of ``splits``, or ``None`` without a splits file.
        record: what ``pilot.json`` records about the data.
    """

    train: torch.Tensor
    validation: torch.Tensor
    rows: npt.NDArray[np.int64]
    train_labels: npt.NDArray[np.int64] | None
    validation_labels: npt.NDArray[np.int64] | None
    screen: ScreenSplit | None
    record: dict[str, Any] = field(default_factory=dict)


def _channels_first(split_images: npt.NDArray[Any], *, float32: bool) -> torch.Tensor:
    array = np.asarray(split_images, dtype=np.float32)
    if not float32:
        array = array / np.float32(255.0)
    return torch.from_numpy(np.ascontiguousarray(array.transpose(0, 3, 1, 2)))


def load_pilot_data(config: PilotConfig) -> PilotData:
    """Read the layout (and the splits file) of ``config`` into host memory.

    Raises:
        DataError: if the layout or the splits file is unusable, or the splits file was
            made for other data (``verify_splits``).
        ConfigError: if fewer stamps remain than one batch.
    """
    path = Path(config.data)
    meta = read_meta(path)
    data = load_prepared(path)
    train, val = data.splits["Train"], data.splits["Validation"]
    float32 = train.images_f32 is not None and val.images_f32 is not None
    if train.images_f32 is not None and val.images_f32 is not None:
        train_images = _channels_first(train.images_f32, float32=True)
        val_images = _channels_first(val.images_f32, float32=True)
    else:
        train_images = _channels_first(train.images, float32=False)
        val_images = _channels_first(val.images, float32=False)
    rows = np.arange(len(train), dtype=np.int64)
    screen = None
    excluded: tuple[int, ...] = ()
    splits_sha = None
    if config.splits is not None:
        splits_path = Path(config.splits)
        splits = read_splits(splits_path)
        verify_splits(splits, path)
        screen = splits.screen
        excluded = splits.screen_training_excluded()
        rows = np.setdiff1d(rows, np.asarray(excluded, dtype=np.int64))
        splits_sha = sha256_file(splits_path)
    if rows.size < config.batch_size:
        raise ConfigError(
            f"{rows.size} pretraining stamps, fewer than a batch of {config.batch_size}"
        )
    record = {
        "path": str(path),
        "source_sha256": meta["source"]["sha256"],
        "kind": data.kind,
        "input": "images_f32" if float32 else "images/255 (uint8: a recorded deviation)",
        "n_train": len(train),
        "n_pretrain": int(rows.size),
        "excluded_train": list(excluded),
        "splits_sha256": splits_sha,
    }
    return PilotData(train_images, val_images, rows, train.labels, val.labels, screen, record)


@dataclass(frozen=True, slots=True)
class BatchOrder:
    """The stream of batches: pass permutations keyed by ``(seed, pass)`` (module docstring)."""

    n: int
    batch_size: int
    seed: int

    @property
    def steps_per_pass(self) -> int:
        """Whole batches in one pass."""
        return self.n // self.batch_size

    def batch(self, step: int) -> npt.NDArray[np.int64]:
        """Positions (into the pretraining rows) of the batch of step ``step``."""
        pass_index, i = divmod(step, self.steps_per_pass)
        perm = host_generator(self.seed, pass_index, RESERVED_STREAMS["simclr_batch_order"])
        order = perm.permutation(self.n)
        return order[i * self.batch_size : (i + 1) * self.batch_size].astype(np.int64)


def resolve_device(name: str) -> torch.device:
    """``auto`` -> CUDA, else MPS, else CPU; other names as given.

    Raises:
        ConfigError: for an unknown name or an unavailable accelerator.
    """
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name not in ("cpu", "cuda", "mps"):
        raise ConfigError(f"unknown device {name!r}; use auto, cpu, cuda or mps")
    available = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }
    if not available[name]:
        raise ConfigError(f"device {name!r} is not available")
    return torch.device(name)


def _apply_device_policy(policy: DevicePolicy) -> dict[str, Any]:
    """Set the precision and determinism flags of ``policy`` (ADR 0010, ADR 0014)."""
    cast(Any, torch.backends.cudnn).conv.fp32_precision = policy.fp32_precision
    torch.set_float32_matmul_precision(_MATMUL_PRECISION[policy.fp32_precision])
    flags = policy.torch_flags()
    if flags["CUBLAS_WORKSPACE_CONFIG"] is not None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(flags["CUBLAS_WORKSPACE_CONFIG"])
    torch.backends.cudnn.benchmark = bool(flags["cudnn.benchmark"])
    torch.use_deterministic_algorithms(bool(flags["use_deterministic_algorithms"]))
    return {"fp32_precision": policy.fp32_precision, **flags}


@torch.no_grad()
def embed(
    model: SimCLRNetwork,
    images: torch.Tensor,
    rows: npt.NDArray[np.int64] | None,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(h, z)`` float32 on the CPU of the centred 57-pixel views of ``images[rows]``.

    ``rows=None`` embeds every stamp. Runs in eval mode (BatchNorm on its running
    statistics) and restores the model's mode afterwards.
    """
    was_training = model.training
    model.eval()
    index = np.arange(images.shape[0]) if rows is None else rows
    hs, zs = [], []
    for start in range(0, index.size, batch_size):
        chunk = torch.from_numpy(index[start : start + batch_size]).to(images.device)
        h, z = model(centre_crop(images[chunk]).to(device))
        hs.append(h.float().cpu())
        zs.append(z.float().cpu())
    model.train(was_training)
    if not hs:
        return torch.empty(0, REPRESENTATION_DIM), torch.empty(0, PROJECTION_DIM)
    return torch.cat(hs), torch.cat(zs)


def _knn_bank(data: PilotData, config: PilotConfig) -> npt.NDArray[np.int64] | None:
    """The kNN monitor's bank: up to ``knn_bank_per_class`` pretraining stamps per class."""
    if data.train_labels is None:
        return None
    counts = np.bincount(data.train_labels[data.rows], minlength=len(LABEL_NAMES))
    per_class = min(config.knn_bank_per_class, int(counts.min()))
    if per_class < 1:
        return None
    bank = draw_label_subsets(
        data.train_labels, data.rows, per_class=per_class, n_subsets=1, seed=config.seed
    )
    return bank.subsets[0]


@torch.no_grad()
def diagnose(
    model: SimCLRNetwork,
    data: PilotData,
    images: torch.Tensor,
    config: PilotConfig,
    device: torch.device,
) -> dict[str, Any]:
    """The diagnostics of the model now (module docstring), as a JSON-ready dict.

    The sample (``diag_samples`` pretraining stamps) is drawn from ``(seed, 0,
    simclr_diagnostics)`` and its two views from ``(seed, 1, simclr_diagnostics)``, without
    noise: the same stamps and views at every call, so the values are comparable along the
    run. The kNN monitor (``knn_selection_bacc``) needs labels and a splits file; it is
    ``None`` otherwise.
    """
    was_training = model.training
    model.eval()
    stream = RESERVED_STREAMS["simclr_diagnostics"]
    size = min(config.diag_samples, data.rows.size)
    sample = np.sort(host_generator(config.seed, 0, stream).choice(data.rows, size, replace=False))
    x = images[torch.from_numpy(sample).to(images.device)].to(device)
    params = draw_view_params(config.seed, 1, size, stream=stream)
    v1, v2 = make_views(x, params, ViewConfig())
    _, z1 = model(v1)
    _, z2 = model(v2)
    h, z = model(centre_crop(x))
    z_std = dimension_std(z)
    result: dict[str, Any] = {
        "effective_rank_h": effective_rank(h),
        "z_std_mean": float(z_std.mean()),
        "z_std_min": float(z_std.min()),
        "alignment": alignment(z1, z2),
        "uniformity": uniformity(z),
        "knn_selection_bacc": None,
    }
    bank = _knn_bank(data, config)
    if bank is not None and data.screen is not None and data.validation_labels is not None:
        assert data.train_labels is not None  # noqa: S101 - a bank needs labels
        bank_h, _ = embed(model, images, bank, device, config.embed_batch)
        query = data.screen.selection
        query_h, _ = embed(model, data.validation, query, device, config.embed_batch)
        result["knn_selection_bacc"] = knn_balanced_accuracy(
            bank_h,
            torch.from_numpy(data.train_labels[bank]),
            query_h,
            torch.from_numpy(data.validation_labels[query]),
            n_classes=len(LABEL_NAMES),
        )
    model.train(was_training)
    return result


def dump_embeddings(
    model: SimCLRNetwork,
    data: PilotData,
    out_dir: Path,
    device: torch.device,
    *,
    batch_size: int,
    step: int,
    source: dict[str, Any],
) -> Path:
    """Write ``h`` of every Train and Validation stamp (centred view, eval mode) to ``out_dir``.

    Files: ``train_h.npy``, ``validation_h.npy`` (float32 (N, 512), file order), the label
    arrays when the layout is labelled, and ``meta.json`` (:data:`EMBEDDINGS_FORMAT`, the
    step, the data record and ``source``: the run's config or ``random_init``). Returns
    ``out_dir``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "format": EMBEDDINGS_FORMAT,
        "format_version": _EMBEDDINGS_VERSION,
        "view": "centre-57",
        "step": step,
        "data": data.record,
        "source": source,
        "files": {},
    }
    for split, images, labels in (
        ("train", data.train, data.train_labels),
        ("validation", data.validation, data.validation_labels),
    ):
        h, _ = embed(model, images, None, device, batch_size)
        np.save(out_dir / f"{split}_h.npy", h.numpy())
        meta["files"][f"{split}_h"] = f"{split}_h.npy"
        if labels is not None:
            np.save(out_dir / f"{split}_labels.npy", np.asarray(labels, dtype=np.int64))
            meta["files"][f"{split}_labels"] = f"{split}_labels.npy"
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return out_dir


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def _truncate_jsonl(path: Path, steps_done: int) -> None:
    """Keep only the lines of ``path`` with ``step <= steps_done`` (a resumed run's log)."""
    if not path.exists():
        return
    kept = [
        line for line in path.read_text().splitlines() if json.loads(line)["step"] <= steps_done
    ]
    path.write_text("".join(f"{line}\n" for line in kept))


def _save(payload: dict[str, Any], path: Path) -> None:
    """``torch.save`` through a temporary file and a rename (no half-written checkpoint)."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


class PilotRun:
    """One pilot run: model, optimizer, stream and state (module docstring)."""

    def __init__(self, config: PilotConfig, data: PilotData, device: torch.device) -> None:
        """Build the model on the CPU from ``seed``, move it, build LARS; place the stamps."""
        self.config = config
        self.data = data
        self.device = device
        self.order = BatchOrder(int(data.rows.size), config.batch_size, config.seed)
        per_pass = self.order.steps_per_pass
        self.total_steps = config.epochs * per_pass if config.epochs is not None else 0
        if config.steps is not None:
            self.total_steps = config.steps
        self.warmup_steps = min(round(config.warmup_epochs * per_pass), self.total_steps - 1)
        self.base_lr = base_learning_rate(config.batch_size, config.lr_per_256)
        self.keep_every = (
            None if config.keep_every_epochs is None else config.keep_every_epochs * per_pass
        )
        torch.manual_seed(config.seed)
        self.model = SimCLRNetwork().to(device)
        self.optimizer = build_lars(self.model, lr=self.base_lr, weight_decay=config.weight_decay)
        self.loss = SimCLRLoss(config.temperature)
        self.steps_done = 0
        self.images = data.train.to(device) if config.tier == "device" else data.train
        self.rows = torch.from_numpy(data.rows)
        self.fits: torch.Tensor | None = None
        if config.views.noise:
            fits = stamp_fits(data.train.numpy())
            self.fits = torch.from_numpy(fits).to(self.images.device)

    def schedule(self) -> dict[str, Any]:
        """The run's length and rates, for ``pilot.json``."""
        return {
            "steps_per_pass": self.order.steps_per_pass,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "base_lr": self.base_lr,
            "lr_min": self.config.lr_min,
        }

    def lr(self, step: int) -> float:
        """The rate of the step after ``step`` steps."""
        return warmup_cosine_lr(
            step,
            base_lr=self.base_lr,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            lr_min=self.config.lr_min,
        )

    def step(self, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Train on the batch of step ``step``; returns ``(loss, pos, neg, lr)`` (no sync)."""
        config = self.config
        index = self.rows[torch.from_numpy(self.order.batch(step))].to(self.images.device)
        x = self.images[index].to(self.device, non_blocking=True)
        fits = None if self.fits is None else self.fits[index].to(self.device)
        generator = noise_generator(config.seed, step, self.device) if config.views.noise else None
        params = draw_view_params(config.seed, step, config.batch_size)
        v1, v2 = make_views(x, params, config.views, fits=fits, generator=generator)
        self.model.train()
        _, z = self.model(torch.cat([v1, v2]))
        loss, pos, neg = self.loss(z[: config.batch_size], z[config.batch_size :])
        lr = self.lr(step)
        set_learning_rate(self.optimizer, lr)
        self.optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(loss)
        self.optimizer.step()
        self.steps_done = step + 1
        return loss.detach(), pos, neg, lr

    def checkpoint(self) -> dict[str, Any]:
        """The checkpoint payload of the current state (plain values only)."""
        return {
            "format": CKPT_FORMAT,
            "format_version": _CKPT_VERSION,
            "config": self.config.to_plain(),
            "data": self.data.record,
            "steps_done": self.steps_done,
            "total_steps": self.total_steps,
            "model": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "optimizer": self.optimizer.state_dict(),
        }

    def restore(self, payload: dict[str, Any]) -> None:
        """Load a checkpoint of this run (same config) into the model and optimizer.

        Raises:
            ConfigError: if the checkpoint is of another format or another config.
        """
        if payload.get("format") != CKPT_FORMAT or payload.get("format_version") != _CKPT_VERSION:
            raise ConfigError("not a pilot checkpoint of this version")
        if payload["config"] != self.config.to_plain():
            raise ConfigError("the checkpoint was written with another config; refusing to resume")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.steps_done = int(payload["steps_done"])


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Read a pilot checkpoint with ``weights_only=True``.

    Raises:
        DataError: if the file is missing or is not a pilot checkpoint.
    """
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError) as err:
        raise DataError(f"{path}: cannot read the checkpoint: {err}") from err
    if not isinstance(payload, dict) or payload.get("format") != CKPT_FORMAT:
        raise DataError(f"{path}: not a {CKPT_FORMAT}")
    return cast(dict[str, Any], payload)


def _record(run: PilotRun, flags: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": run.config.to_plain(),
        "data": run.data.record,
        "schedule": run.schedule(),
        "device": str(run.device),
        "numerics": flags,
        "noise_fields": "device-dependent" if run.config.views.noise else None,
        "versions": {
            "astro_simclr": version("astro-simclr"),
            "torch": str(torch.__version__),
            "python": platform.python_version(),
        },
    }


def _intervals(run: PilotRun, done: int) -> Iterator[str]:
    config = run.config
    final = done == run.total_steps
    if done % config.log_every == 0 or final:
        yield "log"
    if done % config.diagnostics_every == 0 or final:
        yield "diagnostics"
    if done % config.checkpoint_every == 0 or final:
        yield "checkpoint"
    if run.keep_every is not None and done % run.keep_every == 0:
        yield "keep"


def _after_step(
    run: PilotRun,
    out_dir: Path,
    done: int,
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, float],
) -> None:
    """Log, diagnose, checkpoint and keep at the intervals that ``done`` steps reach."""
    loss, pos, neg, lr = outputs
    for event in _intervals(run, done):
        if event == "log":
            value = float(loss)
            if not np.isfinite(value):
                raise FloatingPointError(f"loss {value} at step {done}")
            row = {"step": done, "loss": value, "pos": float(pos), "neg": float(neg), "lr": lr}
            _append_jsonl(out_dir / "metrics.jsonl", row)
        elif event == "diagnostics":
            diag = diagnose(run.model, run.data, run.images, run.config, run.device)
            _append_jsonl(out_dir / "diagnostics.jsonl", {"step": done, **diag})
        elif event == "checkpoint":
            _save(run.checkpoint(), out_dir / "ckpt" / "last.pt")
        else:
            _save(run.checkpoint(), out_dir / "ckpt" / f"step-{done:07d}.pt")


def train_pilot(
    config: PilotConfig,
    out_dir: Path,
    *,
    resume: bool = False,
    data: PilotData | None = None,
    stop_at: int | None = None,
) -> PilotRun:
    """Run (or resume) the pilot into ``out_dir`` and return the run.

    ``data`` skips :func:`load_pilot_data` (tests, or a caller that already read it).
    ``stop_at`` ends this call once that many steps are done (a time-boxed job): it writes
    ``ckpt/last.pt`` there and returns, and a later ``resume`` continues the same run (the
    schedule is still the whole run's). Without it, or when the run reaches its end, the
    call writes ``ckpt/final.pt`` and the embeddings.

    Raises:
        ConfigError: if ``out_dir`` already holds a run and ``resume`` is False, or the
            checkpoint to resume is of another config; or as :class:`PilotConfig`.
        DataError: as :func:`load_pilot_data`.
        FloatingPointError: if the loss becomes NaN or infinite (checked at log steps).
    """
    ckpt_dir = out_dir / "ckpt"
    last = ckpt_dir / "last.pt"
    if last.exists() and not resume:
        raise ConfigError(f"{out_dir} already holds a run; pass resume to continue it")
    device = resolve_device(config.device)
    flags = _apply_device_policy(
        DevicePolicy(cast(Any, device.type), config.deterministic, config.fp32_precision)
    )
    data = load_pilot_data(config) if data is None else data
    run = PilotRun(config, data, device)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if resume and last.exists():
        run.restore(load_checkpoint(last))
    for name in ("metrics.jsonl", "diagnostics.jsonl"):
        _truncate_jsonl(out_dir / name, run.steps_done)
    (out_dir / "pilot.json").write_text(json.dumps(_record(run, flags), indent=1) + "\n")
    end = run.total_steps if stop_at is None else min(stop_at, run.total_steps)
    for step in range(run.steps_done, end):
        _after_step(run, out_dir, step + 1, run.step(step))
    if run.steps_done < run.total_steps:
        _save(run.checkpoint(), last)
        return run
    _save(run.checkpoint(), ckpt_dir / "final.pt")
    dump_embeddings(
        run.model,
        data,
        out_dir / "embeddings",
        device,
        batch_size=config.embed_batch,
        step=run.steps_done,
        source={"config": config.to_plain()},
    )
    return run
