"""Tests of the plain engine's frozen contracts (``astro_simclr.engine``, task B0, ADR 0014).

Two kinds:

- the contracts themselves (dataclass checks, the pure policy rules, the checkpoint key set,
  the generators, the engine-free import rule);
- **Lightning reference facts**: what Lightning 2.6.6 and today's ``training`` code do, as
  ADR 0014's checklist states them, pinned here so that a Lightning upgrade that changes
  one fails before the 4.7(a) differential test is written against it. Each policy rule is
  checked against the Lightning-side code it replaces (``HorizonCallback._improves``,
  ``_set_torch_flags``, ``class_augment.augment_rng``).
"""

from __future__ import annotations

import io
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pytest
import torch
from lightning.pytorch.trainer.connectors.accelerator_connector import _set_torch_flags
from torch import nn
from torch.utils.data import DataLoader

from astro_simclr import rj23_regime
from astro_simclr.config import ConfigError, Settings
from astro_simclr.engine import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_KEYS,
    CUBLAS_WORKSPACE_CONFIG,
    RESERVED_STREAMS,
    DataTier,
    DevicePolicy,
    EarlyStop,
    Horizon,
    Monitor,
    Schedule,
    TierChoice,
    ValidationGrid,
    check_resumable,
    device_generator,
    host_generator,
    mean_of_batch_losses,
)
from astro_simclr.engine import data as engine_data
from astro_simclr.engine import stream as engine_stream
from astro_simclr.training import class_augment, device_data, flux_rescale
from astro_simclr.training.schedule import HorizonCallback

# --- the engine-free rule and the frozen batch format ---------------------------------


def test_engine_imports_no_lightning() -> None:
    code = (
        "import sys, astro_simclr.engine; "
        "roots = ('lightning', 'pytorch_lightning'); "
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in roots); "
        "assert not bad, bad"
    )
    # A fresh interpreter: this process has imported Lightning already.
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603 (fixed argv)


def test_batch_format_is_device_data_s() -> None:
    assert engine_data.Batch == device_data.Batch
    assert engine_data.TrainBatch == device_data.TrainBatch


# --- monitor ----------------------------------------------------------------------------

_VALUES = [0.5, 0.25, 0.5000001, float("nan"), float("inf"), -1.0]


@pytest.mark.parametrize("mode", ["min", "max"])
def test_monitor_rule_is_horizon_callback_s(mode: str, tmp_path: Path) -> None:
    monitor = Monitor("loss_val", mode)  # type: ignore[arg-type]
    callback = HorizonCallback(dirpath=tmp_path, mode=mode)  # type: ignore[arg-type]
    for best in [None, *_VALUES]:
        callback.best_model_score = None if best is None else torch.tensor(best)
        stored = None if best is None else float(torch.tensor(best))
        for value in _VALUES:
            assert monitor.improves(value, stored) == callback._improves(value), (best, value)


def test_monitor_is_strict_and_nan_never_improves() -> None:
    monitor = Monitor()
    assert monitor.improves(1.0, None)
    assert not monitor.improves(math.nan, None)
    assert not monitor.improves(1.0, 1.0)
    assert not monitor.improves(math.nan, 1.0)
    assert Monitor("accuracy_val", "max").improves(0.6, 0.5)


def test_monitor_checks() -> None:
    with pytest.raises(ConfigError, match="name"):
        Monitor("")
    with pytest.raises(ConfigError, match="mode"):
        Monitor("loss_val", "lowest")  # type: ignore[arg-type]


def test_monitor_value_is_the_mean_of_batch_means() -> None:
    # Two full batches and a partial one: the batch means weigh equally (TF's rule), so the
    # result is not the per-sample mean.
    per_sample = [torch.full((64,), 1.0), torch.full((64,), 2.0), torch.full((52,), 4.0)]
    losses = [values.mean() for values in per_sample]
    value = mean_of_batch_losses(losses)
    assert value.dtype == torch.float32
    assert torch.equal(value, torch.stack(losses).mean())
    assert float(value) == pytest.approx(7 / 3)
    assert float(value) != pytest.approx(float(torch.cat(per_sample).mean()))


def test_monitor_value_checks() -> None:
    with pytest.raises(ValueError, match="no validation batch"):
        mean_of_batch_losses([])
    with pytest.raises(ValueError, match="0-d float32"):
        mean_of_batch_losses([torch.tensor(1.0, dtype=torch.float64)])
    with pytest.raises(ValueError, match="0-d float32"):
        mean_of_batch_losses([torch.ones(2)])


# --- validation grid, horizon, early stop, schedule -------------------------------------


def test_grid_validates_on_multiples_only() -> None:
    grid = ValidationGrid(10)
    assert [s for s in range(0, 41) if grid.due(s)] == [10, 20, 30, 40]
    assert not grid.due(1_136)  # the epoch boundary of `red` triggers nothing
    assert grid.due(1_140)
    with pytest.raises(ConfigError):
        ValidationGrid(0)
    with pytest.raises(ConfigError):
        ValidationGrid(True)


def test_horizon_rule() -> None:
    horizon = Horizon(30_000, 10_000)
    assert horizon.extended(30_000, 4_780) == 30_000
    assert horizon.extended(30_000, 25_000) == 35_000
    assert Horizon(100, 0).extended(100, 100) == 100  # a fixed cap
    assert horizon.resumed(35_000) == 35_000
    assert Horizon(40_000, 10_000).resumed(35_000) == 40_000
    with pytest.raises(ConfigError):
        Horizon(0, 1)
    with pytest.raises(ConfigError):
        Horizon(1, -1)


def test_early_stop_checks() -> None:
    assert EarlyStop(5).min_delta == 0.0
    with pytest.raises(ConfigError, match="patience"):
        EarlyStop(0)
    with pytest.raises(ConfigError, match="min_delta"):
        EarlyStop(5, min_delta=-1.0)


def test_default_schedule_is_the_carrasco2021_settings() -> None:
    settings, schedule = Settings(), Schedule()
    assert schedule.horizon == Horizon(settings.max_steps, settings.horizon_increment)
    assert schedule.validation.every_n_steps == settings.val_every_n_steps
    assert schedule.checkpoint_every_n_steps == settings.checkpoint_every_n_steps
    assert schedule.monitor == Monitor("loss_val", "min")
    assert schedule.keep == ("best",)
    assert schedule.early_stop is None


def test_schedule_checks() -> None:
    assert Schedule(keep=("stop", "best"), early_stop=EarlyStop(5)).keep == ("stop", "best")
    assert Schedule(checkpoint_every_n_steps=None).checkpoint_every_n_steps is None
    for keep in [(), ("best", "best"), ("last",)]:
        with pytest.raises(ConfigError, match="keep"):
            Schedule(keep=keep)  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="multiple"):
        Schedule(checkpoint_every_n_steps=15)
    with pytest.raises(ConfigError, match="multiple"):
        Schedule(checkpoint_every_n_steps=0)


# --- Lightning reference facts (ADR 0014 checklist) -------------------------------------

_STEPS_PER_EPOCH = 7  # not a multiple of the validation interval


class _Probe(L.LightningModule):
    """Records what the model sees at every fit validation."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.head = nn.Dropout(0.5)
        self.head.eval()  # a submodule left in eval: Lightning restores per submodule
        self.seen: list[dict[str, Any]] = []
        self.train_steps: list[int] = []

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        self.train_steps.append(self.global_step)
        return self.w.sum()

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self.seen.append(
            {
                "step": self.global_step,
                "training": self.training,
                "grad": self.w.grad,
                "grad_enabled": torch.is_grad_enabled(),
                "inference": torch.is_inference_mode_enabled(),
                "fn": self.trainer.state.fn,
            }
        )

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        # After a validation, the module is back in train mode with the eval child kept.
        assert self.training and not self.head.training

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def train_dataloader(self) -> DataLoader[torch.Tensor]:
        return DataLoader(torch.zeros(_STEPS_PER_EPOCH, 1), batch_size=1)  # type: ignore[arg-type]

    def val_dataloader(self) -> DataLoader[torch.Tensor]:
        return DataLoader(torch.zeros(1, 1), batch_size=1)  # type: ignore[arg-type]


def _fit(max_steps: int, tmp_path: Path) -> tuple[_Probe, L.Trainer]:
    module = _Probe()
    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=-1,
        max_steps=max_steps,
        val_check_interval=10,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(module)
    return module, trainer


@pytest.mark.parametrize("max_steps", [25, 30])
def test_lightning_validation_grid_and_epochs(max_steps: int, tmp_path: Path) -> None:
    module, trainer = _fit(max_steps, tmp_path)
    grid = ValidationGrid(10)
    # On the grid only: not at the 7-step epoch ends, not forced after a final step off it.
    assert [seen["step"] for seen in module.seen] == [
        s for s in range(1, max_steps + 1) if grid.due(s)
    ]
    # global_step inside training_step is the steps done before the step.
    assert module.train_steps == list(range(max_steps))
    # metrics.json's last_epoch (current_epoch - 1) is the epoch of the last step.
    assert trainer.current_epoch - 1 == (max_steps - 1) // _STEPS_PER_EPOCH


def test_lightning_fit_validation_mode(tmp_path: Path) -> None:
    module, _ = _fit(20, tmp_path)
    for seen in module.seen:
        assert seen["training"] is False  # eval()
        assert seen["grad"] is None  # zero_grad(set_to_none=True) before the validation
        # The fit's validations run under no_grad, NOT inference_mode (the epoch loop
        # builds its validation loop with inference_mode=False).
        assert seen["grad_enabled"] is False
        assert seen["inference"] is False


def test_lightning_validate_uses_inference_mode(tmp_path: Path) -> None:
    module, trainer = _fit(10, tmp_path)
    module.seen.clear()
    trainer.validate(module, verbose=False)
    assert module.seen and all(seen["inference"] for seen in module.seen)


@pytest.mark.parametrize("deterministic", [False, True])
def test_device_flags_are_lightning_s(deterministic: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    saved = (
        torch.backends.cudnn.benchmark,
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        _set_torch_flags(deterministic=deterministic, benchmark=not deterministic)
        lightning = {
            "cudnn.benchmark": torch.backends.cudnn.benchmark,
            "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        }
    finally:
        torch.backends.cudnn.benchmark = saved[0]
        torch.use_deterministic_algorithms(saved[1], warn_only=saved[2])
    assert DevicePolicy("cpu", deterministic=deterministic).torch_flags() == lightning


# --- device, data tiers -----------------------------------------------------------------


def test_device_policy_refuses_multi_gpu_and_bad_values() -> None:
    assert DevicePolicy("cuda").cudnn_benchmark
    assert not DevicePolicy("cuda", deterministic=True).cudnn_benchmark
    assert DevicePolicy("cpu", deterministic=True).torch_flags()["CUBLAS_WORKSPACE_CONFIG"] == (
        CUBLAS_WORKSPACE_CONFIG
    )
    with pytest.raises(ConfigError, match="one device only"):
        DevicePolicy("cuda", world_size=2)
    with pytest.raises(ConfigError, match="accelerator"):
        DevicePolicy("tpu")  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="fp32_precision"):
        DevicePolicy("cpu", fp32_precision="bf16")  # type: ignore[arg-type]


def test_quantised_storage_is_a_deviation() -> None:
    assert not TierChoice().is_deviation
    assert not TierChoice(DataTier.PINNED).is_deviation
    assert TierChoice(DataTier.DEVICE, "uint8").is_deviation
    assert TierChoice(DataTier.MEMMAP, "float16").is_deviation
    assert [tier.value for tier in DataTier] == ["device", "pinned", "memmap"]
    with pytest.raises(ConfigError, match="tier"):
        TierChoice("disk")  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="dtype"):
        TierChoice(DataTier.DEVICE, "bfloat16")  # type: ignore[arg-type]


# --- randomness -------------------------------------------------------------------------


def test_reserved_streams_are_the_augmentations_numbers() -> None:
    assert RESERVED_STREAMS == {
        "aug_noise": class_augment.NOISE_STREAM,
        "aug_feature_jitter": class_augment.JITTER_STREAM,
        "synthetic_bogus": class_augment.SYNTHETIC_STREAM,
        "flux_rescale": flux_rescale.FLUX_STREAM,
        "block_shuffle": engine_stream.BLOCK_STREAM_WORD,
        # Phase 5's SimCLR pilot (astro_simclr.simclr).
        "simclr_views": 6,
        "simclr_view_noise": 7,
        "simclr_batch_order": 8,
        "simclr_diagnostics": 9,
        "rj23_class_uniform": rj23_regime.STREAM,
    }
    assert len(set(RESERVED_STREAMS.values())) == len(RESERVED_STREAMS)


def test_reserved_stream_numbers_are_unique() -> None:
    # Two augmentations on one stream number would draw correlated numbers at every step.
    numbers = list(RESERVED_STREAMS.values())
    assert len(set(numbers)) == len(numbers)
    assert all(isinstance(n, int) and not isinstance(n, bool) and n >= 1 for n in numbers)
    assert RESERVED_STREAMS["rj23_class_uniform"] == 10


def test_host_generator_is_augment_rng() -> None:
    for seed, step, stream in [(0, 0, 1), (7, 4_780, 4), (2**40, 3, 2)]:
        ours = host_generator(seed, step, stream).random(5)
        theirs = class_augment.augment_rng(seed, step, stream).random(5)
        assert np.array_equal(ours, theirs)


def test_device_generator_is_seeded_by_every_word() -> None:
    cpu = torch.device("cpu")

    def draw(seed: int, step: int, stream: int) -> torch.Tensor:
        return torch.randn(8, generator=device_generator(seed, step, stream, cpu))

    assert torch.equal(draw(1, 2, 3), draw(1, 2, 3))
    others = [draw(2, 2, 3), draw(1, 3, 3), draw(1, 2, 4), draw(1 + 2**32, 2, 3)]
    assert not any(torch.equal(draw(1, 2, 3), other) for other in others)
    assert device_generator(0, 0, 0, cpu).device == cpu


@pytest.mark.parametrize("words", [(-1, 0, 0), (0, -1, 0), (0, 0, -1)])
def test_generators_refuse_negative_words(words: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError, match=">= 0"):
        host_generator(*words)
    with pytest.raises(ValueError, match=">= 0"):
        device_generator(*words, torch.device("cpu"))


# --- checkpoint -------------------------------------------------------------------------


def test_checkpoint_key_set_is_frozen() -> None:
    expected = {
        "format_version",
        "engine",
        "identity",
        "steps_done",
        "model",
        "optimizer",
        "task",
        "policy",
        "stream",
        "rng",
        "numerics",
        "versions",
    }
    assert expected == CHECKPOINT_KEYS


def _example_checkpoint() -> dict[str, Any]:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    numpy_state = np.random.get_state(legacy=False)
    numpy_state["state"]["key"] = torch.from_numpy(numpy_state["state"]["key"].astype(np.int64))
    return {
        "format_version": CHECKPOINT_FORMAT,
        "engine": "plain",
        "identity": {"run_id": "abc123", "attempt": 1, "split_run": None},
        "steps_done": 1_000,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "task": {},
        "policy": {
            "monitor": "loss_val",
            "mode": "min",
            "horizon": 30_000,
            "best_score": 0.75,
            "best_step": 990,
            "wait_count": 0,
            "best_model": {k: v.clone() for k, v in model.state_dict().items()},
        },
        "stream": {"n": 72_704, "batch_size": 64, "seed": 0},
        "rng": {
            "python": random.getstate(),
            "numpy": numpy_state,
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": None,
            "torch_mps": None,
        },
        "numerics": {"fp32_precision": "ieee", "deterministic": False},
        "versions": {"astro_simclr": "0.0.0", "torch": str(torch.__version__), "git_sha": None},
    }


def test_checkpoint_is_plain_values_and_resumable() -> None:
    checkpoint = _example_checkpoint()
    assert set(checkpoint) == CHECKPOINT_KEYS
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=True)  # plain values only
    check_resumable(loaded)
    assert loaded["rng"]["python"] == checkpoint["rng"]["python"]
    assert torch.equal(loaded["rng"]["torch_cpu"], checkpoint["rng"]["torch_cpu"])


def test_check_resumable_refuses_other_checkpoints() -> None:
    lightning = {"pytorch-lightning_version": L.__version__, "state_dict": {}, "loops": {}}
    with pytest.raises(ConfigError, match="Lightning checkpoint"):
        check_resumable(lightning)
    with pytest.raises(ConfigError, match="engine"):
        check_resumable({**_example_checkpoint(), "engine": "other"})
    with pytest.raises(ConfigError, match="format"):
        check_resumable({**_example_checkpoint(), "format_version": CHECKPOINT_FORMAT + 1})
    incomplete = _example_checkpoint()
    del incomplete["rng"]
    with pytest.raises(ConfigError, match="lacks rng"):
        check_resumable(incomplete)
