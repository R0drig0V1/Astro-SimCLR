"""Tests of the plain engine's frozen contracts (``astro_simclr.engine``, task B0, ADR 0014).

The contracts themselves: the dataclass checks, the pure policy rules (the monitor's strict
comparison, the validation grid, the horizon, the device flags), the checkpoint key set,
the generators and the engine-free import rule. Until task 4.8 several rules were checked
against the Lightning code they replaced (``HorizonCallback._improves``, Lightning's
``_set_torch_flags``); those expectations are now written out here, and the loop that
follows them is ``test_engine_loop.py``'s.
"""

from __future__ import annotations

import io
import math
import random
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

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
from astro_simclr.training import class_augment, flux_rescale, tensor_splits

# --- the engine-free rule and the frozen batch format ---------------------------------


def test_engine_imports_no_lightning() -> None:
    code = (
        "import sys, astro_simclr.engine; "
        "roots = ('lightning', 'pytorch_lightning', 'torchmetrics'); "
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in roots); "
        "assert not bad, bad"
    )
    # A fresh interpreter: another test of this process may have imported them.
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603 (fixed argv)


def test_batch_format_is_the_tensor_splits_one() -> None:
    assert engine_data.Batch == tensor_splits.Batch
    assert engine_data.TrainBatch == tensor_splits.TrainBatch


# --- monitor ----------------------------------------------------------------------------

#: The best as the engine stores it: the Python float of a float32 value.
_F32_HALF_PLUS = float(torch.tensor(0.5000001))  # 0.50000011920928955...


@pytest.mark.parametrize(
    ("mode", "best", "value", "improves"),
    [
        # No best yet: anything but NaN is one (inf too).
        ("min", None, 0.5, True),
        ("min", None, math.inf, True),
        ("min", None, math.nan, False),
        ("max", None, -math.inf, True),
        ("max", None, math.nan, False),
        # Strict: an equal value is not a new best.
        ("min", 0.5, 0.5, False),
        ("max", 0.5, 0.5, False),
        ("min", 0.5, 0.25, True),
        ("min", 0.5, -1.0, True),
        ("min", 0.5, math.inf, False),
        ("max", 0.5, math.inf, True),
        ("max", 0.5, -1.0, False),
        # The comparison is on Python floats: 0.5000001 is above 0.5, and below the float32
        # rounding of itself (which is how a stored best holds it).
        ("min", 0.5, 0.5000001, False),
        ("max", 0.5, 0.5000001, True),
        ("min", _F32_HALF_PLUS, 0.5000001, True),
        ("max", _F32_HALF_PLUS, 0.5000001, False),
        # NaN never improves, and nothing improves on a NaN best.
        ("min", 0.5, math.nan, False),
        ("max", 0.5, math.nan, False),
        ("min", math.nan, -1.0, False),
        ("max", math.nan, 1.0, False),
        ("min", math.inf, 0.5, True),
        ("max", -1.0, math.inf, True),
    ],
)
def test_monitor_rule(mode: str, best: float | None, value: float, improves: bool) -> None:
    assert Monitor("loss_val", mode).improves(value, best) is improves  # type: ignore[arg-type]


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


# --- device flags ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("accelerator", "deterministic", "expected"),
    [
        # cudnn.benchmark is on unless deterministic (the legacy benchmark=True), as the
        # Lightning Trainer set it; CUBLAS_WORKSPACE_CONFIG only for a deterministic run.
        (
            "cuda",
            False,
            {
                "cudnn.benchmark": True,
                "use_deterministic_algorithms": False,
                "CUBLAS_WORKSPACE_CONFIG": None,
            },
        ),
        (
            "cpu",
            True,
            {
                "cudnn.benchmark": False,
                "use_deterministic_algorithms": True,
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            },
        ),
        (
            "cpu",
            False,
            {
                "cudnn.benchmark": True,
                "use_deterministic_algorithms": False,
                "CUBLAS_WORKSPACE_CONFIG": None,
            },
        ),
    ],
)
def test_device_flags(accelerator: str, deterministic: bool, expected: dict[str, object]) -> None:
    policy = DevicePolicy(accelerator, deterministic=deterministic)  # type: ignore[arg-type]
    assert policy.torch_flags() == expected
    assert CUBLAS_WORKSPACE_CONFIG == ":4096:8"


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
        # Task 5.1's corpus index (build-corpus-index --duplicates random).
        "corpus_duplicates": 11,
        # The pilot's online linear probe (astro_simclr.simclr.monitor.side_batch).
        "simclr_online_probe": 12,
        # The go/no-go's supervised scratch baseline (astro_simclr.simclr.scratch).
        "simclr_scratch": 13,
    }
    assert len(set(RESERVED_STREAMS.values())) == len(RESERVED_STREAMS)


def test_reserved_stream_numbers_are_unique() -> None:
    # Two augmentations on one stream number would draw correlated numbers at every step.
    numbers = list(RESERVED_STREAMS.values())
    assert len(set(numbers)) == len(numbers)
    assert all(isinstance(n, int) and not isinstance(n, bool) and n >= 1 for n in numbers)
    assert RESERVED_STREAMS["rj23_class_uniform"] == 10
    assert RESERVED_STREAMS["simclr_online_probe"] == 12  # assigned by the integrator
    assert RESERVED_STREAMS["simclr_scratch"] == 13  # assigned by the integrator


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
    lightning = {"pytorch-lightning_version": "2.6.6", "state_dict": {}, "loops": {}}
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
