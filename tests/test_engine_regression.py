"""The plain engine's golden regression set (task 4.8; ADR 0014, amendment "4.8").

The 4.7(a) differential test ran Lightning's ``run_command`` and the plain ``run_plain`` in
one process and held them bit for bit equal. Lightning is gone, so its outputs are
replaced by **goldens**: the plain engine's outputs for every case, recorded (before
Lightning was removed, while that test passed) in
``tests/fixtures/engine_golden/<platform>/<case>.json``, and asserted bit for bit here:

- the initial weights (``build_paper_task`` / ``RJ23Task`` right after the seed), as a
  SHA-256 of every tensor;
- every fit validation: its step, the monitored value (the same Python float) and a
  SHA-256 of the weights then;
- ``metrics.json``, except what names the engine, the machine or the run and the
  temporary paths (``_engine_cases.computed``): the steps, horizon, epochs and
  ``monitor_best``, ``paper_data``, ``preset``, ``protocol``, every evaluation entry
  (accuracies, macro-F1, recalls, both confusion matrices) and the rj23 schedule record;
- every array of every ``predictions_*.npz``, as a SHA-256;
- the manifest: ``engine: plain``, ``finished``.

The cases (``_engine_cases``) are the differential test's: every 2.5 and D3 flag
(``FLAGS``), split runs (a fold and the screen) and the rj23 scenarios, protocols,
sensitivity flags and both stop reasons. The resume tests are in
``test_engine_resume.py``.

**Platforms.** CPU kernels differ in their rounding between instruction sets (x86 AVX2 /
AVX-512 against arm64 NEON, or another BLAS), so a golden holds on the platform it was
recorded on: ``<sys.platform>-<machine>-<torch CPU capability>``
(``_engine_cases.platform_key``: ``darwin-arm64-default``, ``linux-x86_64-avx2``, ...). On a
platform without goldens every case is skipped, naming it. A golden recorded with another torch version still applies:
a torch upgrade that changes these numbers is a numerics change (bump ``NUMERICS_EPOCH``)
and fails here until the goldens are recorded again.

**Recording** (a new flag, a new platform, or a deliberate numerics change, said in the
PR)::

    ASTRO_SIMCLR_RECORD_GOLDEN=1 uv run pytest tests/test_engine_regression.py

writes this platform's goldens (every case passes) instead of checking them; review the
diff of ``tests/fixtures/engine_golden/`` before committing it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from _engine_cases import (
    FLAGS,
    RJ23_CASES,
    SCHEDULE,
    SPLIT_RUNS,
    PlainProbe,
    computed,
    cpu_state,
    make_split_data,
    paper_settings,
    platform_key,
    predictions_digests,
    rj23_settings,
    state_digest,
)
from _rj23_support import DATASET, rj23_layout

from astro_simclr.config import Settings
from astro_simclr.engine import host_generator, seed_everything
from astro_simclr.protocol.split_runs import SplitRequest
from astro_simclr.rj23_regime import STREAM, ClassUniformStream
from astro_simclr.tasks.paper import build_paper_task
from astro_simclr.tasks.plain import run_plain
from astro_simclr.tasks.rj23 import DevicePart, RJ23Request, RJ23Task, RJ23TrainSource, rj23_inputs

#: Where the goldens are, one directory per platform.
GOLDEN_DIR = Path(__file__).parent / "fixtures" / "engine_golden"
#: Set to 1 to record this platform's goldens instead of checking them (module docstring).
RECORD = os.environ.get("ASTRO_SIMCLR_RECORD_GOLDEN") == "1"
#: The golden document's format; bumped if its keys change.
GOLDEN_FORMAT = 1


def _golden_path(case: str) -> Path:
    return GOLDEN_DIR / platform_key() / f"{case}.json"


def _check_golden(case: str, outputs: dict[str, Any]) -> None:
    """Record ``outputs`` as the golden of ``case`` (``RECORD``), or assert them equal to it."""
    path = _golden_path(case)
    document = {"format": GOLDEN_FORMAT, "case": case, "torch": torch.__version__, **outputs}
    if RECORD:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return
    if not path.is_file():
        pytest.skip(f"no golden for platform {platform_key()} (record it: module docstring)")
    golden = json.loads(path.read_text(encoding="utf-8"))
    assert golden["format"] == GOLDEN_FORMAT
    # In order: the first difference is the most telling (initial weights, then training).
    for key in ("initial", "validations", "metrics", "predictions"):
        assert outputs[key] == golden[key], (
            f"{case}: {key} differ from the golden (recorded with torch {golden['torch']}, "
            f"running {torch.__version__}); a deliberate numerics change re-records it"
        )


def _validations(probe: PlainProbe) -> list[list[Any]]:
    return [[step, value, state_digest(state)] for step, value, state in probe.validations]


def _assert_finished_plain(run_dir: Path) -> None:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert (manifest["engine"], manifest["status"]) == ("plain", "finished")


def _paper_outputs(
    settings: Settings, split: SplitRequest | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the paper case; return (the golden outputs, the metrics)."""
    probe = PlainProbe()
    metrics = run_plain("train-supervised", settings, split=split, on_validation=probe)
    seed_everything(settings.seed)
    initial = cpu_state(build_paper_task(settings, {}).module)
    _assert_finished_plain(settings.output_dir)
    assert metrics["machine"]["engine"] == "plain"
    outputs = {
        "initial": state_digest(initial),
        "validations": _validations(probe),
        "metrics": computed(metrics),
        "predictions": predictions_digests(settings.output_dir),
    }
    return outputs, metrics


@pytest.fixture(scope="module")
def split_data(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """(prepared dir, splits file) with 25 stamps per split (``make_split_data``)."""
    return make_split_data(tmp_path_factory)


@pytest.fixture(scope="module")
def rj23_data(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """(data directory, object splits) of a synthetic 5-class ``raw-alerts`` layout."""
    return rj23_layout(tmp_path_factory.mktemp("golden_rj23"))


def test_platform_key_names_the_cpu() -> None:
    key = platform_key()
    assert key == key.lower() and key.count("-") >= 2
    assert sys.platform in key


@pytest.mark.parametrize("flags", FLAGS)
def test_paper_case_matches_its_golden(
    flags: dict[str, Any], classifier_f32_dir: Path, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    settings = paper_settings(classifier_f32_dir, tmp_path / "plain", **flags)
    outputs, metrics = _paper_outputs(settings)
    if settings.horizon_increment:
        assert metrics["horizon"] > SCHEDULE["max_steps"]  # the schedule did extend
    else:
        assert metrics["last_step"] == metrics["horizon"] == settings.max_steps
    _check_golden(f"paper-{request.node.callspec.id}", outputs)


@pytest.mark.parametrize(("flags", "fold"), SPLIT_RUNS)
def test_split_run_matches_its_golden(
    flags: dict[str, Any],
    fold: int | None,
    split_data: tuple[Path, Path],
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    prepared, path = split_data
    settings = paper_settings(prepared, tmp_path / "plain", **flags)
    outputs, metrics = _paper_outputs(settings, SplitRequest(path, fold))
    assert "protocol" in metrics and "evaluation_accuracy" in metrics
    _check_golden(f"split-{request.node.callspec.id}", outputs)


def _assert_batches_from_stream(settings: Settings, request: RJ23Request, steps: int) -> None:
    """Each training batch's classes are drawn on the class-uniform stream's reserved number."""
    assert STREAM == 10
    inputs = rj23_inputs(settings, request)
    train = inputs.parts[inputs.plan.train]
    n_classes = inputs.data.n_classes
    stream = ClassUniformStream(
        train.labels, settings.batch_size, settings.seed, n_classes=n_classes
    )
    source = RJ23TrainSource(DevicePart.of(train, "cpu"), stream, inputs.identity)
    for step in range(steps):
        labels = source.batch(step)[2].numpy()
        drawn = host_generator(settings.seed, step, 10).integers(0, n_classes, settings.batch_size)
        assert np.array_equal(labels, drawn), step


@pytest.mark.parametrize(("preset", "changes", "mode", "reason"), RJ23_CASES)
def test_rj23_case_matches_its_golden(
    preset: str,
    changes: dict[str, Any],
    mode: Any,
    reason: str,
    rj23_data: tuple[Path, Path],
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    data_dir, splits = rj23_data
    settings = rj23_settings(data_dir, tmp_path / "plain", DATASET, preset, **changes)
    rj23 = RJ23Request(splits, mode)
    probe = PlainProbe()
    metrics = run_plain("train-supervised", settings, rj23=rj23, on_validation=probe)
    seed_everything(settings.seed)
    initial = cpu_state(RJ23Task(settings, 5).module)
    assert probe.validations[0][0] == 0  # the initial weights are validated
    assert metrics["schedule"]["stopped_by"] == reason
    assert metrics["machine"]["engine"] == "plain"
    _assert_finished_plain(settings.output_dir)
    _assert_batches_from_stream(settings, rj23, metrics["last_step"])
    outputs = {
        "initial": state_digest(initial),
        "validations": _validations(probe),
        "metrics": computed(metrics),
        "predictions": predictions_digests(settings.output_dir),
    }
    _check_golden(f"rj23-{request.node.callspec.id}", outputs)


def test_golden_directory_has_no_stray_case() -> None:
    # Every recorded golden belongs to a current case: a renamed or dropped case must not
    # leave a golden nothing checks.
    cases = {f"paper-{p.id}" for p in FLAGS}
    cases |= {f"split-{p.id}" for p in SPLIT_RUNS}
    cases |= {f"rj23-{p.id}" for p in RJ23_CASES}
    recorded = sorted(GOLDEN_DIR.glob("*/*.json"))
    assert recorded, "no golden recorded on any platform"
    for path in recorded:
        assert path.stem in cases, path
