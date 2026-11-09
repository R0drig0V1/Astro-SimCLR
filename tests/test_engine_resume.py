"""Resumes of the plain engine are exact (ADR 0014, C10; task 4.8).

The 4.7(a) differential test resumed the plain engine mid-run and compared it with an
uninterrupted Lightning run. Lightning is gone, so each resume is compared with an
uninterrupted plain run of the same config, which ``test_engine_regression.py`` holds
equal to its golden (what that Lightning run computed). Bit for bit: the validations after
the resume point (steps, values, weights), ``metrics.json`` (``_engine_cases.computed``),
the predictions files and the manifests.

- the paper classifier, with C5-C7 and with C5-C7 and D3 (whose EMA and scheduler state
  the checkpoint carries), from a mid-run ``last.pt``;
- the semantics of the three Lightning-bound resume tests of task 2.2a, ported:
  ``test_resume_from_mid_epoch_last_checkpoint``,
  ``test_resume_at_the_horizon_still_writes_best`` and ``test_augmented_resume_is_exact``;
- the ``rj23`` schedule: a mid-run resume, a resume of the finished run, and resumes two
  validations, one validation and zero steps before a patience stop.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _engine_cases import (
    AUGMENT,
    D3,
    PlainProbe,
    assert_same_states,
    assert_same_validations,
    computed,
    make_split_data,
    paper_settings,
    rj23_settings,
)
from _rj23_support import DATASET, rj23_layout

from astro_simclr.engine import Task, Validation, load_best, load_checkpoint, read_scalars
from astro_simclr.protocol.split_runs import SplitRequest
from astro_simclr.tasks.plain import run_plain
from astro_simclr.tasks.rj23 import RJ23Request


class _CopyLast(PlainProbe):
    """Also copy ``ckpt/last.pt`` (written just before) to ``dst`` at the step-``step`` one."""

    def __init__(self, step: int, src: Path, dst: Path) -> None:
        super().__init__()
        self.step, self.src, self.dst = step, src, dst

    def __call__(self, record: Validation, task: Task) -> None:
        super().__call__(record, task)
        if record.steps_done == self.step:
            self.dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.src, self.dst)


class _CopyLastAt(PlainProbe):
    """Copy ``ckpt/last.pt`` (written just before) to ``dst / f"{step}.pt"`` at each step."""

    def __init__(self, steps: set[int], src: Path, dst: Path) -> None:
        super().__init__()
        self.steps, self.src, self.dst = steps, src, dst

    def __call__(self, record: Validation, task: Task) -> None:
        super().__call__(record, task)
        if record.steps_done in self.steps:
            self.dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.src, self.dst / f"{record.steps_done}.pt")


def _assert_same_outputs(expected_dir: Path, actual_dir: Path) -> None:
    """The predictions files, array by array, and two finished plain manifests."""
    names = sorted(p.name for p in expected_dir.glob("predictions_*.npz"))
    assert names and names == sorted(p.name for p in actual_dir.glob("predictions_*.npz"))
    for name in names:
        with np.load(expected_dir / name) as left, np.load(actual_dir / name) as right:
            assert sorted(left.files) == sorted(right.files)
            for key in left.files:
                assert left[key].dtype == right[key].dtype
                assert np.array_equal(left[key], right[key]), (name, key)
    for path in (expected_dir, actual_dir):
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        assert (manifest["engine"], manifest["status"]) == ("plain", "finished")


@pytest.fixture(scope="module")
def split_data(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """(prepared dir, splits file) with 25 stamps per split (``make_split_data``)."""
    return make_split_data(tmp_path_factory)


@pytest.fixture(scope="module")
def rj23_data(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """(data directory, object splits) of a synthetic 5-class ``raw-alerts`` layout."""
    return rj23_layout(tmp_path_factory.mktemp("resume_rj23"))


# --- the paper classifier ------------------------------------------------------------------


@pytest.mark.parametrize("flags", [AUGMENT, {**AUGMENT, **D3}], ids=["C5-C7", "C5-C7-D3"])
def test_mid_run_resume_equals_the_uninterrupted_run(
    flags: dict[str, Any], classifier_f32_dir: Path, tmp_path: Path
) -> None:
    # With D3, last.pt carries the EMA (in the weights) and the scheduler (task state).
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    saved = tmp_path / "mid" / "last.pt"
    copy = _CopyLast(9, whole / "ckpt" / "last.pt", saved)
    expected = run_plain(
        "train-supervised", paper_settings(classifier_f32_dir, whole, **flags), on_validation=copy
    )
    probe = PlainProbe()
    metrics = run_plain(
        "train-supervised",
        paper_settings(classifier_f32_dir, resumed, **flags),
        resume=str(saved),
        on_validation=probe,
    )
    assert_same_validations([v for v in copy.validations if v[0] > 9], probe.validations)
    assert metrics["resumed_from"] == str(saved)
    assert {**computed(metrics), "resumed_from": False} == computed(expected)
    _assert_same_outputs(whole, resumed)


def test_resume_from_mid_epoch_last_checkpoint(classifier_f32_dir: Path, tmp_path: Path) -> None:
    # last.pt every 3 steps, on the validation grid, in epochs of 2 steps: the step-9 copy
    # is mid-epoch and was written after the step-9 validation, which the resume must not
    # lose nor repeat.
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    saved = resumed / "ckpt" / "last.pt"
    copy = _CopyLast(9, whole / "ckpt" / "last.pt", saved)
    expected = run_plain(
        "train-supervised", paper_settings(classifier_f32_dir, whole), on_validation=copy
    )
    checkpoint = load_checkpoint(saved)
    assert checkpoint["steps_done"] == 9
    assert checkpoint["policy"]["horizon"] > 10  # extended before the checkpoint
    assert checkpoint["policy"]["best_step"] is not None

    probe = PlainProbe()
    metrics = run_plain(
        "train-supervised",
        paper_settings(classifier_f32_dir, resumed),
        resume="auto",
        on_validation=probe,
    )
    assert [s for s, _, _ in copy.validations][:3] == [3, 6, 9]
    assert_same_validations([v for v in copy.validations if v[0] > 9], probe.validations)
    for key in ("best_step", "last_step", "horizon", "monitor_best", "val_acc_best"):
        assert metrics[key] == expected[key], key
    assert metrics["resumed_from"] == str(saved)
    assert metrics["timing"]["steps"] == expected["last_step"] - 9
    assert_same_states(
        load_best(whole / "ckpt" / "best.pt")["model"],
        load_best(resumed / "ckpt" / "best.pt")["model"],
    )
    # The resumed directory's scalars: only its own validations (none was logged there).
    steps = [record["step"] for record in read_scalars(resumed / "scalars.jsonl")]
    assert steps == [s for s, _, _ in probe.validations]


def test_resume_at_the_horizon_still_writes_best(classifier_f32_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    settings = paper_settings(classifier_f32_dir, out)
    expected = run_plain("train-supervised", settings)
    best = load_best(out / "ckpt" / "best.pt")
    (out / "ckpt" / "best.pt").unlink()
    assert load_checkpoint(out / "ckpt" / "last.pt")["steps_done"] == expected["horizon"]
    # The final last.pt is at the horizon: no step runs, and best.pt is written again.
    metrics = run_plain("train-supervised", settings, resume="auto")
    assert metrics["timing"]["steps"] == 0
    for key in ("best_step", "monitor_best", "val_acc_best", "test_acc_at_best", "evaluation"):
        assert metrics[key] == expected[key], key
    assert metrics["best_step"] == best["steps_done"]
    assert metrics["monitor_best"] == best["score"]
    assert_same_states(best["model"], load_best(out / "ckpt" / "best.pt")["model"])
    # A resume after "finished" is the next attempt of the same run (ADR 0015, amendment A1).
    assert metrics["run"]["attempt"] == expected["run"]["attempt"] + 1
    assert metrics["run"]["run_id"] == expected["run"]["run_id"]


def test_augmented_resume_is_exact(split_data: tuple[Path, Path], tmp_path: Path) -> None:
    # The C5 / C6 / C7 draws are a function of [seed, step]: a resumed fold run repeats them,
    # so its predictions are the uninterrupted run's, bit for bit.
    prepared, path = split_data
    changes = {
        "max_steps": 6,
        "val_every_n_steps": 2,
        "horizon_increment": 0,
        "checkpoint_every_n_steps": 2,
        **AUGMENT,
    }
    whole, mid = tmp_path / "whole", tmp_path / "mid.pt"
    split = SplitRequest(path, 2)
    expected = run_plain(
        "train-supervised",
        paper_settings(prepared, whole, **changes),
        split=split,
        on_validation=_CopyLast(2, whole / "ckpt" / "last.pt", mid),
    )
    resumed = tmp_path / "resumed"
    metrics = run_plain(
        "train-supervised",
        paper_settings(prepared, resumed, **changes),
        split=split,
        resume=str(mid),
    )
    assert metrics["last_step"] == expected["last_step"] == 6
    assert metrics["evaluation"] == expected["evaluation"]
    with (
        np.load(whole / "predictions_evaluation.npz") as left,
        np.load(resumed / "predictions_evaluation.npz") as right,
    ):
        assert np.array_equal(left["probabilities"], right["probabilities"])


# --- the rj23 schedule (task 3.4b) ---------------------------------------------------------


def test_rj23_mid_run_resume_equals_the_uninterrupted_run(
    rj23_data: tuple[Path, Path], tmp_path: Path
) -> None:
    data_dir, splits = rj23_data
    changes = {"patience": 50, "max_steps": 8}
    request = RJ23Request(splits)
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    saved = tmp_path / "mid" / "last.pt"
    copy = _CopyLast(4, whole / "ckpt" / "last.pt", saved)
    expected = run_plain(
        "train-supervised",
        rj23_settings(data_dir, whole, DATASET, "rj23-lowres", **changes),
        rj23=request,
        on_validation=copy,
    )
    assert load_checkpoint(saved)["policy"]["history"] == [v for _, v, _ in copy.validations[:3]]
    probe = PlainProbe()
    settings = rj23_settings(data_dir, resumed, DATASET, "rj23-lowres", **changes)
    metrics = run_plain(
        "train-supervised", settings, rj23=request, resume=str(saved), on_validation=probe
    )
    assert_same_validations([v for v in copy.validations if v[0] > 4], probe.validations)
    assert metrics["resumed_from"] == str(saved)
    assert metrics["timing"]["steps"] == 4
    assert {**computed(metrics), "resumed_from": False} == computed(expected)
    _assert_same_outputs(whole, resumed)

    # A resume of the finished run trains nothing and reports the same results.
    again = run_plain("train-supervised", settings, rj23=request, resume="auto")
    assert again["timing"]["steps"] == 0
    assert again["schedule"] == metrics["schedule"] and again["keep"] == metrics["keep"]


def test_rj23_resumes_around_a_patience_stop_are_exact(
    rj23_data: tuple[Path, Path], tmp_path: Path
) -> None:
    """Resumed two validations, one validation and zero steps before a patience stop.

    The patience count and the best are rebuilt from the checkpoint's history and policy
    state: a wrong wait would stop the resumed run early or late, a lost best would keep
    other weights (or count a non-improvement as one).
    """
    data_dir, splits = rj23_data
    request = RJ23Request(splits)
    whole, saved = tmp_path / "whole", tmp_path / "saved"
    probe = PlainProbe()
    expected = run_plain(
        "train-supervised",
        rj23_settings(data_dir, whole, DATASET, "rj23-lowres"),
        rj23=request,
        on_validation=probe,
    )
    assert expected["schedule"]["stopped_by"] == "patience"
    steps = [step for step, _, _ in probe.validations]
    stop = steps[-1]
    resume_at = steps[-3:]  # two validations before the stop, one, and the stop itself
    assert resume_at[0] > 0  # last.pt is never written at step 0
    # The best precedes the earliest resume point, so each resume must restore it.
    assert expected["best_step"] <= resume_at[0]
    again = tmp_path / "again"
    copies = _CopyLastAt(set(resume_at), again / "ckpt" / "last.pt", saved)
    run_plain(
        "train-supervised",
        rj23_settings(data_dir, again, DATASET, "rj23-lowres"),
        rj23=request,
        on_validation=copies,
    )
    assert_same_validations(probe.validations, copies.validations)  # a rerun is exact
    for step in resume_at:
        resumed = tmp_path / f"resumed-{step}"
        later = PlainProbe()
        metrics = run_plain(
            "train-supervised",
            rj23_settings(data_dir, resumed, DATASET, "rj23-lowres"),
            rj23=request,
            resume=str(saved / f"{step}.pt"),
            on_validation=later,
        )
        assert_same_validations([v for v in probe.validations if v[0] > step], later.validations)
        assert metrics["timing"]["steps"] == stop - step, step
        assert metrics["schedule"]["stopped_by"] == "patience", step
        assert {**computed(metrics), "resumed_from": False} == computed(expected)
        _assert_same_outputs(whole, resumed)
