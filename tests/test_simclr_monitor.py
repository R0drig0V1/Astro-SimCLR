"""Tests of the pilot's probe monitors (``astro_simclr.simclr.monitor`` and its wiring).

A CPU pilot on a small prepared layout (4 Train and 4 Validation stamps per class) with
stored splits (2 + 2 per class in the screen halves) and a stored label-budget file drawn
from them at budgets 1 and 2 objects per class (one repeat). Checked: off changes nothing;
on writes the keys; the selection-only guard; images only in the eval view; the encoder is
bit-identical with the online probe on; the abort rule; an exact resume.
"""

from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from _protocol_support import make_layout

import astro_simclr.simclr.monitor as monitor
import astro_simclr.simclr.pilot as pilot
from astro_simclr.cli import main
from astro_simclr.config import ConfigError
from astro_simclr.engine.randomness import RESERVED_STREAMS
from astro_simclr.protocol.label_budgets import (
    BudgetPart,
    cv_parts_from_files,
    make_label_budgets,
    write_label_budgets,
)
from astro_simclr.protocol.probe import fit_probe
from astro_simclr.protocol.splits import make_splits, read_splits, write_splits
from astro_simclr.simclr import commands
from astro_simclr.simclr.monitor import (
    OnlineProbe,
    budget_train_rows,
    eval_h,
    probe_key,
    side_batch,
)
from astro_simclr.simclr.network import SimCLRNetwork
from astro_simclr.simclr.pilot import (
    PilotConfig,
    PilotData,
    load_checkpoint,
    load_pilot_data,
    train_pilot,
)
from astro_simclr.simclr.views import centre_crop

PER_CLASS = 4
BUDGETS = (1, 2)
#: The diagnostics keys of a run without the monitors.
BASE_KEYS = {
    "step",
    "effective_rank_h",
    "z_std_mean",
    "z_std_min",
    "alignment",
    "uniformity",
    "knn_selection_bacc",
}
MONITOR_KEYS = {
    probe_key(1),
    probe_key(2),
    "online_probe_train_acc",
    "online_probe_selection_bacc",
}


@pytest.fixture(scope="module")
def files(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """(prepared dir, splits file, label-budget file drawn from the splits)."""
    root = tmp_path_factory.mktemp("monitor")
    prepared = make_layout(root / "layout", train=PER_CLASS, validation=PER_CLASS, test=1)
    write_splits(make_splits(prepared, selection_per_class=1, screen_per_class=2), root / "s.json")
    source = cv_parts_from_files(root / "s.json", prepared)
    budgets = make_label_budgets(
        source.parts,
        source=source.source,
        label_names=source.label_names,
        object_id=source.object_id,
        budgets=BUDGETS,
        n_repeats=1,
        selection_fraction=0,
    )
    write_label_budgets(budgets, root / "budgets.json")
    return prepared, root / "s.json", root / "budgets.json"


def _config(files: tuple[Path, Path, Path], **changes: Any) -> PilotConfig:
    prepared, splits, _ = files
    base = PilotConfig(
        data=str(prepared),
        splits=str(splits),
        epochs=2,
        batch_size=4,
        warmup_epochs=0.3,
        device="cpu",
        log_every=1,
        checkpoint_every=2,
        diagnostics_every=3,
        keep_every_epochs=None,
        diag_samples=8,
        knn_bank_per_class=2,
        embed_batch=16,
    )
    return replace(base, **changes)


def _on(files: tuple[Path, Path, Path], **changes: Any) -> PilotConfig:
    """Both monitors on: the probe every pass, the online probe on side-batches of 8."""
    monitors: dict[str, Any] = {
        "probe_every": 1,
        "label_budgets": str(files[2]),
        "probe_budgets": BUDGETS,
        "online_probe": True,
        "online_probe_batch": 8,
    }
    return _config(files, **{**monitors, **changes})


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _weights(path: Path) -> dict[str, torch.Tensor]:
    model: dict[str, torch.Tensor] = load_checkpoint(path)["model"]
    return model


@pytest.fixture(scope="module")
def runs(
    files: tuple[Path, Path, Path], tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, Path]:
    """(a finished run without monitors, the same run with both monitors on)."""
    off, on = tmp_path_factory.mktemp("off"), tmp_path_factory.mktemp("on")
    train_pilot(_config(files), off)
    train_pilot(_on(files), on)
    return off, on


# --- configuration --------------------------------------------------------------------------


def test_config_validation(files: tuple[Path, Path, Path], tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="probe_every needs label_budgets"):
        _config(files, probe_every=1)
    with pytest.raises(ConfigError, match="probe_every needs"):
        _config(files, probe_every=1, label_budgets=str(files[2]), splits=None)
    with pytest.raises(ConfigError, match="needs the probe monitor"):
        _config(files, abort_if_probe_below_random=1)
    with pytest.raises(ConfigError, match="probe_every >= 1"):
        _config(files, probe_every=0, label_budgets=str(files[2]))
    with pytest.raises(ConfigError, match="increasing positive"):
        _on(files, probe_budgets=(2, 1))
    with pytest.raises(ConfigError, match="online_probe_lr"):
        _config(files, online_probe=True, online_probe_lr=0.0)
    with pytest.raises(ConfigError, match="past the run's end"):
        train_pilot(_on(files, abort_if_probe_below_random=3), tmp_path)


def test_monitor_refusals(files: tuple[Path, Path, Path], tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"budgets \[3\] are not in"):
        train_pilot(_on(files, probe_budgets=(1, 3)), tmp_path / "a")
    with pytest.raises(ConfigError, match="no part 'fold9'"):
        train_pilot(_on(files, probe_part="fold9"), tmp_path / "b")
    with pytest.raises(ConfigError, match="repeat 1 is not in"):
        train_pilot(_on(files, probe_repeat=1), tmp_path / "c")
    data = load_pilot_data(_config(files))
    unlabelled = replace(data, train_labels=None)
    with pytest.raises(ConfigError, match="labelled layout"):
        train_pilot(_on(files), tmp_path / "d", data=unlabelled)
    with pytest.raises(ConfigError, match="online probe needs Train labels"):
        train_pilot(_config(files, online_probe=True), tmp_path / "e", data=unlabelled)


def test_cli_defaults_match_the_monitor() -> None:
    assert commands.PROBE_BUDGETS == monitor.PROBE_BUDGETS
    assert commands.DEFAULT_PART == monitor.DEFAULT_PART
    assert commands.DEFAULT_REPEAT == monitor.DEFAULT_REPEAT
    assert commands.ONLINE_BATCH == monitor.ONLINE_BATCH
    assert commands.ONLINE_LR == monitor.ONLINE_LR


def test_online_probe_has_its_own_stream() -> None:
    assert RESERVED_STREAMS["simclr_online_probe"] == 12
    rows = np.arange(10, 30, dtype=np.int64)
    first = side_batch(3, 5, rows, 8)
    expected = np.random.default_rng([3, 5, 12]).choice(rows, 8, replace=False)
    np.testing.assert_array_equal(first, expected)
    assert np.unique(first).size == 8 and set(first.tolist()) <= set(rows.tolist())
    assert not np.array_equal(first, side_batch(3, 6, rows, 8))
    assert side_batch(0, 0, rows[:3], 8).size == 8  # fewer rows than the batch: replacement


# --- off changes nothing ------------------------------------------------------------------


def test_off_writes_the_files_of_before(runs: tuple[Path, Path]) -> None:
    off, _ = runs
    record = json.loads((off / "pilot.json").read_text())
    assert set(record) == {
        "config",
        "data",
        "schedule",
        "device",
        "numerics",
        "noise_fields",
        "versions",
    }
    assert set(record["config"]).isdisjoint(pilot._MONITOR_FIELDS)
    assert all(set(line) == BASE_KEYS for line in _lines(off / "diagnostics.jsonl"))
    assert [d["step"] for d in _lines(off / "diagnostics.jsonl")] == [3, 6, 9, 10]
    payload = load_checkpoint(off / "ckpt" / "final.pt")
    assert "monitor" not in payload and set(payload["config"]) == set(record["config"])


def test_monitors_do_not_change_the_training(runs: tuple[Path, Path]) -> None:
    # The same seed and steps: the encoder's weights (BatchNorm statistics included), the
    # optimizer's steps and every logged loss are bit-identical with both monitors on.
    off, on = runs
    assert _lines(off / "metrics.jsonl") == _lines(on / "metrics.jsonl")
    left, right = _weights(off / "ckpt" / "final.pt"), _weights(on / "ckpt" / "final.pt")
    assert left.keys() == right.keys()
    assert all(torch.equal(left[k], right[k]) for k in left)
    np.testing.assert_array_equal(
        np.load(off / "embeddings" / "train_h.npy"), np.load(on / "embeddings" / "train_h.npy")
    )
    # The diagnostics the run had before keep their values (and gain the probe points).
    by_step = {d["step"]: d for d in _lines(on / "diagnostics.jsonl")}
    for line in _lines(off / "diagnostics.jsonl"):
        assert {k: by_step[line["step"]][k] for k in BASE_KEYS} == line


def test_encoder_gradients_ignore_the_online_probe(files: tuple[Path, Path, Path]) -> None:
    data = load_pilot_data(_config(files))
    grads = []
    for online in (False, True):
        run = pilot.PilotRun(_config(files, online_probe=online), data, torch.device("cpu"))
        captured: dict[str, torch.Tensor] = {}

        def keep(
            *_: Any, store: dict[str, torch.Tensor] = captured, model: Any = run.model
        ) -> None:
            store.update({n: p.grad.clone() for n, p in model.named_parameters()})

        handle = run.optimizer.register_step_pre_hook(keep)
        run.step(0)
        handle.remove()
        grads.append(captured)
    assert grads[0].keys() == grads[1].keys()
    assert all(torch.equal(grads[0][k], grads[1][k]) for k in grads[0])


# --- on writes the keys -------------------------------------------------------------------


def test_on_writes_the_monitor_keys(runs: tuple[Path, Path]) -> None:
    _, on = runs
    lines = _lines(on / "diagnostics.jsonl")
    # Probe points every pass (5 steps) and at the end, plus the diagnostics every 3.
    assert [d["step"] for d in lines] == [3, 5, 6, 9, 10]
    assert all(set(d) == BASE_KEYS | MONITOR_KEYS for d in lines)
    for d in lines:
        probed = d["step"] in (5, 10)
        assert (d[probe_key(1)] is not None) == probed
        assert (d[probe_key(2)] is not None) == probed
        assert 0.0 <= d["online_probe_train_acc"] <= 1.0
        assert 0.0 <= d["online_probe_selection_bacc"] <= 1.0
    record = json.loads((on / "pilot.json").read_text())
    assert record["status"] == "complete"
    probe = record["monitors"]["probe"]
    assert set(probe["random_init"]) == {probe_key(1), probe_key(2)}
    assert probe["every_steps"] == 5 and probe["part"] == "fold0" and probe["repeat"] == 0
    assert probe["abort"] is None
    assert record["monitors"]["online_probe"]["stream"] == 12
    monitor_state = load_checkpoint(on / "ckpt" / "final.pt")["monitor"]
    assert monitor_state["random_init"] == probe["random_init"]
    assert set(monitor_state["online"]) == {"head", "optimizer", "correct", "seen"}


# --- the selection-only guard -----------------------------------------------------------------


def test_budget_train_rows_keep_usable_train_stamps_only() -> None:
    # A part of a 10-stamp Train and 10-stamp Validation pool: rows 10.. are Validation.
    rows = np.arange(20, dtype=np.int64)
    labels = np.tile(np.arange(5), 4).astype(np.int64)
    part = BudgetPart("fold0", rows, rows.copy(), labels, np.empty(0, dtype=np.int64))
    stored = make_label_budgets(
        [part],
        source={"kind": "cv-splits"},
        label_names=["a", "b", "c", "d", "e"],
        object_id="stamp",
        budgets=(1, 3),
        n_repeats=2,
        selection_fraction=0,
    )
    usable = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9], dtype=np.int64)  # 4 is screen-excluded
    kept = budget_train_rows(
        stored,
        part,
        repeat=1,
        wanted=(1, 3),
        n_train=10,
        usable=usable,
        train_labels=labels[:10],
    )
    for budget in (1, 3):
        chosen = kept.rows[budget]
        assert set(chosen.tolist()) <= set(usable.tolist())
        record = kept.record[str(budget)]
        assert record["n"] == chosen.size
        assert record["n"] + record["dropped_validation"] + record["dropped_excluded"] == 5 * budget
    assert set(kept.rows[1].tolist()) <= set(kept.rows[3].tolist())  # still nested
    with pytest.raises(ValueError, match="repeat 2"):
        budget_train_rows(
            stored, part, repeat=2, wanted=(1,), n_train=10, usable=usable, train_labels=labels
        )
    with pytest.raises(ValueError, match="no usable Train stamp"):
        budget_train_rows(
            stored,
            part,
            repeat=0,
            wanted=(1,),
            n_train=10,
            usable=np.empty(0, dtype=np.int64),
            train_labels=labels,
        )


def test_monitors_read_only_the_selection_half_and_usable_train_stamps(
    files: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _on(files, abort_if_probe_below_random=1)
    data = load_pilot_data(config)
    splits = read_splits(Path(files[1]))
    assert data.screen is not None
    selection, decision = data.screen.selection, data.screen.decision
    # Every Validation stamp outside the selection half is NaN: reading one would make a
    # monitor's number NaN (or its fit raise).
    poisoned = data.validation.clone()
    outside = np.setdiff1d(np.arange(poisoned.shape[0]), selection)
    assert np.intersect1d(outside, decision).size == decision.size
    poisoned[torch.from_numpy(outside)] = float("nan")
    data = replace(data, validation=poisoned)
    # Test and features are not in the pilot's data at all.
    assert {f.name for f in fields(PilotData)}.isdisjoint({"test", "features"})

    reads: list[tuple[str, np.ndarray]] = []
    real_eval_h, real_update = monitor.eval_h, OnlineProbe.update

    def recording_eval_h(model: Any, images: torch.Tensor, rows: Any, *a: Any) -> torch.Tensor:
        split = "validation" if images.data_ptr() == poisoned.data_ptr() else "train"
        reads.append((split, np.array(rows)))
        return real_eval_h(model, images, rows, *a)

    def recording_update(self: OnlineProbe, model: Any, images: Any, rows: Any, d: Any) -> None:
        reads.append(("online-train", np.array(rows)))
        real_update(self, model, images, rows, d)

    monkeypatch.setattr(pilot, "eval_h", recording_eval_h)
    monkeypatch.setattr(OnlineProbe, "update", recording_update)
    train_pilot(config, tmp_path, data=data)

    fold0 = splits.folds[0]
    fold0_evaluation = np.setdiff1d(
        np.arange(splits.n_train + splits.n_validation),
        np.concatenate([fold0.training, fold0.selection]),
    )
    probe_rows: set[int] = set()
    for split, rows in reads:
        if split == "validation":
            np.testing.assert_array_equal(rows, selection)
        else:
            assert set(rows.tolist()) <= set(data.rows.tolist())  # usable Train stamps
            if split == "train":
                probe_rows |= set(rows.tolist())
    assert {split for split, _ in reads} == {"validation", "train", "online-train"}
    # The periodic probe's Train stamps are in fold0's training part, never its evaluation
    # part (pool indices of Train stamps are their Train indices).
    assert probe_rows <= set(fold0.training.tolist())
    assert probe_rows.isdisjoint(fold0_evaluation.tolist())
    for line in _lines(tmp_path / "diagnostics.jsonl"):
        values = [v for v in line.values() if v is not None]
        assert all(np.isfinite(v) for v in values)


def test_probe_input_is_h_of_the_centre_crop(
    files: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The first fit is the random-init baseline at the smallest budget: its input must be
    # exactly h (512-d, nothing concatenated) of the centred crop of those Train stamps,
    # from the network as initialised, in eval mode.
    config = _on(files, online_probe=False, epochs=1)
    data = load_pilot_data(config)
    fitted: list[torch.Tensor] = []
    real_fit = fit_probe

    def capture(h: torch.Tensor, labels: torch.Tensor, **kwargs: Any) -> Any:
        fitted.append(h.clone())
        return real_fit(h, labels, **kwargs)

    monkeypatch.setattr(monitor, "fit_probe", capture)
    run = train_pilot(config, tmp_path, data=data)
    assert run.probe_rows is not None
    rows = run.probe_rows.rows[1]
    torch.manual_seed(config.seed)
    fresh = SimCLRNetwork().eval()
    with torch.no_grad():
        expected, _ = fresh(centre_crop(data.train[torch.from_numpy(rows)]))
    assert fitted[0].shape == (rows.size, 512)
    torch.testing.assert_close(fitted[0], expected.double(), rtol=0, atol=1e-6)
    np.testing.assert_array_equal(
        eval_h(fresh, data.train, rows, torch.device("cpu"), 3).numpy(), expected.numpy()
    )


def test_online_probe_input_is_detached_h_of_the_centre_crop(
    files: tuple[Path, Path, Path],
) -> None:
    data = load_pilot_data(_config(files))
    assert data.train_labels is not None
    torch.manual_seed(0)
    model = SimCLRNetwork().train()
    probe = OnlineProbe(data.train_labels, data.rows, torch.device("cpu"))
    seen: list[torch.Tensor] = []
    probe.head.register_forward_hook(lambda _m, inputs, _o: seen.append(inputs[0]))
    rows = side_batch(0, 0, data.rows, 8)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    probe.update(model, data.train, rows, torch.device("cpu"))
    assert model.training  # the mode is restored
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())  # no BN move
    assert all(p.grad is None for p in model.parameters())  # stop-gradient
    with torch.no_grad():
        expected, _ = model.eval()(centre_crop(data.train[torch.from_numpy(rows)]))
    assert not seen[0].requires_grad and seen[0].shape == (8, 512)
    torch.testing.assert_close(seen[0], expected, rtol=0, atol=0)
    assert probe.take_train_accuracy() is not None and probe.take_train_accuracy() is None


# --- the abort rule -----------------------------------------------------------------------


@pytest.mark.parametrize("beats", [False, True])
def test_abort_rule(
    files: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    beats: bool,
) -> None:
    monkeypatch.setattr(pilot, "probe_beats_baseline", lambda value, baseline: beats)
    config = _on(files, online_probe=False, abort_if_probe_below_random=1)
    run = train_pilot(config, tmp_path)
    record = json.loads((tmp_path / "pilot.json").read_text())
    outcome = record["monitors"]["probe"]["abort"]["outcome"]
    assert outcome["step"] == 5 and outcome["key"] == probe_key(2)
    assert outcome["aborted"] is (not beats)
    if beats:
        assert run.status == record["status"] == "complete" and run.steps_done == 10
        return
    assert run.status == record["status"] == "aborted" and run.steps_done == 5
    assert not (tmp_path / "ckpt" / "final.pt").exists()
    assert not (tmp_path / "embeddings").exists()
    assert load_checkpoint(tmp_path / "ckpt" / "last.pt")["steps_done"] == 5
    assert [d["step"] for d in _lines(tmp_path / "diagnostics.jsonl")] == [3, 5]
    with pytest.raises(ConfigError, match="aborted at step 5"):
        train_pilot(config, tmp_path, resume=True)
    # The CLI ends cleanly and says why.
    prepared, splits, budgets = files
    args: list[object] = ["simclr-pilot", prepared, "--output-dir", tmp_path / "cli"]
    args += ["--device", "cpu", "--batch-size", 4, "--epochs", 2, "--splits", splits]
    args += ["--label-budgets", budgets, "--probe-every", 1, "--probe-budgets", 1, 2]
    args += ["--abort-if-probe-below-random", 1]
    capsys.readouterr()
    assert main([str(a) for a in args]) == 0
    assert "aborted at step 5" in capsys.readouterr().out


# --- resume ---------------------------------------------------------------------------------


def test_resume_restores_the_monitors_exactly(
    files: tuple[Path, Path, Path],
    runs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, on = runs
    config = _on(files)
    stopped = train_pilot(config, tmp_path, stop_at=6)  # past the probe point at 5
    assert stopped.status == "stopped"
    assert json.loads((tmp_path / "pilot.json").read_text())["status"] == "stopped"
    # The baseline comes from the checkpoint: the resumed call never scores random init.
    calls: list[int] = []
    real_scores = pilot._probe_scores

    def counting(run: Any, selection_h: torch.Tensor) -> dict[str, float]:
        calls.append(run.steps_done)
        return real_scores(run, selection_h)

    monkeypatch.setattr(pilot, "_probe_scores", counting)
    resumed = train_pilot(config, tmp_path, resume=True)
    assert resumed.status == "complete" and 0 not in calls
    assert _lines(tmp_path / "diagnostics.jsonl") == _lines(on / "diagnostics.jsonl")
    assert _lines(tmp_path / "metrics.jsonl") == _lines(on / "metrics.jsonl")
    whole = load_checkpoint(on / "ckpt" / "final.pt")["monitor"]
    again = load_checkpoint(tmp_path / "ckpt" / "final.pt")["monitor"]
    assert again["random_init"] == whole["random_init"]
    assert all(
        torch.equal(whole["online"]["head"][k], v) for k, v in again["online"]["head"].items()
    )
    record = json.loads((on / "pilot.json").read_text())
    assert json.loads((tmp_path / "pilot.json").read_text())["monitors"] == record["monitors"]
