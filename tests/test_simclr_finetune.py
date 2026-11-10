"""Tests of the fine-tuning arm (``simclr.finetune``, ``simclr-finetune``; plan 5.4).

On CPU and synthetic data: reading a pilot checkpoint's encoder and its refusals; the
encoder at step 0 is the checkpoint's and the head the scratch arm's; with a "checkpoint"
equal to the scratch init the fine-tune is the scratch arm bit for bit (nothing else
differs: the draws, the head, the recipe); determinism; the subset and selection half
alone reach training and selection; and ``simclr-finetune`` end to end from a real
``simclr-pilot`` checkpoint, with its report paired in ``simclr-compare`` against the
scratch and probe reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from _engine_cases import PILOT_DATA, cpu_state, pilot_checkpoint_payload
from _protocol_support import make_layout

from astro_simclr.cli import main
from astro_simclr.config import ConfigError
from astro_simclr.data.schema import DataError
from astro_simclr.engine import seed_everything
from astro_simclr.engine.device import DevicePolicy
from astro_simclr.protocol.splits import make_splits, write_splits
from astro_simclr.simclr import pilot
from astro_simclr.simclr.compare import REPORT_VERSION
from astro_simclr.simclr.finetune import (
    ENCODER_PREFIX,
    KNOWN_CKPT_VERSIONS,
    PilotInit,
    check_init_data,
    encoder_state,
    load_pilot_init,
)
from astro_simclr.simclr.network import SimCLRNetwork
from astro_simclr.simclr.scratch import (
    EvalStamps,
    ScratchNetwork,
    ScratchRecipe,
    ScratchStream,
    ScratchTask,
    fit_trial,
    scratch_screen,
    subset_seed,
)

CPU = DevicePolicy("cpu")
N_CLASSES = 5
SEED = 4


def _stamps(n: int, seed: int = 0) -> torch.Tensor:
    """(n, 3, 63, 63) stamps with the stored statistics (per-channel min-max)."""
    x = torch.from_numpy(np.random.default_rng(seed).random((n, 3, 63, 63), dtype=np.float32))
    low = x.amin(dim=(-2, -1), keepdim=True)
    high = x.amax(dim=(-2, -1), keepdim=True)
    return (x - low) / (high - low)


def _labels(per_class: int) -> np.ndarray[Any, np.dtype[np.int64]]:
    return np.repeat(np.arange(N_CLASSES, dtype=np.int64), per_class)


def _save(payload: dict[str, Any], path: Path) -> Path:
    torch.save(payload, path)
    return path


def _encoder_entries(model: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k[len(ENCODER_PREFIX) :]: v for k, v in model.items() if k.startswith(ENCODER_PREFIX)}


def _assert_same(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    assert list(left) == list(right)
    for key in left:
        assert torch.equal(left[key], right[key]), key


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    payload = pilot_checkpoint_payload(11)
    return _save(payload, tmp_path_factory.mktemp("ckpt") / "final.pt"), payload


# --- reading the checkpoint -----------------------------------------------------------------


def test_known_versions_hold_the_pilots_current_one() -> None:
    assert pilot._CKPT_VERSION in KNOWN_CKPT_VERSIONS


def test_load_pilot_init_reads_the_encoder_and_records_the_checkpoint(
    checkpoint: tuple[Path, dict[str, Any]],
) -> None:
    path, payload = checkpoint
    init = load_pilot_init(path)
    _assert_same(dict(init.encoder), _encoder_entries(payload["model"]))
    assert not any(k.startswith("head") for k in init.encoder)
    assert any(k.endswith("running_mean") for k in init.encoder)  # the BN buffers too
    record = init.record
    assert record["checkpoint"] == str(path) and len(init.sha256) == 64
    assert (record["steps_done"], record["total_steps"], record["pretraining_complete"]) == (
        3,
        3,
        True,
    )
    assert record["format_version"] == KNOWN_CKPT_VERSIONS[-1]
    assert record["pretraining"] == {
        "kind": "cd21-layout",
        "eval_layout": "layout",
        "source_sha256": PILOT_DATA["source_sha256"],
        "splits_sha256": PILOT_DATA["splits_sha256"],
        "n_pretrain": 18,
        "n_excluded_train": 2,
        "cd21_train_in_pretraining": True,
    }
    json.dumps(record)  # plain values only: it goes into the report


def test_load_pilot_init_records_a_corpus_pretraining(tmp_path: Path) -> None:
    corpus = {
        **PILOT_DATA,
        "n_pretrain": 1000,
        "pretrain": {
            "kind": "archive-corpus",
            "index": "index.npz",
            "index_sha256": "2" * 64,
            "exclusion_sha256": "3" * 64,
        },
    }
    payload = pilot_checkpoint_payload(11, data=corpus)
    payload["steps_done"] = 2
    init = load_pilot_init(_save(payload, tmp_path / "last.pt"))
    pretraining = init.record["pretraining"]
    assert pretraining["kind"] == "archive-corpus" and pretraining["index_sha256"] == "2" * 64
    assert pretraining["cd21_train_in_pretraining"] is False
    assert init.record["pretraining_complete"] is False


def _broken(change: str) -> dict[str, Any]:
    payload = pilot_checkpoint_payload(11)
    model = payload["model"]
    if change == "version":
        payload["format_version"] = 99
    elif change in ("model", "data", "steps_done"):
        del payload[change]
    elif change == "model-list":
        payload["model"] = [1]
    elif change == "missing":
        del model["encoder.conv1.weight"]
    elif change == "unexpected":
        model["encoder.extra"] = torch.zeros(1)
    elif change == "shape":
        model["encoder.conv1.weight"] = torch.zeros(64, 3, 5, 5)
    elif change == "dtype":
        model["encoder.conv1.weight"] = model["encoder.conv1.weight"].double()
    elif change == "nan":
        model["encoder.bn1.running_var"] = torch.full((64,), float("nan"))
    elif change == "format":
        payload["format"] = "something-else"
    return payload


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("version", "version 99"),
        ("model", "no 'model'"),
        ("data", "no 'data'"),
        ("steps_done", "no 'steps_done'"),
        ("model-list", "not a mapping"),
        ("missing", "missing \\['conv1.weight'\\]"),
        ("unexpected", "unexpected \\['extra'\\]"),
        ("shape", "conv1.weight is \\(64, 3, 5, 5\\)"),
        ("dtype", "torch.float64"),
        ("nan", "non-finite"),
        ("format", "not a astro-simclr-pilot-checkpoint"),
    ],
)
def test_load_pilot_init_refuses_other_files(tmp_path: Path, change: str, message: str) -> None:
    path = _save(_broken(change), tmp_path / "bad.pt")
    with pytest.raises(DataError, match=message):
        load_pilot_init(path)


def test_load_pilot_init_refuses_unreadable_files(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="cannot read"):
        load_pilot_init(tmp_path / "missing.pt")
    (tmp_path / "junk.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(DataError, match="cannot read"):
        load_pilot_init(tmp_path / "junk.pt")


def test_encoder_state_refuses_a_non_tensor_entry() -> None:
    model: dict[str, Any] = dict(pilot_checkpoint_payload(11)["model"])
    model["encoder.conv1.weight"] = [0.0]
    with pytest.raises(DataError, match="not a tensor"):
        encoder_state(model)


def test_encoder_state_leaves_the_global_generator_alone() -> None:
    torch.manual_seed(5)
    expected = torch.rand(3)
    torch.manual_seed(5)
    encoder_state(pilot_checkpoint_payload(11)["model"])
    assert torch.equal(torch.rand(3), expected)


def test_check_init_data_refuses_another_layout_or_splits(
    checkpoint: tuple[Path, dict[str, Any]],
) -> None:
    init = load_pilot_init(checkpoint[0])
    good = {"path": "layout", "source_sha256": "0" * 64, "splits_sha256": "1" * 64}
    check_init_data(init, good)
    with pytest.raises(ConfigError, match="another layout"):
        check_init_data(init, {**good, "source_sha256": "f" * 64})
    with pytest.raises(ConfigError, match="another splits file"):
        check_init_data(init, {**good, "splits_sha256": None})


# --- the task and the trial ------------------------------------------------------------------


def test_task_starts_from_the_checkpoint_encoder_and_the_scratch_head(
    checkpoint: tuple[Path, dict[str, Any]],
) -> None:
    init = load_pilot_init(checkpoint[0])
    recipe = ScratchRecipe(steps=2)
    seed_everything(7)
    scratch = ScratchTask(recipe, 1e-3, 0.05).module
    seed_everything(7)
    tuned = ScratchTask(recipe, 1e-3, 0.05, encoder_init=init.encoder).module
    assert isinstance(tuned, ScratchNetwork) and isinstance(scratch, ScratchNetwork)
    _assert_same(cpu_state(tuned.encoder), _encoder_entries(checkpoint[1]["model"]))
    _assert_same(cpu_state(tuned.head), cpu_state(scratch.head))
    # The task copied the state: training it does not write into the checkpoint's tensors.
    assert all(
        p.data_ptr() != init.encoder[k].data_ptr() for k, p in tuned.encoder.state_dict().items()
    )


def test_task_refuses_a_state_of_another_encoder() -> None:
    with pytest.raises(RuntimeError, match="state_dict"):
        ScratchTask(
            ScratchRecipe(steps=2), 1e-3, 0.05, encoder_init={"conv1.weight": torch.zeros(1)}
        )


@pytest.fixture(scope="module")
def screen_inputs() -> dict[str, Any]:
    """Train (4 per class), two subsets of 2 per class, and the two screen halves."""
    return {
        "train": _stamps(20, seed=1),
        "train_y": _labels(4),
        "subsets": [
            np.array([0, 1, 4, 5, 8, 9, 12, 13, 16, 17]),
            np.array([2, 3, 6, 7, 10, 11, 14, 15, 18, 19]),
        ],
        "selection": (_stamps(10, seed=2), _labels(2)),
        "decision": (_stamps(10, seed=3), _labels(2)),
        "recipe": ScratchRecipe(
            steps=2,
            batch_per_class=2,
            lr_grid=(1e-3, 1e-2),
            weight_decay_grid=(0.05,),
            eval_batch=4,
        ),
    }


def _screen(inputs: dict[str, Any], **changes: Any) -> list[dict[str, Any]]:
    args = {**inputs, **changes}
    return scratch_screen(
        args["train"],
        args["train_y"],
        args["subsets"],
        args["selection"],
        args["decision"],
        args["recipe"],
        seed=SEED,
        device=CPU,
        encoder_init=args.get("encoder_init"),
    )


def _scratch_init_checkpoint(tmp_path: Path, seed: int) -> PilotInit:
    """A pilot "checkpoint" whose encoder is the scratch arm's random init from ``seed``."""
    seed_everything(seed)
    encoder = ScratchTask(ScratchRecipe(steps=2), 1e-3, 0.05).module
    assert isinstance(encoder, ScratchNetwork)
    pilot_model = cpu_state(SimCLRNetwork())  # its head: any values
    pilot_model.update({ENCODER_PREFIX + k: v for k, v in cpu_state(encoder.encoder).items()})
    payload = pilot_checkpoint_payload(0, model=pilot_model)
    return load_pilot_init(_save(payload, tmp_path / "scratch-init.pt"))


def test_finetune_from_the_scratch_init_is_the_scratch_arm_bit_for_bit(
    screen_inputs: dict[str, Any], tmp_path: Path
) -> None:
    # The strongest check that only the initialisation differs: the same draws (stream 13),
    # the same head, the same recipe; with the scratch init loaded, the same numbers.
    init = _scratch_init_checkpoint(tmp_path, subset_seed(SEED, 0))
    one = screen_inputs["subsets"][:1]
    assert _screen(screen_inputs, subsets=one, encoder_init=init.encoder) == _screen(
        screen_inputs, subsets=one
    )
    inputs = screen_inputs
    subset = inputs["subsets"][0]
    stamps = inputs["train"][torch.from_numpy(subset)]
    recipe = ScratchRecipe(steps=3, batch_per_class=2, noise_scale=0.0)
    selection = EvalStamps(
        inputs["selection"][0], inputs["selection"][1], torch.device("cpu"), batch_size=4
    )
    seed = subset_seed(SEED, 0)
    states = []
    for encoder_init in (None, init.encoder):
        stream = ScratchStream(stamps, inputs["train_y"][subset], None, seed=seed, recipe=recipe)
        task, _ = fit_trial(
            stream,
            selection,
            recipe,
            lr=1e-3,
            weight_decay=0.05,
            seed=seed,
            device=CPU,
            encoder_init=encoder_init,
        )
        states.append(cpu_state(task.module))
    _assert_same(states[0], states[1])


def test_finetune_screen_is_deterministic(
    screen_inputs: dict[str, Any], checkpoint: tuple[Path, dict[str, Any]]
) -> None:
    init = load_pilot_init(checkpoint[0])
    one = screen_inputs["subsets"][1:]
    tuned = _screen(screen_inputs, subsets=one, encoder_init=init.encoder)
    assert _screen(screen_inputs, subsets=one, encoder_init=init.encoder) == tuned
    assert tuned[0]["seed"] == subset_seed(SEED, 0)  # the scratch arm's seed of that row


def test_finetune_reads_only_the_subset_and_selects_without_the_decision_half(
    screen_inputs: dict[str, Any], checkpoint: tuple[Path, dict[str, Any]]
) -> None:
    # Every Train stamp outside the subset and every decision stamp poisoned with NaN:
    # training and selection are bit-identical (only the decision scores could move).
    init = load_pilot_init(checkpoint[0])
    one = screen_inputs["subsets"][:1]
    clean = _screen(screen_inputs, subsets=one, encoder_init=init.encoder)
    train = screen_inputs["train"].clone()
    used = np.zeros(20, dtype=bool)
    used[one[0]] = True
    train[torch.from_numpy(~used)] = float("nan")
    decision = (torch.full_like(screen_inputs["decision"][0], float("nan")), _labels(2))
    poisoned = _screen(
        screen_inputs, train=train, subsets=one, decision=decision, encoder_init=init.encoder
    )
    keys = ("seed", "lr", "weight_decay", "selection_bacc", "trials")
    assert {k: poisoned[0][k] for k in keys} == {k: clean[0][k] for k in keys}


# --- the command ------------------------------------------------------------------------------


def _cli(*args: str | Path | float) -> int:
    return main([str(a) for a in args])


@pytest.fixture(scope="module")
def layout(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A layout, its splits, subsets of 2 per class and a 2-step pilot's checkpoints."""
    root = tmp_path_factory.mktemp("finetune")
    prepared = make_layout(root / "layout", train=4, validation=4, test=1)
    splits = root / "splits.json"
    write_splits(make_splits(prepared, selection_per_class=1, screen_per_class=2), splits)
    subsets = root / "subsets.json"
    made = ("make-label-subsets", prepared, "--splits", splits, "--per-class", 2)
    assert _cli(*made, "--n-subsets", 2, "--out", subsets) == 0
    pilot_args = ("simclr-pilot", prepared, "--device", "cpu", "--batch-size", 4, "--steps", 2)
    assert _cli(*pilot_args, "--splits", splits, "--output-dir", root / "pilot") == 0
    assert _cli(*pilot_args, "--output-dir", root / "pilot-nosplits") == 0
    return {
        "prepared": prepared,
        "splits": splits,
        "subsets": subsets,
        "checkpoint": root / "pilot" / "ckpt" / "final.pt",
        "no_splits": root / "pilot-nosplits" / "ckpt" / "final.pt",
    }


def _trial_args(layout: dict[str, Path], out: Path) -> tuple[str | Path | float, ...]:
    return (
        layout["prepared"], "--subsets", layout["subsets"], "--splits", layout["splits"],
        "--out", out, "--device", "cpu", "--steps", 2, "--lr", 1e-3, "--weight-decay", 0.05,
        "--batch-per-class", 2, "--eval-batch", 8,
    )  # fmt: skip


def test_cli_finetune_report_resume_and_compare(
    layout: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "finetune.json"
    finetune = ("simclr-finetune", "--checkpoint", layout["checkpoint"], *_trial_args(layout, out))
    assert _cli(*finetune) == 0
    assert "simclr-finetune: 2 per class" in capsys.readouterr().out
    report = json.loads(out.read_text())
    scratch_out = tmp_path / "scratch.json"
    assert _cli("simclr-scratch", *_trial_args(layout, scratch_out)) == 0
    scratch = json.loads(scratch_out.read_text())

    # The scratch report's shape, plus the init record.
    assert report["report"] == "simclr-finetune" and report["report_version"] == REPORT_VERSION
    assert set(report) == set(scratch) | {"init", "init_sha256"}
    for key in ("recipe", "grid", "seed", "decision_rows", "decision_labels", "subsets_file"):
        assert report[key] == scratch[key], key
    assert [r["seed"] for r in report["subsets"]] == [r["seed"] for r in scratch["subsets"]]
    init = report["init"]
    assert init["checkpoint"] == str(layout["checkpoint"]) and init["steps_done"] == 2
    assert init["sha256"] == report["init_sha256"] and init["pretraining_complete"]
    assert init["pretraining"]["kind"] == "cd21-layout"
    assert init["pretraining"]["cd21_train_in_pretraining"] is True

    # Resume: kept unchanged; another checkpoint, or the other arm's report, refused.
    assert _cli(*finetune) == 2
    before = out.read_text()
    capsys.readouterr()
    assert _cli(*finetune, "--resume") == 0
    assert "subset 0" not in capsys.readouterr().out and out.read_text() == before
    other = tmp_path / "other.pt"
    payload = torch.load(layout["checkpoint"], weights_only=True)
    payload["steps_done"] = 1
    torch.save(payload, other)
    moved = ("simclr-finetune", "--checkpoint", other, *_trial_args(layout, out), "--resume")
    assert _cli(*moved) == 2
    assert "another init_sha256" in capsys.readouterr().err
    as_scratch = ("simclr-scratch", *_trial_args(layout, out), "--resume")
    assert _cli(*as_scratch) == 2
    assert "another report" in capsys.readouterr().err

    # simclr-compare pairs it, unchanged, with the scratch and the probe reports.
    embeddings = tmp_path / "embeddings"
    embed = ("simclr-embed", layout["prepared"], "--device", "cpu", "--embed-batch", 16)
    assert _cli(*embed, "--checkpoint", layout["checkpoint"], "--output-dir", embeddings) == 0
    probe = tmp_path / "probe.json"
    assert _cli("simclr-probe", embeddings, "--subsets", layout["subsets"], "--splits",
                layout["splits"], "--out", probe, "--weight-decay", 0.01) == 0  # fmt: skip
    for baseline in (scratch_out, probe):
        result = tmp_path / f"vs-{baseline.stem}.json"
        compare = ("simclr-compare", "--baseline", baseline, "--variant", out)
        assert _cli(*compare, "--splits", layout["splits"], "--out", result, "--n-boot", 50) == 0
        compared = json.loads(result.read_text())
        assert compared["variant"]["report"] == "simclr-finetune"
        assert compared["n_subsets"] == 2 and compared["verdict"] is None


def test_cli_finetune_refuses_a_checkpoint_of_other_pretraining(
    layout: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "x.json"
    args = _trial_args(layout, out)
    assert _cli("simclr-finetune", "--checkpoint", layout["no_splits"], *args) == 2
    assert "another splits file" in capsys.readouterr().err
    assert _cli("simclr-finetune", "--checkpoint", tmp_path / "missing.pt", *args) == 1
    assert "cannot read" in capsys.readouterr().err
    assert not out.exists()
