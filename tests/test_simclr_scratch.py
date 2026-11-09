"""Tests of the go/no-go's scratch baseline and comparison (``simclr.scratch``, ``.compare``).

On CPU and synthetic data: the recipe, the per-step draws (stream 13, one host generator),
the view, the stream and the evaluation source; ``scratch_screen``'s determinism, its
separation of the subset, the selection half and the decision half; the paired comparison
and the verdict; and the ``simclr-scratch`` / ``simclr-compare`` commands end to end, with
``simclr-probe``'s version-2 report.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from _protocol_support import make_layout

from astro_simclr.cli import main
from astro_simclr.config import ConfigError
from astro_simclr.data.schema import DataError
from astro_simclr.engine.device import DevicePolicy
from astro_simclr.engine.randomness import RESERVED_STREAMS
from astro_simclr.protocol.splits import make_splits, read_splits, write_splits
from astro_simclr.protocol.stats import BootstrapCI
from astro_simclr.simclr import commands
from astro_simclr.simclr.compare import (
    GO_MARGIN,
    PRIMARY_PER_CLASS,
    REPORT_VERSION,
    compare_reports,
    decision_groups,
    margin_verdict,
    read_report,
    summary,
)
from astro_simclr.simclr.scratch import (
    BATCH_PER_CLASS,
    DEFAULT_STEPS,
    EVAL_BATCH,
    NOISE_SCALE,
    SCRATCH_STREAM,
    WARMUP_FRACTION,
    EvalStamps,
    ScratchNetwork,
    ScratchRecipe,
    ScratchStream,
    ScratchTask,
    default_steps,
    draw_step,
    scratch_screen,
    scratch_view,
    selection_score,
    subset_seed,
)
from astro_simclr.simclr.views import (
    ViewConfig,
    background_noise,
    crop,
    dihedral,
    noise_with_fields,
)
from astro_simclr.stamp_relation import stamp_fits

CPU = DevicePolicy("cpu")
N_CLASSES = 5


def _stamps(n: int, seed: int = 0) -> torch.Tensor:
    """(n, 3, 63, 63) stamps with the stored statistics (per-channel min-max)."""
    x = torch.from_numpy(np.random.default_rng(seed).random((n, 3, 63, 63), dtype=np.float32))
    low = x.amin(dim=(-2, -1), keepdim=True)
    high = x.amax(dim=(-2, -1), keepdim=True)
    return (x - low) / (high - low)


def _labels(per_class: int) -> np.ndarray[Any, np.dtype[np.int64]]:
    return np.repeat(np.arange(N_CLASSES, dtype=np.int64), per_class)


# --- the recipe and the draws --------------------------------------------------------------


def test_stream_13_is_the_scratch_stream() -> None:
    assert SCRATCH_STREAM == RESERVED_STREAMS["simclr_scratch"] == 13


def test_recipe_defaults_trials_and_validation() -> None:
    recipe = ScratchRecipe(steps=1000)
    assert recipe.trials == ((5e-4, 0.05), (5e-4, 0.5), (2e-3, 0.05), (2e-3, 0.5))
    assert recipe.warmup_steps == 50 and ScratchRecipe(steps=1).warmup_steps == 0
    assert recipe.views == ViewConfig(noise_scale=NOISE_SCALE)
    plain = recipe.to_plain()
    assert plain["steps"] == 1000 and plain["stream"] == 13 and plain["kept_weights"] == "final"
    for bad in (
        {"steps": 0},
        {"steps": 5, "batch_per_class": 0},
        {"steps": 5, "warmup_fraction": 1.0},
        {"steps": 5, "noise_scale": -1.0},
        {"steps": 5, "lr_grid": ()},
        {"steps": 5, "weight_decay_grid": (0.1, 0.1)},
        {"steps": 5, "lr_grid": (0.0,)},
        {"steps": 5, "weight_decay_grid": (-0.1,)},
    ):
        with pytest.raises(ConfigError):
            ScratchRecipe(**bad)


def test_default_steps_are_declared_per_budget() -> None:
    assert default_steps(10) == 1000 and default_steps(100) == 2000
    assert dict(DEFAULT_STEPS) == {10: 1000, 100: 2000}
    with pytest.raises(ConfigError, match="give --steps"):
        default_steps(1000)


def test_command_defaults_agree_with_the_modules() -> None:
    # The parser keeps copies so that --help does not import torch.
    assert commands.SCRATCH_BATCH_PER_CLASS == BATCH_PER_CLASS
    assert commands.SCRATCH_WARMUP_FRACTION == WARMUP_FRACTION
    assert commands.SCRATCH_NOISE_SCALE == NOISE_SCALE
    assert commands.SCRATCH_EVAL_BATCH == EVAL_BATCH
    assert commands.GO_MARGIN == GO_MARGIN


def test_subset_seed_is_a_seed_sequence_word_per_subset() -> None:
    expected = np.random.SeedSequence([7, 2, 13]).generate_state(1, np.uint32)[0]
    assert subset_seed(7, 2) == int(expected)
    seeds = {subset_seed(0, r) for r in range(20)}
    assert len(seeds) == 20 and all(0 <= s < 2**32 for s in seeds)
    with pytest.raises(ValueError, match=">= 0"):
        subset_seed(-1, 0)


def test_draw_step_replays_one_host_generator_in_order() -> None:
    by_class = [np.arange(c * 4, c * 4 + 4, dtype=np.int64) for c in range(N_CLASSES)]
    draw = draw_step(3, 11, by_class, 2, noise=True)
    rng = np.random.default_rng([3, 11, 13])
    positions = np.concatenate([rng.permutation(rows)[:2] for rows in by_class])
    np.testing.assert_array_equal(draw.positions, positions)
    np.testing.assert_array_equal(draw.dihedral, rng.integers(0, 8, size=10, dtype=np.int64))
    np.testing.assert_array_equal(draw.offsets, rng.integers(0, 7, size=(10, 2), dtype=np.int64))
    np.testing.assert_array_equal(draw.noise_u, rng.random((10, 2)))
    assert draw.noise_z is not None
    np.testing.assert_array_equal(
        draw.noise_z, rng.standard_normal((10, 2, 63, 63), dtype=np.float32)
    )
    # Class-stratified: 2 of every class, each from its own rows.
    np.testing.assert_array_equal(draw.positions // 4, np.repeat(np.arange(N_CLASSES), 2))
    # The noise fields are drawn last: turning them off changes nothing else.
    quiet = draw_step(3, 11, by_class, 2, noise=False)
    assert quiet.noise_z is None
    np.testing.assert_array_equal(quiet.offsets, draw.offsets)
    np.testing.assert_array_equal(quiet.noise_u, draw.noise_u)
    other = draw_step(3, 12, by_class, 2, noise=False)
    assert not np.array_equal(other.offsets, draw.offsets)
    with pytest.raises(ValueError, match="per_class"):
        draw_step(0, 0, by_class, 5, noise=False)


def test_noise_with_fields_is_background_noise_with_its_draw() -> None:
    x = _stamps(4)
    fits = torch.from_numpy(stamp_fits(x.numpy()))
    u = np.random.default_rng(1).random((4, 2))
    config = ViewConfig(noise_scale=3.0, noise_min_r2=-1.0)
    generator = torch.Generator().manual_seed(5)
    expected = background_noise(x, fits, u, generator, config)
    z = torch.randn((4, 2, 63, 63), generator=torch.Generator().manual_seed(5))
    torch.testing.assert_close(noise_with_fields(x, fits, u, z, config), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="z must be"):
        noise_with_fields(x, fits, u, z[:, :1], config)


def test_scratch_view_is_the_pilots_view_of_the_drawn_stamps() -> None:
    stamps = _stamps(10)
    fits = torch.from_numpy(stamp_fits(stamps.numpy()))
    by_class = [np.array([2 * c, 2 * c + 1]) for c in range(N_CLASSES)]
    config = ViewConfig(noise_scale=3.0, noise_min_r2=-1.0)
    draw = draw_step(0, 4, by_class, 1, noise=True)
    view = scratch_view(stamps, fits, draw, config)
    assert view.shape == (5, 3, 57, 57)
    index = torch.from_numpy(draw.positions)
    assert draw.noise_z is not None
    noisy = noise_with_fields(
        stamps[index], fits[index], draw.noise_u, torch.from_numpy(draw.noise_z), config
    )
    torch.testing.assert_close(
        view, crop(dihedral(noisy, draw.dihedral), draw.offsets), rtol=0, atol=0
    )
    plain = scratch_view(stamps, None, draw, ViewConfig())
    torch.testing.assert_close(
        plain, crop(dihedral(stamps[index], draw.dihedral), draw.offsets), rtol=0, atol=0
    )
    with pytest.raises(ValueError, match="fits"):
        scratch_view(stamps, None, draw, config)


# --- the stream, the evaluation source and the task -----------------------------------------


def test_stream_serves_class_stratified_views_of_the_subset_only() -> None:
    stamps, labels = _stamps(15), _labels(3)
    recipe = ScratchRecipe(steps=4, batch_per_class=2, noise_scale=0.0)
    stream = ScratchStream(stamps, labels, None, seed=9, recipe=recipe)
    assert stream.per_class == 2 and stream.batches_per_pass == 1
    views, features, onehot = stream.batch(0)[:3]
    assert views.shape == (10, 3, 57, 57) and features.shape == (10, 0)
    draw = draw_step(9, 0, stream.by_class, 2, noise=False)
    np.testing.assert_array_equal(onehot.argmax(dim=1).numpy(), labels[draw.positions])
    np.testing.assert_array_equal(np.bincount(onehot.argmax(dim=1).numpy()), [2] * N_CLASSES)
    torch.testing.assert_close(stream.batch(3)[0], stream.batch(3)[0], rtol=0, atol=0)
    stream.resume(stream.state(), 2)
    with pytest.raises(DataError, match="stream"):
        stream.resume({**stream.state(), "seed": 1}, 2)
    big = ScratchStream(
        _stamps(50),
        _labels(10),
        None,
        seed=0,
        recipe=ScratchRecipe(steps=1, batch_per_class=2, noise_scale=0),
    )
    assert big.batches_per_pass == 5
    with pytest.raises(DataError, match="misses a class"):
        ScratchStream(stamps[:3], labels[:3], None, seed=0, recipe=recipe)
    with pytest.raises(DataError, match="fits"):
        ScratchStream(stamps, labels, None, seed=0, recipe=ScratchRecipe(steps=1))


def test_eval_source_is_the_centred_view_in_order() -> None:
    stamps, labels = _stamps(7), np.array([0, 1, 2, 3, 4, 0, 1])
    source = EvalStamps(stamps, labels, torch.device("cpu"), batch_size=3)
    batches = list(source)
    assert len(source) == len(batches) == 3 and batches[-1][0].shape[0] == 1
    torch.testing.assert_close(
        torch.cat([b[0] for b in batches]), stamps[..., 3:60, 3:60], rtol=0, atol=0
    )
    np.testing.assert_array_equal(torch.cat([b[2] for b in batches]).argmax(1).numpy(), labels)
    with pytest.raises(ValueError, match="one label per stamp"):
        EvalStamps(stamps, labels[:3], torch.device("cpu"))


def test_task_network_optimizer_and_selection_score() -> None:
    torch.manual_seed(0)
    task = ScratchTask(ScratchRecipe(steps=4), lr=1e-3, weight_decay=0.5)
    assert isinstance(task.module, ScratchNetwork)
    optimizer = task.make_optimizer()
    assert isinstance(optimizer, torch.optim.AdamW)
    decays = [group["weight_decay"] for group in optimizer.param_groups]
    assert decays == [0.5, 0.0]  # norms and biases are not decayed
    assert task.state() == {}
    task.load_state({})
    task.after_step(0)
    batch = (_stamps(5)[..., :57, :57], torch.zeros(5, 0), torch.eye(5))
    loss = task.train_step(batch, 0)
    assert loss.ndim == 0 and optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    task.module.eval()
    with torch.no_grad():
        output = task.eval_step(batch)
    assert output.probabilities.shape == (5, 5) and output.loss.dtype == torch.float32
    perfect = type(output)(output.loss, torch.eye(5), torch.arange(5))
    assert selection_score([perfect]) == 1.0


# --- scratch_screen ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def screen_inputs() -> dict[str, Any]:
    """Train (4 per class), two subsets of 2 per class, and the two screen halves."""
    train, train_y = _stamps(20, seed=1), _labels(4)
    subsets = [
        np.array([0, 1, 4, 5, 8, 9, 12, 13, 16, 17]),
        np.array([2, 3, 6, 7, 10, 11, 14, 15, 18, 19]),
    ]
    selection = (_stamps(10, seed=2), _labels(2))
    decision = (_stamps(10, seed=3), _labels(2))
    recipe = ScratchRecipe(
        steps=2, batch_per_class=2, lr_grid=(1e-3, 1e-2), weight_decay_grid=(0.05,), eval_batch=4
    )
    return {
        "train": train,
        "train_y": train_y,
        "subsets": subsets,
        "selection": selection,
        "decision": decision,
        "recipe": recipe,
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
        seed=4,
        device=CPU,
        done=args.get("done"),
        on_subset=args.get("on_subset"),
    )


@pytest.fixture(scope="module")
def screened(screen_inputs: dict[str, Any]) -> list[dict[str, Any]]:
    return _screen(screen_inputs)


def test_scratch_screen_rows(screened: list[dict[str, Any]]) -> None:
    assert [row["subset"] for row in screened] == [0, 1]
    for r, row in enumerate(screened):
        assert row["seed"] == subset_seed(4, r)
        assert [(t["lr"], t["weight_decay"]) for t in row["trials"]] == [(1e-3, 0.05), (1e-2, 0.05)]
        scores = [t["selection_bacc"] for t in row["trials"]]
        chosen = scores.index(max(scores))  # the first best in grid order
        assert (row["lr"], row["selection_bacc"]) == (row["trials"][chosen]["lr"], max(scores))
        assert len(row["decision_predictions"]) == 10
        truth = _labels(2)
        recall = [
            np.mean(np.asarray(row["decision_predictions"])[truth == c] == c) for c in range(5)
        ]
        assert row["decision_bacc"] == pytest.approx(float(np.mean(recall)))


def test_scratch_screen_is_deterministic(
    screen_inputs: dict[str, Any], screened: list[dict[str, Any]]
) -> None:
    assert _screen(screen_inputs) == screened


def test_scratch_screen_reads_only_the_subset_and_selects_without_the_decision_half(
    screen_inputs: dict[str, Any], screened: list[dict[str, Any]]
) -> None:
    # Every Train stamp outside the subsets and every decision stamp poisoned with NaN:
    # training and selection are bit-identical (only the decision scores could move).
    train = screen_inputs["train"].clone()
    used = np.zeros(20, dtype=bool)
    used[screen_inputs["subsets"][0]] = True
    train[torch.from_numpy(~used)] = float("nan")
    decision = (torch.full_like(screen_inputs["decision"][0], float("nan")), _labels(2))
    poisoned = _screen(
        screen_inputs, train=train, subsets=screen_inputs["subsets"][:1], decision=decision
    )
    keys = ("seed", "lr", "weight_decay", "selection_bacc", "trials")
    assert {k: poisoned[0][k] for k in keys} == {k: screened[0][k] for k in keys}


def test_scratch_screen_keeps_finished_subsets(
    screen_inputs: dict[str, Any], screened: list[dict[str, Any]]
) -> None:
    seen: list[int] = []
    rows = _screen(
        screen_inputs, done={0: screened[0]}, on_subset=lambda row: seen.append(row["subset"])
    )
    assert rows == screened and seen == [1]


# --- the comparison ---------------------------------------------------------------------------


def _report(predictions: list[list[int]], *, per_class: int = 10, **changes: Any) -> dict[str, Any]:
    labels = _labels(4).tolist()
    rows = []
    for r, predicted in enumerate(predictions):
        bacc = float(
            np.mean(
                [np.mean(np.asarray(predicted)[np.asarray(labels) == c] == c) for c in range(5)]
            )
        )
        rows.append({"subset": r, "decision_bacc": bacc, "decision_predictions": predicted})
    report = {
        "report": "test",
        "report_version": REPORT_VERSION,
        "subsets_file": {"path": "a", "sha256": "s"},
        "splits_file": {"path": "b", "sha256": "t"},
        "decision_rows": list(range(20)),
        "decision_labels": labels,
        "per_class": per_class,
        "subsets": rows,
        **summary(rows),
    }
    report.update(changes)
    return report


def test_summary_mean_and_std() -> None:
    assert summary([{"decision_bacc": 0.5}]) == {
        "decision_bacc_mean": 0.5,
        "decision_bacc_std": 0.0,
    }
    two = summary([{"decision_bacc": 0.4}, {"decision_bacc": 0.6}])
    assert two["decision_bacc_mean"] == pytest.approx(0.5)
    assert two["decision_bacc_std"] == pytest.approx(np.std([0.4, 0.6], ddof=1))


def test_margin_verdict_needs_the_lower_bound_above_the_margin() -> None:
    def ci(low: float) -> BootstrapCI:
        return BootstrapCI("balanced_accuracy", low + 0.01, low, low + 0.02, 0.95, 10)

    assert margin_verdict(ci(0.0201)) == "go"
    assert margin_verdict(ci(0.02)) == "no-go"  # equal is not above
    assert margin_verdict(ci(0.12 - 0.1)) == "no-go"  # 0.019999999999999997, rounded
    assert margin_verdict(ci(0.5), margin=0.0) == "go"
    with pytest.raises(ValueError, match="fraction"):
        margin_verdict(ci(0.5), margin=2.0)


def test_compare_reports_interval_and_verdict() -> None:
    truth = _labels(4).tolist()
    noisy = [[(t + (i % 2)) % 5 for i, t in enumerate(truth)] for _ in range(3)]  # half right
    baseline, variant = _report(noisy), _report([truth] * 3)
    result = compare_reports(baseline, variant, groups=None, n_boot=500)
    assert result["difference"] == pytest.approx(0.5)
    assert result["ci_low"] > GO_MARGIN and result["verdict"] == "go" and result["primary"]
    assert result["n_subsets"] == 3 and result["n_objects"] == 20
    same = compare_reports(baseline, baseline, groups=None, n_boot=50)
    assert (same["difference"], same["ci_low"], same["ci_high"]) == (0.0, 0.0, 0.0)
    assert same["verdict"] == "no-go"
    secondary = compare_reports(
        _report(noisy, per_class=100), _report([truth] * 3, per_class=100), groups=None, n_boot=50
    )
    assert secondary["verdict"] is None and not secondary["primary"]
    assert PRIMARY_PER_CLASS == 10
    grouped = compare_reports(baseline, variant, groups=np.arange(20) // 2, n_boot=50)
    assert grouped["n_objects"] == 10


def test_compare_reports_refuses_unpaired_or_old_reports() -> None:
    truth = _labels(4).tolist()
    good = _report([truth] * 2)
    for key, value in (
        ("per_class", 100),
        ("subsets_file", {"path": "a", "sha256": "other"}),
        ("splits_file", {"path": "b", "sha256": "other"}),
        ("decision_rows", list(range(1, 21))),
        ("decision_labels", truth[::-1]),
    ):
        with pytest.raises(DataError, match=key):
            compare_reports(good, {**good, key: value}, groups=None, n_boot=10)
    old = {k: v for k, v in good.items() if k != "report_version"}
    with pytest.raises(DataError, match="version-1"):
        compare_reports(good, old, groups=None, n_boot=10)
    missing = {**good, "subsets": [{"subset": 0, "decision_bacc": 1.0}]}
    with pytest.raises(DataError, match="no decision_predictions"):
        compare_reports(missing, good, groups=None, n_boot=10)
    with pytest.raises(DataError, match="subsets differ"):
        compare_reports(good, _report([truth] * 3), groups=None, n_boot=10)


def test_read_report_refuses_other_files(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{")
    (tmp_path / "list.json").write_text("[]")
    for name in ("bad.json", "list.json", "missing.json"):
        with pytest.raises(DataError):
            read_report(tmp_path / name)


# --- the commands -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def layout(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """(prepared dir, splits file, label subsets of 2 per class)."""
    root = tmp_path_factory.mktemp("scratch")
    prepared = make_layout(root / "layout", train=4, validation=4, test=1)
    splits = root / "splits.json"
    write_splits(make_splits(prepared, selection_per_class=1, screen_per_class=2), splits)
    subsets = root / "subsets.json"
    assert (
        _cli(
            "make-label-subsets",
            prepared,
            "--splits",
            splits,
            "--per-class",
            2,
            "--n-subsets",
            2,
            "--out",
            subsets,
        )
        == 0
    )
    return prepared, splits, subsets


def _cli(*args: str | Path | float) -> int:
    return main([str(a) for a in args])


def _scratch_args(layout: tuple[Path, Path, Path], out: Path) -> tuple[str | Path | float, ...]:
    prepared, splits, subsets = layout
    return (
        "simclr-scratch", prepared, "--subsets", subsets, "--splits", splits, "--out", out,
        "--device", "cpu", "--steps", 2, "--lr", 1e-3, "--weight-decay", 0.05,
        "--batch-per-class", 2, "--eval-batch", 8,
    )  # fmt: skip


def test_cli_scratch_probe_and_compare(
    layout: tuple[Path, Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prepared, splits, subsets = layout
    out = tmp_path / "scratch.json"
    assert _cli(*_scratch_args(layout, out)) == 0
    assert "over 2 subsets (2 trials)" in capsys.readouterr().out
    report = json.loads(out.read_text())
    assert report["report"] == "simclr-scratch" and report["report_version"] == REPORT_VERSION
    assert report["complete"] and report["per_class"] == 2 and report["n_trials"] == 2
    assert report["grid"] == [{"lr": 1e-3, "weight_decay": 0.05}]
    assert report["recipe"]["steps"] == 2 and report["recipe"]["noise_scale"] == NOISE_SCALE
    screen = read_splits(splits).screen
    assert report["decision_rows"] == screen.decision.tolist()
    assert len(report["subsets"]) == 2
    assert all(
        len(row["decision_predictions"]) == screen.decision.size for row in report["subsets"]
    )

    # An existing report: refused, resumed (nothing recomputed), or another config refused.
    assert _cli(*_scratch_args(layout, out)) == 2
    before = out.read_text()
    assert _cli(*_scratch_args(layout, out), "--resume") == 0
    assert "subset 0" not in capsys.readouterr().out and out.read_text() == before
    assert _cli(*_scratch_args(layout, out), "--resume", "--seed", 1) == 2
    assert "another seed" in capsys.readouterr().err

    embeddings = tmp_path / "random"
    assert _cli("simclr-embed", prepared, "--random-init", "--device", "cpu",
                "--embed-batch", 16, "--output-dir", embeddings) == 0  # fmt: skip
    probe = tmp_path / "probe.json"
    assert _cli("simclr-probe", embeddings, "--subsets", subsets, "--splits", splits,
                "--out", probe, "--weight-decay", 0.01) == 0  # fmt: skip
    probed = json.loads(probe.read_text())
    assert probed["report_version"] == REPORT_VERSION and probed["report"] == "simclr-probe"
    assert probed["decision_rows"] == report["decision_rows"]
    assert all("decision_predictions" in row for row in probed["subsets"])

    capsys.readouterr()
    result = tmp_path / "compare.json"
    compare = ("simclr-compare", "--baseline", out, "--variant", probe, "--splits", splits)
    assert _cli(*compare, "--out", result, "--n-boot", 100) == 0
    assert "secondary budget, no verdict" in capsys.readouterr().out
    compared = json.loads(result.read_text())
    assert compared["per_class"] == 2 and compared["verdict"] is None
    assert compared["variant"]["path"] == str(probe)

    # A version-1 report (no predictions) and a wrong splits file are refused.
    old = tmp_path / "old.json"
    old.write_text(json.dumps({k: probed[k] for k in ("per_class", "subsets")}))
    assert _cli("simclr-compare", "--baseline", out, "--variant", old, "--splits", splits) == 1
    assert "version-1" in capsys.readouterr().err
    other = tmp_path / "other-splits.json"
    write_splits(make_splits(prepared, selection_per_class=1, screen_per_class=2, seed=5), other)
    assert _cli("simclr-compare", "--baseline", out, "--variant", probe, "--splits", other) == 1
    assert "another splits file" in capsys.readouterr().err
    assert _cli(*compare, "--margin", 3) == 2
    assert "fraction" in capsys.readouterr().err


def test_cli_scratch_refuses_subsets_of_other_candidates(
    layout: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    # Subsets whose candidates are not the Train stamps the splits file leaves usable.
    prepared, splits, subsets = layout
    data = json.loads(subsets.read_text())
    data["candidates_sha256"] = "0" * 64
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps(data))
    args = list(_scratch_args((prepared, splits, forged), tmp_path / "x.json"))
    assert _cli(*args) == 1
    assert not (tmp_path / "x.json").exists()


def test_decision_groups_follow_the_splits_grouping(layout: tuple[Path, Path, Path]) -> None:
    splits = read_splits(layout[1])
    rows = splits.screen.decision
    groups = decision_groups(splits, rows)
    assert splits.grouping is not None and groups is not None
    np.testing.assert_array_equal(groups, splits.grouping.group_of[splits.n_train + rows])
    ungrouped = replace(splits, grouping=None)
    assert decision_groups(ungrouped, rows) is None
