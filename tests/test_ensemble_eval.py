"""Tests of ``ensemble-eval`` on tiny synthetic split runs (plan row D4).

A module fixture trains, on the synthetic prepared data and its stored splits (25 stamps
per split, as ``test_engine_differential``'s), six short split runs on the CPU: three
plain screen runs of one config (seeds 0-2), a plain screen run of another ``lr``, a
Lightning screen run (seed 0) and a plain run of fold 0. Then:

- ``rot4`` reproduces each run's own ``predictions_evaluation.npz`` bit for bit (plain and
  Lightning weights; screen and fold parts), and the members' balanced accuracies equal
  the runs' stored ones;
- the ensemble arithmetic: its probabilities are the float64 mean of the members', its
  gain is the ensemble's balanced accuracy minus the members' mean;
- ``d4`` and ``d4_shift`` run, with their views, and the centre crop re-taken from the
  stamps is the run's own input;
- ``--test`` adds the Test split; without it Test is never predicted;
- the output is a run for ``compare-variants --splits`` (the screen decision);
- the refusals: other folds, configs, engines, statuses, splits files, data, repeated
  seeds, an unsupported TTA mode, and missing weights.
"""

from __future__ import annotations

import dataclasses
import json
import pickle
import shutil
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from astro_simclr.cli import main
from astro_simclr.config import ConfigError, Settings
from astro_simclr.data import prepare
from astro_simclr.data.synthetic import CLASSIFIER_FILES, make_synthetic
from astro_simclr.ensemble_eval import (
    check_members,
    ensemble_eval,
    evaluation_parts,
    member_settings,
    read_member,
    shifted_images,
)
from astro_simclr.experiments import run_command
from astro_simclr.protocol.compare import compare_decision, load_split_run
from astro_simclr.protocol.predictions import load_predictions
from astro_simclr.protocol.split_runs import SplitRequest, file_sha256, open_split_run
from astro_simclr.protocol.splits import make_splits, write_splits
from astro_simclr.protocol.stats import balanced_accuracy
from astro_simclr.runs.layout import BEST_CHECKPOINT, LIGHTNING_BEST
from astro_simclr.tasks.plain import run_plain

_SCHEDULE: dict[str, Any] = {
    "schedule": "carrasco2021",
    "supervised_model": "paper_stamp_classifier",
    "image_size": 21,
    "eval_batch_size": 64,
    "early_stopping": False,
    "monitor": "loss_val",
    "optimizer": "Adam",
    "lr": 3e-2,
    "beta_loss": 0.5,
    "batch_size": 4,
    "max_steps": 6,
    "val_every_n_steps": 3,
    "horizon_increment": 0,
    "checkpoint_every_n_steps": 3,
}


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The prepared data, the splits file and the six run directories (module docstring)."""
    raw = tmp_path_factory.mktemp("ens_raw")
    out = tmp_path_factory.mktemp("ens_prepared")
    make_synthetic(raw, n_per_split=25, seed=1)
    source = raw / CLASSIFIER_FILES[0]
    # Our own synthetic file, written just above: unpickling it is safe.
    content = pickle.loads(source.read_bytes())  # noqa: S301
    for key in ("images", "features"):
        content["Train"][key][3] = content["Validation"][key][7]  # the leaked stamp
    source.write_bytes(pickle.dumps(content))
    data = prepare(source, out, images="float32")
    splits = tmp_path_factory.mktemp("ens_splits") / "splits.json"
    write_splits(make_splits(data, selection_per_class=1, screen_per_class=1), splits)
    root = tmp_path_factory.mktemp("ens_runs")

    def settings(name: str, **changes: Any) -> Settings:
        base = Settings(
            accelerator="cpu",
            deterministic=True,
            data_dir=data.parent,
            dataset=data.name,
            output_dir=root / name,
            **_SCHEDULE,
        )
        return base.replace(**changes)

    screen = SplitRequest(splits, None)
    dirs: dict[str, Path] = {}
    for seed in (0, 1, 2):
        dirs[f"screen{seed}"] = root / f"screen{seed}"
        run_plain("train-supervised", settings(f"screen{seed}", seed=seed), split=screen)
    run_plain("train-supervised", settings("other_lr", seed=3, lr=1e-2), split=screen)
    run_command("train-supervised", settings("lightning"), progress_bar=False, split=screen)
    run_plain("train-supervised", settings("fold0"), split=SplitRequest(splits, 0))
    for name in ("other_lr", "lightning", "fold0"):
        dirs[name] = root / name
    return {"data": data, "splits": splits, "dirs": dirs}


def _stored(run_dir: Path, stage: str = "evaluation") -> np.ndarray:
    return load_predictions(run_dir / f"predictions_{stage}.npz").probabilities


def _written(out: Path, stage: str = "evaluation") -> Any:
    return load_predictions(out.parent / f"{out.stem}.predictions_{stage}.npz")


@pytest.mark.parametrize("name", ["screen0", "lightning", "fold0"])
def test_rot4_reproduces_the_runs_own_predictions(
    name: str, runs: dict[str, Any], tmp_path: Path
) -> None:
    run_dir = runs["dirs"][name]
    out = tmp_path / f"{name}.json"
    fold = 0 if name == "fold0" else None
    report = ensemble_eval([run_dir], runs["splits"], fold=fold, out=out)
    stored = load_predictions(run_dir / "predictions_evaluation.npz")
    written = _written(out)
    assert np.array_equal(written.probabilities, stored.probabilities)
    assert np.array_equal(written.labels, stored.labels)
    assert np.array_equal(written.index, stored.index)
    member = report.payload["members"][0]["evaluation"]
    assert member["balanced_accuracy"] == member["stored_balanced_accuracy"]
    assert report.payload["seed"] == 0
    assert report.payload["tta"]["views"] == ["identity"]
    assert "test" not in report.payload["predictions"]
    assert not any("WARN" in line for line in report.lines)
    assert json.loads(out.read_text(encoding="utf-8")) == report.payload


def test_ensemble_arithmetic(runs: dict[str, Any], tmp_path: Path) -> None:
    dirs = [runs["dirs"][f"screen{seed}"] for seed in (2, 0, 1)]
    out = tmp_path / "ensemble.json"
    report = ensemble_eval(dirs, runs["splits"], fold=None, out=out)
    stored = [_stored(path) for path in dirs]
    mean = np.mean([p.astype(np.float64) for p in stored], axis=0).astype(np.float32)
    written = _written(out)
    assert np.array_equal(written.probabilities, mean)
    result = report.payload["results"]["evaluation"]
    members = [row["evaluation"]["balanced_accuracy"] for row in report.payload["members"]]
    stored_ba = [row["evaluation"]["stored_balanced_accuracy"] for row in report.payload["members"]]
    assert members == stored_ba
    assert result["member_mean_balanced_accuracy"] == pytest.approx(statistics.mean(members))
    assert result["member_std_balanced_accuracy"] == pytest.approx(statistics.stdev(members))
    ensemble_ba = balanced_accuracy(written.correct.astype(np.float64), written.labels)
    assert result["ensemble"]["balanced_accuracy"] == pytest.approx(ensemble_ba)
    assert report.payload["evaluation_balanced_accuracy"] == result["ensemble"]["balanced_accuracy"]
    gain = result["ensemble"]["balanced_accuracy"] - result["member_mean_balanced_accuracy"]
    assert result["ensemble_minus_member_mean"] == pytest.approx(gain)
    assert report.payload["seed"] == [0, 1, 2]
    assert load_split_run(out).seed == [0, 1, 2]


@pytest.mark.parametrize(("mode", "n_views"), [("d4", 2), ("d4_shift", 18)])
def test_extended_tta_runs(mode: str, n_views: int, runs: dict[str, Any], tmp_path: Path) -> None:
    run_dir = runs["dirs"]["fold0"]
    out = tmp_path / f"{mode}.json"
    report = ensemble_eval([run_dir], runs["splits"], fold=0, tta=mode, test=True, out=out)
    assert len(report.payload["tta"]["views"]) == n_views
    assert report.payload["tta"]["symmetry"] == "c4"
    written = _written(out)
    stored = load_predictions(run_dir / "predictions_evaluation.npz")
    assert np.array_equal(written.index, stored.index)
    assert not np.array_equal(written.probabilities, stored.probabilities)
    np.testing.assert_allclose(written.probabilities.sum(axis=1), 1.0, rtol=1e-5)
    test = _written(out, "test")
    assert report.payload["predictions"]["test"]["split"] == "Test"
    assert np.array_equal(test.index, np.arange(test.index.size))
    assert "test" in report.payload["results"]


def test_centre_crop_from_the_stamps_is_the_runs_input(runs: dict[str, Any]) -> None:
    member = read_member(runs["dirs"]["fold0"])
    settings = member_settings(member)
    split_run = open_split_run(SplitRequest(runs["splits"], 0), runs["data"])
    for part in evaluation_parts(settings, runs["data"], split_run, test=True):
        centre = shifted_images(settings, runs["data"], part, (0, 0))
        assert np.array_equal(centre, part.split.images)
        assert not np.array_equal(shifted_images(settings, runs["data"], part, (1, 0)), centre)


def test_the_output_is_a_run_for_compare_variants(runs: dict[str, Any], tmp_path: Path) -> None:
    baseline = [runs["dirs"][f"screen{seed}"] for seed in (0, 1)]
    variant = []
    for path in baseline:
        out = tmp_path / f"{path.name}_d4.json"
        ensemble_eval([path], runs["splits"], fold=None, tta="d4", out=out)
        variant.append(out)
    decision = compare_decision(baseline, variant, runs["splits"])
    assert decision.mode == "screen"
    assert decision.verdict in ("KEEP", "DROP")
    # rot4 outputs are the runs themselves: no stamp differs.
    same = []
    for path in baseline:
        out = tmp_path / f"{path.name}_rot4.json"
        ensemble_eval([path], runs["splits"], fold=None, out=out)
        same.append(out)
    identical = compare_decision(baseline, same, runs["splits"])
    assert identical.mcnemar is not None
    assert (identical.mcnemar.only_baseline, identical.mcnemar.only_variant) == (0, 0)


def test_cli(runs: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "cli.json"
    dirs = [str(runs["dirs"][f"screen{seed}"]) for seed in (0, 1)]
    args = ["ensemble-eval", *dirs, "--splits", str(runs["splits"]), "--screen"]
    assert main([*args, "--tta", "d4", "--out", str(out), "--accelerator", "cpu"]) == 0
    printed = capsys.readouterr().out
    assert "ensemble (mean of probabilities)" in printed
    assert "tta d4 (2 view(s): identity, flip)" in printed
    assert out.is_file()
    assert main([*args[:-1], "--fold", "0", "--out", str(out)]) == 2
    assert "not of fold 0" in capsys.readouterr().err
    assert main([*args, "--out", str(tmp_path / "out.txt")]) == 2
    assert ".json" in capsys.readouterr().err


def _refused(runs: dict[str, Any], names: list[str], match: str, tmp_path: Path) -> None:
    dirs = [runs["dirs"][name] for name in names]
    with pytest.raises(ConfigError, match=match):
        ensemble_eval(dirs, runs["splits"], fold=None, out=tmp_path / "refused.json")


def test_refusals_of_other_runs(runs: dict[str, Any], tmp_path: Path) -> None:
    _refused(runs, ["screen0", "fold0"], "not of the screen", tmp_path)
    _refused(runs, ["screen0", "other_lr"], "differ in config", tmp_path)
    _refused(runs, ["screen0", "lightning"], "differ in engine", tmp_path)
    _refused(runs, ["screen0", "screen0"], "repeat a run", tmp_path)


def test_refuses_d4_shift_where_the_input_cannot_shift(
    runs: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import astro_simclr.ensemble_eval as module

    original = module.member_settings

    def uint8(member: Any) -> Settings:
        return original(member).replace(paper_input="uint8")

    monkeypatch.setattr(module, "member_settings", uint8)
    with pytest.raises(ConfigError, match="d4_shift re-crops the float32"):
        ensemble_eval(
            [runs["dirs"]["screen0"]],
            runs["splits"],
            fold=None,
            tta="d4_shift",
            out=tmp_path / "u.json",
        )


def test_refusals_on_the_manifest_fields(runs: dict[str, Any]) -> None:
    """Each identity field, changed on a valid manifest, is refused by name."""
    first = read_member(runs["dirs"]["screen0"])
    second = read_member(runs["dirs"]["screen1"])
    sha = file_sha256(runs["splits"])
    check_members([first, second], splits_sha256=sha, part="screen")

    def with_identity(member: Any, **changes: Any) -> Any:
        identity = dataclasses.replace(member.manifest.identity, **changes)
        return dataclasses.replace(
            member, manifest=dataclasses.replace(member.manifest, identity=identity)
        )

    other = "0" * 64
    for changes, match in (
        ({"data_sha256": other}, "differ in data"),
        ({"machine_class": "cuda-other"}, "differ in machine class"),
        ({"numerics_epoch": 2}, "differ in numerics epoch"),
    ):
        with pytest.raises(ConfigError, match=match):
            check_members(
                [first, with_identity(second, **changes)], splits_sha256=sha, part="screen"
            )
    with pytest.raises(ConfigError, match="another splits file"):
        check_members([first], splits_sha256=other, part="screen")
    with pytest.raises(ConfigError, match="not of fold 3"):
        check_members([first], splits_sha256=sha, part=3)
    failed = dataclasses.replace(
        first,
        manifest=dataclasses.replace(first.manifest, status="failed"),
    )
    with pytest.raises(ConfigError, match="failed, not finished"):
        check_members([failed], splits_sha256=sha, part="screen")
    # The command is hashed with the config, so a valid manifest cannot change it alone.
    stub = SimpleNamespace(
        **{f.name: getattr(first.manifest, f.name) for f in dataclasses.fields(first.manifest)}
    )
    stub.command = "pretrain"
    pretrain = dataclasses.replace(first, manifest=cast(Any, stub))
    with pytest.raises(ConfigError, match="reads train-supervised runs"):
        check_members([pretrain], splits_sha256=sha, part="screen")
    other_machine = dataclasses.replace(
        second, metrics={**second.metrics, "machine": {**second.metrics["machine"], "gpu": "X"}}
    )
    with pytest.raises(ConfigError, match="differ in training machine"):
        check_members([first, other_machine], splits_sha256=sha, part="screen")
    with pytest.raises(ConfigError, match="at least one"):
        check_members([], splits_sha256=sha, part="screen")


def test_refusals_of_data_and_weights(runs: dict[str, Any], tmp_path: Path) -> None:
    run_dir = runs["dirs"]["screen0"]
    with pytest.raises(ConfigError, match="not the data the runs trained on"):
        make_synthetic(tmp_path / "raw", n_per_split=25, seed=7)
        other = prepare(
            tmp_path / "raw" / CLASSIFIER_FILES[0], tmp_path / "other", images="float32"
        )
        ensemble_eval([run_dir], runs["splits"], fold=None, out=tmp_path / "x.json", data=other)
    copy = tmp_path / "no_best"
    shutil.copytree(run_dir, copy)
    (copy / BEST_CHECKPOINT).unlink()
    with pytest.raises(ConfigError, match="selected weights are missing"):
        ensemble_eval([copy], runs["splits"], fold=None, out=tmp_path / "y.json")
    with pytest.raises(ConfigError, match="cannot read"):
        ensemble_eval([run_dir], tmp_path / "missing.json", fold=None, out=tmp_path / "z.json")
    with pytest.raises(ConfigError, match="manifest"):
        ensemble_eval([tmp_path], runs["splits"], fold=None, out=tmp_path / "w.json")


def test_out_is_never_overwritten_or_written_into_a_run(
    runs: dict[str, Any], tmp_path: Path
) -> None:
    run_dir = runs["dirs"]["screen0"]
    metrics = (run_dir / "metrics.json").read_bytes()
    for target in (run_dir / "metrics.json", run_dir / "sub" / "ens.json"):
        for overwrite in (False, True):
            with pytest.raises(ConfigError, match="inside the run directory"):
                ensemble_eval([run_dir], runs["splits"], fold=None, out=target, overwrite=overwrite)
    assert (run_dir / "metrics.json").read_bytes() == metrics
    assert not (run_dir / "sub").exists()
    out = tmp_path / "once.json"
    ensemble_eval([run_dir], runs["splits"], fold=None, out=out)
    first = out.read_bytes()
    with pytest.raises(ConfigError, match="exists; pass --overwrite"):
        ensemble_eval([run_dir], runs["splits"], fold=None, out=out)
    # A leftover predictions file is refused too.
    out.unlink()
    with pytest.raises(ConfigError, match=r"once\.predictions_evaluation\.npz exists"):
        ensemble_eval([run_dir], runs["splits"], fold=None, out=out)
    ensemble_eval([run_dir], runs["splits"], fold=None, out=out, overwrite=True)
    assert out.read_bytes() == first
    assert not list(tmp_path.glob("*.tmp"))


def test_a_failed_evaluation_writes_nothing(runs: dict[str, Any], tmp_path: Path) -> None:
    good, broken = runs["dirs"]["screen0"], tmp_path / "no_best"
    shutil.copytree(runs["dirs"]["screen1"], broken)
    (broken / BEST_CHECKPOINT).unlink()
    out = tmp_path / "outputs" / "ens.json"
    with pytest.raises(ConfigError, match="selected weights are missing"):
        ensemble_eval([good, broken], runs["splits"], fold=None, out=out)
    assert not out.parent.exists()


def test_lightning_weights_are_checked_against_their_run(
    runs: dict[str, Any], tmp_path: Path
) -> None:
    copy = tmp_path / "lightning"
    shutil.copytree(runs["dirs"]["lightning"], copy)
    metrics_file = copy / "metrics.json"
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    metrics_file.write_text(json.dumps({**metrics, "best_step": -1}), encoding="utf-8")
    with pytest.raises(ConfigError, match="not the run's best_step"):
        ensemble_eval([copy], runs["splits"], fold=None, out=tmp_path / "a.json")
    metrics_file.write_text(json.dumps(metrics), encoding="utf-8")
    best = copy / LIGHTNING_BEST
    checkpoint = torch.load(best, map_location="cpu", weights_only=True)
    checkpoint["hyper_parameters"]["lr"] = 0.5
    torch.save(checkpoint, best)
    with pytest.raises(ConfigError, match=r"trained with lr=0\.5"):
        ensemble_eval([copy], runs["splits"], fold=None, out=tmp_path / "b.json")
    checkpoint["hyper_parameters"]["lr"] = metrics["config"]["lr"]
    checkpoint["hyper_parameters"]["augment_seed"] = 9
    torch.save(checkpoint, best)
    with pytest.raises(ConfigError, match="augment_seed=9"):
        ensemble_eval([copy], runs["splits"], fold=None, out=tmp_path / "c.json")
