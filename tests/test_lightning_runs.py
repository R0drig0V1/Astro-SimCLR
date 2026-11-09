"""Stored Lightning runs stay readable after task 4.8 removed the engine.

``tests/fixtures/lightning_run/`` is a split run (fold 2) trained by the Lightning engine
before its removal (``scripts/characterize/lightning_run/generate.py``, from main
``36bfb75``): its ``manifest.json``, ``metrics.json``, ``predictions_evaluation.npz``,
``checkpoints/best.ckpt`` and the splits file. Its model is plan row B9's hybrid with
non-default shapes (``hybrid_dim`` 8, ``hybrid_depth`` 1, ``hybrid_heads`` 1). Its data is
``_engine_cases.make_split_data``'s, rebuilt here; the run's recorded data and splits hashes
prove it is the same.

Every reader of stored runs reads it without Lightning (``best.ckpt`` with
``torch.load(weights_only=True)``):

- ``ensemble-eval`` rebuilds the model from the manifest's config, the hybrid shapes
  included (the PR #60 review fix: ``member_net`` once dropped them and failed to load a
  B9 run), and reproduces the run's own predictions (bit for bit on the platform that
  trained it); it checks the Lightning weights against their run (step, seed,
  hyperparameters) and reads ``last.ckpt`` for ``checkpoint_selection: final``; it refuses
  to mix the Lightning run with a plain one;
- ``compare-variants --splits`` loads it as a split run, whose machine signature names its
  engine (a comparison never mixes signatures);
- ``report-phase2`` reads its facts and provenance;
- the registry indexes it (``runs reindex``, ``runs ls``) as ``engine: lightning``.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from _engine_cases import make_split_data, platform_key

from astro_simclr.cli import main
from astro_simclr.config import ConfigError, settings_from_mapping
from astro_simclr.ensemble_eval import ensemble_eval, member_net, member_settings, read_member
from astro_simclr.paper_variants import build_paper_model
from astro_simclr.protocol.compare import load_split_run
from astro_simclr.protocol.phase2_report import read_run_facts
from astro_simclr.protocol.predictions import load_predictions
from astro_simclr.protocol.split_runs import SplitRequest, file_sha256
from astro_simclr.registry.home import HOME_ENV
from astro_simclr.registry.readers import read_run_dir
from astro_simclr.runs.canonical import config_sha256
from astro_simclr.runs.layout import LIGHTNING_BEST, LIGHTNING_LAST
from astro_simclr.runs.manifest import read_manifest, write_manifest
from astro_simclr.tasks.plain import run_plain

#: The stored run (never written to: each test copies it).
FIXTURE = Path(__file__).parent / "fixtures" / "lightning_run"
#: The platform the fixture was trained on: its predictions are reproduced bit for bit there.
RECORDED_ON = "darwin-arm64-default"
#: The fold the fixture trained.
FOLD = 2


@pytest.fixture(scope="module")
def split_data(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """The fixture's data and splits, rebuilt (the run's recorded hashes are checked)."""
    prepared, splits = make_split_data(tmp_path_factory)
    manifest = read_manifest(FIXTURE)
    assert file_sha256(splits) == manifest.identity.splits_sha256
    assert file_sha256(FIXTURE / "splits.json") == manifest.identity.splits_sha256
    return prepared, splits


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A copy of the stored Lightning run."""
    return Path(shutil.copytree(FIXTURE, tmp_path / "lightning"))


def _assert_reproduced(written: np.ndarray, stored: np.ndarray) -> None:
    """Bit for bit on the training platform; float rounding elsewhere."""
    if platform_key() == RECORDED_ON:
        assert np.array_equal(written, stored)
    else:
        np.testing.assert_allclose(written, stored, rtol=1e-5, atol=1e-6)


def test_the_fixture_is_a_lightning_run_read_without_lightning(run_dir: Path) -> None:
    manifest = read_manifest(run_dir)
    assert (manifest.engine, manifest.status, manifest.identity.split) == (
        "lightning",
        "finished",
        FOLD,
    )
    checkpoint = torch.load(run_dir / LIGHTNING_BEST, map_location="cpu", weights_only=True)
    assert "pytorch-lightning_version" in checkpoint
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert checkpoint["global_step"] == metrics["best_step"]
    assert metrics["versions"]["lightning"]
    assert "lightning" not in sys.modules  # nothing here imported it


def test_member_net_builds_the_hybrid_shapes_of_the_run(run_dir: Path) -> None:
    # The PR #60 review fix: the model is built with the run's hybrid_* shapes.
    member = read_member(run_dir)
    settings = member_settings(member)
    assert (settings.hybrid_dim, settings.hybrid_depth, settings.hybrid_heads) == (8, 1, 1)
    net = member_net(member, settings)
    assert not net.training
    state = torch.load(run_dir / LIGHTNING_BEST, map_location="cpu", weights_only=True)
    for key, tensor in state["state_dict"].items():
        assert torch.equal(net.state_dict()[key], tensor), key
    # The default shapes (what member_net built before the fix) do not fit these weights.
    default = build_paper_model(settings.paper_arch)
    weights = {k.removeprefix("model."): v for k, v in state["state_dict"].items()}
    with pytest.raises(RuntimeError, match="size mismatch"):
        default.load_state_dict(weights)


def test_ensemble_eval_reproduces_the_lightning_predictions(
    run_dir: Path, split_data: tuple[Path, Path], tmp_path: Path
) -> None:
    prepared, splits = split_data
    out = tmp_path / "member.json"
    report = ensemble_eval([run_dir], splits, fold=FOLD, out=out, data=prepared)
    stored = load_predictions(run_dir / "predictions_evaluation.npz")
    written = load_predictions(tmp_path / "member.predictions_evaluation.npz")
    _assert_reproduced(written.probabilities, stored.probabilities)
    assert np.array_equal(written.labels, stored.labels)
    assert np.array_equal(written.index, stored.index)
    member = report.payload["members"][0]["evaluation"]
    if platform_key() == RECORDED_ON:
        assert member["balanced_accuracy"] == member["stored_balanced_accuracy"]
    assert report.payload["seed"] == 0


def test_the_cli_reads_the_lightning_run(
    run_dir: Path, split_data: tuple[Path, Path], tmp_path: Path
) -> None:
    prepared, splits = split_data
    out = tmp_path / "cli.json"
    argv = ["ensemble-eval", str(run_dir), "--splits", str(splits), "--fold", str(FOLD)]
    assert main([*argv, "--out", str(out), "--data", str(prepared)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["protocol"]["fold"] == FOLD


def test_lightning_weights_are_checked_against_their_run(
    run_dir: Path, split_data: tuple[Path, Path], tmp_path: Path
) -> None:
    prepared, splits = split_data

    def evaluate(name: str) -> None:
        ensemble_eval([run_dir], splits, fold=FOLD, out=tmp_path / name, data=prepared)

    metrics_file = run_dir / "metrics.json"
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    metrics_file.write_text(json.dumps({**metrics, "best_step": -1}), encoding="utf-8")
    with pytest.raises(ConfigError, match="not the run's best_step"):
        evaluate("a.json")
    metrics_file.write_text(json.dumps(metrics), encoding="utf-8")
    best = run_dir / LIGHTNING_BEST
    checkpoint = torch.load(best, map_location="cpu", weights_only=True)
    checkpoint["hyper_parameters"]["lr"] = 0.5
    torch.save(checkpoint, best)
    with pytest.raises(ConfigError, match=r"trained with lr=0\.5"):
        evaluate("b.json")
    checkpoint["hyper_parameters"]["lr"] = metrics["config"]["lr"]
    checkpoint["hyper_parameters"]["augment_seed"] = 9
    torch.save(checkpoint, best)
    with pytest.raises(ConfigError, match="augment_seed=9"):
        evaluate("c.json")
    del checkpoint["state_dict"]
    torch.save(checkpoint, best)
    with pytest.raises(ConfigError, match="is not a Lightning checkpoint"):
        evaluate("d.json")


def test_a_final_selection_lightning_run_scores_last_ckpt(
    run_dir: Path, split_data: tuple[Path, Path], tmp_path: Path
) -> None:
    # The fixture's best step is its last step, so its weights stand for the last ones of a
    # checkpoint_selection: final run (plan row D3), whose manifest is rewritten to say so.
    prepared, splits = split_data
    manifest = read_manifest(run_dir)
    config = {**manifest.config, "checkpoint_selection": "final"}
    digest = config_sha256(manifest.command, settings_from_mapping(dict(config)))
    identity = dataclasses.replace(manifest.identity, config_sha256=digest)
    write_manifest(run_dir, dataclasses.replace(manifest, config=config, identity=identity))
    (run_dir / LIGHTNING_BEST).rename(run_dir / LIGHTNING_LAST)
    out = tmp_path / "final.json"
    ensemble_eval([run_dir], splits, fold=FOLD, out=out, data=prepared)
    stored = load_predictions(run_dir / "predictions_evaluation.npz")
    written = load_predictions(tmp_path / "final.predictions_evaluation.npz")
    _assert_reproduced(written.probabilities, stored.probabilities)
    (run_dir / LIGHTNING_LAST).unlink()
    with pytest.raises(ConfigError, match="selected weights are missing"):
        ensemble_eval([run_dir], splits, fold=FOLD, out=tmp_path / "gone.json", data=prepared)


def test_engines_are_never_mixed(
    run_dir: Path, split_data: tuple[Path, Path], tmp_path: Path
) -> None:
    # A plain rerun of the stored config: the same config hash, data and part, another
    # engine (a run-id component).
    prepared, splits = split_data
    manifest = read_manifest(run_dir)
    settings = settings_from_mapping(
        {**manifest.config, "data_dir": prepared.parent, "output_dir": tmp_path / "plain"}
    )
    request = SplitRequest(splits, FOLD)
    metrics = run_plain("train-supervised", settings, split=request)
    assert metrics["run"]["config_sha256"] == manifest.identity.config_sha256
    assert metrics["run"]["run_id"] != manifest.run_id
    with pytest.raises(ConfigError, match="differ in engine"):
        ensemble_eval(
            [run_dir, tmp_path / "plain"], splits, fold=FOLD, out=tmp_path / "m.json", data=prepared
        )
    # compare-variants refuses runs of different machine signatures, the engine included
    # (protocol.compare._check_machines); the two differ only in it.
    lightning, plain = load_split_run(run_dir), load_split_run(tmp_path / "plain")
    assert lightning.machine is not None and plain.machine is not None
    assert lightning.machine[-1] == "lightning" and plain.machine[-1] == "plain"
    assert lightning.machine[:-1] == plain.machine[:-1]


def test_compare_variants_loads_the_lightning_split_run(run_dir: Path) -> None:
    run = load_split_run(run_dir)
    assert (run.mode, run.fold, run.seed) == ("fold", FOLD, 0)
    assert run.machine is not None and "lightning" in run.machine
    stored = load_predictions(run_dir / "predictions_evaluation.npz")
    assert np.array_equal(run.predictions.probabilities, stored.probabilities)


def test_report_phase2_reads_the_lightning_run(run_dir: Path) -> None:
    facts = read_run_facts(run_dir)
    manifest = read_manifest(run_dir)
    assert facts.has_manifest
    assert facts.config_sha256 == manifest.identity.config_sha256
    assert facts.git_sha == manifest.git.sha
    assert facts.seconds_per_step is not None


def test_the_registry_indexes_the_lightning_run(
    run_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = read_run_dir(run_dir)
    manifest = read_manifest(run_dir)
    assert (record.engine, record.run_id, record.split) == ("lightning", manifest.run_id, FOLD)
    assert record.metrics["evaluation.balanced_accuracy"] == pytest.approx(
        json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))[
            "evaluation_balanced_accuracy"
        ]
    )
    monkeypatch.setenv(HOME_ENV, str(tmp_path / "home"))
    assert main(["runs", "reindex", str(run_dir.parent)]) == 0
    capsys.readouterr()
    assert main(["runs", "ls", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [(row["engine"], row["run_id"]) for row in listed] == [("lightning", manifest.run_id)]
