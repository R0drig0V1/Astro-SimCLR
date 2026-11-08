"""Task A1: every ``run_command`` run writes ``manifest.json``; comparisons never mix engines.

The manifest's schema is ``tests/test_runs_manifest.py``'s; here, what a real (dry, CPU)
run records, the statuses of failed and interrupted fits, and the engine in the machine
signature that ``compare-variants``, ``bench-report`` and ``compare-reference`` share.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightning as L
import pytest
from _protocol_support import MACHINE, fake_run, predictions

from astro_simclr.config import ConfigError, Settings
from astro_simclr.experiments import ENGINE, METRICS_FILE, run_command
from astro_simclr.protocol.compare import compare_variants
from astro_simclr.reference import machine_signature
from astro_simclr.runs.manifest import MANIFEST_FILE, read_manifest

LABELS = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]


def _settings(data_dir: Path, out: Path, **changes: object) -> Settings:
    return Settings(
        data_dir=data_dir,
        output_dir=out,
        accelerator="cpu",
        dataset="td_ztf_stamp_17_06_20_sup_1",
        batch_size=5,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
    ).replace(**changes)


@pytest.fixture(scope="module")
def data_dir(synthetic_prepared: dict[str, Path]) -> Path:
    return next(iter(synthetic_prepared.values())).parent


class _Raise(L.Callback):
    """Raise ``error`` when the fit starts."""

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        raise self.error


def test_a_run_writes_a_finished_manifest(data_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    metrics = run_command("train-supervised", _settings(data_dir, out), progress_bar=False)
    manifest = read_manifest(out)
    assert (manifest.status, manifest.engine, manifest.attempt) == ("finished", ENGINE, 1)
    assert manifest.command == "train-supervised"
    assert manifest.started_at is not None and manifest.ended_at is not None
    assert manifest.machine is not None and manifest.machine["engine"] == "lightning"
    assert metrics["run"] == {
        "run_id": manifest.run_id,
        "attempt": 1,
        "config_sha256": manifest.identity.config_sha256,
    }
    assert metrics["machine"]["engine"] == "lightning"
    on_disk = json.loads((out / METRICS_FILE).read_text(encoding="utf-8"))
    assert on_disk["run"]["run_id"] == manifest.run_id
    # The same command again is the next attempt of the same run.
    again = run_command("train-supervised", _settings(data_dir, out), progress_bar=False)
    assert again["run"]["run_id"] == manifest.run_id
    assert again["run"]["attempt"] == 2


def test_a_failed_fit_is_recorded(data_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="boom"):
        run_command(
            "train-supervised",
            _settings(data_dir, out),
            callbacks=[_Raise(RuntimeError("boom"))],
            progress_bar=False,
        )
    assert read_manifest(out).status == "failed"
    assert not (out / METRICS_FILE).exists()


def test_an_interrupted_fit_is_cancelled(data_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    # Lightning catches the Ctrl-C, shuts down and exits: SystemExit reaches run_command.
    with pytest.raises((KeyboardInterrupt, SystemExit)):
        run_command(
            "train-supervised",
            _settings(data_dir, out),
            callbacks=[_Raise(KeyboardInterrupt())],
            progress_bar=False,
        )
    assert read_manifest(out).status == "cancelled"


def test_an_invalid_run_leaves_no_manifest(data_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    with pytest.raises(ConfigError):
        run_command("train-supervised", _settings(data_dir, out, monitor="nope"))
    assert not (out / MANIFEST_FILE).exists()


# --- The engine in the machine signature -----------------------------------------------------


def test_machine_signature_holds_the_engine() -> None:
    plain = machine_signature({"machine": {**MACHINE, "engine": "plain"}})
    assert plain is not None and plain[-1] == "plain"
    # A run written before task A1 has no engine: it ran on Lightning.
    old = machine_signature({"machine": MACHINE})
    assert old == machine_signature({"machine": {**MACHINE, "engine": "lightning"}})
    assert old != plain


def _run(root: Path, name: str, machine: dict[str, Any]) -> Path:
    preds = {"test": predictions(LABELS, LABELS), "val": predictions(LABELS, LABELS)}
    return fake_run(root, name, preds, seed=0, machine=machine)


def test_compare_variants_refuses_mixed_engines(tmp_path: Path) -> None:
    lightning = [_run(tmp_path, "lightning", {**MACHINE, "engine": "lightning"})]
    plain = [_run(tmp_path, "plain", {**MACHINE, "engine": "plain"})]
    with pytest.raises(ConfigError, match="engines"):
        compare_variants(lightning, plain, n_boot=10)
    allowed = compare_variants(lightning, plain, n_boot=10, allow_mixed_machines=True)
    assert any("WARN: mixed machines" in line and "plain" in line for line in allowed.lines)
    # An archived run (no engine, no manifest) and a new Lightning run compare as before.
    archived = [_run(tmp_path, "archived", MACHINE)]
    assert not (archived[0] / MANIFEST_FILE).exists()
    result = compare_variants(archived, lightning, n_boot=10)
    assert not any("mixed" in line for line in result.lines)
