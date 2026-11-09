"""Task A1: every ``run_plain`` run writes ``manifest.json``; comparisons never mix engines.

The manifest's schema is ``tests/test_runs_manifest.py``'s, the record's attempts
``tests/test_runs_record.py``'s; here, what a real (tiny, CPU) plain run records, the
statuses of failed and interrupted fits (an exception or a Ctrl-C raised from the loop's
``on_step`` hook), and the engine in the machine signature that ``compare-variants``,
``bench-report`` and ``compare-reference`` share.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from _protocol_support import MACHINE, fake_run, predictions

from astro_simclr.config import ConfigError, Settings
from astro_simclr.protocol.compare import compare_variants
from astro_simclr.reference import machine_signature
from astro_simclr.runs.layout import METRICS_FILE
from astro_simclr.runs.manifest import MANIFEST_FILE, read_manifest
from astro_simclr.tasks.plain import ENGINE, run_plain

LABELS = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

#: A two-step ``carrasco2021`` run of the paper classifier (the plain runner's path).
_CD21: dict[str, Any] = {
    "schedule": "carrasco2021",
    "supervised_model": "paper_stamp_classifier",
    "image_size": 21,
    "eval_batch_size": 64,
    "early_stopping": False,
    "batch_size": 4,
    "max_steps": 2,
    "val_every_n_steps": 2,
    "horizon_increment": 0,
    "checkpoint_every_n_steps": 2,
}


def _settings(data: Path, out: Path, **changes: object) -> Settings:
    base = Settings(
        accelerator="cpu",
        deterministic=True,
        data_dir=data.parent,
        dataset=data.name,
        output_dir=out,
        **_CD21,
    )
    return base.replace(**changes)


def _raise_at_step(error: BaseException) -> Callable[[int], None]:
    """An ``on_step`` hook that raises ``error`` after the first optimizer step."""

    def on_step(steps_done: int) -> None:
        raise error

    return on_step


def test_a_run_writes_a_finished_manifest(classifier_f32_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    metrics = run_plain("train-supervised", _settings(classifier_f32_dir, out))
    manifest = read_manifest(out)
    assert (manifest.status, manifest.engine, manifest.attempt) == ("finished", ENGINE, 1)
    assert manifest.command == "train-supervised"
    assert manifest.started_at is not None and manifest.ended_at is not None
    assert manifest.machine is not None and manifest.machine["engine"] == "plain"
    assert metrics["run"] == {
        "run_id": manifest.run_id,
        "attempt": 1,
        "config_sha256": manifest.identity.config_sha256,
    }
    assert metrics["machine"]["engine"] == "plain"
    on_disk = json.loads((out / METRICS_FILE).read_text(encoding="utf-8"))
    assert on_disk["run"]["run_id"] == manifest.run_id
    # The same command again is the next attempt of the same run.
    again = run_plain("train-supervised", _settings(classifier_f32_dir, out))
    assert again["run"]["run_id"] == manifest.run_id
    assert again["run"]["attempt"] == 2


def test_a_failed_fit_is_recorded(classifier_f32_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="boom"):
        run_plain(
            "train-supervised",
            _settings(classifier_f32_dir, out),
            on_step=_raise_at_step(RuntimeError("boom")),
        )
    assert read_manifest(out).status == "failed"
    assert not (out / METRICS_FILE).exists()


def test_an_interrupted_fit_is_cancelled(classifier_f32_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        run_plain(
            "train-supervised",
            _settings(classifier_f32_dir, out),
            on_step=_raise_at_step(KeyboardInterrupt()),
        )
    assert read_manifest(out).status == "cancelled"
    assert not (out / METRICS_FILE).exists()


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
    # Stored Lightning runs (up to the tag lightning-final) still compare among themselves.
    lightning = [_run(tmp_path, "lightning", {**MACHINE, "engine": "lightning"})]
    plain = [_run(tmp_path, "plain", {**MACHINE, "engine": "plain"})]
    with pytest.raises(ConfigError, match="engines"):
        compare_variants(lightning, plain, n_boot=10)
    allowed = compare_variants(lightning, plain, n_boot=10, allow_mixed_machines=True)
    assert any("WARN: mixed machines" in line and "plain" in line for line in allowed.lines)
    # An archived run (no engine, no manifest) and a stored Lightning run compare as before.
    archived = [_run(tmp_path, "archived", MACHINE)]
    assert not (archived[0] / MANIFEST_FILE).exists()
    result = compare_variants(archived, lightning, n_boot=10)
    assert not any("mixed" in line for line in result.lines)
