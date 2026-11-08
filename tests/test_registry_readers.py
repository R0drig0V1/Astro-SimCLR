"""Tests of ``astro_simclr.registry.readers``: manifest runs, legacy backfill, config formats.

The legacy reader is tested on the real archived runs of ``results/phase2/replication``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from _registry_support import (
    DATA,
    GTX1050,
    REPLICATION,
    SPLITS,
    case_config_sha256,
    legacy_run,
    manifest_run,
)

from astro_simclr.config import Settings
from astro_simclr.registry.readers import (
    CONFIG_FORMATS,
    DEFAULT_CONFIG_FORMAT,
    FLAT_1_FILLS,
    RecordError,
    is_run_dir,
    lower_flat_1,
    metrics_summary,
    read_run_dir,
)
from astro_simclr.runs.canonical import HASH_NEUTRAL_DEFAULTS

ARCHIVED = sorted(REPLICATION.glob("*/seed-*"))


def test_the_archive_is_there() -> None:
    assert len(ARCHIVED) == 10
    assert all(is_run_dir(run) for run in ARCHIVED)
    assert not is_run_dir(REPLICATION)


@pytest.mark.parametrize("run_dir", ARCHIVED, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_archived_runs_backfill(run_dir: Path) -> None:
    record = read_run_dir(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert record.legacy
    assert record.source == "metrics"
    assert record.config_format == DEFAULT_CONFIG_FORMAT == "flat-1"
    # Missing provenance is null, never guessed.
    for name in ("run_id", "attempt", "data_sha256", "pretrained_sha256", "numerics_epoch"):
        assert getattr(record, name) is None, name
    assert (record.git_sha, record.git_dirty, record.uv_lock_sha256) == (None, None, None)
    assert (record.created_at, record.started_at, record.ended_at) == (None, None, None)
    # Derived from facts of the code at the time.
    assert (record.status, record.engine, record.machine_class) == (
        "finished",
        "lightning",
        GTX1050,
    )
    assert record.command == "train-supervised"
    assert record.seed == int(run_dir.name.removeprefix("seed-"))
    assert record.split is None and record.splits_sha256 is None
    assert record.data_source_sha256 == metrics["data"]["source"]["sha256"]
    assert record.config == metrics["config"]
    assert record.key.startswith("legacy-")
    for stage in ("val", "test"):
        for key in ("accuracy", "balanced_accuracy", "macro_f1"):
            assert record.metrics[f"{stage}.{key}"] == metrics["evaluation"][stage][key]


def test_archived_config_hash_is_todays_preset_hash() -> None:
    """flat-1 lowering: the archived runs hash like today's --preset of the same experiment."""
    for arm, case in (
        ("with-features", "carrasco2021"),
        ("no-features", "carrasco2021-no-features"),
    ):
        hashes = {read_run_dir(run).config_sha256 for run in REPLICATION.glob(f"{arm}/seed-*")}
        assert hashes == {case_config_sha256(case)}, arm
        presets = {read_run_dir(run).preset for run in REPLICATION.glob(f"{arm}/seed-*")}
        assert presets == {case}


def test_flat_1_fills_are_the_defaults_and_the_archive_lacks_them() -> None:
    defaults = Settings()
    archived = json.loads((ARCHIVED[0] / "metrics.json").read_text(encoding="utf-8"))["config"]
    lacking = {f.name for f in dataclasses.fields(Settings)} - set(archived)
    # Fields added after A0 are filled from HASH_NEUTRAL_DEFAULTS (ADR 0015 rule 9).
    assert set(FLAT_1_FILLS) == lacking - set(HASH_NEUTRAL_DEFAULTS)
    assert not set(FLAT_1_FILLS) & set(HASH_NEUTRAL_DEFAULTS)
    for key in lacking & set(HASH_NEUTRAL_DEFAULTS):
        assert lower_flat_1(archived)[key] == HASH_NEUTRAL_DEFAULTS[key] == getattr(defaults, key)
    for key, (value, reason) in FLAT_1_FILLS.items():
        assert getattr(defaults, key) == value, key
        assert reason
    assert CONFIG_FORMATS["flat-1"] is lower_flat_1


def test_lower_flat_1_refuses_unknown_and_unfilled_keys() -> None:
    config = json.loads((ARCHIVED[0] / "metrics.json").read_text(encoding="utf-8"))["config"]
    with pytest.raises(RecordError, match=r"unknown to today's Settings \['bogus'\]"):
        lower_flat_1({**config, "bogus": 1})
    del config["lr"]
    with pytest.raises(RecordError, match=r"no reader fills \['lr'\]"):
        lower_flat_1(config)


def test_legacy_run_of_todays_settings(tmp_path: Path) -> None:
    record = read_run_dir(legacy_run(tmp_path / "run", seed=3, accuracy=0.8))
    assert record.config_sha256 == case_config_sha256("carrasco2021")
    assert record.seed == 3
    assert record.metrics["val.accuracy"] == 0.8
    assert record.metrics["seconds_per_step"] == 0.05


def test_legacy_split_runs(tmp_path: Path) -> None:
    fold = {"mode": "fold", "fold": 2, "splits_sha256": SPLITS}
    record = read_run_dir(legacy_run(tmp_path / "fold", protocol=fold))
    assert (record.split, record.splits_sha256) == (2, SPLITS)
    screen = {"mode": "screen", "fold": None, "splits_sha256": SPLITS}
    assert read_run_dir(legacy_run(tmp_path / "screen", protocol=screen)).split == "screen"


@pytest.mark.parametrize(
    ("protocol", "match"),
    [
        ("fold", "not a mapping"),
        ({"mode": "fold", "fold": 1}, "no splits_sha256"),
        ({"mode": "fold", "fold": None, "splits_sha256": SPLITS}, "not a split run"),
        ({"mode": "screen", "fold": 1, "splits_sha256": SPLITS}, "not a split run"),
        ({"mode": "fold", "fold": True, "splits_sha256": SPLITS}, "not a split run"),
    ],
)
def test_legacy_bad_protocol(tmp_path: Path, protocol: object, match: str) -> None:
    with pytest.raises(RecordError, match=match):
        read_run_dir(legacy_run(tmp_path / "run", protocol=protocol))


def _rewrite(run_dir: Path, **changes: object) -> Path:
    path = run_dir / "metrics.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    for key, value in changes.items():
        if value is None:
            doc.pop(key, None)
        else:
            doc[key] = value
    path.write_text(json.dumps(doc), encoding="utf-8")
    return run_dir


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"command": None}, "not a training run"),
        ({"config": None}, "not a training run"),
        ({"seed": 7}, "not the config's seed"),
        ({"seed": True}, "not the config's seed"),
        ({"config_format": "nested-1"}, "has no reader"),
        ({"config_format": 1}, "has no reader"),
        ({"command": "pretend"}, "config: command 'pretend'"),
    ],
)
def test_legacy_unreadable(tmp_path: Path, changes: dict[str, object], match: str) -> None:
    run = _rewrite(legacy_run(tmp_path / "run"), **changes)
    with pytest.raises(RecordError, match=match):
        read_run_dir(run)


@pytest.mark.parametrize(
    ("text", "match"),
    [("{", "not valid JSON"), ("[]", "not a JSON object"), (b"\xff", "not valid JSON")],
)
def test_garbled_metrics(tmp_path: Path, text: str | bytes, match: str) -> None:
    (tmp_path / "metrics.json").write_bytes(text if isinstance(text, bytes) else text.encode())
    with pytest.raises(RecordError, match=match):
        read_run_dir(tmp_path)


def test_garbled_manifest(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RecordError, match="not an astro-simclr-run-manifest"):
        read_run_dir(tmp_path)


def test_unreadable_metrics(tmp_path: Path) -> None:
    path = legacy_run(tmp_path / "run") / "metrics.json"
    path.chmod(0)
    try:
        with pytest.raises(RecordError, match=r"cannot read metrics\.json"):
            read_run_dir(path.parent)
    finally:
        path.chmod(0o644)


def test_neither_file(tmp_path: Path) -> None:
    with pytest.raises(RecordError, match="has neither"):
        read_run_dir(tmp_path)


def test_legacy_machine_class_needs_a_machine(tmp_path: Path) -> None:
    run = legacy_run(tmp_path / "a")
    assert read_run_dir(_rewrite(run, machine=None)).machine_class is None
    assert read_run_dir(_rewrite(run, machine={"device": "cpu"})).machine_class is None
    odd = {"device": "cuda", "gpu": 3, "processor": "x86_64"}
    assert read_run_dir(_rewrite(run, machine=odd)).machine_class is None
    cpu = {"device": "cpu", "gpu": None, "processor": "arm64"}
    assert read_run_dir(_rewrite(run, machine=cpu)).machine_class == "cpu-arm64"


def test_manifest_run(tmp_path: Path) -> None:
    run = manifest_run(tmp_path / "run", seed=2, attempt=3, split=4, accuracy=0.7)
    record = read_run_dir(run)
    assert not record.legacy
    assert record.source == "manifest"
    assert record.run_id is not None and record.attempt == 3
    assert record.key == f"{record.run_id[:16]}-a3"
    assert record.config_sha256 == case_config_sha256("carrasco2021")
    assert (record.seed, record.split, record.splits_sha256) == (2, 4, SPLITS)
    assert (record.data_sha256, record.machine_class, record.numerics_epoch) == (DATA, GTX1050, 1)
    assert (record.git_sha, record.git_dirty) == ("0" * 40, False)
    assert record.status == "finished" and record.ended_at is not None
    assert record.metrics["val.accuracy"] == 0.7
    assert record.metrics_sha256 is not None
    assert record.data_source_sha256 == "a" * 64


def test_manifest_run_without_metrics_yet(tmp_path: Path) -> None:
    record = read_run_dir(manifest_run(tmp_path / "run", status="running"))
    assert record.status == "running"
    assert record.metrics == {} and record.metrics_sha256 is None
    assert record.preset is None


def test_invalid_manifest_is_not_backfilled(tmp_path: Path) -> None:
    run = manifest_run(tmp_path / "run")
    doc = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    doc["attempt"] = 0
    (run / "manifest.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(RecordError, match="attempt"):
        read_run_dir(run)


def test_metrics_summary_skips_what_is_not_a_finite_number() -> None:
    summary = metrics_summary(
        {
            "evaluation": {
                "val": {"accuracy": 0.5, "macro_f1": float("nan"), "ece": 0.25},
                "x": 3,
            },
            "best_step": True,
            "last_step": 7,
            "timing": None,
        }
    )
    assert summary == {"val.accuracy": 0.5, "val.ece": 0.25, "last_step": 7.0}
    # A run that records its scored weights' step (plan row D3) keeps it too.
    assert metrics_summary({"best_step": 3, "selection": {"scored_step": 9}}) == {
        "best_step": 3.0,
        "scored_step": 9.0,
    }
    assert metrics_summary({}) == {}
