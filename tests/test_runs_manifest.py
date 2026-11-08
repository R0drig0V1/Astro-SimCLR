"""Tests of ``astro_simclr.runs.manifest``: the schema, the writer and the validating reader."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from _canonical_cases import case_settings

from astro_simclr.runs import canonical
from astro_simclr.runs.identity import GitState, run_identity
from astro_simclr.runs.manifest import (
    ENGINES,
    FORMAT,
    FORMAT_VERSION,
    MANIFEST_FILE,
    STATUSES,
    TERMINAL_STATUSES,
    Manifest,
    ManifestError,
    is_finished,
    new_manifest,
    read_manifest,
    transition,
    utc_now,
    write_manifest,
)

MACHINE = {
    "device": "cpu",
    "gpu": None,
    "torch": "2.14.1",
    "cuda": None,
    "fp32_precision": "ieee",
    "cudnn": None,
    "platform": "macOS-27.0-arm64",
}
T0, T1, T2 = "2026-11-08T10:00:00+00:00", "2026-11-08T10:05:00+00:00", "2026-11-08T11:00:00+00:00"
SHA = "0123456789abcdef0123456789abcdef01234567"


def _queued(command: str = "train-supervised", case: str = "carrasco2021") -> Manifest:
    _, settings = case_settings(case)
    identity = run_identity(
        command, settings, data_sha256="d" * 64, machine_class="cpu-arm64", engine="lightning"
    )
    return new_manifest(
        command,
        settings,
        identity,
        git=GitState(SHA, False),
        uv_lock_sha256="e" * 64,
        created_at=T0,
    )


def _finished() -> Manifest:
    running = transition(_queued(), "running", at=T1, machine=MACHINE)
    return transition(running, "finished", at=T2)


def test_enums() -> None:
    assert ENGINES == ("lightning", "plain")
    assert STATUSES == ("queued", "running", "finished", "failed", "cancelled")
    assert frozenset({"finished", "failed", "cancelled"}) == TERMINAL_STATUSES


def test_new_manifest_is_queued() -> None:
    manifest = _queued()
    assert manifest.status == "queued"
    assert manifest.attempt == 1
    assert manifest.seeds == {"seed": 0}
    assert manifest.machine is None
    assert manifest.started_at is None
    assert manifest.run_id == manifest.identity.run_id


def test_life_cycle() -> None:
    running = transition(_queued(), "running", at=T1, machine=MACHINE)
    assert (running.status, running.started_at, running.machine) == ("running", T1, MACHINE)
    requeued = transition(running, "queued")  # an expired lease
    again = transition(requeued, "running", at=T2, machine=MACHINE)
    assert again.started_at == T1  # the first start
    done = transition(again, "finished", at=T2)
    assert (done.status, done.ended_at) == ("finished", T2)
    assert is_finished(done.status)


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_failed_and_cancelled_are_never_finished(status: str) -> None:
    running = transition(_queued(), "running", at=T1, machine=MACHINE)
    ended = transition(running, status, at=T2)  # type: ignore[arg-type]
    assert not is_finished(ended.status)
    cancelled = transition(_queued(), "cancelled", at=T1)
    assert cancelled.started_at is None
    assert not is_finished(cancelled.status)


@pytest.mark.parametrize(
    ("start", "status"),
    [
        ("queued", "finished"),
        ("queued", "failed"),
        ("finished", "running"),
        ("cancelled", "queued"),
    ],
)
def test_forbidden_transitions(start: str, status: str) -> None:
    manifest = _queued()
    if start == "finished":
        manifest = _finished()
    elif start == "cancelled":
        manifest = transition(manifest, "cancelled", at=T1)
    with pytest.raises(ManifestError, match="cannot become"):
        transition(manifest, status)  # type: ignore[arg-type]


def test_running_needs_the_machine() -> None:
    with pytest.raises(ManifestError, match="machine record"):
        transition(_queued(), "running", at=T1)


def test_transition_defaults_to_now() -> None:
    running = transition(_queued(), "running", machine=MACHINE)
    assert running.started_at is not None
    assert running.started_at >= T0


def test_utc_now_has_an_offset() -> None:
    assert utc_now().endswith("+00:00")


def test_write_then_read(tmp_path: Path) -> None:
    for manifest in (_queued(), _finished(), _queued("pretrain", "pretrain-legacy")):
        path = write_manifest(tmp_path, manifest)
        assert path == tmp_path / MANIFEST_FILE
        assert read_manifest(tmp_path) == manifest
        assert read_manifest(path) == manifest
    data = json.loads(path.read_text(encoding="utf-8"))
    assert (data["format"], data["format_version"]) == (FORMAT, FORMAT_VERSION)
    assert list(data)[:3] == ["format", "format_version", "run_id"]
    assert not list(tmp_path.glob("*.tmp"))


def test_the_manifest_records_the_full_config(tmp_path: Path) -> None:
    write_manifest(tmp_path, _finished())
    data = json.loads((tmp_path / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert data["config"]["output_dir"] == "outputs"  # non-semantic, recorded all the same
    assert data["config_sha256"] == data["identity"]["config_sha256"]
    assert data["machine"]["platform"] == MACHINE["platform"]  # extra keys kept


def test_an_older_manifest_reads_after_a_hash_neutral_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Written before `rot90_augment` existed (simulated): its hash must still check out.
    neutral = {**canonical.HASH_NEUTRAL_DEFAULTS, "rot90_augment": False}
    monkeypatch.setattr(canonical, "HASH_NEUTRAL_DEFAULTS", neutral)
    manifest = _finished()
    data = manifest.to_plain()
    config = dict(manifest.config)
    del config["rot90_augment"]
    data["config"] = config
    (tmp_path / MANIFEST_FILE).write_text(json.dumps(data), encoding="utf-8")
    assert read_manifest(tmp_path).config == config


def _write(tmp_path: Path, edit: Callable[[dict[str, Any]], object]) -> Path:
    data = _finished().to_plain()
    edit(data)
    (tmp_path / MANIFEST_FILE).write_text(json.dumps(data), encoding="utf-8")
    return tmp_path


def _set(path: str, value: object) -> Callable[[dict[str, Any]], object]:
    def edit(data: dict[str, Any]) -> None:
        *parents, last = path.split(".")
        target = data
        for key in parents:
            target = target[key]
        target[last] = value

    return edit


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (_set("format", "other"), "not an astro-simclr-run-manifest"),
        (_set("format_version", 2), "format version 2"),
        (lambda d: d.pop("engine"), r"missing keys \['engine'\]"),
        (_set("extra", 1), r"unknown \['extra'\]"),
        (_set("canonical_version", 2), "canonical_version 2"),
        (_set("config.lr", 0.5), "config hashes to"),
        (_set("config.seed", 3), "config seed"),
        (_set("run_id", "f" * 64), "run_id"),
        (_set("config_sha256", "f" * 64), "config_sha256"),
        (_set("identity.seed", 3), "config hashes to|seed"),
        (_set("identity.split", 1), "go together"),
        (_set("command", "pretrain"), "config hashes to"),
        (_set("command", 5), "command must be a string"),
        (_set("command", "run"), "command 'run'"),
        (_set("attempt", 0), "attempt"),
        (_set("concurrent_jobs", True), "concurrent_jobs"),
        (_set("engine", "torch"), "engine 'torch'"),
        (_set("engine", "plain"), "is not identity.engine"),
        (_set("format_version", True), "format version True"),
        (_set("format_version", 1.0), "format version 1.0"),
        (_set("canonical_version", True), "canonical_version True"),
        (_set("config.seed", 0.0), "config seed"),
        (_set("config.seed", False), "config seed"),
        (_set("seeds", {"seed": False}), "seeds must map"),
        (_set("seeds", {"seed": 0, "views": -1}), "integer >= 0"),
        (_set("config.pretrained", "encoder.ckpt"), "go together"),
        (_set("identity.pretrained_sha256", "c" * 64), "go together"),
        (_set("status", "done"), "status 'done'"),
        (_set("status", "running"), "ended_at"),
        (_set("seeds", {"seed": 1}), "seeds must map"),
        (_set("seeds", {"seed": 0, "views": "1"}), "seeds: 'views'"),
        (_set("seeds", [0]), "seeds must be a mapping"),
        (_set("git", {"sha": SHA}), "git needs exactly"),
        (_set("git.sha", "HEAD"), "git.sha"),
        (_set("git.dirty", "no"), "git.dirty"),
        (_set("git", {"sha": None, "dirty": False}), "git.dirty needs git.sha"),
        (_set("uv_lock_sha256", "x"), "uv_lock_sha256"),
        (_set("machine", None), "needs started_at and machine"),
        (_set("machine", "cpu"), "machine must be a mapping"),
        (lambda d: d["machine"].pop("torch"), "machine lacks torch"),
        (_set("machine.device", ""), "machine.device"),
        (_set("machine.gpu", 1), "machine.gpu"),
        (_set("machine.fp32_precision", "bf16"), "fp32_precision"),
        (_set("machine.extra", float("nan")), "no JSON form|machine"),
        (_set("timestamps", {"created_at": T0}), "timestamps exactly"),
        (_set("timestamps.created_at", 5), "ISO 8601 string"),
        (_set("timestamps.created_at", "yesterday"), "not ISO 8601"),
        (_set("timestamps.created_at", "2026-11-08T10:00:00"), "no UTC offset"),
        (_set("timestamps.created_at", "2026-11-09T10:00:00+00:00"), "created_at <= started_at"),
        (_set("timestamps.started_at", None), "needs started_at"),
        (_set("config", [1]), "config must be a mapping"),
        (_set("identity", {"seed": 0}), "identity: missing"),
    ],
)
def test_the_reader_refuses_invalid_manifests(
    tmp_path: Path, edit: Callable[[dict[str, Any]], object], match: str
) -> None:
    with pytest.raises(ManifestError, match=match):
        read_manifest(_write(tmp_path, edit))


def test_machine_needs_a_start(tmp_path: Path) -> None:
    def edit(data: dict[str, Any]) -> None:
        data["status"] = "failed"
        data["timestamps"]["started_at"] = None

    with pytest.raises(ManifestError, match="machine is recorded when the attempt starts"):
        read_manifest(_write(tmp_path, edit))


def test_the_constructor_checks_the_machine_type() -> None:
    with pytest.raises(ManifestError, match="machine must be a mapping"):
        dataclasses.replace(_finished(), machine="cpu")  # type: ignore[arg-type]


def test_corrupt_files_are_manifest_errors(tmp_path: Path) -> None:
    file = tmp_path / MANIFEST_FILE
    for raw in (b"\xff\xfe{", b"[" * 100_000, b'{"status": "queued", "status": "finished"}'):
        file.write_bytes(raw)
        with pytest.raises(ManifestError, match=str(tmp_path)):
            read_manifest(tmp_path)


def test_a_duplicate_key_is_refused(tmp_path: Path) -> None:
    text = json.dumps(_finished().to_plain())
    text = text.replace('"status": "finished"', '"status": "finished", "status": "failed"')
    (tmp_path / MANIFEST_FILE).write_text(text, encoding="utf-8")
    with pytest.raises(ManifestError, match="'status' appears twice"):
        read_manifest(tmp_path)


def test_a_validated_manifest_is_read_only() -> None:
    manifest = _finished()
    with pytest.raises(TypeError):
        manifest.config["lr"] = 0.5  # type: ignore[index]
    with pytest.raises(TypeError):
        manifest.seeds["seed"] = 1  # type: ignore[index]


def test_a_linear_eval_manifest_carries_the_encoder() -> None:
    _, settings = case_settings("train-supervised-legacy")
    settings = settings.replace(pretrained="encoder.ckpt")
    identity = run_identity(
        "linear-eval",
        settings,
        data_sha256="d" * 64,
        machine_class="cpu-arm64",
        engine="lightning",
        pretrained_sha256="c" * 64,
    )
    manifest = new_manifest(
        "linear-eval", settings, identity, git=GitState(None, None), uv_lock_sha256=None
    )
    assert manifest.config["pretrained"] == "encoder.ckpt"
    with pytest.raises(ManifestError, match="go together"):
        new_manifest(
            "linear-eval",
            settings.replace(pretrained=None),
            identity,
            git=GitState(None, None),
            uv_lock_sha256=None,
        )


def test_the_reader_refuses_unreadable_files(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="cannot read"):
        read_manifest(tmp_path)
    (tmp_path / MANIFEST_FILE).write_text("{", encoding="utf-8")
    with pytest.raises(ManifestError, match="invalid JSON"):
        read_manifest(tmp_path)
    (tmp_path / MANIFEST_FILE).write_text("[]", encoding="utf-8")
    with pytest.raises(ManifestError, match=str(tmp_path)):
        read_manifest(tmp_path)


def test_new_manifest_refuses_an_identity_of_other_settings() -> None:
    manifest = _queued()
    _, other = case_settings("carrasco2021+p4")
    with pytest.raises(ManifestError, match="config hashes to"):
        new_manifest(
            manifest.command,
            other,
            manifest.identity,
            git=GitState(None, None),
            uv_lock_sha256=None,
        )
