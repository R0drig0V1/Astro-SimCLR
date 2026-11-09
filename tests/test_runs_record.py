"""Tests of ``astro_simclr.runs.record``: the manifest ``run_command`` writes (task A1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from astro_simclr.config import ConfigError, Settings
from astro_simclr.data.files import sha256_file
from astro_simclr.protocol.split_runs import SplitRequest
from astro_simclr.runs.identity import SCREEN, WHOLE, data_sha256
from astro_simclr.runs.manifest import MANIFEST_FILE, ManifestError, read_manifest
from astro_simclr.runs.record import (
    CONCURRENT_JOBS_ENV,
    QueueAttempt,
    RunRecord,
    concurrent_jobs_from_env,
    source_root,
    start_run,
)

MACHINE = {
    "device": "cpu",
    "gpu": None,
    "torch": "2.14.1",
    "cuda": None,
    "fp32_precision": "ieee",
    "processor": "arm64",
    "engine": "plain",
}
T0, T1, T2 = "2026-11-08T10:00:00+00:00", "2026-11-08T10:05:00+00:00", "2026-11-08T11:00:00+00:00"


def _settings(tmp_path: Path, **changes: object) -> Settings:
    return Settings(output_dir=tmp_path / "out").replace(**changes)


def test_a_fresh_run_is_running_attempt_one(classifier_dir: Path, tmp_path: Path) -> None:
    settings = _settings(tmp_path, seed=3)
    record = start_run(
        "train-supervised", settings, data=classifier_dir, machine=MACHINE, root=tmp_path, at=T0
    )
    manifest = read_manifest(settings.output_dir)
    assert manifest == record.manifest
    assert (manifest.status, manifest.attempt, manifest.engine) == ("running", 1, "plain")
    assert manifest.identity.seed == 3
    assert manifest.identity.data_sha256 == data_sha256(classifier_dir)
    assert manifest.identity.machine_class == "cpu-arm64"
    assert manifest.identity.split is None
    assert (manifest.git.sha, manifest.uv_lock_sha256) == (None, None)  # tmp_path: no checkout
    assert manifest.concurrent_jobs == 1
    assert record.summary() == {
        "run_id": manifest.run_id,
        "attempt": 1,
        "config_sha256": manifest.identity.config_sha256,
    }


@pytest.mark.parametrize("status", ["finished", "failed", "cancelled"])
def test_end_writes_the_final_status_once(
    classifier_dir: Path, tmp_path: Path, status: str
) -> None:
    settings = _settings(tmp_path)
    record = start_run(
        "train-supervised", settings, data=classifier_dir, machine=MACHINE, root=tmp_path, at=T0
    )
    record.end(status, at=T1)  # type: ignore[arg-type]
    record.end("failed", at=T2)  # ignored: the attempt has ended
    manifest = read_manifest(settings.output_dir)
    assert (manifest.status, manifest.ended_at) == (status, T1)


def test_end_needs_a_terminal_status(classifier_dir: Path, tmp_path: Path) -> None:
    record = start_run(
        "train-supervised", _settings(tmp_path), data=classifier_dir, machine=MACHINE, root=tmp_path
    )
    with pytest.raises(ManifestError, match="not running"):
        record.end("running")


def _start(
    classifier_dir: Path, settings: Settings, *, resumed: bool = False, at: str = T1
) -> RunRecord:
    return start_run(
        "train-supervised",
        settings,
        data=classifier_dir,
        machine=MACHINE,
        resumed=resumed,
        root=settings.output_dir.parent,
        at=at,
    )


def test_a_rerun_is_the_next_attempt(classifier_dir: Path, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    first = _start(classifier_dir, settings, at=T0)
    first.end("finished", at=T1)
    second = _start(classifier_dir, settings.replace(num_workers=0, accelerator="cpu"), at=T2)
    assert second.manifest.run_id == first.manifest.run_id
    assert second.manifest.attempt == 2
    assert second.manifest.started_at == T2
    # A resume after a terminal status is a new attempt too.
    second.end("failed", at=T2)
    third = _start(classifier_dir, settings, resumed=True, at=T2)
    assert third.manifest.attempt == 3


def test_a_resume_of_a_killed_attempt_continues_it(classifier_dir: Path, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    killed = _start(classifier_dir, settings, at=T0)  # never ended: kill -9
    resumed = _start(classifier_dir, settings, resumed=True, at=T2)
    assert resumed.manifest.attempt == killed.manifest.attempt == 1
    assert resumed.manifest.started_at == T0  # the first start
    assert resumed.manifest.status == "running"
    # Without --resume, the same command is a new attempt.
    restarted = _start(classifier_dir, settings, at=T2)
    assert restarted.manifest.attempt == 2


def test_another_run_in_the_directory_starts_at_attempt_one(
    classifier_dir: Path, tmp_path: Path
) -> None:
    settings = _settings(tmp_path)
    _start(classifier_dir, settings, at=T0).end("finished", at=T1)
    other = _start(classifier_dir, settings.replace(lr=0.5), at=T2)
    assert other.manifest.attempt == 1


def test_an_unreadable_manifest_is_never_replaced(classifier_dir: Path, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.output_dir.mkdir()
    (settings.output_dir / MANIFEST_FILE).write_text("{", encoding="utf-8")
    with pytest.raises(ConfigError, match="cannot be read"):
        _start(classifier_dir, settings)
    assert (settings.output_dir / MANIFEST_FILE).read_text(encoding="utf-8") == "{"


def test_split_and_pretrained_are_hashed(classifier_dir: Path, tmp_path: Path) -> None:
    splits = tmp_path / "splits.json"
    splits.write_text('{"x": 1}', encoding="utf-8")
    encoder = tmp_path / "encoder.ckpt"
    encoder.write_bytes(b"weights")
    settings = _settings(tmp_path, pretrained=encoder)
    record = start_run(
        "linear-eval",
        settings,
        data=classifier_dir,
        machine=MACHINE,
        split=SplitRequest(splits, None),
        pretrained_file=encoder,
        root=tmp_path,
    )
    identity = record.manifest.identity
    assert (identity.split, identity.splits_sha256) == (SCREEN, sha256_file(splits))
    assert identity.pretrained_sha256 == sha256_file(encoder)
    with pytest.raises(ConfigError, match="checkpoint file to hash"):
        start_run("linear-eval", settings, data=classifier_dir, machine=MACHINE, root=tmp_path)


def test_the_source_checkout_is_recorded(classifier_dir: Path, tmp_path: Path) -> None:
    root = source_root()
    assert root is not None
    assert (root / "pyproject.toml").is_file()
    record = start_run(
        "train-supervised", _settings(tmp_path), data=classifier_dir, machine=MACHINE
    )
    manifest = record.manifest
    assert manifest.uv_lock_sha256 == sha256_file(root / "uv.lock")
    data = json.loads((tmp_path / "out" / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert data["git"] == {"sha": manifest.git.sha, "dirty": manifest.git.dirty}


def test_concurrent_jobs(
    classifier_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert concurrent_jobs_from_env({}) == 1
    assert concurrent_jobs_from_env({CONCURRENT_JOBS_ENV: ""}) == 1
    assert concurrent_jobs_from_env({CONCURRENT_JOBS_ENV: "2"}) == 2
    for bad in ("0", "-1", "two", "1.5"):
        with pytest.raises(ConfigError, match=CONCURRENT_JOBS_ENV):
            concurrent_jobs_from_env({CONCURRENT_JOBS_ENV: bad})
    monkeypatch.setenv(CONCURRENT_JOBS_ENV, "3")
    record = _start(classifier_dir, _settings(tmp_path))
    assert record.manifest.concurrent_jobs == 3


def test_the_machine_engine_must_agree(classifier_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match=r"machine\.engine"):
        start_run(
            "train-supervised",
            _settings(tmp_path),
            data=classifier_dir,
            machine={**MACHINE, "engine": "lightning"},
            root=tmp_path,
        )


def test_object_splits_are_the_whole_split_part(classifier_dir: Path, tmp_path: Path) -> None:
    splits = tmp_path / "object_splits.json"
    splits.write_text("{}", encoding="utf-8")
    settings = _settings(tmp_path)
    start_run(
        "train-supervised",
        settings,
        data=classifier_dir,
        machine=MACHINE,
        object_splits=splits,
        root=tmp_path,
    )
    identity = read_manifest(settings.output_dir).identity
    assert (identity.split, identity.splits_sha256) == (WHOLE, sha256_file(splits))
    with pytest.raises(ConfigError, match="one splits file"):
        start_run(
            "train-supervised",
            settings,
            data=classifier_dir,
            machine=MACHINE,
            split=SplitRequest(splits, None),
            object_splits=splits,
            root=tmp_path,
        )


# --- queued runs (task A3) ----------------------------------------------------------------------


def test_a_queued_run_takes_the_queues_attempt_and_machine_class(
    classifier_dir: Path, tmp_path: Path
) -> None:
    settings = _settings(tmp_path)
    queued = QueueAttempt(attempt=3, machine_class="cuda-nvidia-geforce-gtx-1050")
    start_run(
        "train-supervised",
        settings,
        data=classifier_dir,
        machine=MACHINE,
        root=tmp_path,
        at=T0,
        queued=queued,
    )
    manifest = read_manifest(settings.output_dir)
    assert manifest.attempt == 3
    assert manifest.identity.machine_class == "cuda-nvidia-geforce-gtx-1050"
    # A crash before any checkpoint (resumed=False): the same attempt continues.
    again = start_run(
        "train-supervised",
        settings,
        data=classifier_dir,
        machine=MACHINE,
        root=tmp_path,
        at=T1,
        queued=queued,
    )
    assert (again.manifest.attempt, again.manifest.started_at) == (3, T0)
    again.end("finished", at=T2)
    # A finished attempt, or another attempt number, is never overwritten.
    for attempt in (3, 4):
        with pytest.raises(ConfigError, match="not a resumable run"):
            start_run(
                "train-supervised",
                settings,
                data=classifier_dir,
                machine=MACHINE,
                root=tmp_path,
                queued=QueueAttempt(attempt=attempt, machine_class=queued.machine_class),
            )


def test_a_queued_run_with_another_run_id_is_refused(classifier_dir: Path, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    queued = QueueAttempt(attempt=1, machine_class="cpu-arm64", run_id="0" * 64)
    with pytest.raises(ConfigError, match="disagree"):
        start_run(
            "train-supervised",
            settings,
            data=classifier_dir,
            machine=MACHINE,
            root=tmp_path,
            queued=queued,
        )
    assert not (settings.output_dir / MANIFEST_FILE).exists()
