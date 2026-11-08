"""Tests of ``astro_simclr.runs.identity``: the run id and the provenance helpers."""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from _canonical_cases import FIXTURE, case_settings, identity_cases

from astro_simclr.config import Settings
from astro_simclr.data.files import sha256_file
from astro_simclr.data.raw_alerts import make_index, write_raw_alerts
from astro_simclr.data.schema import DataError
from astro_simclr.data.synthetic import synthetic_raw_alerts
from astro_simclr.protocol.alerts import RJ23_LABEL_NAMES as RJ23_NAMES
from astro_simclr.protocol.split_runs import SplitRequest
from astro_simclr.runs.canonical import config_sha256
from astro_simclr.runs.identity import (
    NUMERICS_EPOCH,
    SCREEN,
    GitState,
    IdentityError,
    RunIdentity,
    data_sha256,
    default_machine_class,
    git_state,
    prepared_data_sha256,
    run_identity,
    split_part,
    uv_lock_sha256,
)

A, B, C = "a" * 64, "b" * 64, "c" * 64


def _identity(**changes: Any) -> RunIdentity:
    values: dict[str, Any] = {
        "config_sha256": A,
        "seed": 0,
        "split": None,
        "data_sha256": B,
        "splits_sha256": None,
        "pretrained_sha256": None,
        "machine_class": "cpu-arm64",
        "engine": "lightning",
    }
    return RunIdentity(**{**values, **changes})


def test_run_ids_match_the_fixture() -> None:
    pinned = json.loads(FIXTURE.read_text(encoding="utf-8"))["run_ids"]
    for name, identity in identity_cases().items():
        assert identity.to_plain() == pinned[name]["identity"]
        assert identity.run_id == pinned[name]["run_id"]


def test_numerics_epoch_is_a_positive_int() -> None:
    assert isinstance(NUMERICS_EPOCH, int)
    assert NUMERICS_EPOCH >= 1
    assert _identity().numerics_epoch == NUMERICS_EPOCH


@pytest.mark.parametrize(
    "changes",
    [
        {"config_sha256": C},
        {"seed": 1},
        {"split": 0, "splits_sha256": C},
        {"split": SCREEN, "splits_sha256": C},
        {"data_sha256": C},
        {"pretrained_sha256": C},
        {"machine_class": "cuda-nvidia-geforce-gtx-1050"},
        {"engine": "plain"},
        {"numerics_epoch": NUMERICS_EPOCH + 1},
    ],
)
def test_every_component_moves_the_run_id(changes: dict[str, Any]) -> None:
    assert _identity(**changes).run_id != _identity().run_id


def test_fold_and_splits_file_both_count() -> None:
    fold0 = _identity(split=0, splits_sha256=C)
    assert fold0.run_id != _identity(split=1, splits_sha256=C).run_id
    assert fold0.run_id != _identity(split=0, splits_sha256=A).run_id


def test_run_id_is_a_stable_hex_digest() -> None:
    run_id = _identity().run_id
    assert run_id == _identity().run_id
    assert len(run_id) == 64
    assert int(run_id, 16) >= 0


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"config_sha256": "A" * 64}, "config_sha256"),
        ({"data_sha256": "ab"}, "data_sha256"),
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
        ({"split": -1, "splits_sha256": C}, "fold"),
        ({"split": "fold", "splits_sha256": C}, "fold"),
        ({"split": 0}, "go together"),
        ({"splits_sha256": C}, "go together"),
        ({"split": 0, "splits_sha256": "x"}, "splits_sha256"),
        ({"pretrained_sha256": "x"}, "pretrained_sha256"),
        ({"machine_class": "GTX 1050"}, "slug"),
        ({"machine_class": ""}, "slug"),
        ({"machine_class": "x" * 65}, "slug"),
        ({"numerics_epoch": 0}, "numerics_epoch"),
        ({"engine": "torch"}, "engine 'torch'"),
    ],
)
def test_invalid_identities(changes: dict[str, Any], match: str) -> None:
    with pytest.raises(IdentityError, match=match):
        _identity(**changes)


def test_plain_round_trip() -> None:
    identity = _identity(split=SCREEN, splits_sha256=C, pretrained_sha256=A)
    assert RunIdentity.from_plain(identity.to_plain()) == identity
    folded = _identity(split=3, splits_sha256=C)
    assert RunIdentity.from_plain(folded.to_plain()) == folded


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"extra": 1}, "unknown"),
        ({"seed": "0"}, "seed"),
        ({"split": 1.5}, "fold"),
        ({"config_sha256": 5}, "config_sha256 must be a string"),
        ({"splits_sha256": 5}, "splits_sha256 must be a string"),
        ({"numerics_epoch": None}, "numerics_epoch"),
    ],
)
def test_from_plain_refuses_bad_records(changes: dict[str, Any], match: str) -> None:
    with pytest.raises(IdentityError, match=match):
        RunIdentity.from_plain({**_identity().to_plain(), **changes})
    plain = _identity().to_plain()
    del plain["seed"]
    with pytest.raises(IdentityError, match="missing"):
        RunIdentity.from_plain(plain)


def test_run_identity_takes_the_seed_and_hash_from_the_settings() -> None:
    command, settings = case_settings("carrasco2021")
    identity = run_identity(
        command, settings.replace(seed=4), data_sha256=B, machine_class="cpu", engine="plain"
    )
    assert identity.seed == 4
    assert identity.config_sha256 == config_sha256(command, settings)


def test_run_identity_needs_the_pretrained_hash_with_a_pretrained_path() -> None:
    settings = Settings(pretrained=Path("encoder.ckpt"))
    identity = run_identity(
        "linear-eval",
        settings,
        data_sha256=B,
        machine_class="cpu",
        engine="plain",
        pretrained_sha256=C,
    )
    assert identity.pretrained_sha256 == C
    with pytest.raises(IdentityError, match="pretrained_sha256 goes with"):
        run_identity("linear-eval", settings, data_sha256=B, machine_class="cpu", engine="plain")
    with pytest.raises(IdentityError, match="pretrained_sha256 goes with"):
        run_identity(
            "train-supervised",
            Settings(),
            data_sha256=B,
            machine_class="cpu",
            engine="plain",
            pretrained_sha256=C,
        )


def test_split_part() -> None:
    assert split_part(None) is None
    assert split_part(SplitRequest(Path("s.json"), 2)) == 2
    assert split_part(SplitRequest(Path("s.json"), None)) == SCREEN


# --- Data, lock and git provenance -----------------------------------------------------------


def _meta(**changes: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "format": "astro-simclr-prepared",
        "format_version": 2,
        "kind": "labelled",
        "created_by": "astro-simclr 0.1.0",
        "source": {"name": "x.pkl", "size_bytes": 10, "sha256": A},
        "splits": {
            "Train": {"n": 3, "class_counts": {}, "files": {"images.npy": {"sha256": B}}},
            "Test": {"n": 2, "class_counts": {}, "files": {"images.npy": {"sha256": C}}},
        },
    }
    return {**meta, **changes}


def test_data_hash_ignores_the_writer_version_and_format_version() -> None:
    base = prepared_data_sha256(_meta())
    assert prepared_data_sha256(_meta(created_by="astro-simclr 9")) == base
    assert prepared_data_sha256(_meta(format_version=1)) == base


def test_data_hash_follows_the_content() -> None:
    base = prepared_data_sha256(_meta())
    splits = _meta()["splits"]
    splits["Train"]["files"]["images_f32.npy"] = {"sha256": C}
    assert prepared_data_sha256(_meta(splits=splits)) != base
    assert prepared_data_sha256(_meta(source={"sha256": C})) != base
    assert prepared_data_sha256(_meta(kind="unlabelled")) != base


@pytest.mark.parametrize(
    "changes",
    [
        {"source": None},
        {"splits": None},
        {"source": {"sha256": "x"}},
        {"splits": {"Train": {"n": 1}}},
        {"splits": {"Train": 3}},
        {"splits": {"Train": {"n": 1, "files": {"images.npy": {}}}}},
        {"splits": {"Train": {"n": 1, "files": {"images.npy": "abc"}}}},
    ],
)
def test_data_hash_needs_the_recorded_hashes(changes: dict[str, Any]) -> None:
    with pytest.raises(IdentityError):
        prepared_data_sha256(_meta(**changes))


def test_data_sha256_reads_a_prepared_layout(classifier_dir: Path) -> None:
    meta = json.loads((classifier_dir / "meta.json").read_text(encoding="utf-8"))
    assert data_sha256(classifier_dir) == prepared_data_sha256(meta)
    with pytest.raises(DataError):
        data_sha256(classifier_dir.parent / "missing")


def test_raw_alerts_layouts_hash_their_pool(tmp_path: Path) -> None:
    cutouts, columns = synthetic_raw_alerts(RJ23_NAMES, n_per_class=2, seed=0)
    path = write_raw_alerts(
        tmp_path / "pool", cutouts, make_index(columns), RJ23_NAMES, {"run": "x"}
    )
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    base = data_sha256(path)
    assert base == prepared_data_sha256(meta)
    assert prepared_data_sha256({**meta, "source": {"run": "y"}, "created_by": "z"}) == base
    renamed = {**meta, "schema": {**meta["schema"], "label_names": ["a", "b"]}}
    assert prepared_data_sha256(renamed) != base
    with pytest.raises(IdentityError, match="pool"):
        prepared_data_sha256({**meta, "files": None})


def test_uv_lock_sha256(tmp_path: Path) -> None:
    assert uv_lock_sha256(tmp_path) is None
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    assert uv_lock_sha256(tmp_path) == sha256_file(tmp_path / "uv.lock")


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)  # noqa: S603, S607


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_git_state(tmp_path: Path) -> None:
    assert git_state(tmp_path) == GitState(None, None)
    _git(tmp_path, "init", "-q")
    assert git_state(tmp_path) == GitState(None, None)  # no commit yet
    (tmp_path / ".gitignore").write_text("outputs/\n", encoding="utf-8")
    _git(tmp_path, "add", ".gitignore")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "init")
    clean = git_state(tmp_path)
    assert clean.dirty is False
    assert clean.sha is not None
    assert len(clean.sha) in (40, 64)
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "metrics.json").write_text("{}", encoding="utf-8")
    assert git_state(tmp_path).dirty is False  # ignored run outputs
    (tmp_path / "new.py").write_text("", encoding="utf-8")
    assert git_state(tmp_path) == GitState(clean.sha, True)


def test_git_state_without_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def missing(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", missing)
    assert git_state(tmp_path) == GitState(None, None)


def test_git_state_when_status_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        code = 0 if "rev-parse" in argv else 128
        return subprocess.CompletedProcess(argv, code, stdout="f" * 40 + "\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake)
    assert git_state(tmp_path) == GitState("f" * 40, None)


@pytest.mark.parametrize(
    ("device", "gpu", "processor", "expected"),
    [
        ("cuda", "NVIDIA GeForce GTX 1050", "x86_64", "cuda-nvidia-geforce-gtx-1050"),
        ("cuda", "NVIDIA GeForce RTX 5090", "AMD64", "cuda-nvidia-geforce-rtx-5090"),
        ("cuda", None, "x86_64", "cuda-unknown"),
        ("cpu", None, "arm64", "cpu-arm64"),
        ("cpu", None, "x86_64", "cpu-x86_64"),
        ("mps", None, "arm64", "mps-arm64"),
        ("cpu", None, "", "cpu-unknown"),
    ],
)
def test_default_machine_class(device: str, gpu: str | None, processor: str, expected: str) -> None:
    machine_class = default_machine_class(device, gpu, processor)
    assert machine_class == expected
    _identity(machine_class=machine_class)  # a valid slug


def test_default_machine_class_is_cut_to_a_slug() -> None:
    machine_class = default_machine_class("cuda", "X" * 100, "x86_64")
    assert len(machine_class) == 64
    _identity(machine_class=machine_class)


def test_identity_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _identity().seed = 1  # type: ignore[misc]
