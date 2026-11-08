"""Tests of ``astro_simclr.runs.layout``: run-directory names, artefact kinds, retention."""

from __future__ import annotations

from pathlib import Path

import pytest
from _canonical_cases import case_settings

from astro_simclr.experiments import BEST_CHECKPOINT as LIGHTNING_BEST_NAME
from astro_simclr.experiments import CHECKPOINT_DIR as LIGHTNING_DIR
from astro_simclr.experiments import LAST_CHECKPOINT as LIGHTNING_LAST_NAME
from astro_simclr.experiments import METRICS_FILE as EXPERIMENTS_METRICS
from astro_simclr.runs.identity import GitState, IdentityError, run_identity
from astro_simclr.runs.layout import (
    LIGHTNING_BEST,
    LIGHTNING_LAST,
    METRICS_FILE,
    ArtefactKind,
    classify_artefact,
    retention,
    run_dir_name,
)
from astro_simclr.runs.manifest import Manifest, Status, new_manifest, transition

MACHINE = {"device": "cpu", "gpu": None, "torch": "2.14.1", "cuda": None, "fp32_precision": "ieee"}
T0, T1, T2 = "2026-11-08T10:00:00+00:00", "2026-11-08T10:05:00+00:00", "2026-11-08T11:00:00+00:00"

ARTEFACTS = [
    "manifest.json",
    "metrics.json",
    "scalars.jsonl",
    "tb/events.out.tfevents.1",
    "predictions_val.npz",
    "predictions_evaluation.npz",
    "confusion_matrix_test.png",
    "ckpt/last.pt",
    "ckpt/best.pt",
    "checkpoints/last.ckpt",
    "checkpoints/best.ckpt",
    "lightning_logs/version_0/hparams.yaml",
    "notes.txt",
    "ckpt/epoch=3.pt",
]


def _manifest(command: str, case: str, status: Status) -> Manifest:
    _, settings = case_settings(case)
    identity = run_identity(
        command, settings, data_sha256="d" * 64, machine_class="cpu-arm64", engine="lightning"
    )
    manifest = new_manifest(
        command,
        settings,
        identity,
        git=GitState(None, None),
        uv_lock_sha256=None,
        created_at=T0,
    )
    if status == "queued":
        return manifest
    if status == "cancelled":
        return transition(manifest, "cancelled", at=T1)
    running = transition(manifest, "running", at=T1, machine=MACHINE)
    return running if status == "running" else transition(running, status, at=T2)


def test_lightning_names_match_experiments() -> None:
    assert f"{LIGHTNING_DIR}/{LIGHTNING_LAST_NAME}" == LIGHTNING_LAST
    assert f"{LIGHTNING_DIR}/{LIGHTNING_BEST_NAME}" == LIGHTNING_BEST
    assert EXPERIMENTS_METRICS == METRICS_FILE


def test_run_dir_name() -> None:
    assert run_dir_name("0123456789abcdef" + "0" * 48, 2) == "0123456789abcdef-a2"
    with pytest.raises(IdentityError):
        run_dir_name("xyz", 1)
    for attempt in (0, True):
        with pytest.raises(ValueError, match="attempt"):
            run_dir_name("a" * 64, attempt)


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        ("manifest.json", ArtefactKind.MANIFEST),
        ("metrics.json", ArtefactKind.METRICS),
        ("scalars.jsonl", ArtefactKind.SCALARS),
        ("tb/events.out.tfevents.1", ArtefactKind.TENSORBOARD),
        ("lightning_logs/version_0/hparams.yaml", ArtefactKind.TENSORBOARD),
        ("predictions_test.npz", ArtefactKind.PREDICTIONS),
        ("confusion_matrix_val.png", ArtefactKind.FIGURE),
        ("ckpt/last.pt", ArtefactKind.LAST_CHECKPOINT),
        ("checkpoints/last.ckpt", ArtefactKind.LAST_CHECKPOINT),
        ("ckpt/best.pt", ArtefactKind.BEST_CHECKPOINT),
        ("checkpoints/best.ckpt", ArtefactKind.BEST_CHECKPOINT),
        ("ckpt/epoch=3.pt", ArtefactKind.OTHER),
        ("sub/predictions_val.npz", ArtefactKind.OTHER),
        ("tb", ArtefactKind.OTHER),
        ("notes.txt", ArtefactKind.OTHER),
    ],
)
def test_classify_artefact(path: str, kind: ArtefactKind) -> None:
    assert classify_artefact(path) is kind


def _deleted(manifest: Manifest) -> set[str]:
    decisions = retention(manifest, ARTEFACTS)
    assert set(decisions) == set(ARTEFACTS)
    return {path for path, decision in decisions.items() if decision.action == "delete"}


def test_a_finished_pretraining_run_keeps_best() -> None:
    manifest = _manifest("pretrain", "pretrain-legacy", "finished")
    assert _deleted(manifest) == {"ckpt/last.pt", "checkpoints/last.ckpt"}
    decision = retention(manifest, ["checkpoints/best.ckpt"])["checkpoints/best.ckpt"]
    assert (decision.action, decision.kind) == ("keep", ArtefactKind.BEST_CHECKPOINT)


@pytest.mark.parametrize(
    ("command", "case"),
    [
        ("train-supervised", "carrasco2021"),
        ("linear-eval", "train-supervised-legacy"),
        ("fine-tune", "train-supervised-legacy"),
    ],
)
def test_a_finished_classifier_drops_both_checkpoints(command: str, case: str) -> None:
    deleted = _deleted(_manifest(command, case, "finished"))
    assert deleted == {
        "ckpt/last.pt",
        "ckpt/best.pt",
        "checkpoints/last.ckpt",
        "checkpoints/best.ckpt",
    }


@pytest.mark.parametrize("status", ["queued", "running", "failed", "cancelled"])
def test_an_unfinished_attempt_keeps_everything(status: Status) -> None:
    assert _deleted(_manifest("train-supervised", "carrasco2021", status)) == set()


def test_records_results_and_unknown_files_are_kept() -> None:
    decisions = retention(_manifest("train-supervised", "carrasco2021", "finished"), ARTEFACTS)
    for path in ("metrics.json", "predictions_val.npz", "notes.txt", "ckpt/epoch=3.pt"):
        assert decisions[path].action == "keep"
        assert decisions[path].reason


def test_retention_touches_no_file(tmp_path: Path) -> None:
    (tmp_path / "ckpt").mkdir()
    last = tmp_path / "ckpt" / "last.pt"
    last.write_bytes(b"x")
    decisions = retention(
        _manifest("train-supervised", "carrasco2021", "finished"), ["ckpt/last.pt"]
    )
    assert decisions["ckpt/last.pt"].action == "delete"
    assert last.read_bytes() == b"x"
