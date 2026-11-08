"""Tests of ``report-phase2`` (:mod:`astro_simclr.protocol.phase2_report`).

The screen and CV runs are hand-made split runs (``_protocol_support.fake_split_run``) on a
small stored splits file, with a chosen number of errors per fold and seed, as in
``test_protocol_decision.py``. The replication section reads the archived replication
runs (``results/phase2/replication``), whose CD21 gate is known to pass (faithful). The
report must repeat, verbatim, what ``compare-variants --splits`` and ``compare-reference``
print for the same runs.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from _protocol_support import fake_split_run, make_layout, predictions
from _registry_support import REPLICATION, manifest_run

from astro_simclr.cli import main
from astro_simclr.config import ConfigError
from astro_simclr.protocol.compare import compare_decision, load_split_run
from astro_simclr.protocol.phase2_report import (
    ReportInputs,
    build_report,
    find_arms,
    pooled_class_scores,
    read_run_facts,
)
from astro_simclr.protocol.predictions import Predictions
from astro_simclr.protocol.split_runs import file_sha256
from astro_simclr.protocol.splits import ProtocolSplits, make_splits, write_splits
from astro_simclr.reference import compare_reference
from astro_simclr.registry.index import reindex
from astro_simclr.registry.store import open_registry

TRAIN_PER_CLASS, VAL_PER_CLASS = 30, 10
N_BOOT = 200
Stored = tuple[ProtocolSplits, Path]


@pytest.fixture(scope="module")
def stored(tmp_path_factory: pytest.TempPathFactory) -> Stored:
    root = tmp_path_factory.mktemp("report")
    layout = make_layout(root / "layout", train=TRAIN_PER_CLASS, validation=VAL_PER_CLASS, test=2)
    splits = make_splits(layout, selection_per_class=3, screen_per_class=5)
    path = root / "splits.json"
    write_splits(splits, path)
    return splits, path


def _labels(splits: ProtocolSplits, pool: np.ndarray) -> np.ndarray:
    train, val = splits.locate(pool)
    return np.concatenate([train // TRAIN_PER_CLASS, val // VAL_PER_CLASS]).astype(np.int64)


def _on(index: np.ndarray, labels: np.ndarray, wrong: int, offset: int = 0) -> Predictions:
    """Predictions with ``wrong`` errors (the next class), from position ``offset`` on."""
    predicted = labels.copy()
    at = (np.arange(wrong) + offset) % index.size
    predicted[at] = (predicted[at] + 1) % 5
    return predictions(labels.tolist(), predicted.tolist(), index=index.tolist())


def _timing(seconds: float) -> dict[str, float]:
    """``metrics.json["timing"]`` of a 100-step fit at ``seconds`` per step."""
    return {"fit_seconds": 100 * seconds, "steps": 100, "seconds_per_step": seconds}


def _cv_arm(
    root: Path, arm: str, stored: Stored, errors: Callable[[int, int], int], seconds: float = 0.05
) -> list[Path]:
    splits, path = stored
    runs = []
    for fold in splits.folds:
        labels = _labels(splits, fold.evaluation)
        for seed in (0, 1):
            pred = _on(fold.evaluation, labels, errors(fold.fold, seed), offset=3 * seed)
            runs.append(
                fake_split_run(
                    root,
                    f"{arm}/fold{fold.fold}/seed{seed}",
                    pred,
                    sha256=file_sha256(path),
                    seed=seed,
                    fold=fold.fold,
                    timing=_timing(seconds + 0.001 * seed),
                )
            )
    return runs


def _screen_arm(root: Path, arm: str, stored: Stored, wrong: list[int]) -> list[Path]:
    splits, path = stored
    decision = splits.screen.decision
    labels = (decision // VAL_PER_CLASS).astype(np.int64)
    return [
        fake_split_run(
            root,
            f"{arm}/seed{seed}",
            _on(decision, labels, n),
            sha256=file_sha256(path),
            seed=seed,
            timing=_timing(0.02),
        )
        for seed, n in enumerate(wrong)
    ]


@pytest.fixture
def cv_root(tmp_path: Path, stored: Stored) -> Path:
    root = tmp_path / "cv"
    _cv_arm(root, "rot4", stored, lambda fold, seed: 6)
    _cv_arm(root, "p4", stored, lambda fold, seed: seed, seconds=0.02)
    _cv_arm(root, "ctx", stored, lambda fold, seed: 12)
    return root


@pytest.fixture
def screen_root(tmp_path: Path, stored: Stored) -> Path:
    root = tmp_path / "screen"
    _screen_arm(root, "rot4", stored, [2, 3, 2])
    _screen_arm(root, "p4", stored, [1, 2, 2])
    _screen_arm(root, "wide", stored, [12, 13, 12])
    return root


def _inputs(stored: Stored, **values: object) -> ReportInputs:
    return ReportInputs(n_boot=N_BOOT, **values)  # type: ignore[arg-type]


def _between(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def test_full_report_repeats_the_protocol_outputs(
    stored: Stored, cv_root: Path, screen_root: Path
) -> None:
    _, splits = stored
    inputs = _inputs(
        stored,
        replication=REPLICATION,
        screen=screen_root,
        screen_splits=splits,
        cv=cv_root,
        cv_splits=splits,
        non_inferiority=frozenset({"p4"}),
    )
    text = build_report(inputs)
    assert build_report(inputs) == text  # deterministic
    assert text.startswith("<!-- Generated by `astro-simclr report-phase2`")
    assert f"    --cv {cv_root} \\" in text and "    --seed 0\n```" in text

    # Replication: per-seed rows, the mean row and the gate of compare-reference.
    replication = _between(text, "## Replication", "## Screen")
    assert "### no-features" in replication and "### with-features" in replication
    assert "| 0 | 18,930 | 0.946 | 0.942 | 0.850 | 0.0502 |" in replication
    assert "| **mean ± std (ddof=1)** |  | 0.9428 ± 0.0027 | 0.9428 ± 0.0041 |" in replication
    runs = [run for arm in find_arms(REPLICATION) for run in arm.runs]
    gate = compare_reference("CD21", runs, oracle_report=REPLICATION / "m1_oracle.json")
    assert "\n".join(gate.lines) in replication
    assert "**Gate: CD21: PASS (faithful)**" in replication

    # Screen: wide is dropped, p4 kept; the compare-variants lines are verbatim.
    screen = _between(text, "## Screen", "## Cross-validation")
    rot4 = sorted((screen_root / "rot4").glob("seed*"))
    wide = compare_decision(rot4, sorted((screen_root / "wide").glob("seed*")), splits)
    assert "\n".join(wide.lines) in screen
    assert wide.verdict == "DROP" and wide.mcnemar is not None
    assert "| wide | 3 | 0.9067 | 0.5067 | -0.4000 |" in screen
    assert "| DROP |" in screen and "| KEEP |" in screen

    # CV: per run, per fold, pooled, both verdicts, the pre-registered rule.
    cv = _between(text, "## Cross-validation", "## Throughput")
    base = sorted(p.parent for p in (cv_root / "rot4").rglob("metrics.json"))
    p4 = sorted(p.parent for p in (cv_root / "p4").rglob("metrics.json"))
    ctx = sorted(p.parent for p in (cv_root / "ctx").rglob("metrics.json"))
    p4_decision = compare_decision(base, p4, splits, rule="non-inferiority", n_boot=N_BOOT)
    ctx_decision = compare_decision(base, ctx, splits, n_boot=N_BOOT)
    assert "\n".join(p4_decision.lines) in cv and "\n".join(ctx_decision.lines) in cv
    assert "#### p4 vs rot4 (non-inferiority)" in cv and "#### ctx vs rot4 (superiority)" in cv
    assert p4_decision.cv is not None and ctx_decision.cv is not None
    pooled = (
        f"| **pooled** | {p4_decision.cv.baseline_pooled:.4f} | "
        f"{ctx_decision.cv.variant_pooled:.4f} | {p4_decision.cv.variant_pooled:.4f} |"
    )
    assert "| Fold | rot4 | ctx | p4 |" in cv and pooled in cv
    assert "| p4 | " in cv and "| adopt | adopt | non-inferiority: **ADOPT** |" in cv
    assert "| worse | worse | superiority: **NOT ADOPTED** |" in cv
    assert cv.count("| 0 | 0 | ") == 1 and "| 4 | 1 | " in cv  # 5 folds x 2 seeds

    # Per-class recall and the confusion matrices of every arm, baseline included.
    assert "| Arm | AGN | SN | VS | Asteroid | Bogus | Balanced accuracy |" in cv
    for arm in ("rot4", "p4", "ctx"):
        assert f"#### {arm} (" in cv
    assert "| True \\ predicted | AGN | SN | VS | Asteroid | Bogus |" in cv

    # Throughput: every arm, no manifest anywhere.
    throughput = _between(text, "## Throughput", "## Provenance")
    assert "| CV | p4 | 10 | cpu | 0.0205 ± 0.0005 | not recorded for 10 of 10 runs |" in throughput
    assert "| replication | with-features | 5 | NVIDIA GeForce GTX 1050 | 0.0502" in throughput
    assert "| screen | wide | 3 | cpu | 0.0200 ± 0.0000 |" in throughput
    assert "No registry was given." in text and "| CV | rot4 | 0 of 10 | — | — |" in text


def test_pooled_class_scores_match_the_pooled_balanced_accuracy(
    stored: Stored, cv_root: Path
) -> None:
    _, splits = stored
    base = sorted(p.parent for p in (cv_root / "rot4").rglob("metrics.json"))
    variant = sorted(p.parent for p in (cv_root / "p4").rglob("metrics.json"))
    decision = compare_decision(base, variant, splits, n_boot=N_BOOT)
    assert decision.cv is not None
    for runs, pooled in (
        (base, decision.cv.baseline_pooled),
        (variant, decision.cv.variant_pooled),
    ):
        folds: dict[int, list[Predictions]] = {}
        for run in map(load_split_run, runs):
            assert run.fold is not None
            folds.setdefault(run.fold, []).append(run.predictions)
        recall, counts = pooled_class_scores(folds)
        assert recall.mean() == pytest.approx(pooled)
        per_class = TRAIN_PER_CLASS + VAL_PER_CLASS
        assert counts.sum(axis=1) == pytest.approx(np.full(5, per_class))
        assert np.diag(counts) / counts.sum(axis=1) == pytest.approx(recall)
    with pytest.raises(ValueError, match="no folds"):
        pooled_class_scores({})
    only_agn = predictions([0, 0], [0, 1])
    with pytest.raises(ValueError, match="no stamp of class"):
        pooled_class_scores({0: [only_agn]})


def _manifest(run_dir: Path, tmp: Path, jobs: int) -> None:
    """Give ``run_dir`` a valid ``manifest.json`` recording ``jobs`` concurrent jobs."""
    source = manifest_run(tmp / f"m-{run_dir.parent.name}-{run_dir.name}")
    data = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    data["concurrent_jobs"] = jobs
    (run_dir / "manifest.json").write_text(json.dumps(data), encoding="utf-8")


def test_throughput_flags_runs_measured_under_concurrency(
    tmp_path: Path, stored: Stored, screen_root: Path
) -> None:
    _, splits = stored
    runs = sorted((screen_root / "p4").glob("seed*"))
    _manifest(runs[0], tmp_path, 2)
    for run in sorted((screen_root / "rot4").glob("seed*")):
        _manifest(run, tmp_path, 1)
    text = build_report(_inputs(stored, screen=screen_root, screen_splits=splits))
    assert (
        "| screen | p4 | 3 | cpu | 0.0200 ± 0.0000 | **measured under concurrency** (up to 2 "
        "jobs; 1 of 3 runs); not recorded for 2 of 3 runs |" in text
    )
    assert "| screen | rot4 | 3 | cpu | 0.0200 ± 0.0000 | alone |" in text
    assert "| screen | rot4 | 3 of 3 | 0000000 | 1 |" in text
    facts = read_run_facts(runs[0])
    assert facts.has_manifest and facts.concurrent_jobs == 2 and facts.git_sha == "0" * 40

    (runs[1] / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"unreadable manifest\.json"):
        read_run_facts(runs[1])


def test_provenance_flags_mixed_configs(tmp_path: Path, stored: Stored, screen_root: Path) -> None:
    _, splits = stored
    runs = sorted((screen_root / "wide").glob("seed*"))
    _manifest(runs[0], tmp_path, 1)
    other = manifest_run(tmp_path / "other", case="carrasco2021-no-features")
    shutil.copy(other / "manifest.json", runs[1] / "manifest.json")
    text = build_report(_inputs(stored, screen=screen_root, screen_splits=splits))
    assert "| screen | wide | 2 of 3 | 0000000 | 2 (**mixed configs**) |" in text


def test_registry_counts_indexed_and_stale_runs(tmp_path: Path) -> None:
    replication = tmp_path / "replication"
    shutil.copytree(REPLICATION, replication)
    registry = tmp_path / "home" / "registry.sqlite"
    conn = open_registry(registry)
    reindex(conn, [replication / "with-features"], specs=tmp_path / "specs")
    conn.close()
    metrics = replication / "with-features" / "seed-0" / "metrics.json"
    metrics.write_text(metrics.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    text = build_report(ReportInputs(replication=replication, registry=registry))
    assert "| replication | no-features | 0 of 5 | — | — | 0 of 5 indexed |" in text
    assert (
        "| replication | with-features | 0 of 5 | — | — | 5 of 5 indexed, **1 stale** "
        "(metrics.json changed since reindex) |" in text
    )
    with pytest.raises(ConfigError, match="no registry file"):
        build_report(ReportInputs(replication=replication, registry=tmp_path / "none.sqlite"))
    (tmp_path / "bad.sqlite").write_text("not sqlite", encoding="utf-8")
    with pytest.raises(ConfigError, match="registry"):
        build_report(ReportInputs(replication=replication, registry=tmp_path / "bad.sqlite"))


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({}, "at least one of"),
        ({"screen": Path("s")}, "--screen needs --screen-splits"),
        ({"cv_splits": Path("f"), "replication": Path("r")}, "--cv-splits is only used"),
        ({"oracle_report": Path("o"), "cv": Path("c"), "cv_splits": Path("f")}, "oracle"),
        ({"replication": Path("r"), "non_inferiority": frozenset({"p4"})}, "needs --cv"),
        (
            {"cv": Path("c"), "cv_splits": Path("f"), "non_inferiority": frozenset({"rot4"})},
            "no rule of its own",
        ),
        ({"replication": Path("r"), "margin": 0.5}, "fraction of balanced accuracy"),
    ],
)
def test_inconsistent_inputs_are_refused(values: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        build_report(ReportInputs(**values))  # type: ignore[arg-type]


def test_arm_layout_errors(tmp_path: Path, stored: Stored, cv_root: Path) -> None:
    _, splits = stored
    with pytest.raises(ConfigError, match="not a directory"):
        find_arms(tmp_path / "missing")
    (tmp_path / "empty" / "notes").mkdir(parents=True)
    (tmp_path / "empty" / "README.md").write_text("x", encoding="utf-8")
    with pytest.raises(ConfigError, match="no arm directories"):
        find_arms(tmp_path / "empty")
    with pytest.raises(ConfigError, match="is a run directory"):
        find_arms(cv_root / "rot4" / "fold0")
    with pytest.raises(ConfigError, match="no baseline arm 'base'"):
        build_report(_inputs(stored, cv=cv_root, cv_splits=splits, baseline_arm="base"))
    with pytest.raises(ConfigError, match="names no arm here"):
        build_report(
            _inputs(stored, cv=cv_root, cv_splits=splits, non_inferiority=frozenset({"x"}))
        )
    lonely = tmp_path / "lonely"
    shutil.copytree(cv_root / "rot4", lonely / "rot4")
    with pytest.raises(ConfigError, match="nothing to compare"):
        build_report(_inputs(stored, cv=lonely, cv_splits=splits))


def test_sections_refuse_runs_of_the_other_mode(
    stored: Stored, cv_root: Path, screen_root: Path
) -> None:
    _, splits = stored
    shutil.rmtree(screen_root / "rot4")
    shutil.copytree(cv_root / "rot4", screen_root / "rot4")
    with pytest.raises(ConfigError, match="mix CV folds and the screen"):
        build_report(_inputs(stored, screen=screen_root, screen_splits=splits))
    shutil.rmtree(cv_root / "ctx")
    shutil.rmtree(cv_root / "p4")
    shutil.copytree(screen_root / "p4", cv_root / "p4")
    shutil.copytree(screen_root / "wide", cv_root / "wide")
    shutil.rmtree(cv_root / "rot4")
    shutil.copytree(screen_root / "p4", cv_root / "rot4")
    with pytest.raises(ConfigError, match="not a CV fold run"):
        build_report(_inputs(stored, cv=cv_root, cv_splits=splits))


def test_cli_writes_the_report(
    tmp_path: Path,
    stored: Stored,
    cv_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, splits = stored
    out = tmp_path / "results-phase2.md"
    argv = ["report-phase2", "--cv", str(cv_root), "--cv-splits", str(splits)]
    argv += ["--n-boot", str(N_BOOT), "--non-inferiority", "p4", "--margin", "0.01"]
    assert main([*argv, "--out", str(out)]) == 0
    assert f"wrote {out}" in capsys.readouterr().err
    text = out.read_text(encoding="utf-8")
    assert "Non-inferiority (1.00 pt)" in text and "--margin 0.01" in text
    assert main(argv) == 0
    assert capsys.readouterr().out == text

    home = tmp_path / "home"
    monkeypatch.setenv("ASTRO_SIMCLR_HOME", str(home))
    open_registry(home / "registry.sqlite").close()
    assert main([*argv, "--registry"]) == 0
    assert "| CV | p4 | 0 of 10 | — | — | 0 of 10 indexed |" in capsys.readouterr().out
    monkeypatch.setenv("ASTRO_SIMCLR_HOME", str(Path(__file__).parent / "home"))
    assert main([*argv, "--registry"]) == 2
    assert "inside the git working tree" in capsys.readouterr().err
