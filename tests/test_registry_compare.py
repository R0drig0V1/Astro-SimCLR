"""Tests of ``astro_simclr.registry.compare`` and ``registry.spec_view``: specs to runs."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
from _registry_support import (
    GTX1050,
    MACHINE,
    REPLICATION,
    SPLITS,
    FakeArm,
    FakeSplits,
    case_config_sha256,
    hide_spec_module,
    install_fake_spec_module,
    legacy_run,
    manifest_run,
    replication_spec,
    write_spec,
)

from astro_simclr.registry.compare import (
    ComparisonError,
    arm_summary,
    default_metrics,
    format_report,
    resolution_to_plain,
    resolve,
)
from astro_simclr.registry.readers import RunRecord, read_run_dir
from astro_simclr.registry.spec_view import (
    ComparisonPlan,
    PlannedCell,
    SpecUnavailableError,
    load_spec_file,
    plan_from_spec,
)

CD21 = case_config_sha256("carrasco2021")
NOFEAT = case_config_sha256("carrasco2021-no-features")
SPEC_ID = "f" * 64


def _plan(
    seeds: tuple[int, ...] = (0, 1),
    splits: tuple[object, ...] = (None,),
    arms: dict[str, str] | None = None,
    **changes: object,
) -> ComparisonPlan:
    arms = {"base": CD21} if arms is None else arms
    cells = tuple(PlannedCell(a, s, p) for a in arms for s in seeds for p in splits)  # type: ignore[arg-type]
    fields: dict[str, object] = {
        "spec_id": SPEC_ID,
        "name": "test",
        "command": "train-supervised",
        "arms": arms,
        "cells": cells,
        "machine_class": GTX1050,
        **changes,
    }
    return ComparisonPlan(**fields)  # type: ignore[arg-type]


def _records(*dirs: Path) -> list[RunRecord]:
    return [read_run_dir(d) for d in dirs]


def _problems(err: pytest.ExceptionInfo[ComparisonError]) -> str:
    return "\n".join(err.value.problems)


def test_archived_replication_resolves() -> None:
    plan = plan_from_spec(replication_spec())
    records = _records(*sorted(REPLICATION.glob("*/seed-*")))
    resolution = resolve(plan, records)
    assert len(resolution.chosen) == 10
    for cell, record in resolution.chosen.items():
        assert record.run_dir == (REPLICATION / cell.arm / f"seed-{cell.seed}").resolve()
    assert resolution.outside == ()
    assert any("source pickle" in note for note in resolution.notes)
    metrics = default_metrics(resolution.chosen.values())
    assert metrics == [
        "val.balanced_accuracy",
        "val.macro_f1",
        "test.balanced_accuracy",
        "test.macro_f1",
    ]
    report = "\n".join(format_report(resolution, metrics))
    assert "2 arms x 5 seeds x splits [none] = 10 cells, all resolved" in report
    plain = resolution_to_plain(resolution, ["val.balanced_accuracy"])
    summary = plain["arms"]["with-features"]["val.balanced_accuracy"]  # type: ignore[index]
    values = [
        r.metrics["val.balanced_accuracy"]
        for c, r in resolution.chosen.items()
        if c.arm == "with-features"
    ]
    assert summary["n"] == 5
    assert summary["mean"] == pytest.approx(sum(values) / 5)


def test_archive_copies_are_one_attempt(tmp_path: Path) -> None:
    """results/phase2 is a byte-exact copy of ~/runs: two directories, one attempt each."""
    import shutil

    copy = tmp_path / "runs"
    shutil.copytree(REPLICATION, copy)
    records = _records(*REPLICATION.glob("*/seed-*"), *copy.glob("*/seed-*"))
    resolution = resolve(plan_from_spec(replication_spec()), records)
    assert len(resolution.chosen) == 10
    assert sum(len(c) for c in resolution.copies.values()) == 10
    assert "10 copies" in "\n".join(format_report(resolution, ["val.accuracy"]))


def test_missing_cells_fail(tmp_path: Path) -> None:
    records = _records(manifest_run(tmp_path / "s0", seed=0))
    with pytest.raises(ComparisonError) as err:
        resolve(_plan(seeds=(0, 1, 2)), records)
    assert _problems(err) == "base/seed=1: no finished run\nbase/seed=2: no finished run"
    assert "spec test (ffffffffffffffff) does not resolve" in str(err.value)


def test_unfinished_and_other_runs_do_not_count(tmp_path: Path) -> None:
    records = _records(
        manifest_run(tmp_path / "s0", seed=0),
        manifest_run(tmp_path / "s1-running", seed=1, status="running"),
        manifest_run(tmp_path / "s1-failed", seed=1, attempt=2, status="failed"),
        manifest_run(tmp_path / "other-config", seed=1, case="carrasco2021-no-features"),
    )
    with pytest.raises(ComparisonError, match="base/seed=1: no finished run"):
        resolve(_plan(), records)


def test_several_attempts_need_a_choice(tmp_path: Path) -> None:
    first = manifest_run(tmp_path / "a1", attempt=1, ended_at="2026-11-08T11:00:00+00:00")
    second = manifest_run(
        tmp_path / "a2", attempt=2, accuracy=0.5, ended_at="2026-11-08T12:00:00+00:00"
    )
    records = _records(first, second, manifest_run(tmp_path / "s1", seed=1))
    plan = _plan()
    with pytest.raises(ComparisonError) as err:
        resolve(plan, records)
    assert "base/seed=0: 2 attempts" in _problems(err)
    assert "--use KEY or --attempts latest" in _problems(err)
    cell = PlannedCell("base", 0, None)
    key2 = read_run_dir(second).key
    assert resolve(plan, records, attempts="latest").chosen[cell].key == key2
    by_key = resolve(plan, records, use=[read_run_dir(first).key])
    assert by_key.chosen[cell].run_dir == first.resolve()
    by_dir = resolve(plan, records, use=[str(first)])
    assert by_dir.chosen[cell].run_dir == first.resolve()
    unnormalised = str(first.parent / ".." / first.parent.name / first.name)
    assert resolve(plan, records, use=[unnormalised]).chosen[cell].run_dir == first.resolve()


def test_use_errors(tmp_path: Path) -> None:
    a1 = manifest_run(tmp_path / "a1", attempt=1)
    a2 = manifest_run(tmp_path / "a2", attempt=2)
    records = _records(a1, a2, manifest_run(tmp_path / "s1", seed=1))
    key1, key2 = read_run_dir(a1).key, read_run_dir(a2).key
    common = key1[:6]
    assert key2.startswith(common)
    with pytest.raises(ComparisonError) as err:
        resolve(_plan(), records, use=["nothing", common, "abc"])
    text = _problems(err)
    assert "--use nothing: matches no candidate" in text
    assert f"--use {common}: ambiguous in base/seed=0" in text
    assert "--use abc: matches no candidate" in text  # too short for a prefix
    with pytest.raises(ComparisonError, match="two attempts chosen for base/seed=0"):
        resolve(_plan(), records, use=[key1, key2])


def test_use_takes_a_key_prefix(tmp_path: Path) -> None:
    l1 = legacy_run(tmp_path / "l1", accuracy=0.8)
    records = _records(
        l1, legacy_run(tmp_path / "l2", accuracy=0.7), legacy_run(tmp_path / "s1", seed=1)
    )
    key = read_run_dir(l1).key
    resolution = resolve(_plan(), records, use=[key[:12]])
    assert resolution.chosen[PlannedCell("base", 0, None)].key == key


def test_latest_needs_end_times(tmp_path: Path) -> None:
    records = _records(
        legacy_run(tmp_path / "l1", accuracy=0.8),
        legacy_run(tmp_path / "l2", accuracy=0.7),
        legacy_run(tmp_path / "s1", seed=1),
    )
    with pytest.raises(ComparisonError, match="without an end time"):
        resolve(_plan(), records, attempts="latest")


def test_copies_that_differ_fail(tmp_path: Path) -> None:
    a = manifest_run(tmp_path / "a")
    b = manifest_run(tmp_path / "b", accuracy=0.1)  # the same attempt, other metrics
    records = _records(a, b, manifest_run(tmp_path / "s1", seed=1))
    assert records[0].key == records[1].key
    with pytest.raises(ComparisonError, match=r"copies of .* differ in metrics\.json"):
        resolve(_plan(), records)


@pytest.mark.parametrize(
    ("kind", "change"),
    [
        ("machines", {"machine": {**MACHINE, "fp32_precision": "tf32"}}),
        ("engines", {"engine": "plain"}),
        ("numerics", {"numerics_epoch": 2}),
    ],
)
def test_mixes_fail_unless_allowed(tmp_path: Path, kind: str, change: dict[str, Any]) -> None:
    records = _records(
        manifest_run(tmp_path / "s0", seed=0), manifest_run(tmp_path / "s1", seed=1, **change)
    )
    with pytest.raises(ComparisonError, match=f"the runs mix {kind}.*--allow-mixed {kind}"):
        resolve(_plan(), records)
    resolution = resolve(_plan(), records, allow_mixed=[kind])  # type: ignore[list-item]
    assert any(note.startswith(f"mixed {kind} allowed") for note in resolution.notes)


def test_legacy_and_manifest_runs_mix_numerics(tmp_path: Path) -> None:
    records = _records(manifest_run(tmp_path / "s0", seed=0), legacy_run(tmp_path / "s1", seed=1))
    with pytest.raises(ComparisonError, match=r"mix numerics: 1 \(1 runs\); unknown \(legacy\)"):
        resolve(_plan(), records)
    assert resolve(_plan(), records, allow_mixed=["numerics"])


def test_different_data_never_mixes(tmp_path: Path) -> None:
    records = _records(
        manifest_run(tmp_path / "s0", seed=0),
        manifest_run(tmp_path / "s1", seed=1, data_sha256="c" * 64),
    )
    with pytest.raises(ComparisonError, match="different data: cccccccccccc, dddddddddddd"):
        resolve(_plan(), records, allow_mixed=["machines", "engines", "numerics"])


def test_outside_runs_are_reported(tmp_path: Path) -> None:
    records = _records(
        manifest_run(tmp_path / "s0", seed=0),
        manifest_run(tmp_path / "s1", seed=1),
        manifest_run(tmp_path / "s9", seed=9),
        manifest_run(tmp_path / "cpu", seed=0, attempt=2, machine_class="cpu-arm64"),
    )
    resolution = resolve(_plan(), records)
    assert len(resolution.outside) == 2
    assert any("2 finished attempts of these arms are outside" in n for n in resolution.notes)
    assert "note: 2 finished attempts" in "\n".join(format_report(resolution, ["val.accuracy"]))


def test_split_cells_and_splits_file(tmp_path: Path) -> None:
    runs = [manifest_run(tmp_path / f"f{fold}", split=fold) for fold in (0, 1)]
    records = _records(*runs)
    plan = _plan(seeds=(0,), splits=(0, 1), splits_sha256=SPLITS)
    resolution = resolve(plan, records)
    assert [c.split for c in resolution.chosen] == [0, 1]
    other = _plan(seeds=(0,), splits=(0, 1), splits_sha256="9" * 64)
    with pytest.raises(ComparisonError, match="split=0: no finished run"):
        resolve(other, records)
    assert "x splits [0, 1] = 2 cells" in "\n".join(format_report(resolution, ["val.accuracy"]))


def test_missing_metric(tmp_path: Path) -> None:
    resolution = resolve(
        _plan(), _records(*(manifest_run(tmp_path / f"s{s}", seed=s) for s in (0, 1)))
    )
    with pytest.raises(ComparisonError, match=r"has no metric val\.recall"):
        format_report(resolution, ["val.recall"])
    with pytest.raises(ComparisonError):
        resolution_to_plain(resolution, ["val.recall"])


def test_arm_summary() -> None:
    assert arm_summary([0.5]) == (0.5, None, 1)
    mean, std, n = arm_summary([1.0, 2.0, 3.0])
    assert (mean, n) == (2.0, 3)
    assert std == pytest.approx(1.0)
    assert default_metrics([]) == []


def test_single_seed_report(tmp_path: Path) -> None:
    resolution = resolve(_plan(seeds=(0,)), _records(manifest_run(tmp_path / "s0")))
    lines = format_report(resolution, ["val.accuracy"])
    assert lines[-1].split() == ["base", "1", "0.9000"]


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"cells": ()}, "no cells"),
        ({"arms": {"base": CD21, "same": CD21}}, "two arms share a config hash"),
        ({"cells": (PlannedCell("x", 0, None),)}, "unknown arms"),
        ({"arms": {"base": CD21, "idle": NOFEAT}}, r"arms without cells \['idle'\]"),
        ({"cells": (PlannedCell("base", 0, None),) * 2}, "appears twice"),
        ({"splits_sha256": "nope"}, "splits_sha256"),
        ({"spec_id": "nope"}, "spec_id"),
        ({"arms": {"base": "nope"}}, "config_sha256"),
    ],
)
def test_plan_validation(changes: dict[str, object], match: str) -> None:
    base = _plan()
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(base, **changes)  # type: ignore[arg-type]


def test_plan_orders_cells() -> None:
    plan = _plan(seeds=(1, 0), splits=("screen", 1, None))
    assert plan.seeds == (0, 1)
    assert plan.splits == (None, 1, "screen")
    assert plan.cells[0].label() == "base/seed=0"
    assert plan.cells[1].label() == "base/seed=0/split=1"


def test_plan_from_spec_hashes_the_splits_file(tmp_path: Path) -> None:
    splits = tmp_path / "splits.json"
    splits.write_text("{}", encoding="utf-8")
    spec = dataclasses.replace(
        replication_spec(seeds=(0,)),
        arms=(FakeArm("only", CD21),),
        split_parts=(0, 1),
        splits=FakeSplits(splits),
    )
    plan = plan_from_spec(spec)
    assert plan.spec_id == spec.spec_id
    assert plan.splits_sha256 == "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
    assert plan.arms == {"only": CD21}
    assert len(plan.cells) == 2


def test_load_spec_file_keeps_other_import_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(name: str) -> object:
        raise ModuleNotFoundError("No module named 'yaml'", name="yaml")

    monkeypatch.setattr("importlib.import_module", broken)
    with pytest.raises(ModuleNotFoundError, match="yaml"):
        load_spec_file(Path("spec.yaml"))


def test_load_spec_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_spec(tmp_path / "spec.yaml", replication_spec())
    hide_spec_module(monkeypatch)
    with pytest.raises(SpecUnavailableError, match="need task A3"):
        load_spec_file(path)
    install_fake_spec_module(monkeypatch)
    assert load_spec_file(path).name == "replication"
