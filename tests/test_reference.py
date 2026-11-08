"""Tests of ``compare-reference``: the pass rule, the gates of R1-R4 and the exit codes."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from astro_simclr.cli import main
from astro_simclr.config import ConfigError
from astro_simclr.reference import REFERENCES, compare_reference, exit_code, pass_rule


def _run(
    root: Path,
    name: str,
    *,
    command: str = "train-supervised",
    seed: int = 0,
    val: float | None = 0.95,
    test: float | None = 0.94,
    model: str = "stamp_classifier",
    dataset: str = "td_ztf_stamp_17_06_20",
    validation_counts: list[int] | None = None,
    **extra: Any,
) -> Path:
    out = root / name
    out.mkdir()
    metrics = {
        "command": command,
        "seed": seed,
        "val_acc_best": val,
        "test_acc_at_best": test,
        "config": {"supervised_model": model, "dataset": dataset},
        "data": {"class_counts": {"Validation": validation_counts or [100] * 5}},
        **extra,
    }
    (out / "metrics.json").write_text(json.dumps(metrics))
    return out


def test_minimal_set_rule() -> None:
    # One run: |x - m| <= max(2 s sqrt(1 + 1/n), 0.01); with R1's s the floor 0.01 wins.
    passed, diff, tolerance = pass_rule([0.9406], 0.9496, 0.002191, 5)
    assert tolerance == 0.01
    assert diff == pytest.approx(0.009)
    assert passed
    assert not pass_rule([0.9390], 0.9496, 0.002191, 5)[0]
    # R2's larger s: 2 * 0.009866 * sqrt(1 + 1/3) = 0.02278.
    _, _, tolerance = pass_rule([0.9], 0.902667, 0.009866, 3)
    assert tolerance == pytest.approx(2 * 0.009866 * math.sqrt(4 / 3))


def test_full_set_rule_uses_ddof_1() -> None:
    values = [0.93, 0.95, 0.94, 0.96, 0.92]
    std_new = 0.015811388300841896  # sample std (ddof=1) of values
    passed, diff, tolerance = pass_rule(values, 0.9496, 0.002191, 5)
    assert diff == pytest.approx(abs(0.94 - 0.9496))
    assert tolerance == pytest.approx(2 * math.sqrt(0.002191**2 / 5 + std_new**2 / 5))
    assert passed
    with pytest.raises(ValueError):
        pass_rule([], 0.9, 0.01, 5)


def test_r1_pass_and_fail(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = [_run(tmp_path, f"s{s}", seed=s, val=0.948 + s / 1000) for s in range(3)]
    comparison = compare_reference("R1", good)
    assert comparison.result == "PASS"
    assert "full-set rule" in "\n".join(comparison.lines)
    assert "Carrasco-Davis" in "\n".join(comparison.lines)
    assert main(["compare-reference", *map(str, good), "--ref", "R1"]) == 0
    assert capsys.readouterr().out.rstrip().endswith("R1: PASS")

    bad = _run(tmp_path, "bad", val=0.90)
    assert compare_reference("R1", [bad]).result == "FAIL"
    assert main(["compare-reference", str(bad), "--ref", "R1"]) == 1
    nan = _run(tmp_path, "nan", val=None)
    assert compare_reference("R1", [nan]).result == "FAIL"


def test_r2_needs_the_resnet(tmp_path: Path) -> None:
    run = _run(tmp_path, "r2", val=0.90, model="resnet18")
    assert compare_reference("R2", [run]).result == "PASS"
    with pytest.raises(ConfigError, match="supervised_model 'stamp_classifier'"):
        compare_reference("R2", [_run(tmp_path, "stamp", val=0.90)])


def test_r3_baseline_gate(tmp_path: Path) -> None:
    pretrain = _run(
        tmp_path,
        "pre",
        command="pretrain",
        val=None,
        test=None,
        monitor="loss_train_enc",
        monitor_best=4.2,
    )
    fractions = [
        _run(tmp_path, f"le{i}", command="linear-eval", dataset=d, val=0.5, test=0.45)
        for i, d in enumerate(("sup_1", "sup_10", "full"))
    ]
    assert compare_reference("R3", [pretrain, *fractions]).result == "PASS"
    # Imbalanced validation: the majority baseline is 0.6, so 0.5 is not above it.
    imbalanced = _run(
        tmp_path,
        "imb",
        command="linear-eval",
        dataset="x",
        val=0.5,
        validation_counts=[60, 10, 10, 10, 10],
    )
    assert compare_reference("R3", [imbalanced]).result == "FAIL"
    diverged = _run(
        tmp_path, "nanpre", command="pretrain", val=None, test=None, monitor_best=None, dataset="y"
    )
    assert compare_reference("R3", [diverged, fractions[0]]).result == "FAIL"
    assert compare_reference("R3", [pretrain]).result == "FAIL"  # no linear-eval run


def test_r4_reports(tmp_path: Path) -> None:
    runs = [_run(tmp_path, f"s{s}", seed=s, val=0.6) for s in range(2)]
    comparison = compare_reference("R4", runs)
    assert comparison.result == "REPORT"
    assert exit_code(comparison.result) == 0
    assert exit_code("FAIL") == 1


def test_input_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(ConfigError, match="unknown reference"):
        compare_reference("R9", [])
    with pytest.raises(ConfigError, match="at least one run"):
        compare_reference("R1", [])
    with pytest.raises(ConfigError, match="cannot read"):
        compare_reference("R1", [tmp_path / "missing"])
    pretrain = _run(tmp_path, "pre", command="pretrain")
    with pytest.raises(ConfigError, match="'pretrain' run cannot be compared with R1"):
        compare_reference("R1", [pretrain])
    twins = [_run(tmp_path, "a"), _run(tmp_path, "b")]
    with pytest.raises(ConfigError, match="same run"):
        compare_reference("R1", twins)
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "metrics.json").write_text("{")
    assert main(["compare-reference", str(broken), "--ref", "R1"]) == 2
    assert "invalid JSON" in capsys.readouterr().err
    (broken / "metrics.json").write_text("[]")
    with pytest.raises(ConfigError, match="not a metrics object"):
        compare_reference("R1", [broken / "metrics.json"])


def test_every_reference_is_defined() -> None:
    assert sorted(REFERENCES) == ["R1", "R2", "R3", "R4"]
    assert REFERENCES["R1"].mean == 0.9496
    assert REFERENCES["R2"].n == 3
