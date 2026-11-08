"""Tests of ``compare-reference``: the pass rule, the gates of R1-R4 and CD21, the exit codes."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from astro_simclr.cli import main
from astro_simclr.config import ConfigError
from astro_simclr.hparams import LABEL_FRACTION_DATASETS
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
    dataset: str = "td_ztf_stamp_17_06_20_red-002",
    pretrained: str | None = None,
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
        "config": {"supervised_model": model, "dataset": dataset, "pretrained": pretrained},
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


def _r3_runs(root: Path, **linear: Any) -> list[Path]:
    pretrain = _run(
        root,
        "pre",
        command="pretrain",
        val=None,
        test=None,
        monitor="loss_train_enc",
        monitor_best=4.2,
        dataset="td_ztf_stamp_simclr_300",
    )
    fractions = [
        _run(
            root,
            f"le{fraction}",
            command="linear-eval",
            dataset=dataset,
            val=0.5,
            test=0.45,
            pretrained=str(pretrain / "checkpoints" / "best.ckpt")
            if fraction != 10
            else str(pretrain),
            **linear,
        )
        for fraction, dataset in LABEL_FRACTION_DATASETS.items()
    ]
    return [pretrain, *fractions]


def test_r3_passes_with_every_fraction(tmp_path: Path) -> None:
    comparison = compare_reference("R3", _r3_runs(tmp_path))
    assert comparison.result == "PASS", comparison.lines


def test_r3_needs_every_fraction_and_the_pretrain_run(tmp_path: Path) -> None:
    pretrain, one, ten, full = _r3_runs(tmp_path)
    comparison = compare_reference("R3", [pretrain, one, ten])
    assert comparison.result == "FAIL"
    assert "incomplete, no linear-eval run at 100 % (td_ztf_stamp_17_06_20_red-002)" in (
        "\n".join(comparison.lines)
    )
    comparison = compare_reference("R3", [one, ten, full])
    assert comparison.result == "FAIL"
    assert "no pretrain run given" in "\n".join(comparison.lines)


def test_r3_runs_must_come_from_the_given_pretrain(tmp_path: Path) -> None:
    pretrain, one, ten, _ = _r3_runs(tmp_path)
    other = _run(
        tmp_path,
        "other",
        command="linear-eval",
        dataset=LABEL_FRACTION_DATASETS[100],
        val=0.5,
        test=0.45,
        pretrained="/somewhere/else",
    )
    comparison = compare_reference("R3", [pretrain, one, ten, other])
    assert comparison.result == "FAIL"
    assert "not on the given pretrain run" in "\n".join(comparison.lines)


def test_r3_baseline_and_nan(tmp_path: Path) -> None:
    root = tmp_path / "imbalanced"
    root.mkdir()
    # Imbalanced validation: the majority baseline is 0.6, so 0.5 is not above it.
    runs = _r3_runs(root, validation_counts=[60, 10, 10, 10, 10])
    assert compare_reference("R3", runs).result == "FAIL"
    root = tmp_path / "nan"
    root.mkdir()
    pretrain, *fractions = _r3_runs(root)
    metrics = json.loads((pretrain / "metrics.json").read_text())
    metrics["monitor_best"] = None
    (pretrain / "metrics.json").write_text(json.dumps(metrics))
    assert compare_reference("R3", [pretrain, *fractions]).result == "FAIL"


def test_missing_config_keys_read_as_their_defaults(tmp_path: Path) -> None:
    # A run written before the 2.5 flags existed (task 2.4) has no paper_arch, context,
    # rot90_augment or C5 / C6 keys: they read as the defaults, the faithful model.
    run = _cd21(tmp_path, [0.941])[0]
    text = "\n".join(compare_reference("CD21", [run]).lines)
    assert "WARN" in text  # the stub config misses real keys (schedule, ...)
    for key in ("paper_arch", "context", "rot90_augment", "aug_noise", "synthetic_bogus"):
        assert f"{key}=" not in text
    metrics = json.loads((run / "metrics.json").read_text())
    metrics["config"]["aug_noise"] = 1.0
    (run / "metrics.json").write_text(json.dumps(metrics))
    assert "aug_noise=1.0 (expected 0.0)" in "\n".join(compare_reference("CD21", [run]).lines)


def test_config_differences_warn(tmp_path: Path) -> None:
    # An R4-like run (no features, 1 % labels) given as R1 is judged, but flagged.
    run = _run(tmp_path, "r4", val=0.95, dataset=LABEL_FRACTION_DATASETS[1])
    comparison = compare_reference("R1", [run])
    text = "\n".join(comparison.lines)
    assert "WARN:" in text
    assert (
        "dataset='td_ztf_stamp_17_06_20_sup_1' (expected 'td_ztf_stamp_17_06_20_red-002')" in text
    )


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
    assert sorted(REFERENCES) == ["CD21", "R1", "R2", "R3", "R4"]
    assert REFERENCES["R1"].mean == 0.9496
    assert REFERENCES["R2"].n == 3


# --- CD21: the two-tier gate of the paper replication (task 2.3) ---------------------------

MACHINE = {"device": "cuda", "gpu": "NVIDIA GeForce GTX 1050", "torch": "2.14.0", "cuda": "12.6"}


def _cd21(
    root: Path,
    tests: list[float],
    *,
    vals: list[float] | None = None,
    sn: list[float] | None = None,
    with_features: bool = True,
    prefix: str = "f",
    machine: dict[str, object] | None = None,
) -> list[Path]:
    runs = []
    for seed, test in enumerate(tests):
        out = _run(
            root,
            f"{prefix}{seed}",
            seed=seed,
            val=(vals or [0.95] * len(tests))[seed],
            test=test,
            model="paper_stamp_classifier",
            evaluation={"test": {"recall_per_class": {"SN": (sn or [0.87] * len(tests))[seed]}}},
            machine=machine or MACHINE,
            best_step=4780,
        )
        metrics = json.loads((out / "metrics.json").read_text())
        metrics["config"]["with_features"] = with_features
        (out / "metrics.json").write_text(json.dumps(metrics))
        runs.append(out)
    return runs


def _oracle(root: Path, passed: object) -> Path:
    path = root / "oracle.json"
    path.write_text(json.dumps({"passed": passed, "port_equivalence": {"max_abs_logit": 1e-6}}))
    return path


def test_cd21_faithful_needs_the_oracle(tmp_path: Path) -> None:
    runs = _cd21(tmp_path, [0.940, 0.944, 0.942, 0.946, 0.943])  # mean 0.943
    comparison = compare_reference("CD21", runs, oracle_report=_oracle(tmp_path, True))
    text = "\n".join(comparison.lines)
    assert (comparison.result, comparison.tier) == ("PASS", "faithful")
    assert text.endswith("CD21: PASS (faithful)")
    assert "ddof=0" in text and "about 0.0045 with ddof=1" in text
    assert "best_step: [4780, 4780, 4780, 4780, 4780]" in text
    without = compare_reference("CD21", runs)
    assert (without.result, without.tier) == ("PASS", "acceptable")
    assert "faithful needs the M1 oracle" in "\n".join(without.lines)
    failed = compare_reference("CD21", runs, oracle_report=_oracle(tmp_path, False))
    assert failed.tier == "acceptable"
    assert "M1 oracle: FAILED" in "\n".join(failed.lines)


def test_cd21_tiers_and_boundaries(tmp_path: Path) -> None:
    oracle = _oracle(tmp_path, True)
    # |0.946 - 0.941| = 0.005 exactly (float noise rounded): still faithful.
    edge = _cd21(tmp_path, [0.946] * 5, prefix="edge")
    assert compare_reference("CD21", edge, oracle_report=oracle).tier == "faithful"
    acceptable = _cd21(tmp_path, [0.949] * 5, prefix="acc")
    assert compare_reference("CD21", acceptable, oracle_report=oracle).tier == "acceptable"
    low = _cd21(tmp_path, [0.930] * 5, prefix="low")
    assert compare_reference("CD21", low, oracle_report=oracle).result == "FAIL"
    assert main(["compare-reference", *map(str, low), "--ref", "CD21"]) == 1


def test_cd21_needs_sn_recall_and_validation(tmp_path: Path) -> None:
    oracle = _oracle(tmp_path, True)
    sn = _cd21(tmp_path, [0.941] * 5, sn=[0.82] * 5, prefix="sn")
    comparison = compare_reference("CD21", sn, oracle_report=oracle)
    assert comparison.result == "FAIL"
    assert "SN test recall: 0.8200" in "\n".join(comparison.lines)
    assert compare_reference("CD21", _cd21(tmp_path, [0.941] * 5, sn=[0.91] * 5, prefix="sn2")).tier
    val = _cd21(tmp_path, [0.941] * 5, vals=[0.938] * 5, prefix="val")
    assert compare_reference("CD21", val, oracle_report=oracle).result == "FAIL"
    old = _run(tmp_path, "old", test=0.941, model="paper_stamp_classifier")  # before 2.3
    comparison = compare_reference("CD21", [old])
    assert comparison.result == "FAIL"
    assert "written since task 2.3" in "\n".join(comparison.lines)


def test_cd21_needs_the_paper_model(tmp_path: Path) -> None:
    legacy = _run(tmp_path, "legacy", test=0.941)  # stamp_classifier, the phase-1 port
    with pytest.raises(ConfigError, match="paper_stamp_classifier"):
        compare_reference("CD21", [legacy])
    lines = compare_reference("CD21", _cd21(tmp_path, [0.941] * 5)).lines
    # A missing schedule reads as the default, the legacy 'epochs'.
    assert any("schedule='epochs' (expected 'carrasco2021')" in line for line in lines)


def test_cd21_is_defined_on_five_seeds(tmp_path: Path) -> None:
    three = _cd21(tmp_path, [0.941] * 3)
    comparison = compare_reference("CD21", three, oracle_report=_oracle(tmp_path, True))
    assert (comparison.result, comparison.tier) == ("REPORT", None)
    assert "it would be PASS (faithful)" in "\n".join(comparison.lines)
    one = _cd21(tmp_path, [0.90], prefix="one")
    assert "it would be FAIL" in "\n".join(compare_reference("CD21", one).lines)


def test_cd21_without_features_is_advisory(tmp_path: Path) -> None:
    without = _cd21(tmp_path, [0.880, 0.885, 0.884, 0.878, 0.890], with_features=False, prefix="n")
    only = compare_reference("CD21", without)
    text = "\n".join(only.lines)
    assert only.result == "REPORT"
    assert "without features (advisory, not gating)" in text and "0.883 ± 0.006" in text
    # Both sets together: the same seeds are allowed, and only the with-features set gates.
    both = compare_reference("CD21", [*_cd21(tmp_path, [0.941] * 5), *without])
    assert (both.result, both.tier) == ("PASS", "acceptable")
    broken = _cd21(tmp_path, [0.88], with_features=False, prefix="b")
    metrics = json.loads((broken[0] / "metrics.json").read_text())
    metrics["test_acc_at_best"] = None
    (broken[0] / "metrics.json").write_text(json.dumps(metrics))
    assert "missing or not finite" in "\n".join(compare_reference("CD21", broken).lines)


def test_cd21_warns_on_mixed_machines(tmp_path: Path) -> None:
    runs = _cd21(tmp_path, [0.941] * 4)
    runs += _cd21(tmp_path, [0.941], prefix="m", machine={**MACHINE, "gpu": "RTX 5090"})
    metrics = json.loads((runs[-1] / "metrics.json").read_text())
    metrics["seed"] = 4
    (runs[-1] / "metrics.json").write_text(json.dumps(metrics))
    lines = compare_reference("CD21", runs).lines
    assert any("different machines, builds or float32" in line for line in lines)


def test_oracle_report_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs = _cd21(tmp_path, [0.941] * 5)
    with pytest.raises(ConfigError, match="boolean 'passed'"):
        compare_reference("CD21", runs, oracle_report=_oracle(tmp_path, "yes"))
    with pytest.raises(ConfigError, match="cannot read"):
        compare_reference("CD21", runs, oracle_report=tmp_path / "missing.json")
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(ConfigError, match="invalid JSON"):
        compare_reference("CD21", runs, oracle_report=tmp_path / "bad.json")
    with pytest.raises(ConfigError, match="only used by CD21"):
        compare_reference("R1", runs, oracle_report=_oracle(tmp_path, True))
    oracle = str(_oracle(tmp_path, True))
    argv = ["compare-reference", *map(str, runs), "--ref", "CD21", "--oracle-report", oracle]
    assert main(argv) == 0
    assert capsys.readouterr().out.rstrip().endswith("CD21: PASS (faithful)")
