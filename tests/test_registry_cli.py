"""Tests of ``astro-simclr runs reindex | ls | show | compare`` (``registry.commands``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _registry_support import (
    copy_replication,
    hide_spec_module,
    install_fake_spec_module,
    legacy_run,
    manifest_run,
    replication_spec,
    write_spec,
)

from astro_simclr.cli import main
from astro_simclr.registry import parser
from astro_simclr.registry.compare import MIXABLE
from astro_simclr.registry.home import HOME_ENV
from astro_simclr.runs.manifest import STATUSES


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    state = tmp_path / "home"
    monkeypatch.setenv(HOME_ENV, str(state))
    return state


@pytest.fixture
def archive(tmp_path: Path, home: Path) -> Path:
    root = copy_replication(tmp_path / "archive")
    assert main(["runs", "reindex", str(root)]) == 0
    return root


def _json(capsys: pytest.CaptureFixture[str]) -> object:
    return json.loads(capsys.readouterr().out)


def test_parser_choices_match_the_modules() -> None:
    assert parser.MIXABLE == MIXABLE
    assert parser.STATUSES == STATUSES


def test_reindex_prints_roots_and_skips(
    tmp_path: Path, home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "root"
    legacy_run(root / "good")
    (root / "bad").mkdir()
    (root / "bad" / "metrics.json").write_text("[]", encoding="utf-8")
    assert main(["runs", "reindex", str(root)]) == 0
    captured = capsys.readouterr()
    assert f"{root.resolve()}: 1 runs" in captured.out
    assert "specs: 0" in captured.out
    assert "astro-simclr: skipped" in captured.err and "not a JSON object" in captured.err
    assert (home / "registry.sqlite").is_file()
    for path in (root / "good").iterdir():
        path.unlink()
    (root / "good").rmdir()
    (root / "bad" / "metrics.json").unlink()
    (root / "bad").rmdir()
    root.rmdir()
    assert main(["runs", "reindex"]) == 0
    assert "remembered root" in capsys.readouterr().err
    assert main(["runs", "reindex", str(root)]) == 1
    assert "not a directory" in capsys.readouterr().err


def test_registry_option_and_worktree_refusal(
    tmp_path: Path, home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = legacy_run(tmp_path / "root" / "run").parent
    other = tmp_path / "other.sqlite"
    assert main(["runs", "reindex", str(root), "--registry", str(other)]) == 0
    assert other.is_file() and not (home / "registry.sqlite").exists()
    capsys.readouterr()
    inside = Path(__file__).parent / "registry.sqlite"
    assert main(["runs", "ls", "--registry", str(inside)]) == 1
    assert "outside every worktree" in capsys.readouterr().err
    assert not inside.exists()


def test_ls(archive: Path, capsys: pytest.CaptureFixture[str]) -> None:
    capsys.readouterr()
    assert main(["runs", "ls"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].split()[:3] == ["key", "command", "config"]
    assert out.rstrip().endswith("10 runs")
    assert main(["runs", "ls", "--seed", "3", "--json"]) == 0
    rows = _json(capsys)
    assert isinstance(rows, list) and len(rows) == 2
    assert {row["seed"] for row in rows} == {3}
    assert all(row["legacy"] and row["run_id"] is None for row in rows)
    config = rows[0]["config_sha256"]
    assert main(["runs", "ls", "--config", config[:8], "--json"]) == 0
    assert len(_json(capsys)) == 5  # type: ignore[arg-type]
    assert main(["runs", "ls", "--no-legacy", "--json"]) == 0
    assert _json(capsys) == []
    assert main(["runs", "ls", "--status", "running", "--command", "pretrain"]) == 0
    assert capsys.readouterr().out.rstrip().endswith("0 runs")


def test_show(archive: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    capsys.readouterr()
    seed0 = archive / "with-features" / "seed-0"
    assert main(["runs", "show", str(seed0), "--json"]) == 0
    shown = _json(capsys)
    assert isinstance(shown, dict)
    assert shown["run_dir"] == str(seed0.resolve())
    assert {a["path"] for a in shown["artefacts"]} == {
        "metrics.json",
        "predictions_val.npz",
        "predictions_test.npz",
        "confusion_matrix_val.png",
        "confusion_matrix_test.png",
    }
    assert shown["copies"] == []
    key = shown["key"]
    assert main(["runs", "show", key[:12]]) == 0
    text = capsys.readouterr().out
    assert f"key: {key}" in text and "val.balanced_accuracy: 0.946" in text
    assert "predictions_val.npz  predictions" in text
    assert main(["runs", "show", "legacy-"]) == 1
    assert "ambiguous" in capsys.readouterr().err
    assert main(["runs", "show", "zzzzzzzz"]) == 1
    assert "no indexed run matches" in capsys.readouterr().err
    assert main(["runs", "show", "short"]) == 1  # a short token is never a prefix
    capsys.readouterr()
    # A copy of one attempt in another root is shown as a copy.
    copy = tmp_path / "copy"
    copy.mkdir()
    (copy / "metrics.json").write_bytes((seed0 / "metrics.json").read_bytes())
    assert main(["runs", "reindex", str(copy)]) == 0
    capsys.readouterr()
    assert main(["runs", "show", key]) == 0
    assert f"copy: {copy.resolve()}" in capsys.readouterr().out


def test_show_a_manifest_run(
    tmp_path: Path, home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = manifest_run(tmp_path / "root" / "run")
    assert main(["runs", "reindex", str(run.parent)]) == 0
    capsys.readouterr()
    assert main(["runs", "ls", "--json"]) == 0
    rows = _json(capsys)
    assert isinstance(rows, list) and len(rows) == 1
    assert main(["runs", "show", rows[0]["run_id"][:20], "--json"]) == 0
    shown = _json(capsys)
    assert shown["attempt"] == 1 and shown["git"] == {"sha": "0" * 40, "dirty": False}  # type: ignore[index]


def test_compare_by_file_id_and_name(
    archive: Path,
    tmp_path: Path,
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    install_fake_spec_module(monkeypatch)
    spec = replication_spec()
    path = write_spec(tmp_path / "spec.yaml", spec)
    capsys.readouterr()
    assert main(["runs", "compare", str(path)]) == 0
    out = capsys.readouterr().out
    assert f"spec replication ({spec.spec_id}): train-supervised" in out
    assert "10 cells, all resolved" in out
    assert "with-features  5" in out
    # Indexed from $ASTRO_SIMCLR_HOME/specs: by id, prefix and name.
    write_spec(home / "specs" / f"{spec.spec_id}.yaml", spec)
    assert main(["runs", "reindex"]) == 0
    assert "specs: 1" in capsys.readouterr().out
    for token in (spec.spec_id, spec.spec_id[:8], "replication"):
        assert main(["runs", "compare", token, "--json", "--metric", "val.accuracy"]) == 0
        plain = _json(capsys)
        assert plain["spec_id"] == spec.spec_id  # type: ignore[index]
        assert plain["metrics"] == ["val.accuracy"]  # type: ignore[index]
        assert len(plain["cells"]) == 10  # type: ignore[index]
    assert main(["runs", "compare", "unknown-spec"]) == 1
    assert "no spec file and no indexed spec" in capsys.readouterr().err


def test_compare_ambiguous_name(
    archive: Path, home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_fake_spec_module(monkeypatch)
    for seeds in ((0, 1), (2, 3)):
        spec = replication_spec(seeds=seeds, name="same")
        write_spec(home / "specs" / f"{spec.spec_id}.yaml", spec)
    assert main(["runs", "reindex"]) == 0
    capsys.readouterr()
    assert main(["runs", "compare", "same"]) == 1
    assert "is ambiguous" in capsys.readouterr().err


def test_compare_fails_on_missing_cells(
    archive: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    install_fake_spec_module(monkeypatch)
    path = write_spec(tmp_path / "spec.yaml", replication_spec(seeds=range(6)))
    capsys.readouterr()
    assert main(["runs", "compare", str(path)]) == 1
    err = capsys.readouterr().err
    assert "with-features/seed=5: no finished run" in err
    assert "no-features/seed=5: no finished run" in err


def test_compare_bad_spec_and_no_a3(
    archive: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = write_spec(tmp_path / "spec.yaml", replication_spec())
    hide_spec_module(monkeypatch)
    capsys.readouterr()
    assert main(["runs", "compare", str(path)]) == 1
    assert "need task A3" in capsys.readouterr().err
    install_fake_spec_module(monkeypatch)
    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    assert main(["runs", "compare", str(bad)]) == 1
    assert "not a spec" in capsys.readouterr().err


def test_runs_needs_a_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["runs"])
    assert exit_info.value.code == 2
