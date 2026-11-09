"""Tests of the experiment-spec format (``runs.spec``, task A3; shared with the registry, A2)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from _queue_support import SHA, spec_text

from astro_simclr.cli import build_parser, training_settings
from astro_simclr.runs.canonical import config_sha256
from astro_simclr.runs.identity import SCREEN, WHOLE, Engine, run_identity
from astro_simclr.runs.spec import (
    Cell,
    ExperimentSpec,
    SpecError,
    check_trainable,
    expected_run_ids,
    load_spec,
    parse_spec,
)

FULL = """\
format: astro-simclr-experiment-spec
version: 1
name: c5-screen
command: train-supervised
preset: carrasco2021
settings: {max_steps: 300}
arms:
  baseline: {}
  flux: {flux_rescale: 0.5}
seeds: [0, 1]
splits: {file: configs/splits/x.json, folds: [1, 0]}
machine: {class: cuda-nvidia-geforce-gtx-1050, extra: cu126}
"""


def test_parse_full_spec() -> None:
    spec = parse_spec(FULL)
    assert spec.name == "c5-screen"
    assert spec.preset == "carrasco2021"
    assert [arm.name for arm in spec.arms] == ["baseline", "flux"]
    assert spec.seeds == (0, 1)
    assert spec.splits is not None and spec.splits.file == Path("configs/splits/x.json")
    assert spec.split_parts() == (1, 0)
    assert spec.machine_class == "cuda-nvidia-geforce-gtx-1050"
    assert spec.extra == "cu126"
    assert spec.scratch is False
    cells = spec.cells()
    assert len(cells) == 8
    assert cells[0] == Cell("baseline", 0, 0)
    assert [cell.key for cell in cells[:2]] == ["baseline/s0/f0", "baseline/s0/f1"]


def test_spec_id_ignores_comments_and_key_order_but_not_content() -> None:
    reordered = FULL.replace("seeds: [0, 1]\n", "") + "seeds: [0, 1]  # moved\n"
    assert parse_spec(reordered).spec_id == parse_spec(FULL).spec_id
    assert parse_spec(FULL.replace("[0, 1]", "[0, 2]")).spec_id != parse_spec(FULL).spec_id
    scratch = FULL + "scratch: true\n"
    assert parse_spec(scratch).spec_id != parse_spec(FULL).spec_id


def test_to_plain_round_trips() -> None:
    import yaml

    spec = parse_spec(FULL)
    assert parse_spec(yaml.safe_dump(spec.to_plain())) == spec
    screen = parse_spec(FULL.replace("folds: [1, 0]", "screen: true"))
    assert screen.split_parts() == (SCREEN,)
    assert parse_spec(yaml.safe_dump(screen.to_plain())) == screen


def test_arm_config_hash_matches_the_command_line() -> None:
    spec = parse_spec(FULL)
    args = build_parser().parse_args(
        [
            "train-supervised",
            "--preset",
            "carrasco2021",
            "--set",
            "max_steps=300",
            "flux_rescale=0.5",
        ]
    )
    flat = training_settings("train-supervised", args)
    assert spec.arm_config_sha256("flux") == config_sha256("train-supervised", flat)
    assert spec.arm_config_sha256(spec.arms[1]) == spec.arm_config_sha256("flux")
    assert spec.arm_config_sha256("flux") != spec.arm_config_sha256("baseline")
    # Seed, data_dir and output_dir do not change the config hash.
    one = spec.arm_settings("flux", 1, data_dir=Path("/d"), output_dir=Path("/o"))
    assert config_sha256("train-supervised", one) == spec.arm_config_sha256("flux")
    assert one.seed == 1 and one.data_dir == Path("/d")


def test_expected_run_ids() -> None:
    spec = parse_spec(FULL)
    ids = expected_run_ids(spec, data_sha256=SHA, splits_sha256="b" * 64)
    assert len(set(ids.values())) == 8
    cell = Cell("flux", 1, 0)
    identity = run_identity(
        "train-supervised",
        spec.arm_settings("flux", 1),
        data_sha256=SHA,
        machine_class=spec.machine_class,
        engine="plain",
        split=0,
        splits_sha256="b" * 64,
    )
    assert ids[cell] == identity.run_id
    with pytest.raises(SpecError, match="splits_sha256"):
        expected_run_ids(spec, data_sha256=SHA, splits_sha256=None)
    with pytest.raises(SpecError, match="SHA-256"):
        expected_run_ids(spec, data_sha256="nope", splits_sha256="b" * 64)
    plain = parse_spec(spec_text())
    assert len(expected_run_ids(plain, data_sha256=SHA, splits_sha256=None)) == 4


def test_the_engine_changes_run_ids_not_config_hashes() -> None:
    """``engine`` is a run-id component, not a setting (ADR 0017); plain by default (4.8)."""
    plain = parse_spec(FULL)
    lightning = parse_spec(FULL + "engine: lightning\n")
    assert lightning.engine == "lightning" and plain.engine == "plain"
    assert plain.spec_id != lightning.spec_id
    assert parse_spec(FULL + "engine: plain\n").spec_id == plain.spec_id
    assert plain.arm_config_sha256("flux") == lightning.arm_config_sha256("flux")

    def ids(spec: ExperimentSpec, engine: Engine | None = None) -> dict[Cell, str]:
        return expected_run_ids(spec, data_sha256=SHA, splits_sha256="b" * 64, engine=engine)

    assert set(ids(plain).values()).isdisjoint(ids(lightning).values())
    assert ids(plain, "lightning") == ids(lightning)
    assert parse_spec(yaml.safe_dump(plain.to_plain())) == plain


RJ23 = """\
format: astro-simclr-experiment-spec
version: 1
name: rj23
command: train-supervised
preset: rj23-lowres
arms: {base: {}}
seeds: [0]
splits: {file: configs/splits/objects.json, whole: true}
machine: {class: cpu-test}
engine: plain
"""


def test_rj23_arms_take_the_object_splits_as_screen_or_whole() -> None:
    whole = parse_spec(RJ23)
    assert whole.split_parts() == (WHOLE,) and whole.cells()[0].key == "base/s0/whole"
    assert parse_spec(yaml.safe_dump(whole.to_plain())) == whole
    screen = parse_spec(RJ23.replace("whole: true", "screen: true"))
    assert screen.split_parts() == (SCREEN,) and screen.spec_id != whole.spec_id
    ids = expected_run_ids(whole, data_sha256=SHA, splits_sha256="b" * 64)
    assert len(ids) == 1
    with pytest.raises(SpecError, match="engine: plain"):
        parse_spec(RJ23.replace("engine: plain", "engine: lightning"))
    with pytest.raises(SpecError, match="not folds"):
        parse_spec(RJ23.replace("whole: true", "folds: [0]"))
    with pytest.raises(SpecError, match="not folds"):
        parse_spec(RJ23.replace("splits: {file: configs/splits/objects.json, whole: true}\n", ""))
    with pytest.raises(SpecError, match="exactly one"):
        parse_spec(RJ23.replace("whole: true", "whole: true, screen: true"))
    with pytest.raises(SpecError, match="true or false"):
        parse_spec(RJ23.replace("whole: true", "whole: 1"))
    with pytest.raises(SpecError, match="for schedule rj23 only"):
        parse_spec(FULL.replace("folds: [1, 0]", "whole: true"))


def test_cell_keys() -> None:
    for cell in (
        Cell("a", 0, None),
        Cell("a.b-c", 12, SCREEN),
        Cell("x", 3, 4),
        Cell("w", 1, WHOLE),
    ):
        assert Cell.parse(cell.key) == cell
    assert Cell("a", 0, None).key == "a/s0/all"
    with pytest.raises(SpecError):
        Cell.parse("a/0/all")
    assert sorted([Cell("a", 0, 1), Cell("a", 0, SCREEN), Cell("a", 0, None)]) == [
        Cell("a", 0, None),
        Cell("a", 0, SCREEN),
        Cell("a", 0, 1),
    ]
    assert Cell("a", 0, None).__lt__(3) is NotImplemented


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (("format: astro-simclr-experiment-spec", "format: other"), "not an"),
        (("version: 1", "version: 2"), "version"),
        (("version: 1", "version: true"), "version"),
        (("name: c5-screen", "name: C5 Screen"), "slug"),
        (("command: train-supervised", "command: pretrain"), "a spec runs"),
        (("preset: carrasco2021", "preset: nope"), "preset"),
        (("{max_steps: 300}", "{seed: 3}"), "may not set seed"),
        (("{max_steps: 300}", "{data_dir: /x}"), "may not set data_dir"),
        (("{max_steps: 300}", "{no_such_setting: 1}"), "unknown setting"),
        (("{max_steps: 300}", "{max_steps: many}"), "max_steps"),
        (("{max_steps: 300}", "[1]"), "mapping of setting names"),
        (("{max_steps: 300}", "{max_steps: .nan}"), "no JSON form"),
        (("flux: {flux_rescale: 0.5}", "flux: {pretrained: x.ckpt}"), "pretrained"),
        (("  baseline: {}\n  flux: {flux_rescale: 0.5}\n", "  {}\n"), "arms"),
        (("seeds: [0, 1]", "seeds: [0, 0]"), "repeats a seed"),
        (("seeds: [0, 1]", "seeds: []"), "non-empty"),
        (("seeds: [0, 1]", "seeds: [true]"), "integer"),
        (("seeds: [0, 1]", "seeds: [-1]"), "integer"),
        (("folds: [1, 0]", "folds: [1, 1]"), "repeats a fold"),
        (("folds: [1, 0]", "folds: []"), "non-empty"),
        (("folds: [1, 0]", "folds: [1], screen: true"), "exactly one"),
        (("folds: [1, 0]", "screen: false"), "exactly one"),
        (("folds: [1, 0]", "screen: 1"), "true or false"),
        (("folds: [1, 0]", "folds: [0], extra: 1"), "unknown keys"),
        (("file: configs/splits/x.json", "file: /abs.json"), "inside the checkout"),
        (("file: configs/splits/x.json", "file: ../x.json"), "inside the checkout"),
        (("extra: cu126", "extra: 'cu 126'"), "uv extra"),
        (("extra: cu126", "gpu: 1"), "unknown keys"),
        (("class: cuda-nvidia-geforce-gtx-1050", "class: GTX"), "slug"),
        (("machine: {", "machine: ["), "YAML"),
        (("name: c5-screen", "name: c5-screen\nowner: me"), "unknown"),
        (("name: c5-screen\n", ""), "missing keys"),
        (("name: c5-screen", "name: c5-screen\nengine: jax"), "engine"),
    ],
)
def test_invalid_specs(edit: tuple[str, str], message: str) -> None:
    text = FULL.replace(*edit)
    assert text != FULL
    with pytest.raises(SpecError, match=message):
        parse_spec(text)


def test_scratch_must_be_a_bool_and_top_level_a_mapping() -> None:
    with pytest.raises(SpecError, match="scratch"):
        parse_spec(FULL + "scratch: yes please\n")
    with pytest.raises(SpecError, match="mapping"):
        parse_spec("- 1\n")
    with pytest.raises(SpecError, match="mapping"):
        parse_spec(
            FULL.replace("splits: {file: configs/splits/x.json, folds: [1, 0]}", "splits: 3")
        )


def test_unknown_arm() -> None:
    spec = parse_spec(FULL)
    with pytest.raises(SpecError, match="no arm"):
        spec.arm("nope")
    with pytest.raises(SpecError, match="no arm"):
        spec.arm_settings("nope", 0)


def test_load_spec(tmp_path: Path) -> None:
    path = tmp_path / "spec.yaml"
    path.write_text(FULL)
    assert load_spec(path) == parse_spec(FULL)
    with pytest.raises(SpecError, match="cannot read"):
        load_spec(tmp_path / "missing.yaml")
    path.write_text(FULL.replace("version: 1", "version: 9"))
    with pytest.raises(SpecError, match=str(path)):
        load_spec(path)


def test_spec_without_preset_or_splits_uses_the_legacy_protocol() -> None:
    spec = parse_spec(spec_text().replace("preset: carrasco2021\n", ""))
    assert spec.preset is None and spec.splits is None
    assert spec.split_parts() == (None,)
    settings = spec.arm_settings("wide", 1)
    # The Settings defaults (the train-supervised legacy protocol), then the spec's.
    assert settings.supervised_model == "stamp_classifier" and settings.max_epochs == 220
    assert settings.lr == 0.01 and settings.batch_size == 4 and settings.seed == 1


def test_the_registry_reads_a_real_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The A2/A3 contract: ``ExperimentSpec`` is the registry's ``SpecLike`` (also for mypy)."""
    from astro_simclr.registry.spec_view import SpecLike, load_spec_file, plan_from_spec

    splits = tmp_path / "configs" / "splits" / "x.json"
    splits.parent.mkdir(parents=True)
    splits.write_text("{}\n")
    path = tmp_path / "spec.yaml"
    path.write_text(FULL)
    spec: SpecLike = parse_spec(FULL)  # a static conformance check
    monkeypatch.chdir(tmp_path)  # the registry hashes splits.file from the checkout root
    loaded = load_spec_file(path)
    assert loaded.spec_id == spec.spec_id
    plan = plan_from_spec(loaded)
    assert plan.spec_id == spec.spec_id and plan.machine_class == spec.machine_class
    assert plan.arms == {arm.name: spec.arm_config_sha256(arm.name) for arm in spec.arms}
    assert len(plan.cells) == len(spec.cells()) == 8
    from astro_simclr.data.files import sha256_file

    assert plan.splits_sha256 == sha256_file(splits)


def test_a_lightning_spec_parses_but_is_not_trainable() -> None:
    # Task 4.8: an old Lightning spec still resolves its run ids (runs compare), but the
    # queue refuses to train it, naming the tag that keeps the engine.
    lightning = parse_spec(FULL + "engine: lightning\n")
    with pytest.raises(SpecError, match="lightning-final"):
        check_trainable(lightning)
    check_trainable(parse_spec(FULL))
