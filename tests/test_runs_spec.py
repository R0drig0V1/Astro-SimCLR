"""Tests of the experiment-spec format (``runs.spec``, task A3; shared with the registry, A2)."""

from __future__ import annotations

from pathlib import Path

import pytest
from _queue_support import SHA, spec_text

from astro_simclr.cli import build_parser, training_settings
from astro_simclr.runs.canonical import config_sha256
from astro_simclr.runs.identity import SCREEN, run_identity
from astro_simclr.runs.spec import (
    Cell,
    SpecError,
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
        engine="lightning",
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


def test_cell_keys() -> None:
    for cell in (Cell("a", 0, None), Cell("a.b-c", 12, SCREEN), Cell("x", 3, 4)):
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
    spec = parse_spec(spec_text())
    assert spec.preset is None and spec.splits is None
    assert spec.split_parts() == (None,)
    settings = spec.arm_settings("wide", 1)
    assert settings.drop_rate == 0.5 and settings.batch_size == 5 and settings.seed == 1
