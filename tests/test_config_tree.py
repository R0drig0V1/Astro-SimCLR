"""Tests of the nested config: sections, tagged unions, ``--set`` paths and the flat shim.

The hash migration test (task 4.2) is ``test_config_tree_presets.py``; here are the
schema's construction rules, the tag rule of the merge, the error messages and the shim.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from astro_simclr.config import ConfigError, Settings
from astro_simclr.config_tree import (
    PRESETS,
    Config,
    config_from_mapping,
    config_from_settings,
    config_json,
    config_sha256,
    config_to_plain,
    preset,
    resolve_config,
    resolve_mapping,
)
from astro_simclr.config_tree.resolve import is_nested, merge_sections, split_overrides
from astro_simclr.config_tree.schema import (
    AdamOptim,
    Carrasco2021Schedule,
    LARSOptim,
    PaperAugment,
    PaperStampClassifier,
    ResNetClassifier,
    Runtime,
    SGDOptim,
    StampClassifier,
)

CD21 = {"preset": "carrasco2021"}


# --- Schema --------------------------------------------------------------------------------


def test_the_default_config_is_the_flat_default() -> None:
    config = Config()
    assert config.command == "train-supervised"
    assert config.to_settings() == Settings()


@pytest.mark.parametrize("name", list(PRESETS))
def test_plain_form_round_trips(name: str) -> None:
    config = PRESETS[name]
    plain = config_to_plain(config)
    assert config_from_mapping(plain) == config
    assert next(iter(plain["model"])) == "kind"
    # Canonical JSON: sorted, compact, and the same for an equal config.
    text = config_json(config)
    assert json.loads(text) == json.loads(json.dumps(plain))
    assert text == config_json(config_from_mapping(plain))


def test_presets_decide_the_command() -> None:
    commands = {name: config.command for name, config in PRESETS.items()}
    assert commands == {
        "carrasco2021": "train-supervised",
        "carrasco2021-no-features": "train-supervised",
        "pretrain-legacy": "pretrain",
        "linear-eval-legacy": "linear-eval",
        "fine-tune-legacy": "fine-tune",
        "train-supervised-legacy": "train-supervised",
    }


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: StampClassifier(kind="resnet"), "StampClassifier kind must be one of"),
        (lambda: AdamOptim(kind="SGD"), "AdamOptim kind must be one of: AdamW, Adam"),
        (lambda: ResNetClassifier(name="resnet7"), "model.name 'resnet7' is not one of"),
        (lambda: Runtime(engine="jax"), "runtime.engine 'jax' is not one of: lightning, plain"),
        (lambda: Config(model=PaperStampClassifier()), "go together"),
        (
            lambda: Config(
                model=PaperStampClassifier(),
                schedule=Carrasco2021Schedule(),
                eval=dataclasses.replace(Config().eval, batch_size=64),
                runtime=Runtime(num_workers=0),
            ),
            "augment.kind paper go together",
        ),
        (lambda: Config(augment=PaperAugment()), "augment.kind paper go together"),
        (lambda: Config(model=AdamOptim()), "section model must be one of"),  # type: ignore[arg-type]
    ],
)
def test_invalid_configs_are_refused(build: object, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        build()  # type: ignore[operator]


def test_settings_errors_name_the_nested_paths() -> None:
    with pytest.raises(ConfigError) as err:
        Config(eval=dataclasses.replace(Config().eval, batch_size=0))
    assert "eval_batch_size must be >= 1" in str(err.value)
    assert "eval_batch_size is eval.batch_size" in str(err.value)


def test_from_mapping_checks_sections_keys_and_types() -> None:
    with pytest.raises(ConfigError, match="unknown section"):
        config_from_mapping({"trainer": {}})
    with pytest.raises(ConfigError, match="section data must be a mapping"):
        config_from_mapping({"data": 3})
    with pytest.raises(ConfigError, match=r"model.kind 'mlp' is not one of: stamp_classifier"):
        config_from_mapping({"model": {"kind": "mlp"}})
    with pytest.raises(ConfigError, match=r"unknown key\(s\) data.size; data takes: dir"):
        config_from_mapping({"data": {"size": 3}})
    with pytest.raises(ConfigError, match=r"run.seed must be an integer, got '3'"):
        config_from_mapping({"run": {"seed": "3"}})
    # A union section without kind is its first variant; YAML 1.1 float strings coerce.
    config = config_from_mapping({"model": {"drop_rate": 0.3}, "optim": {"lr": "1e-3"}})
    assert config.model == StampClassifier(drop_rate=0.3)
    assert config.optim == AdamOptim(lr=1e-3)


def test_paths_coerce_and_lower_as_strings() -> None:
    config = config_from_mapping({"run": {"output_dir": "out/x"}, "data": {"dir": "d"}})
    assert config.run.output_dir == Path("out/x")
    settings = config.to_settings()
    assert settings.output_dir == Path("out/x")
    assert settings.data_dir == Path("d")


def test_engine_and_paths_stay_out_of_the_config_hash() -> None:
    base = preset("carrasco2021")
    moved = dataclasses.replace(base, run=dataclasses.replace(base.run, output_dir=Path("o")))
    assert config_sha256(moved) == config_sha256(base)
    reseeded = dataclasses.replace(base, run=dataclasses.replace(base.run, seed=4))
    assert config_sha256(reseeded) == config_sha256(base)  # the seed is a run-id component
    other = resolve_mapping(CD21, ["model.arch=p4"])
    assert config_sha256(other) != config_sha256(base)


# --- The shim (raising flat settings) ------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "changes", "model", "optim"),
    [
        ("train-supervised", {"supervised_model": "resnet50"}, ResNetClassifier, AdamOptim),
        ("train-supervised", {"optimizer": "SGD"}, StampClassifier, SGDOptim),
        ("train-supervised", {"optimizer": "LARS"}, StampClassifier, LARSOptim),
    ],
)
def test_flat_settings_raise_to_their_variants(
    command: str, changes: dict[str, object], model: type, optim: type
) -> None:
    settings = Settings().replace(**changes)
    config = config_from_settings(command, settings)
    assert isinstance(config.model, model)
    assert isinstance(config.optim, optim)
    assert config.to_settings() == settings


@pytest.mark.parametrize(
    ("command", "changes", "message"),
    [
        ("evaluate", {}, "unknown training command 'evaluate'"),
        ("train-supervised", {"supervised_model": "mlp"}, "supervised_model 'mlp' is not one"),
        ("train-supervised", {"optimizer": "RMSprop"}, "optim.kind 'RMSprop' is not one of"),
        ("train-supervised", {"temperature": 0.5}, "temperature=0.5 is not read by that run"),
        ("pretrain", {"supervised_model": "resnet18"}, "supervised_model='resnet18' is not"),
        ("train-supervised", {"adam_betas": "paper", "optimizer": "SGD"}, "adam_betas='paper'"),
    ],
)
def test_lossy_or_unknown_flat_settings_are_refused(
    command: str, changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        config_from_settings(command, Settings().replace(**changes))


def test_the_shim_keeps_the_engine() -> None:
    config = config_from_settings("train-supervised", Settings())
    assert config.runtime.engine == "lightning"


# --- Resolution ----------------------------------------------------------------------------


def test_is_nested() -> None:
    assert is_nested({})
    assert is_nested({"preset": "x", "model": {"kind": "resnet"}})
    assert not is_nested({"seed": 3})
    assert not is_nested({"schedule": "carrasco2021"})  # a flat field that is also a section


def test_merge_follows_the_tag_rule() -> None:
    base = config_to_plain(preset("train-supervised-legacy"))
    same = merge_sections(base, {"model": {"kind": "stamp_classifier", "drop_rate": 0.4}})
    assert same["model"]["beta_loss"] == base["model"]["beta_loss"]
    switched = merge_sections(base, {"model": {"kind": "resnet"}})
    assert switched["model"] == {"kind": "resnet"}
    with pytest.raises(ConfigError, match="unknown section 'trainer'"):
        merge_sections(base, {"trainer": {}})
    with pytest.raises(ConfigError, match="section model must be a mapping of keys"):
        merge_sections(base, {"model": "resnet"})


def test_split_overrides() -> None:
    patch, flat = split_overrides(["model.arch=p4", "seed=3", "model.arch=p4m", "run.seed=1"])
    assert patch == {"model": {"arch": "p4m"}, "run": {"seed": 1}}
    assert flat == {"seed": 3}
    with pytest.raises(ConfigError, match="unknown section 'trainer'"):
        split_overrides(["trainer.max_steps=1"])
    with pytest.raises(ConfigError, match=r"a path is section.key"):
        split_overrides(["model.arch.x=1"])
    with pytest.raises(ConfigError, match=r"a path is section.key"):
        split_overrides(["model.=1"])
    with pytest.raises(ConfigError, match="not of the form key=value"):
        split_overrides(["model.arch"])


def test_an_unknown_path_is_an_error() -> None:
    # model.group_fields exists, but not for the stamp classifier.
    with pytest.raises(ConfigError, match=r"unknown key\(s\) model.group_fields"):
        resolve_mapping({}, ["model.group_fields=8"])
    with pytest.raises(ConfigError, match=r"unknown key\(s\) schedule.max_steps"):
        resolve_mapping({}, ["schedule.max_steps=100"])


def test_dotted_overrides_switch_a_variant_in_any_order() -> None:
    one = resolve_mapping({}, ["model.kind=resnet", "model.name=resnet50"])
    two = resolve_mapping({}, ["model.name=resnet50", "model.kind=resnet"])
    assert one == two
    assert one.model == ResNetClassifier(name="resnet50")


def test_a_file_without_preset_starts_from_its_command() -> None:
    pretrain = resolve_mapping({"model": {"kind": "simclr", "encoder": "stamp1"}})
    assert pretrain.command == "pretrain"
    assert pretrain.schedule == dataclasses.replace(
        preset("pretrain-legacy").schedule
    )  # max 400 epochs, as `astro-simclr pretrain`
    assert resolve_mapping({}) == preset("train-supervised-legacy")
    # An unknown kind fails when the section is built.
    with pytest.raises(ConfigError, match=r"model.kind 'mlp' is not one of"):
        resolve_mapping({"model": {"kind": "mlp"}})


def test_preset_file_and_overrides_resolve_in_order() -> None:
    data = {"preset": "carrasco2021", "run": {"seed": 2}, "model": {"arch": "p4"}}
    config = resolve_mapping(data, ["model.arch=p4m", "seed=5", "paper_arch=single"])
    # The flat overrides come last: they win over the dotted ones.
    assert config.model.kind == "paper_stamp_classifier"
    assert isinstance(config.model, PaperStampClassifier)
    assert config.model.arch == "single"
    assert config.run.seed == 5


@pytest.mark.parametrize(
    ("data", "command", "message"),
    [
        ({"preset": "carrasco2022"}, None, "unknown preset 'carrasco2022'; presets: carrasco"),
        ({"preset": 3}, None, "preset must be a preset name"),
        ({"model": {"kind": "resnet"}}, "pretrain", "--command is for flat config files"),
        ({"seed": 3}, None, "a flat config .* needs the training command"),
        ({"seed": 3}, "evaluate", "unknown training command 'evaluate'"),
        ({"sed": 3}, "train-supervised", r"unknown setting\(s\) sed"),
    ],
)
def test_resolution_errors(data: dict[str, object], command: str | None, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        resolve_mapping(data, command=command)


def test_flat_files_resolve_as_their_subcommand(tmp_path: Path) -> None:
    flat = tmp_path / "flat.yaml"
    flat.write_text("encoder_name: stamp1\nbatch_size: 4\n")
    config = resolve_config(flat, ["run.seed=7", "max_epochs=2"], command="pretrain")
    settings = config.to_settings()
    assert config.command == "pretrain"
    assert (settings.encoder_name, settings.batch_size, settings.seed) == ("stamp1", 4, 7)
    assert settings.max_epochs == 2
    assert settings.lr == preset("pretrain-legacy").optim.lr  # the legacy protocol
    # An empty file with --command is flat too.
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    assert resolve_config(empty, command="pretrain") == preset("pretrain-legacy")


def test_the_example_nested_configs_resolve() -> None:
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "configs" / "runs").glob("*.yaml"))
    assert files
    for path in files:
        config = resolve_config(path)
        assert config.runtime.accelerator == "cuda", path
