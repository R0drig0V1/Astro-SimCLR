"""Tests of the ConvNeXt stamp classifiers (``convnext_model``; plan row B8, ADR 0022).

1. **Geometry:** the whole 63x63 stamp, padded to 64, gives stage maps of 32-16-8-4 (and
   the timm trap the module avoids: ``patch_size=2`` alone downsamples stage 0 again).
2. **Exact C4 invariance** of the full model in float64, with every parameter perturbed
   (LayerScale's 1e-6 would otherwise make the blocks almost the identity and hide a
   break); a flip, which the model is not invariant to, is checked to change the logits.
3. Parameter and MAC counts (pinned), the head and ``feature_set``, the input checks.
4. **Pretrained weights** without the network: a fake downloader serves a local file; the
   licence gate, the size and SHA-256 checks and the strict key check are exercised. A
   from-scratch model is built and run with sockets disabled.
5. The settings (``image_size`` 63, the crop-only flags refused) and the wiring of
   ``backbone_weights`` into a run. The CPU split run is in ``tests/test_split_runs.py``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import socket
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import timm
import torch
from timm.models.convnext import ConvNeXt
from torch import nn

from astro_simclr import experiments
from astro_simclr.config import ConfigError, Settings, settings_from_mapping
from astro_simclr.convnext_model import (
    BACKBONE_WEIGHTS,
    NATIVE_SIDE,
    STAGE_SIDES,
    ZOOBOT_CONVNEXT_PICO,
    BackboneWeights,
    ConvNeXtStampClassifier,
    WeightsError,
    backbone_weights_for,
    fetch_backbone_weights,
    load_backbone_weights,
)
from astro_simclr.data.float_input import crop_split
from astro_simclr.data.prepared import load_prepared
from astro_simclr.data.schema import DataError
from astro_simclr.paper_variants import (
    VARIANTS,
    build_paper_model,
    count_macs,
    count_parameters,
    paper_input_side,
)
from astro_simclr.presets import CARRASCO2021_PRESET, PRESETS, preset_report
from astro_simclr.tasks.common import check_schedule
from astro_simclr.training.device_data import float_crops, interim_uint8_crops
from astro_simclr.training.modules import PaperClassifierModule

ARCHS = ("convnext_atto", "convnext_pico")

#: (parameters, MACs per stamp: 4 views, count_macs) with the 23 features.
_COUNTS = {
    "convnext_atto": (3_405_227, 714_557_184),
    "convnext_pico": (8_575_667, 1_785_014_016),
}


def _model(arch: str = "convnext_atto", seed: int = 0, **kwargs: Any) -> ConvNeXtStampClassifier:
    torch.manual_seed(seed)
    model = build_paper_model(arch, **kwargs)
    assert isinstance(model, ConvNeXtStampClassifier)
    return model


def _inputs(n: int = 3, dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    images = torch.rand(n, 3, NATIVE_SIDE, NATIVE_SIDE, generator=generator, dtype=dtype)
    return images, torch.randn(n, 23, generator=generator, dtype=dtype)


# --- 1. Geometry ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ARCHS)
def test_stage_maps_are_32_16_8_4(arch: str) -> None:
    model = _model(arch).eval()
    images, _ = _inputs(1, torch.float32)
    with torch.no_grad():
        assert model.stage_sides(images) == STAGE_SIDES == (32, 32, 16, 8, 4)
    stem = model.backbone.stem[0]
    assert isinstance(stem, nn.Conv2d)
    assert (stem.kernel_size, stem.stride, stem.padding) == ((4, 4), (2, 2), (1, 1))
    assert isinstance(model.backbone.stages[0].downsample, nn.Identity)


def test_timm_patch_size_2_alone_downsamples_stage_0() -> None:
    # The trap the module avoids (module docstring, point 2): timm compensates the stride.
    backbone = timm.create_model("convnext_atto", pretrained=False, num_classes=0, patch_size=2)
    assert isinstance(backbone, ConvNeXt)
    x = backbone.stem(torch.zeros(1, 3, 64, 64))
    assert x.shape[-1] == 32
    assert backbone.stages[0](x).shape[-1] == 16


@pytest.mark.parametrize("arch", ARCHS)
def test_forward_shapes(arch: str) -> None:
    model = _model(arch, n_classes=7).eval()
    images, features = _inputs(2, torch.float32)
    with torch.no_grad():
        assert model(images, features).shape == (2, 7)
        assert model.representation(images).shape == (2, 64)


# --- 2. Invariance -------------------------------------------------------------------------


def _perturbed(model: nn.Module) -> nn.Module:
    generator = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for parameter in model.parameters():
            noise = torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
            parameter.add_(0.1 * noise)
    return model


@pytest.mark.parametrize("arch", ARCHS)
def test_rotation_invariance_is_exact_and_a_flip_is_visible(arch: str) -> None:
    model = _perturbed(_model(arch).double()).eval()
    images, features = _inputs()
    with torch.no_grad():
        logits = model(images, features)
        for k in (1, 2, 3):
            rotated = model(torch.rot90(images, k, dims=(2, 3)), features)
            torch.testing.assert_close(rotated, logits, rtol=0, atol=1e-10)
        flipped = model(images.flip(-1), features)
    assert (flipped - logits).abs().max() > 1e-4


def test_dropout_active_applies_only_in_train_mode() -> None:
    model = _model(dropout_active=True, drop_rate=0.5)
    images, features = _inputs(2, torch.float32)
    with torch.no_grad():
        model.eval()
        torch.testing.assert_close(model(images, features), model(images, features))
        model.train()
        torch.manual_seed(0)
        first = model(images, features)
        torch.manual_seed(1)
        assert not torch.equal(first, model(images, features))


# --- 3. Counts, head, checks ---------------------------------------------------------------


@pytest.mark.parametrize("arch", ARCHS)
def test_parameter_and_mac_counts(arch: str) -> None:
    model = _model(arch)
    parameters, macs = _COUNTS[arch]
    assert count_parameters(model) == parameters
    assert count_macs(model) == macs
    # Without features: the feature BN (46) and fc2's 23 x 64 feature weights go.
    assert count_parameters(_model(arch, feature_set="none")) == parameters - 46 - 23 * 64


def test_feature_set_none_is_the_model_without_features() -> None:
    images, features = _inputs(2)
    none = _model(feature_set="none").double().eval()
    without = _model(with_features=False).double().eval()
    assert not none.with_features
    with torch.no_grad():
        torch.testing.assert_close(none(images, features), without(images, None), rtol=0, atol=0)
    position = _model(feature_set="position").eval()
    assert position.head.bn_features is not None
    assert position.head.bn_features.num_features == 6


def test_the_seed_fixes_the_initialisation() -> None:
    first, second, other = _model(seed=3), _model(seed=3), _model(seed=4)
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key]), key
    stem = "stem.0.weight"
    assert not torch.equal(first.backbone.state_dict()[stem], other.backbone.state_dict()[stem])


def test_input_and_construction_checks() -> None:
    model = _model()
    with pytest.raises(ValueError, match=r"needs images of shape \(N, 3, 63, 63\)"):
        model(torch.zeros(1, 3, 21, 21), torch.zeros(1, 23))
    with pytest.raises(ValueError, match=r"needs images of shape"):
        model.stage_sides(torch.zeros(3, 63, 63))
    with pytest.raises(ValueError, match="arch must be one of"):
        ConvNeXtStampClassifier("convnext_nano")
    with pytest.raises(ValueError, match="drop_rate"):
        ConvNeXtStampClassifier(drop_rate=1.0)
    with pytest.raises(ValueError, match="in_channels"):
        ConvNeXtStampClassifier(in_channels=0)
    with pytest.raises(ValueError, match="context must be none"):
        build_paper_model("convnext_atto", context="avgpool63")
    with pytest.raises(ValueError, match="group_fields applies to"):
        build_paper_model("convnext_atto", group_fields=8)
    assert paper_input_side("convnext_pico") == 63
    assert paper_input_side("rot4") == paper_input_side("p4") == 21
    with pytest.raises(ValueError, match="paper_arch must be one of"):
        paper_input_side("vit")


def test_an_unexpected_timm_layout_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    real = timm.create_model

    def stage0_downsampled(*args: Any, **kwargs: Any) -> nn.Module:
        return real(*args, **{**kwargs, "patch_size": 2})

    monkeypatch.setattr(timm, "create_model", stage0_downsampled)
    with pytest.raises(RuntimeError, match="unexpected layout"):
        ConvNeXtStampClassifier("convnext_atto")


# --- 4. Pretrained weights -----------------------------------------------------------------


def _weights_file(tmp_path: Path, state: object) -> tuple[Path, BackboneWeights]:
    path = tmp_path / "pytorch_model.bin"
    torch.save(state, path)
    data = path.read_bytes()
    spec = dataclasses.replace(
        ZOOBOT_CONVNEXT_PICO,
        arch="convnext_atto",
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    return path, spec


def _served(path: Path) -> Any:
    def download(weights: BackboneWeights) -> Path:
        return path

    return download


def test_the_zoobot_pin_and_its_licence_are_recorded() -> None:
    assert BACKBONE_WEIGHTS["zoobot"] == {"convnext_pico": ZOOBOT_CONVNEXT_PICO}
    assert ZOOBOT_CONVNEXT_PICO.licence == "Apache-2.0"
    assert len(ZOOBOT_CONVNEXT_PICO.revision) == 40  # a commit, never a branch
    assert len(ZOOBOT_CONVNEXT_PICO.sha256) == 64
    assert backbone_weights_for("none", "convnext_atto") is None
    assert backbone_weights_for("zoobot", "convnext_pico") is ZOOBOT_CONVNEXT_PICO
    with pytest.raises(WeightsError, match="no zoobot weights exist for 'convnext_atto'"):
        backbone_weights_for("zoobot", "convnext_atto")
    with pytest.raises(WeightsError, match="backbone_weights must be one of"):
        backbone_weights_for("imagenet", "convnext_pico")
    assert ZOOBOT_CONVNEXT_PICO.record()["sha256"] == ZOOBOT_CONVNEXT_PICO.sha256
    assert issubclass(WeightsError, DataError)  # the CLI prints it on one line


def test_weights_load_into_the_backbone_only(tmp_path: Path) -> None:
    source = _model(seed=5)
    path, spec = _weights_file(tmp_path, source.backbone.state_dict())
    model = _model(seed=0)
    fc1 = model.fc1.weight.clone()
    load_backbone_weights(model, spec, fetch_backbone_weights(spec, download=_served(path)))
    for key, value in source.backbone.state_dict().items():
        assert torch.equal(model.backbone.state_dict()[key], value), key
    assert torch.equal(model.fc1.weight, fc1)


def test_unlicensed_weights_are_refused_before_any_download() -> None:
    def never(weights: BackboneWeights) -> Path:
        raise AssertionError("downloaded")

    unlicensed = dataclasses.replace(ZOOBOT_CONVNEXT_PICO, licence=None)
    with pytest.raises(WeightsError, match="no licence is recorded"):
        fetch_backbone_weights(unlicensed, download=never)


def test_a_corrupted_or_missing_download_is_refused(tmp_path: Path) -> None:
    path, spec = _weights_file(tmp_path, _model().backbone.state_dict())
    with pytest.raises(WeightsError, match="bytes, expected"):
        fetch_backbone_weights(dataclasses.replace(spec, size=1), download=_served(path))
    with pytest.raises(WeightsError, match="SHA-256"):
        fetch_backbone_weights(dataclasses.replace(spec, sha256="0" * 64), download=_served(path))

    def offline(weights: BackboneWeights) -> Path:
        raise OSError("offline")

    with pytest.raises(WeightsError, match="cannot download"):
        fetch_backbone_weights(spec, download=offline)


def test_mismatched_weights_are_refused(tmp_path: Path) -> None:
    model = _model()
    state = model.backbone.state_dict()
    path, spec = _weights_file(tmp_path, state)
    with pytest.raises(WeightsError, match="are for convnext_atto, not convnext_pico"):
        load_backbone_weights(_model("convnext_pico"), spec, path)
    extra = tmp_path / "extra"
    extra.mkdir()
    bad, _ = _weights_file(extra, {**state, "head.fc.weight": torch.zeros(1)})
    with pytest.raises(WeightsError, match="unexpected"):
        load_backbone_weights(model, spec, bad)
    shapes = tmp_path / "shapes"
    shapes.mkdir()
    resized, _ = _weights_file(shapes, {**state, "stem.0.bias": torch.zeros(1)})
    with pytest.raises(WeightsError, match="other shapes"):
        load_backbone_weights(model, spec, resized)
    listed = tmp_path / "listed"
    listed.mkdir()
    not_a_state, _ = _weights_file(listed, [torch.zeros(1)])
    with pytest.raises(WeightsError, match="not a state dict"):
        load_backbone_weights(model, spec, not_a_state)
    six = ConvNeXtStampClassifier("convnext_atto", in_channels=6)
    with pytest.raises(WeightsError, match="take 3 channels"):
        load_backbone_weights(six, spec, path)


def test_from_scratch_needs_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network access")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    model = _model("convnext_pico").eval()
    images, features = _inputs(1, torch.float32)
    with torch.no_grad():
        assert model(images, features).shape == (1, 5)


# --- 5. Settings and the run ---------------------------------------------------------------


def _settings(**changes: object) -> Settings:
    return settings_from_mapping({**PRESETS[CARRASCO2021_PRESET], **changes})


def test_the_variants_are_preset_deviations() -> None:
    for name in ("convnext-atto", "convnext-pico", "convnext-pico-zoobot"):
        variant = VARIANTS[name]
        assert variant.row == "B8"
        report = preset_report(_settings(**variant.overrides))
        assert report is not None
        assert not report["faithful"]
        assert set(report["deviations"]) == set(variant.overrides)


def test_settings_refuse_the_crop_only_flags() -> None:
    with pytest.raises(ConfigError, match="exists for paper_arch convnext_pico only"):
        _settings(paper_arch="convnext_atto", backbone_weights="zoobot")
    with pytest.raises(ConfigError, match="exists for paper_arch convnext_pico only"):
        _settings(backbone_weights="zoobot")
    with pytest.raises(ConfigError, match="backbone_weights 'imagenet' is not one of"):
        _settings(backbone_weights="imagenet")
    for flag, value in (
        ("context", "avgpool63"),
        ("aug_noise", 1.0),
        ("synthetic_bogus", 0.1),
        ("flux_rescale", 0.5),
    ):
        with pytest.raises(ConfigError, match=f"{flag} work on the 21x21 crop"):
            _settings(paper_arch="convnext_pico", image_size=63, **{flag: value})
    with pytest.raises(ValueError, match="work on the 21x21 crop"):
        PaperClassifierModule(
            beta_loss=0.5, lr=1e-3, optimizer="Adam", paper_arch="convnext_atto", aug_noise=1.0
        )


def test_image_size_must_be_the_arch_side() -> None:
    with pytest.raises(ConfigError, match="takes 63x63 images: set image_size=63"):
        check_schedule("train-supervised", _settings(paper_arch="convnext_atto"))
    with pytest.raises(ConfigError, match="takes 21x21 images: set image_size=21"):
        check_schedule("train-supervised", _settings(image_size=63))
    check_schedule("train-supervised", _settings(paper_arch="convnext_atto", image_size=63))
    # Another model (the phase-1 stamp classifier) refuses the flag instead of ignoring it.
    phase1 = Settings(paper_arch="convnext_pico", backbone_weights="zoobot")
    with pytest.raises(ConfigError, match="paper_arch, backbone_weights apply only to"):
        check_schedule("train-supervised", phase1)


def test_a_run_loads_the_pinned_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(paper_arch="convnext_pico", image_size=63, backbone_weights="zoobot")
    source = _model("convnext_pico", seed=5)
    path, spec = _weights_file(tmp_path, source.backbone.state_dict())
    spec = dataclasses.replace(spec, arch="convnext_pico")
    monkeypatch.setitem(BACKBONE_WEIGHTS, "zoobot", {"convnext_pico": spec})
    monkeypatch.setattr(experiments, "_DOWNLOAD", _served(path))
    torch.manual_seed(0)
    module = experiments.paper_module(settings)
    info: dict[str, Any] = {}
    experiments._load_backbone_weights(settings, module, info)
    assert isinstance(module.model, ConvNeXtStampClassifier)
    stem = "stem.0.weight"
    assert torch.equal(module.model.backbone.state_dict()[stem], source.backbone.state_dict()[stem])
    assert info["backbone_weights"] == spec.record()
    # From scratch: nothing is loaded or recorded.
    scratch: dict[str, Any] = {}
    experiments._load_backbone_weights(
        _settings(paper_arch="convnext_pico", image_size=63), module, scratch
    )
    assert scratch == {}


def test_the_data_path_gives_whole_stamps(classifier_f32_dir: Path) -> None:
    stamps = float_crops(classifier_f32_dir, names=("Train",), side=63)["Train"].images
    stored = load_prepared(classifier_f32_dir).splits["Train"].images_f32
    assert stored is not None
    expected = crop_split(stored, 63).transpose(0, 3, 1, 2)
    np.testing.assert_array_equal(stamps, expected)
    assert stamps.shape[1:] == (3, 63, 63)
    uint8 = interim_uint8_crops(classifier_f32_dir, names=("Train",), side=63)["Train"].images
    assert uint8.shape == stamps.shape
    with pytest.raises(ValueError, match="side must be one of"):
        float_crops(classifier_f32_dir, side=27)
    with pytest.raises(ValueError, match="context channels go with the 21x21 crop"):
        float_crops(classifier_f32_dir, side=63, context=True)
