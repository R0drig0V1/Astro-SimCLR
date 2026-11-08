"""Tests of the 2.5-models variants of the paper model (``paper_variants``; plan B7, B2', B1').

1. **The faithful model is unchanged.** ``tests/fixtures/paper_model/faithful.npz`` holds the
   seeded initialisation and float64 logits of ``PaperStampClassifier`` *before* the flags
   existed (``scripts/characterize/paper_model/generate.py`` runs the commit before them).
   Once per new flag, the ``carrasco2021`` preset with that flag set explicitly to its
   default must reproduce it: the parameter count (441,075 / 439,557), every initial
   tensor, and the outputs.
2. **Invariance of the full model**, not of one layer: padding and pooling parity can break a
   G-CNN silently. Every invariant configuration is checked in eval mode, in float64 and
   float32, under the rotations by 90 degrees, and under the flips for p4m, with random
   biases and BN affine parameters (the initial zeros would hide an expansion bug). The
   non-invariant cases are checked to fail, so the test can see a break. The check on each
   accelerator runs in float64 on CUDA, where cuDNN's float32 Winograd convolutions move
   the logits by ~1e-5 under a rotation (the comment in the test has the measurements).
3. Parameter and MAC counts, the initialisation, the head's equivalence with the paper's,
   the settings, the context data path and a CPU training smoke run.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from _training_support import ACCELERATORS
from torch import nn

from astro_simclr.config import ConfigError, Settings, settings_from_mapping
from astro_simclr.data.float_input import context_split, crop_split, crop_then_normalise
from astro_simclr.data.float_input import context_then_normalise as context_of
from astro_simclr.experiments import paper_module, run_command
from astro_simclr.group_conv import GroupReadout
from astro_simclr.paper_model import PaperStampClassifier
from astro_simclr.paper_variants import (
    CONTEXTS,
    PAPER_ARCHS,
    VARIANTS,
    GroupStampClassifier,
    PaperHead,
    build_paper_model,
    count_macs,
    count_parameters,
    default_group_fields,
    paper_in_channels,
    paper_input_side,
)
from astro_simclr.presets import (
    CARRASCO2021_NO_FEATURES,
    CARRASCO2021_PRESET,
    PRESETS,
    preset_report,
)
from astro_simclr.training.device_data import TensorSplit, float_crops
from astro_simclr.training.modules import PaperClassifierModule

FIXTURE = Path(__file__).parent / "fixtures" / "paper_model" / "faithful.npz"

#: The new flags and their defaults (the faithful model).
_FLAG_DEFAULTS: dict[str, object] = {
    "paper_arch": "rot4",
    "group_fields": None,
    "context": "none",
    "rot90_augment": False,
    # The features the head reads (ADR 0016).
    "feature_set": "all",
    # The ConvNeXt backbone's initial weights (plan row B8; ADR 0022).
    "backbone_weights": "none",
    # Plan rows C5 and C6 (ADR 0006).
    "aug_noise": 0.0,
    "aug_feature_jitter": 0.0,
    "synthetic_bogus": 0.0,
    "aug_min_r2": 0.5,
    # Plan row C7 (ADR 0007).
    "flux_rescale": 0.0,
    "flux_k_min": 0.5,
    "flux_k_max": 1.5,
}


@pytest.fixture(scope="module")
def faithful() -> Iterator[dict[str, np.ndarray]]:
    with np.load(FIXTURE, allow_pickle=False) as npz:
        yield {key: npz[key] for key in npz.files}


def _module(preset: str, **changes: object) -> PaperClassifierModule:
    torch.manual_seed(0)
    return paper_module(settings_from_mapping({**PRESETS[preset], **changes}))


# --- 1. The faithful model is unchanged --------------------------------------------------


@pytest.mark.parametrize(
    ("preset", "case"),
    [(CARRASCO2021_PRESET, "features"), (CARRASCO2021_NO_FEATURES, "no_features")],
)
@pytest.mark.parametrize("flag", list(_FLAG_DEFAULTS))
def test_each_flag_at_its_default_keeps_the_faithful_model(
    flag: str, preset: str, case: str, faithful: dict[str, np.ndarray]
) -> None:
    module = _module(preset, **{flag: _FLAG_DEFAULTS[flag]})
    model = module.model
    assert type(model) is PaperStampClassifier
    assert count_parameters(model) == int(faithful[f"{case}_params"])
    assert count_parameters(model) == (441_075 if case == "features" else 439_557)
    state = model.state_dict()
    sums = np.array([state[key].double().sum().item() for key in sorted(state)])
    np.testing.assert_allclose(sums, faithful[f"{case}_init"], rtol=1e-12, atol=1e-12)
    images = torch.from_numpy(faithful["images"]).double()
    features = torch.from_numpy(faithful["features"]).double()
    with torch.no_grad():
        logits = model.eval().double()(images, features)
    np.testing.assert_allclose(logits.numpy(), faithful[f"{case}_logits64"], rtol=0, atol=1e-12)


def test_the_flag_defaults_are_the_preset_and_variants_are_deviations() -> None:
    defaults = Settings()
    for name in (CARRASCO2021_PRESET, CARRASCO2021_NO_FEATURES):
        for flag, value in _FLAG_DEFAULTS.items():
            assert PRESETS[name][flag] == value == getattr(defaults, flag)
    faithful = preset_report(settings_from_mapping(PRESETS[CARRASCO2021_PRESET]))
    assert faithful is not None
    assert faithful["faithful"]
    variant = preset_report(
        settings_from_mapping({**PRESETS[CARRASCO2021_PRESET], **VARIANTS["p4-context"].overrides})
    )
    assert variant is not None
    assert not variant["faithful"]
    assert variant["deviations"] == {
        "paper_arch": {"value": "p4", "preset": "rot4"},
        "context": {"value": "avgpool63", "preset": "none"},
    }


# --- 2. Invariance of the full model -----------------------------------------------------

#: (arch, group_fields, context) of every invariant configuration.
_INVARIANT = [
    ("rot4", None, "none"),
    ("rot4", None, "avgpool63"),
    ("p4", None, "none"),
    ("p4", 16, "none"),
    ("p4", None, "avgpool63"),
    ("p4m", None, "none"),
    ("p4m", None, "avgpool63"),
]


def _randomised[M: nn.Module](model: M) -> M:
    """``model`` in eval mode with random biases and BN affine parameters (seed 0)."""
    torch.manual_seed(0)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("bias"):
                parameter.uniform_(-0.2, 0.2)
            elif "bn_features" in name:
                parameter.uniform_(0.5, 1.5)
    return model.eval()


def _variant(
    arch: str, fields: int | None, context: str, *, with_features: bool = True
) -> nn.Module:
    torch.manual_seed(0)
    model = build_paper_model(
        arch, group_fields=fields, context=context, with_features=with_features
    )
    return _randomised(model)


def _inputs(channels: int, dtype: torch.dtype, n: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    images = torch.rand(n, channels, 21, 21, generator=generator, dtype=torch.float64)
    features = torch.randn(n, 23, generator=generator, dtype=torch.float64)
    return images.to(dtype), features.to(dtype)


def _max_change(model: nn.Module, images: torch.Tensor, features: torch.Tensor) -> dict[str, float]:
    """max |logit(T x) - logit(x)| over the 3 rotations and over the 4 flips of D4."""
    with torch.no_grad():
        reference = model(images, features)

        def change(transformed: torch.Tensor) -> float:
            return float((model(transformed, features) - reference).abs().max())

        rotations = [torch.rot90(images, k, dims=(2, 3)) for k in (1, 2, 3)]
        flipped = torch.flip(images, dims=(3,))
        flips = [torch.rot90(flipped, k, dims=(2, 3)) for k in range(4)]
        return {
            "rotation": max(map(change, rotations)),
            "flip": max(map(change, flips)),
            "scale": float(reference.abs().max()),
        }


@pytest.mark.parametrize("with_features", [True, False], ids=["features", "no_features"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["float64", "float32"])
@pytest.mark.parametrize(("arch", "fields", "context"), _INVARIANT)
def test_invariant_models_are_invariant(
    arch: str, fields: int | None, context: str, dtype: torch.dtype, with_features: bool
) -> None:
    model = _variant(arch, fields, context, with_features=with_features).to(dtype)
    changes = _max_change(model, *_inputs(paper_in_channels(context), dtype))
    assert changes["scale"] > 0.05  # the logits are not trivially constant
    assert changes["rotation"] < 1e-5
    if arch == "p4m":
        assert changes["flip"] < 1e-5
    else:  # p4 and rot4 are not flip-invariant: the check can see a break
        assert changes["flip"] > 1e-4
    if dtype == torch.float64:
        assert changes["rotation"] < 1e-10


def test_the_single_pass_model_is_not_invariant() -> None:
    model = _variant("single", None, "none").double()
    changes = _max_change(model, *_inputs(3, torch.float64))
    assert changes["rotation"] > 1e-3


@pytest.mark.parametrize("accelerator", ACCELERATORS)
@pytest.mark.parametrize("arch", ["p4", "p4m"])
def test_invariance_on_each_accelerator(arch: str, accelerator: str) -> None:
    # CUDA runs in float64. In float32, cuDNN runs the 5x5 group convolutions (32 -> 32
    # channels) with its non-fused Winograd F(5x5) on 9x9 tiles, whatever the TF32 policy:
    # 4e-3 relative error per layer against 6e-7 on the CPU (RTX 5090, torch 2.14.1+cu130,
    # cuDNN 9.24), and a rotated input lands on other tiles, so the logits move by up to
    # 1.6e-5 (p4) and 8e-6 (p4m) under `ieee` (2.9e-5 and 1.7e-5 under torch's default
    # TF32). The model is invariant there too: float64 gives 7e-16, and float32 with
    # cuDNN disabled 3.5e-7 on that layer. MPS has no float64 and stays in float32.
    device = torch.device(accelerator)
    dtype = torch.float64 if accelerator == "cuda" else torch.float32
    model = _variant(arch, None, "avgpool63").to(device=device, dtype=dtype)
    images, features = _inputs(6, dtype)
    changes = _max_change(model, images.to(device), features.to(device))
    assert changes["rotation"] < 1e-5
    if arch == "p4m":
        assert changes["flip"] < 1e-5


@pytest.mark.parametrize("k", [1, 2, 3])
def test_invariance_from_the_stored_stamp_through_the_input_path(k: int) -> None:
    # The whole pipeline: rotate the 63x63 stamp, then crop, pool the context, normalise.
    rng = np.random.default_rng(0)
    stamps = rng.uniform(0.0, 1.0, size=(4, 63, 63, 3)).astype(np.float32)
    stamps[:, 10:14, 50:55, 1] = 0  # former NaNs (the approximation must commute too)

    def model_input(batch: np.ndarray) -> torch.Tensor:
        both = np.concatenate((crop_then_normalise(batch), context_of(batch)), axis=-1)
        return torch.from_numpy(np.ascontiguousarray(both.transpose(0, 3, 1, 2)))

    features = torch.from_numpy(rng.standard_normal((4, 23)).astype(np.float32))
    model = _variant("p4", None, "avgpool63")
    with torch.no_grad():
        reference = model(model_input(stamps), features)
        rotated = model(model_input(np.rot90(stamps, k, axes=(1, 2))), features)
    assert float((rotated - reference).abs().max()) < 1e-5


# --- 3. Counts, initialisation, head, errors ---------------------------------------------


@pytest.mark.parametrize(
    ("arch", "fields", "context", "with_features", "parameters", "macs"),
    [
        ("rot4", None, "none", True, 441_075, 210_577_152),
        ("rot4", None, "none", False, 439_557, 210_575_680),
        ("single", None, "none", True, 441_075, 52_651_776),
        ("p4", None, "none", True, 228_531, 53_094_144),
        ("p4", None, "none", False, 227_013, 53_092_672),
        ("p4", 16, "none", True, 587_635, 209_397_504),
        ("p4m", None, "none", True, 193_107, 53_683_968),
        ("p4", None, "avgpool63", True, 228_915, 53_978_880),
        ("rot4", None, "avgpool63", True, 442_611, 214_116_096),
    ],
)
def test_parameter_and_mac_counts(
    arch: str,
    fields: int | None,
    context: str,
    with_features: bool,
    parameters: int,
    macs: int,
) -> None:
    model = build_paper_model(
        arch, group_fields=fields, context=context, with_features=with_features
    )
    assert count_parameters(model) == parameters
    assert count_macs(model, in_channels=paper_in_channels(context)) == macs


def test_mac_count_is_the_analytic_one_and_keeps_the_mode() -> None:
    # The paper model: 4 passes of conv1-conv5 and fc1 (on 24, 24, 12, 12, 12 and 6 px),
    # then the head (87 -> 64 -> 64 -> 5).
    trunk = (
        32 * 3 * 16 * 24**2
        + 32 * 32 * 25 * 24**2
        + 64 * 32 * 25 * 12**2
        + 2 * 64 * 64 * 25 * 12**2
        + 2304 * 64
    )
    head = 87 * 64 + 64 * 64 + 64 * 5
    model = PaperStampClassifier().train()
    assert count_macs(model) == 4 * trunk + head
    assert model.training
    # p4 B7: the expanded kernels have the paper's shapes, one pass; the readout is
    # 256 x (64 * 36), and the context adds conv1's 3 extra channels (+1.7 %).
    p4 = trunk - 2304 * 64 + 256 * 64 * 36 + head
    assert count_macs(build_paper_model("p4")) == p4
    with_context = count_macs(build_paper_model("p4", context="avgpool63"), in_channels=6)
    assert with_context - p4 == 32 * 3 * 16 * 24**2
    assert 0.016 < with_context / p4 - 1 < 0.018
    assert 3.9 < count_macs(PaperStampClassifier()) / p4 < 4.0


def test_group_model_initialisation() -> None:
    torch.manual_seed(0)
    model = GroupStampClassifier("p4")
    assert model.fields == (8, 16)
    assert isinstance(model.fc1, GroupReadout)
    for conv in (model.conv1, model.conv2, model.conv3, model.conv4, model.conv5):
        assert torch.all(conv.bias == 0)
        bound = math.sqrt(6 / sum(conv.fans()))
        assert bound * 0.9 < conv.weight.abs().max() <= bound
    assert model.fc1.fans() == (2304, 256)
    limit = math.sqrt(3 / 64)  # the paper's Glorot fc1 bias
    assert 0.5 * limit < model.fc1.bias.abs().max() <= limit
    head = model.head
    assert head.bn_features is not None
    assert (head.bn_features.eps, head.bn_features.momentum) == (1e-3, 0.0)
    assert torch.all(head.logits.bias == 0)
    torch.manual_seed(0)
    again = GroupStampClassifier("p4").state_dict()
    assert all(torch.equal(again[key], value) for key, value in model.state_dict().items())
    assert default_group_fields("p4") == 8
    assert default_group_fields("p4m") == 4
    assert GroupStampClassifier("p4m").fields == (4, 8)


@pytest.mark.parametrize("with_features", [True, False])
def test_paper_head_is_the_paper_models_head(with_features: bool) -> None:
    paper = _randomised(PaperStampClassifier(with_features=with_features))
    head = PaperHead(with_features=with_features, dropout_active=False).eval()
    head.load_state_dict(
        {
            key: value
            for key, value in paper.state_dict().items()
            if key.split(".")[0] in ("bn_features", "fc2", "fc3", "logits")
        }
    )
    h = torch.randn(5, 64)
    features = torch.randn(5, 23)
    with torch.no_grad():
        torch.testing.assert_close(head(h, features), paper._head(h, features), rtol=0, atol=0)


def test_dropout_of_the_group_model() -> None:
    images, features = _inputs(3, torch.float32)
    inactive = GroupStampClassifier("p4").train()
    assert torch.equal(inactive(images, features), inactive(images, features))
    torch.manual_seed(0)
    active = GroupStampClassifier("p4", dropout_active=True, drop_rate=0.5).train()
    assert not torch.equal(active(images, features), active(images, features))
    active.eval()
    assert torch.equal(active(images, features), active(images, features))


def test_without_features_the_group_model_ignores_them() -> None:
    model = GroupStampClassifier("p4m", with_features=False).eval()
    images, features = _inputs(3, torch.float32)
    assert model.head.bn_features is None
    assert torch.equal(model(images), model(images, features))


def test_model_errors() -> None:
    with pytest.raises(ValueError, match="paper_arch must be one of"):
        build_paper_model("p8")
    with pytest.raises(ValueError, match="context must be one of"):
        build_paper_model("p4", context="avgpool41")
    with pytest.raises(ValueError, match="group_fields applies to p4 and p4m only"):
        build_paper_model("rot4", group_fields=8)
    with pytest.raises(ValueError, match="only the G-CNNs"):
        default_group_fields("rot4")
    with pytest.raises(ValueError, match="group_fields must be >= 1"):
        GroupStampClassifier("p4", group_fields=0)
    with pytest.raises(ValueError, match="drop_rate"):
        GroupStampClassifier("p4", drop_rate=1.0)
    for factory in (
        lambda: GroupStampClassifier(in_channels=0),
        lambda: PaperStampClassifier(in_channels=0),
    ):
        with pytest.raises(ValueError, match="in_channels"):
            factory()
    model = GroupStampClassifier("p4", in_channels=6)
    with pytest.raises(
        ValueError, match=r"GroupStampClassifier needs images of shape \(N, 6, 21, 21\)"
    ):
        model(torch.zeros(2, 3, 21, 21), torch.zeros(2, 23))
    with pytest.raises(ValueError, match="x_feat"):
        model(torch.zeros(2, 6, 21, 21))
    assert PAPER_ARCHS == ("rot4", "single", "p4", "p4m", "convnext_atto", "convnext_pico")
    assert CONTEXTS == ("none", "avgpool63")


# --- Settings, variants, data path and training ------------------------------------------


def _preset(**changes: object) -> Settings:
    return settings_from_mapping({**PRESETS[CARRASCO2021_PRESET], **changes})


def test_settings_validate_the_variant_flags() -> None:
    assert _preset(paper_arch="p4", group_fields=16).group_fields == 16
    with pytest.raises(ConfigError, match="paper_arch 'p8' is not one of"):
        _preset(paper_arch="p8")
    with pytest.raises(ConfigError, match="context 'avgpool41' is not one of"):
        _preset(context="avgpool41")
    with pytest.raises(ConfigError, match="group_fields applies to paper_arch p4 or p4m only"):
        _preset(group_fields=8)
    with pytest.raises(ConfigError, match="group_fields must be >= 1"):
        _preset(paper_arch="p4m", group_fields=0)
    with pytest.raises(ConfigError, match="needs paper_input=float32"):
        _preset(context="avgpool63", paper_input="uint8")


@pytest.mark.parametrize("name", list(VARIANTS))
def test_every_named_variant_builds(name: str) -> None:
    variant = VARIANTS[name]
    module = _module(CARRASCO2021_PRESET, **variant.overrides)
    settings = _preset(**variant.overrides)
    assert module.rot90_augment == settings.rot90_augment
    channels = paper_in_channels(settings.context)
    side = paper_input_side(settings.paper_arch)
    assert settings.image_size == side
    assert module(torch.zeros(2, channels, side, side), torch.zeros(2, 23)).shape == (2, 5)
    assert variant.description


def test_rot90_augmentation_is_seeded_by_step_and_rotates_training_batches() -> None:
    module = _module(CARRASCO2021_PRESET, paper_arch="single", rot90_augment=True, seed=3)
    first = module.rotation_steps(64, 5)
    assert torch.equal(first, module.rotation_steps(64, 5))
    assert not torch.equal(first, module.rotation_steps(64, 6))
    assert set(first.tolist()) == {0, 1, 2, 3}
    other_seed = _module(CARRASCO2021_PRESET, paper_arch="single", rot90_augment=True, seed=4)
    assert not torch.equal(first, other_seed.rotation_steps(64, 5))
    with pytest.raises(ValueError, match="augment_seed"):
        PaperClassifierModule(beta_loss=0.5, lr=1e-3, optimizer="Adam", augment_seed=-1)

    images, features = _inputs(3, torch.float32, n=6)
    targets = torch.eye(5)[torch.tensor([0, 1, 2, 3, 4, 0])]
    k = module.rotation_steps(6, 0)  # no Trainer: global step 0
    expected = torch.stack([torch.rot90(images[i], int(k[i]), dims=(1, 2)) for i in range(6)])
    module.train()
    loss = module.training_step((images, features, targets), 0)
    torch.testing.assert_close(loss, module.compute_loss(module(expected, features), targets))
    plain = _module(CARRASCO2021_PRESET, paper_arch="single")
    plain.load_state_dict(module.state_dict())
    plain.train()
    torch.testing.assert_close(
        plain.training_step((images, features, targets), 0),
        plain.compute_loss(plain(images, features), targets),
    )


def test_tensor_split_takes_the_context_channels() -> None:
    labels = np.zeros(2, dtype=np.int64)
    features = np.zeros((2, 23), dtype=np.float32)
    split = TensorSplit(np.zeros((2, 6, 21, 21), dtype=np.float32), features, labels)
    assert split.images.shape[1] == 6
    with pytest.raises(ValueError, match=r"\(N, 3 or 6, 21, 21\) or \(N, 3, 63, 63\) with N = 2"):
        TensorSplit(np.zeros((2, 4, 21, 21), dtype=np.float32), features, labels)
    # Whole stamps (plan row B8) have 3 channels only.
    assert TensorSplit(np.zeros((2, 3, 63, 63), dtype=np.float32), features, labels)
    with pytest.raises(ValueError, match=r"TensorSplit\.images must be"):
        TensorSplit(np.zeros((2, 6, 63, 63), dtype=np.float32), features, labels)


def test_float_crops_with_context(classifier_f32_dir: Path) -> None:
    plain = float_crops(classifier_f32_dir)
    both = float_crops(classifier_f32_dir, context=True)
    from astro_simclr.data.prepared import load_prepared

    stored = load_prepared(classifier_f32_dir).splits["Train"].images_f32
    assert stored is not None
    train = both["Train"].images
    assert train.shape[1:] == (6, 21, 21)
    np.testing.assert_array_equal(train[:, :3], plain["Train"].images)
    np.testing.assert_array_equal(train[:, :3], crop_split(stored).transpose(0, 3, 1, 2))
    np.testing.assert_array_equal(train[:, 3:], context_split(stored).transpose(0, 3, 1, 2))
    np.testing.assert_array_equal(both["Test"].labels, plain["Test"].labels)


def _smoke_settings(classifier_f32_dir: Path, out: Path, **changes: Any) -> Settings:
    return _preset(
        accelerator="cpu",
        deterministic=True,
        data_dir=classifier_f32_dir.parent,
        dataset=classifier_f32_dir.name,
        output_dir=out,
        batch_size=4,
        max_steps=4,
        val_every_n_steps=2,
        horizon_increment=0,
        checkpoint_every_n_steps=2,
    ).replace(**changes)


@pytest.mark.parametrize("accelerator", ACCELERATORS)
@pytest.mark.parametrize(
    "overrides",
    [VARIANTS["p4-context"].overrides, VARIANTS["single-aug"].overrides, VARIANTS["p4m"].overrides],
    ids=["p4-context", "single-aug", "p4m"],
)
def test_variants_train_through_run_command(
    overrides: dict[str, object], accelerator: str, classifier_f32_dir: Path, tmp_path: Path
) -> None:
    settings = _smoke_settings(classifier_f32_dir, tmp_path, accelerator=accelerator, **overrides)
    metrics = run_command("train-supervised", settings, progress_bar=False)
    assert metrics["last_step"] == 4
    for key, value in overrides.items():
        assert metrics["config"][key] == value
    assert metrics["preset"]["faithful"] is False
    smoke = {"batch_size", "max_steps", "val_every_n_steps", "horizon_increment"}
    assert set(metrics["preset"]["deviations"]) - smoke == set(overrides)
    on_disk = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert on_disk["config"]["paper_arch"] == overrides["paper_arch"]
    # The best checkpoint loads back into the same variant.
    checkpoint = torch.load(tmp_path / "checkpoints" / "best.ckpt", weights_only=True)
    restored = PaperClassifierModule(**checkpoint["hyper_parameters"])
    restored.load_state_dict(checkpoint["state_dict"])


def test_variant_flags_need_the_paper_model(classifier_f32_dir: Path, tmp_path: Path) -> None:
    settings = Settings(
        accelerator="cpu",
        data_dir=classifier_f32_dir.parent,
        dataset=classifier_f32_dir.name,
        output_dir=tmp_path,
        paper_arch="p4",
        rot90_augment=True,
    )
    with pytest.raises(ConfigError, match="paper_arch, rot90_augment apply only to"):
        run_command("train-supervised", settings, progress_bar=False)
