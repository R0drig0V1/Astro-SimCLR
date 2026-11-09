"""Tests of the conv-attention hybrid stamp classifiers (``hybrid_model``; plan row B9).

1. **Shapes:** the trunks' 6x6 grid, the tokens, the logits and the representation.
2. **Exact C4 invariance** of the full models in float64, with every parameter perturbed
   (both trunks, both readouts, with the context channels); a flip, which the models are
   not invariant to, changes the logits. For ``hybrid`` one view alone is *not* invariant
   (the free positional table), so the 4-rotation mean is load-bearing; for ``hybrid_p4``
   one view already is (the orientation mean and the C4-tied positions).
3. The building blocks: the C4 orbits, the tied positional table, the encoder block's
   permutation equivariance.
4. Parameter and MAC counts (pinned, next to the paper model's), initialisation, dropout,
   ``feature_set``, the input and construction checks.
5. The settings and the module: the flags only for a hybrid, the preset deviations, the
   hyperparameters (defaults left out) and a checkpoint round trip. The default flags
   keeping the faithful model bit-identical (the golden fixture) is
   ``tests/test_paper_variants.py``; the CPU split runs are in ``tests/test_split_runs.py``.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest
import torch
from torch import nn

from astro_simclr.config import ConfigError, Settings, settings_from_mapping
from astro_simclr.experiments import paper_module
from astro_simclr.hybrid_model import (
    GRID_SIDE,
    EncoderBlock,
    GridPositions,
    HybridStampClassifier,
    c4_orbits,
    check_hybrid,
)
from astro_simclr.paper_variants import (
    VARIANTS,
    build_paper_model,
    count_macs,
    count_parameters,
)
from astro_simclr.presets import CARRASCO2021_PRESET, PRESETS, preset_report
from astro_simclr.training.modules import PaperClassifierModule

ARCHS = ("hybrid", "hybrid_p4")

#: (parameters, MACs per stamp from count_macs, views) with the 23 features, at the
#: defaults (dim 64, depth 2, heads 4, mean readout; hybrid_p4 at 8/16 fields).
_COUNTS = {
    "hybrid": (371_251, 221_357_824, 4),
    "hybrid_p4": (153_907, 55_246_720, 1),
}
#: The paper model's (rot4) parameters and MACs per stamp (4 views), for comparison.
_PAPER = (441_075, 210_577_152)


def _model(arch: str = "hybrid", seed: int = 0, **kwargs: Any) -> HybridStampClassifier:
    torch.manual_seed(seed)
    model = build_paper_model(arch, **kwargs)
    assert isinstance(model, HybridStampClassifier)
    return model


def _inputs(
    n: int = 3, channels: int = 3, dtype: torch.dtype = torch.float64
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    images = torch.rand(n, channels, 21, 21, generator=generator, dtype=dtype)
    return images, torch.randn(n, 23, generator=generator, dtype=dtype)


def _perturbed[M: nn.Module](model: M) -> M:
    """Every parameter moved off its initialisation (zero biases, unit norms would hide a break)."""
    generator = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for parameter in model.parameters():
            noise = torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
            parameter.add_(0.1 * noise)
    return model


# --- 1. Shapes -------------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("readout", ["mean", "cls"])
def test_forward_shapes(arch: str, readout: str) -> None:
    model = _model(arch, n_classes=7, hybrid_readout=readout).eval()
    images, features = _inputs(2, dtype=torch.float32)
    channels = 64 if arch == "hybrid" else 16
    with torch.no_grad():
        assert model.grid(images).shape == (2, channels, GRID_SIDE, GRID_SIDE)
        assert model(images, features).shape == (2, 7)
        assert model.representation(images).shape == (2, 64)
    assert model.views == (4 if arch == "hybrid" else 1)
    assert (model.cls_token is not None) == (readout == "cls")
    assert len(model.blocks) == 2


def test_the_shape_flags_shape_the_model() -> None:
    model = _model("hybrid_p4", group_fields=4, hybrid_dim=48, hybrid_depth=3, hybrid_heads=6)
    assert model.embed.in_features == 8  # 2F orientation-pooled fields, F = 4
    assert model.embed.out_features == 48
    assert len(model.blocks) == 3
    block = model.blocks[0]
    assert isinstance(block, EncoderBlock)
    assert block.heads == 6
    assert model.positions.table.shape == (9, 48)  # the 9 C4 orbits of the 6x6 grid
    assert _model("hybrid").positions.table.shape == (36, 64)  # one vector per cell


# --- 2. Invariance ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("readout", ["mean", "cls"])
@pytest.mark.parametrize("channels", [3, 6], ids=["crop", "context"])
def test_rotation_invariance_is_exact_and_a_flip_is_visible(
    arch: str, readout: str, channels: int
) -> None:
    context = "avgpool63" if channels == 6 else "none"
    model = _perturbed(_model(arch, hybrid_readout=readout, context=context).double()).eval()
    images, features = _inputs(channels=channels)
    with torch.no_grad():
        logits = model(images, features)
        for k in (1, 2, 3):
            rotated = model(torch.rot90(images, k, dims=(2, 3)), features)
            torch.testing.assert_close(rotated, logits, rtol=0, atol=1e-10)
        flipped = model(images.flip(-1), features)
    assert (flipped - logits).abs().max() > 1e-4


@pytest.mark.parametrize("readout", ["mean", "cls"])
def test_one_view_is_invariant_only_with_the_p4_trunk(readout: str) -> None:
    images, _ = _inputs()
    rotated = torch.rot90(images, 1, dims=(2, 3))
    with torch.no_grad():
        # hybrid: the free positional table breaks the symmetry of one view; the mean over
        # the 4 rotations restores it (the test above).
        plain = _perturbed(_model("hybrid", hybrid_readout=readout).double()).eval()
        assert (plain.encode(rotated) - plain.encode(images)).abs().max() > 1e-4
        # hybrid_p4: the orientation mean and the tied positions make one view invariant.
        p4 = _perturbed(_model("hybrid_p4", hybrid_readout=readout).double()).eval()
        torch.testing.assert_close(p4.encode(rotated), p4.encode(images), rtol=0, atol=1e-10)


@pytest.mark.parametrize("arch", ARCHS)
def test_invariance_in_float32(arch: str) -> None:
    model = _model(arch).eval()
    images, features = _inputs(4, dtype=torch.float32)
    with torch.no_grad():
        logits = model(images, features)
        rotated = model(torch.rot90(images, 3, dims=(2, 3)), features)
    torch.testing.assert_close(rotated, logits, rtol=0, atol=1e-5)


# --- 3. Building blocks ----------------------------------------------------------------------


@pytest.mark.parametrize(("side", "orbits"), [(6, 9), (5, 7), (2, 1), (1, 1)])
def test_c4_orbits(side: int, orbits: int) -> None:
    orbit = c4_orbits(side)
    assert orbit.dtype == torch.int64
    assert orbit.shape == (side * side,)
    assert int(orbit.max()) + 1 == orbits
    grid = orbit.view(side, side)
    for k in (1, 2, 3):
        assert torch.equal(torch.rot90(grid, k), grid)
    counts = torch.bincount(orbit)
    # Orbits of 4 cells, and the centre of an odd grid on its own.
    assert set(counts.tolist()) <= {1, 4}
    assert int((counts == 1).sum()) == side % 2
    assert int(orbit[0]) == 0  # numbered by their smallest cell
    with pytest.raises(ValueError, match="side must be >= 1"):
        c4_orbits(0)


def test_tied_positions_are_c4_invariant_and_untied_ones_are_not() -> None:
    torch.manual_seed(0)
    tied = GridPositions(GRID_SIDE, 8, tied_c4=True)
    tied.reset_parameters()
    untied = GridPositions(GRID_SIDE, 8, tied_c4=False)
    untied.reset_parameters()
    assert untied.assign is None
    for positions, invariant in ((tied, True), (untied, False)):
        grid = positions().detach().view(GRID_SIDE, GRID_SIDE, 8).permute(2, 0, 1)
        same = all(torch.equal(torch.rot90(grid, k, dims=(1, 2)), grid) for k in (1, 2, 3))
        assert same is invariant
    # Each cell's vector is exactly its orbit's row (a copy, not a sum).
    assert torch.equal(tied(), tied.table.detach()[c4_orbits(GRID_SIDE)])
    # The table's gradient is the sum over the orbit's cells (the one-hot product).
    tied().sum(dim=1).pow(2).sum().backward()
    assert tied.table.grad is not None
    assert torch.all(tied.table.grad.abs() > 0)


def test_the_encoder_block_is_permutation_equivariant() -> None:
    torch.manual_seed(0)
    block = _perturbed(EncoderBlock(16, 4).double()).eval()
    tokens = torch.randn(2, 37, 16, dtype=torch.float64)
    order = torch.randperm(37)
    with torch.no_grad():
        out = block(tokens)
        torch.testing.assert_close(block(tokens[:, order]), out[:, order], rtol=0, atol=1e-12)
    assert out.shape == tokens.shape


def test_attention_is_scaled_dot_product_attention() -> None:
    torch.manual_seed(0)
    block = _perturbed(EncoderBlock(16, 4).double()).eval()
    tokens = torch.randn(2, 5, 16, dtype=torch.float64)
    attention = nn.MultiheadAttention(16, 4, batch_first=True, dtype=torch.float64)
    with torch.no_grad():
        attention.in_proj_weight.copy_(block.qkv.weight)
        attention.in_proj_bias.copy_(block.qkv.bias)
        attention.out_proj.weight.copy_(block.proj.weight)
        attention.out_proj.bias.copy_(block.proj.bias)
        normed = block.norm1(tokens)
        expected = tokens + attention(normed, normed, normed, need_weights=False)[0]
        expected = expected + block.fc_out(
            torch.nn.functional.gelu(block.fc_in(block.norm2(expected)))
        )
        torch.testing.assert_close(block(tokens), expected, rtol=0, atol=1e-12)


# --- 4. Counts, initialisation, head, checks -------------------------------------------------


@pytest.mark.parametrize("arch", ARCHS)
def test_parameter_and_mac_counts(arch: str) -> None:
    model = _model(arch)
    parameters, macs, views = _COUNTS[arch]
    assert count_parameters(model) == parameters
    assert count_macs(model) == macs
    assert model.views == views
    # Without features: the feature BN (46) and fc2's 23 x 64 feature weights go.
    assert count_parameters(_model(arch, feature_set="none")) == parameters - 46 - 23 * 64
    # Next to the paper model: fewer parameters, about its MACs per view (52.6 M).
    paper = build_paper_model("rot4")
    assert (count_parameters(paper), count_macs(paper)) == _PAPER
    assert parameters < _PAPER[0]
    assert 1.0 < (macs / views) / (_PAPER[1] / 4) < 1.1


def test_the_attention_part_costs_what_its_shape_says() -> None:
    # Each block: qkv (3 dim^2 per token), proj (dim^2) and the MLP (4 dim^2 at MLP ratio
    # 2), plus the two attention products (2 T^2 dim); the 36 tokens' embedding; fc1. Per
    # view, then the head (fc2, fc3, logits) once per stamp.
    dim, tokens = 64, GRID_SIDE * GRID_SIDE
    head = 87 * 64 + 64 * 64 + 64 * 5
    convs = count_macs(build_paper_model("single")) - 64 * tokens * 64 - head  # minus fc1
    block = tokens * 8 * dim * dim + 2 * tokens * tokens * dim
    attention = tokens * 64 * dim + 2 * block + dim * 64
    assert count_macs(_model("hybrid")) == 4 * (convs + attention) + head
    # The CLS token adds one token to every block.
    cls_block = (tokens + 1) * 8 * dim * dim + 2 * (tokens + 1) ** 2 * dim
    cls = tokens * 64 * dim + 2 * cls_block + dim * 64
    assert count_macs(_model("hybrid", hybrid_readout="cls")) == 4 * (convs + cls) + head


def test_initialisation() -> None:
    model = _model("hybrid_p4", hybrid_readout="cls")
    for parameter in (model.positions.table, model.cls_token, model.embed.weight):
        assert parameter is not None
        tensor = parameter.detach()
        assert float(tensor.abs().max()) <= 0.04  # truncated at 2 sigma, sigma 0.02
        assert 0.01 < float(tensor.std()) < 0.02
    block = model.blocks[0]
    assert isinstance(block, EncoderBlock)
    for linear in (block.qkv, block.proj, block.fc_in, block.fc_out, model.embed):
        assert linear.bias is not None
        assert not linear.bias.any()
    assert torch.equal(block.norm1.weight, torch.ones(64))
    assert float(model.fc1.bias.abs().max()) <= (3 / 64) ** 0.5
    assert model.fc1.bias.any()
    for trunk in (model.trunk, _model("hybrid").trunk):  # LiftingConv, then nn.Conv2d
        bias = trunk.conv1.bias
        assert isinstance(bias, torch.Tensor)
        assert not bias.any()


def test_the_seed_fixes_the_initialisation() -> None:
    for arch in ARCHS:
        first, second, other = _model(arch, seed=3), _model(arch, seed=3), _model(arch, seed=4)
        for key, value in first.state_dict().items():
            assert torch.equal(value, second.state_dict()[key]), key
        assert not torch.equal(first.positions.table, other.positions.table)


def test_dropout_active_applies_only_in_train_mode() -> None:
    model = _model(dropout_active=True, drop_rate=0.5)
    images, features = _inputs(2, dtype=torch.float32)
    with torch.no_grad():
        model.eval()
        torch.testing.assert_close(model(images, features), model(images, features))
        model.train()
        torch.manual_seed(0)
        first = model(images, features)
        torch.manual_seed(1)
        assert not torch.equal(first, model(images, features))
        # Inactive dropout (the paper's training): train mode is deterministic.
        inactive = _model().train()
        torch.testing.assert_close(inactive(images, features), inactive(images, features))


def test_feature_set_none_is_the_model_without_features() -> None:
    images, features = _inputs(2)
    none = _model(feature_set="none").double().eval()
    without = _model(with_features=False).double().eval()
    assert not none.with_features
    with torch.no_grad():
        torch.testing.assert_close(none(images, features), without(images, None), rtol=0, atol=0)
    position = _model("hybrid_p4", feature_set="position").eval()
    assert position.head.bn_features is not None
    assert position.head.bn_features.num_features == 6


def test_input_and_construction_checks() -> None:
    model = _model()
    with pytest.raises(ValueError, match=r"needs images of shape \(N, 3, 21, 21\)"):
        model(torch.zeros(1, 3, 63, 63), torch.zeros(1, 23))
    with pytest.raises(ValueError, match=r"needs images of shape"):
        model.grid(torch.zeros(3, 21, 21))
    with pytest.raises(ValueError, match="x_feat"):
        model(torch.zeros(1, 3, 21, 21))
    with pytest.raises(ValueError, match="arch must be one of"):
        HybridStampClassifier("hybrid_p4m")
    with pytest.raises(ValueError, match="drop_rate"):
        HybridStampClassifier(drop_rate=1.0)
    with pytest.raises(ValueError, match="in_channels"):
        HybridStampClassifier(in_channels=0)
    with pytest.raises(ValueError, match="group_fields applies to hybrid_p4, not to hybrid"):
        HybridStampClassifier("hybrid", group_fields=8)
    with pytest.raises(ValueError, match="group_fields must be >= 1"):
        HybridStampClassifier("hybrid_p4", group_fields=0)
    with pytest.raises(ValueError, match="hybrid_dim, hybrid_readout apply to the hybrids only"):
        build_paper_model("p4", hybrid_dim=32, hybrid_readout="cls")
    with pytest.raises(ValueError, match="group_fields applies to p4, p4m and hybrid_p4 only"):
        build_paper_model("hybrid", group_fields=8)
    assert (
        count_parameters(build_paper_model("hybrid_p4", group_fields=4)) < _COUNTS["hybrid_p4"][0]
    )


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        ({"dim": 0}, "hybrid_dim must be >= 1"),
        ({"depth": 0}, "hybrid_depth must be >= 1"),
        ({"heads": 0}, "hybrid_heads must be >= 1"),
        ({"dim": 30, "heads": 4}, r"hybrid_dim \(30\) must be a multiple of hybrid_heads \(4\)"),
        ({"readout": "max"}, "hybrid_readout must be one of mean, cls"),
    ],
)
def test_check_hybrid(flags: dict[str, Any], message: str) -> None:
    values: dict[str, Any] = {"dim": 64, "depth": 2, "heads": 4, "readout": "mean", **flags}
    with pytest.raises(ValueError, match=message):
        check_hybrid(**values)
    with pytest.raises(ValueError, match=message.split()[0]):
        HybridStampClassifier(**values)


def test_the_module_imports_no_lightning() -> None:
    code = (
        "import sys, astro_simclr.hybrid_model; "
        "bad = sorted(m for m in sys.modules if m.split('.')[0] == 'lightning'); "
        "assert not bad, bad"
    )
    # A fresh interpreter: this process has imported Lightning already (ADR 0014).
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603 (fixed argv)


# --- 5. Settings and the module --------------------------------------------------------------


def _preset(**changes: object) -> Settings:
    return settings_from_mapping({**PRESETS[CARRASCO2021_PRESET], **changes})


def test_settings_accept_the_flags_for_a_hybrid_only() -> None:
    settings = _preset(paper_arch="hybrid_p4", group_fields=16, hybrid_dim=96, hybrid_heads=6)
    assert (settings.group_fields, settings.hybrid_dim, settings.hybrid_heads) == (16, 96, 6)
    with pytest.raises(ConfigError, match="hybrid_depth apply to paper_arch hybrid or hybrid_p4"):
        _preset(hybrid_depth=3)
    with pytest.raises(ConfigError, match="hybrid_readout 'max' is not one of: mean, cls"):
        _preset(paper_arch="hybrid", hybrid_readout="max")
    with pytest.raises(ConfigError, match="hybrid_heads must be >= 1"):
        _preset(paper_arch="hybrid", hybrid_heads=0)
    with pytest.raises(ConfigError, match=r"hybrid_dim \(64\) must be a multiple of"):
        _preset(paper_arch="hybrid", hybrid_heads=5)
    with pytest.raises(
        ConfigError, match="group_fields applies to paper_arch p4, p4m or hybrid_p4"
    ):
        _preset(paper_arch="hybrid", group_fields=8)


def test_the_variants_are_preset_deviations() -> None:
    report = preset_report(_preset(**VARIANTS["hybrid-wide"].overrides))
    assert report is not None
    assert report["deviations"] == {
        "paper_arch": {"value": "hybrid", "preset": "rot4"},
        "hybrid_dim": {"value": 128, "preset": 64},
        "hybrid_depth": {"value": 4, "preset": 2},
    }
    for name in ("hybrid", "hybrid-cls", "hybrid-wide", "hybrid-p4"):
        assert VARIANTS[name].row == "B9"


def test_the_module_records_only_changed_flags_and_reloads() -> None:
    torch.manual_seed(0)
    default = paper_module(_preset(paper_arch="hybrid"))
    assert isinstance(default.model, HybridStampClassifier)
    assert not {key for key in default.hparams if key.startswith("hybrid_")}
    torch.manual_seed(0)
    changed = paper_module(_preset(paper_arch="hybrid", hybrid_readout="cls", hybrid_depth=3))
    assert changed.hparams["hybrid_readout"] == "cls"
    assert changed.hparams["hybrid_depth"] == 3
    assert "hybrid_dim" not in changed.hparams
    restored = PaperClassifierModule(**changed.hparams)
    restored.load_state_dict(changed.state_dict())
    images, features = _inputs(2, dtype=torch.float32)
    with torch.no_grad():
        torch.testing.assert_close(
            restored.eval()(images, features), changed.eval()(images, features), rtol=0, atol=0
        )
