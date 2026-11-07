"""Tests for the ported networks and the encoder factory (astro_simclr.models, task 1.4).

The characterization fixtures in tests/fixtures/models/ come from the legacy networks
(scripts/characterize/models/generate.py). Both sides are filled with the deterministic
weights of tests/model_weights.py and run in eval mode; outputs must match with the
conv-net tolerances of the plan (rtol=1e-4, atol=1e-5).
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from model_weights import fill_deterministic
from torch import nn

from astro_simclr.models import (
    ENCODER_NAMES,
    Identity,
    LinearClassifier,
    PStampsEncoder,
    SimCLRNet,
    StampClassifier,
    StampEncoder,
    ThreeLayerClassifier,
    build_encoder,
)

FIXTURES = Path(__file__).parent / "fixtures" / "models"
RTOL, ATOL = 1e-4, 1e-5


def _resnet18() -> nn.Module:
    return build_encoder("resnet18").module


#: fixture name -> factory of the ported network (same arguments as the generator).
CASES: dict[str, Callable[[], nn.Module]] = {
    "stamp_encoder_k1": lambda: StampEncoder(k=1.0),
    "stamp_encoder_k0_5": lambda: StampEncoder(k=0.5),
    "stamp_encoder_k1_5": lambda: StampEncoder(k=1.5),
    "pstamps": lambda: PStampsEncoder(k=1.0),
    "stamp_classifier_features": lambda: StampClassifier(0.2, with_features=True),
    "stamp_classifier_no_features": lambda: StampClassifier(0.2, with_features=False),
    "linear_classifier": lambda: LinearClassifier(512, 5),
    "three_layers_classifier": lambda: ThreeLayerClassifier(151, 64, 64, 5, 0.2),
    "clr_pstamps": lambda: SimCLRNet(PStampsEncoder(k=1.0), 128, 300),
    "resnet18_27": _resnet18,
    "resnet18_63": _resnet18,
}


def _params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _init_fingerprint(factory: Callable[[], nn.Module]) -> np.ndarray:
    torch.manual_seed(0)
    state = factory().state_dict()
    return np.array(
        [state[k].double().sum().item() for k in sorted(state) if state[k].is_floating_point()]
    )


@pytest.mark.parametrize("case", sorted(CASES))
def test_matches_legacy_fixture(case: str) -> None:
    fixture = np.load(FIXTURES / f"{case}.npz")
    assert str(fixture["legacy_sha"]) in {
        "b357b4390d9f77e74bfdbbf88193299dd5edfc43",
        "c5c7e6f3ccbcd9306267e4e307d7ac7aac8aeb5a",
    }
    net = CASES[case]()
    assert _params(net) == int(fixture["n_params"])
    fill_deterministic(net)
    net.eval()
    inputs = [torch.from_numpy(fixture[k]) for k in sorted(fixture) if k.startswith("input_")]
    with torch.no_grad():
        out = net(*inputs)
    outputs = out if isinstance(out, tuple) else (out,)
    expected = sorted(k for k in fixture if k.startswith("output_"))
    assert len(outputs) == len(expected)
    for got, key in zip(outputs, expected, strict=True):
        np.testing.assert_allclose(got.numpy(), fixture[key], rtol=RTOL, atol=ATOL)


#: Cases whose default init draws only from ``uniform_`` (Conv2d/Linear defaults), which
#: gives the same values on x86 and arm64 for a seed. The ResNets are left out: torchvision
#: initialises their convolutions with ``kaiming_normal_``, and torch's vectorised CPU
#: ``normal_`` (AVX vs NEON) is not bit-identical across architectures, so an arm64 fixture
#: cannot pin their seeded init on x86 CI. Their parameter count and fixed-weight forward
#: pass are still checked against the legacy fixtures.
PORTABLE_INIT_CASES = sorted(case for case in CASES if not case.startswith("resnet"))


@pytest.mark.parametrize("case", PORTABLE_INIT_CASES)
def test_seeded_init_matches_legacy(case: str) -> None:
    # Modules are created in the legacy order, so a seed gives the legacy initial weights.
    fixture = np.load(FIXTURES / f"{case}.npz")
    np.testing.assert_allclose(
        _init_fingerprint(CASES[case]), fixture["init_fingerprint"], rtol=1e-6, atol=1e-6
    )


def test_state_dict_keys_match_legacy_names() -> None:
    assert list(StampEncoder().state_dict())[:2] == ["conv1.weight", "conv1.bias"]
    assert "fc1.bias" not in PStampsEncoder().state_dict()
    assert "bn1.running_mean" in PStampsEncoder().state_dict()
    assert list(LinearClassifier(4, 2).state_dict()) == ["model.weight", "model.bias"]
    assert set(SimCLRNet(StampEncoder(), 128, 8).state_dict()) >= {
        "encoder.fc1.weight",
        "projector.4.bias",
    }


@pytest.mark.parametrize(
    ("name", "n_params"),
    [("resnet34", "resnet34"), ("resnet50", "resnet50"), ("resnet152", "resnet152")],
)
def test_resnet_param_counts_match_legacy(name: str, n_params: str) -> None:
    counts = np.load(FIXTURES / "param_counts.npz")
    assert _params(build_encoder(name).module) == int(counts[n_params])


def test_clr_stamp1_param_count_matches_legacy() -> None:
    counts = np.load(FIXTURES / "param_counts.npz")
    encoder = build_encoder("stamp1")
    assert _params(SimCLRNet(encoder.module, encoder.out_features, 300)) == int(
        counts["clr_stamp1"]
    )


@pytest.mark.parametrize(
    ("name", "out_features", "cls"),
    [
        ("stamp1", 128, StampEncoder),
        ("stamp0.5", 64, StampEncoder),
        ("stamp1.5", 192, StampEncoder),
        ("stamp2", 256, StampEncoder),
        ("stamp.5", 64, StampEncoder),
        ("pstamps", 128, PStampsEncoder),
    ],
)
def test_build_stamp_encoders(name: str, out_features: int, cls: type[nn.Module]) -> None:
    encoder = build_encoder(name)
    assert type(encoder.module) is cls
    assert encoder.out_features == out_features
    encoder.module.eval()
    assert encoder.module(torch.rand(2, 3, 27, 27)).shape == (2, out_features)


@pytest.mark.parametrize(
    ("name", "out_features"),
    [("resnet18", 512), ("resnet34", 512), ("resnet50", 2048), ("resnet152", 2048)],
)
@pytest.mark.parametrize("size", [27, 63])
def test_build_resnets(name: str, out_features: int, size: int) -> None:
    encoder = build_encoder(name)
    assert isinstance(encoder.module.fc, Identity)
    assert encoder.out_features == out_features
    encoder.module.eval()
    with torch.no_grad():
        assert encoder.module(torch.rand(2, 3, size, size)).shape == (2, out_features)


@pytest.mark.parametrize("name", ["stamp", "stampx", "pstamp", "stamps1", "resnet101", "", "1"])
def test_unknown_encoder_lists_valid_names(name: str) -> None:
    with pytest.raises(ValueError, match="unknown encoder") as info:
        build_encoder(name)
    for valid in ENCODER_NAMES:
        assert valid in str(info.value)


@pytest.mark.parametrize("name", ["stamp0", "stamp0.01"])
def test_too_narrow_stamp_encoder_fails(name: str) -> None:
    with pytest.raises(ValueError, match="width k"):
        build_encoder(name)


def test_too_narrow_pstamps_fails() -> None:
    with pytest.raises(ValueError, match="width k"):
        PStampsEncoder(k=0)


@pytest.mark.parametrize("with_features", [True, False])
def test_stamp_classifier_shapes(with_features: bool) -> None:
    net = StampClassifier(0.5, with_features=with_features).eval()
    feat = torch.randn(3, 23) if with_features else None
    assert net(torch.rand(3, 3, 27, 27), feat).shape == (3, 5)


def test_stamp_classifier_without_features_ignores_them() -> None:
    net = StampClassifier(0.0, with_features=False).eval()
    img = torch.rand(2, 3, 27, 27)
    assert torch.equal(net(img), net(img, torch.randn(2, 23)))


@pytest.mark.parametrize("feat", [None, torch.randn(2, 22), torch.randn(3, 23)])
def test_stamp_classifier_rejects_bad_features(feat: torch.Tensor | None) -> None:
    net = StampClassifier(0.0, with_features=True)
    with pytest.raises(ValueError, match="x_feat"):
        net(torch.rand(2, 3, 27, 27), feat)


@pytest.mark.parametrize(
    "net",
    [StampEncoder(), PStampsEncoder(), StampClassifier(0.0, with_features=False)],
    ids=["stamp", "pstamps", "classifier"],
)
@pytest.mark.parametrize("shape", [(2, 3, 63, 63), (2, 3, 28, 28), (2, 1, 27, 27), (3, 27, 27)])
def test_stamp_networks_reject_other_sizes(net: nn.Module, shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="27"):
        net(torch.rand(*shape))


def test_heads_shapes() -> None:
    assert LinearClassifier(10, 5)(torch.randn(4, 10)).shape == (4, 5)
    assert ThreeLayerClassifier(10, 8, 6, 5, 0.1)(torch.randn(4, 10)).shape == (4, 5)
    assert torch.equal(Identity()(x := torch.randn(2, 3)), x)


def test_simclr_net_normalises_before_projection() -> None:
    encoder = build_encoder("stamp1")
    net = SimCLRNet(encoder.module, encoder.out_features, 16).eval()
    x_i, x_j = torch.rand(2, 3, 27, 27), torch.rand(2, 3, 27, 27)
    h_i, h_j, z_i, z_j = net(x_i, x_j)
    assert h_i.shape == h_j.shape == (2, 128)
    assert z_i.shape == z_j.shape == (2, 16)
    # Scaling h does not change z: the projector sees h / ||h||.
    torch.testing.assert_close(net.projector(h_i / h_i.norm(dim=1, keepdim=True)), z_i)
