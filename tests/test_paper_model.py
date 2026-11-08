"""Tests of ``paper_model.PaperStampClassifier``, the paper-faithful network (task 2.2a).

The facts checked here come from ALeRCE's code and M1's meta graph and checkpoint
(``docs/phases/02-stamp-classifier.md``): the parameter counts, the frozen feature
BatchNorm, the inactive dropout and the initialisation.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from astro_simclr.paper_model import PAPER_CROP, PaperStampClassifier

_BIAS_LIMIT = math.sqrt(6 / (64 + 64))  # M1's dense_*/bias/Initializer/random_uniform/max


def _inputs(n: int = 6, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    images = torch.rand(n, 3, PAPER_CROP, PAPER_CROP, generator=generator)
    features = torch.randn(n, 23, generator=generator)
    return images, features


def _trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize(("with_features", "expected"), [(True, 441_075), (False, 439_557)])
def test_parameter_counts_match_m1(with_features: bool, expected: int) -> None:
    assert _trainable(PaperStampClassifier(with_features=with_features)) == expected


def test_output_shape_and_input_checks() -> None:
    model = PaperStampClassifier()
    images, features = _inputs()
    assert model(images, features).shape == (6, 5)
    with pytest.raises(ValueError, match=r"\(N, 3, 21, 21\)"):
        model(torch.zeros(2, 3, 27, 27), features[:2])
    with pytest.raises(ValueError, match="x_feat"):
        model(images)
    with pytest.raises(ValueError, match="x_feat"):
        model(images, features[:, :20])
    with pytest.raises(ValueError, match="drop_rate"):
        PaperStampClassifier(drop_rate=1.0)


def test_without_features_ignores_them() -> None:
    model = PaperStampClassifier(with_features=False).eval()
    images, features = _inputs()
    assert model.bn_features is None
    assert torch.equal(model(images), model(images, features))


def test_initialisation_follows_m1_graph() -> None:
    torch.manual_seed(0)
    model = PaperStampClassifier()
    for conv in (model.conv1, model.conv2, model.conv3, model.conv4, model.conv5):
        assert conv.bias is not None
        assert torch.all(conv.bias == 0)
    assert torch.all(model.logits.bias == 0)
    for dense in (model.fc1, model.fc2, model.fc3):
        # Glorot-uniform biases, not zeros: ALeRCE passes bias_initializer=None.
        assert dense.bias.abs().max() <= _BIAS_LIMIT
        assert dense.bias.abs().max() > 0.5 * _BIAS_LIMIT
    for layer in (model.conv1, model.conv2, model.fc1, model.fc2, model.logits):
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(layer.weight)
        limit = math.sqrt(6 / (fan_in + fan_out))
        assert layer.weight.abs().max() <= limit
        assert layer.weight.abs().max() > 0.9 * limit
    bn = model.bn_features
    assert bn is not None
    assert (bn.eps, bn.momentum) == (1e-3, 0.0)
    assert torch.all(bn.weight == 1)
    assert torch.all(bn.bias == 0)


def test_seeded_initialisation_is_reproducible() -> None:
    torch.manual_seed(3)
    first = PaperStampClassifier().state_dict()
    torch.manual_seed(3)
    second = PaperStampClassifier().state_dict()
    assert all(torch.equal(first[key], second[key]) for key in first)


def test_output_is_invariant_to_rotations() -> None:
    model = PaperStampClassifier().eval()
    images, features = _inputs()
    expected = model(images, features)
    for k in (1, 2, 3):
        rotated = torch.rot90(images, k, dims=(2, 3))
        torch.testing.assert_close(model(rotated, features), expected, rtol=1e-5, atol=1e-5)


def test_train_mode_is_deterministic_with_dropout_inactive() -> None:
    model = PaperStampClassifier(dropout_active=False).train()
    images, features = _inputs()
    assert torch.equal(model(images, features), model(images, features))


def test_active_dropout_is_random_in_train_mode_only() -> None:
    torch.manual_seed(0)
    model = PaperStampClassifier(dropout_active=True, drop_rate=0.5).train()
    images, features = _inputs()
    assert not torch.equal(model(images, features), model(images, features))
    model.eval()
    assert torch.equal(model(images, features), model(images, features))


def test_feature_bn_running_stats_stay_frozen_while_training() -> None:
    torch.manual_seed(0)
    model = PaperStampClassifier().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    images, features = _inputs(n=16)
    features = 3.0 * features + 2.0  # far from (0, 1), so a moving average would move
    for _ in range(3):
        optimizer.zero_grad()
        model(images, features).logsumexp(dim=1).mean().backward()
        optimizer.step()
    bn = model.bn_features
    assert bn is not None
    assert bn.running_mean is not None
    assert bn.running_var is not None
    assert torch.equal(bn.running_mean, torch.zeros(23))
    assert torch.equal(bn.running_var, torch.ones(23))
    assert not torch.all(bn.weight == 1)  # gamma and beta are trained
    assert not torch.all(bn.bias == 0)


def test_feature_bn_in_eval_is_the_frozen_affine_map() -> None:
    model = PaperStampClassifier().eval()
    bn = model.bn_features
    assert bn is not None
    with torch.no_grad():
        bn.weight.uniform_(0.5, 1.5)
        bn.bias.uniform_(-0.5, 0.5)
    _, features = _inputs()
    expected = bn.weight * features / math.sqrt(1 + 1e-3) + bn.bias
    torch.testing.assert_close(bn(features), expected)
