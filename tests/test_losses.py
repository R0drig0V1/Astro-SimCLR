"""Tests for astro_simclr.losses: reference implementations, legacy fixtures, hand-computed cases.

The reference implementations below are written independently of the module: an explicit
loop over the views in float64, straight from the formulas, with no masks and no matrix
tricks. The fixtures in ``tests/fixtures/losses/`` come from the legacy code
(``scripts/characterize/losses/generate.py``).
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from astro_simclr.losses import NT_Xent, P_stamps_loss

FIXTURES = Path(__file__).parent / "fixtures" / "losses"
ATOL = 1e-6  # the shared tolerance for losses (phase 1, "Definition of equivalent")
BATCH_SIZES = (1, 2, 7, 150)


def reference_nt_xent(
    z_i: torch.Tensor, z_j: torch.Tensor, temperature: float, *, legacy: bool
) -> float:
    """NT-Xent from its definition, one view at a time, in float64."""
    z = torch.cat((z_i, z_j)).double()
    batch = z_i.shape[0]
    n = 2 * batch

    def s(a: int, b: int) -> float:
        cos = float(z[a] @ z[b]) / (float(z[a].norm()) * float(z[b].norm()))
        return cos / temperature

    total = 0.0
    for a in range(n):
        p = (a + batch) % n
        negatives = [math.exp(s(a, b)) for b in range(n) if b not in (a, p)]
        positive_term = 1.0 if legacy else math.exp(s(a, p))
        total += -s(a, p) + math.log(positive_term + sum(negatives))
    return total / n


def projections(batch: int, dim: int = 8, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed + batch)
    return torch.randn(batch, dim, generator=gen), torch.randn(batch, dim, generator=gen)


@pytest.mark.parametrize("legacy", [True, False])
@pytest.mark.parametrize("batch", BATCH_SIZES)
@pytest.mark.parametrize("temperature", [0.1791, 0.5])
def test_nt_xent_matches_reference(batch: int, legacy: bool, temperature: float) -> None:
    z_i, z_j = projections(batch)
    loss, _, _ = NT_Xent(temperature, legacy_denominator=legacy)(z_i, z_j)
    expected = reference_nt_xent(z_i, z_j, temperature, legacy=legacy)
    assert loss.dtype == torch.float32
    assert float(loss) == pytest.approx(expected, abs=ATOL)


def test_default_is_the_legacy_denominator() -> None:
    assert NT_Xent(0.5).legacy_denominator is True


def test_batch_of_one() -> None:
    # One pair, no negatives: the paper's loss is exactly 0, the legacy one is -s(pos).
    z_i, z_j = projections(1)
    cos = float(torch.nn.functional.cosine_similarity(z_i, z_j))
    loss, pos, neg = NT_Xent(0.5)(z_i, z_j)
    assert float(loss) == pytest.approx(-cos / 0.5, abs=ATOL)
    assert float(pos) == pytest.approx(cos, abs=ATOL)
    assert torch.isnan(neg)
    paper, _, _ = NT_Xent(0.5, legacy_denominator=False)(z_i, z_j)
    assert float(paper) == pytest.approx(0.0, abs=ATOL)


def test_pos_and_neg_are_mean_cosine_similarities() -> None:
    z_i, z_j = projections(7)
    _, pos, neg = NT_Xent(0.1)(z_i, z_j)
    z = torch.nn.functional.normalize(torch.cat((z_i, z_j)).double(), dim=1)
    cos = z @ z.T
    n = 14
    pos_values = [float(cos[a, (a + 7) % n]) for a in range(n)]
    neg_values = [float(cos[a, b]) for a in range(n) for b in range(n) if b not in (a, (a + 7) % n)]
    assert float(pos) == pytest.approx(np.mean(pos_values), abs=ATOL)
    assert float(neg) == pytest.approx(np.mean(neg_values), abs=ATOL)
    assert not pos.requires_grad
    assert not neg.requires_grad


def test_zero_vector_gives_zero_similarity_not_nan() -> None:
    z_i, z_j = projections(3)
    z_i[0] = 0.0
    loss, pos, neg = NT_Xent(0.5)(z_i, z_j)
    assert torch.isfinite(loss) and torch.isfinite(pos) and torch.isfinite(neg)


def test_identical_views_are_minimal() -> None:
    # Perfectly aligned pairs give a lower loss than random pairs (sanity of the sign).
    z_i, z_j = projections(7)
    aligned, _, _ = NT_Xent(0.5)(z_i, z_i.clone())
    random_pairs, _, _ = NT_Xent(0.5)(z_i, z_j)
    assert aligned < random_pairs


@pytest.mark.parametrize(
    ("shape_i", "shape_j"), [((3, 4), (2, 4)), ((3, 4), (3, 5)), ((0, 4), (0, 4)), ((4,), (4,))]
)
def test_nt_xent_rejects_bad_shapes(shape_i: tuple[int, ...], shape_j: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="shape"):
        NT_Xent(0.5)(torch.zeros(shape_i), torch.zeros(shape_j))


@pytest.mark.parametrize("temperature", [0.0, -1.0, math.nan])
def test_nt_xent_rejects_bad_temperature(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        NT_Xent(temperature)


def _fixture(name: str) -> dict[str, np.ndarray]:
    with np.load(FIXTURES / name) as data:
        return {key: data[key] for key in data.files}


def test_fixtures_record_their_provenance() -> None:
    for name in ("nt_xent.npz", "p_stamps_loss.npz"):
        data = _fixture(name)
        assert len(str(data["legacy_commit"])) == 40
        assert str(data["torch_version"]).startswith("2.")


def _nt_xent_cases() -> list[tuple[int, int]]:
    data = _fixture("nt_xent.npz")
    return [(t, int(b)) for t in range(len(data["temperatures"])) for b in data["batch_sizes"]]


@pytest.mark.parametrize(("t_index", "batch"), _nt_xent_cases())
def test_nt_xent_matches_legacy_fixture(t_index: int, batch: int) -> None:
    data = _fixture("nt_xent.npz")
    key = f"t{t_index}/b{batch}"
    z_i = torch.tensor(data[f"{key}/z_i"], requires_grad=True)
    z_j = torch.tensor(data[f"{key}/z_j"], requires_grad=True)
    loss, pos, neg = NT_Xent(float(data["temperatures"][t_index]))(z_i, z_j)
    loss.backward()
    assert z_i.grad is not None and z_j.grad is not None
    np.testing.assert_allclose(loss.detach().numpy(), data[f"{key}/loss"], rtol=0, atol=ATOL)
    np.testing.assert_allclose(pos.numpy(), data[f"{key}/pos"], rtol=0, atol=ATOL)
    np.testing.assert_allclose(neg.numpy(), data[f"{key}/neg"], rtol=0, atol=ATOL)  # NaN at b1
    np.testing.assert_allclose(z_i.grad.numpy(), data[f"{key}/grad_z_i"], rtol=0, atol=ATOL)
    np.testing.assert_allclose(z_j.grad.numpy(), data[f"{key}/grad_z_j"], rtol=0, atol=ATOL)


def _entropy(probs: list[float]) -> float:
    return -sum(p * math.log(p) for p in probs if p > 0)


@pytest.mark.parametrize("beta", [0.0, 0.0161, 1.0])
def test_p_stamps_loss_hand_computed(beta: float) -> None:
    # Row 0: logits (0, 0) -> p = (1/2, 1/2), target class 1: CE = log 2, H = log 2.
    # Row 1: logits (log 3, 0) -> p = (3/4, 1/4), target class 0: CE = log(4/3),
    #        H = -(3/4 log 3/4 + 1/4 log 1/4).
    logits = torch.tensor([[0.0, 0.0], [math.log(3.0), 0.0]])
    targets = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    cross_entropy = (math.log(2.0) + math.log(4.0 / 3.0)) / 2
    entropy = (_entropy([0.5, 0.5]) + _entropy([0.75, 0.25])) / 2
    loss = P_stamps_loss(beta)(logits, targets)
    assert float(loss) == pytest.approx(cross_entropy - beta * entropy, abs=ATOL)


def test_p_stamps_loss_confident_and_three_classes() -> None:
    # logits (log 1, log 2, log 5) -> p = (1/8, 2/8, 5/8); target class 2: CE = log(8/5).
    logits = torch.tensor([[0.0, math.log(2.0), math.log(5.0)]])
    targets = torch.tensor([[0.0, 0.0, 1.0]])
    expected = math.log(8.0 / 5.0) - 0.3 * _entropy([1 / 8, 2 / 8, 5 / 8])
    assert float(P_stamps_loss(0.3)(logits, targets)) == pytest.approx(expected, abs=ATOL)


def test_p_stamps_loss_uses_the_argmax_of_soft_targets() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    soft = torch.tensor([[0.2, 0.7, 0.1]])
    hard = torch.tensor([[0.0, 1.0, 0.0]])
    assert torch.equal(P_stamps_loss(0.1)(logits, soft), P_stamps_loss(0.1)(logits, hard))


def test_p_stamps_loss_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        P_stamps_loss(0.1)(torch.zeros(2, 3), torch.zeros(2, 4))
    with pytest.raises(ValueError, match="shape"):
        P_stamps_loss(0.1)(torch.zeros(3), torch.zeros(3))


@pytest.mark.parametrize("b_index", range(3))
def test_p_stamps_loss_matches_legacy_fixture(b_index: int) -> None:
    data = _fixture("p_stamps_loss.npz")
    key = f"beta{b_index}"
    logits = torch.tensor(data[f"{key}/logits"], requires_grad=True)
    loss = P_stamps_loss(float(data["betas"][b_index]))(
        logits, torch.tensor(data[f"{key}/targets"])
    )
    loss.backward()
    assert logits.grad is not None
    np.testing.assert_allclose(loss.detach().numpy(), data[f"{key}/loss"], rtol=0, atol=ATOL)
    np.testing.assert_allclose(logits.grad.numpy(), data[f"{key}/grad_logits"], rtol=0, atol=ATOL)
