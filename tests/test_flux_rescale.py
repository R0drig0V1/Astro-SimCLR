"""Tests of the transient flux rescaling (plan row C7; ADR 0007; ``training.flux_rescale``).

1. The central PSF fit recovers a known Gaussian and rejects bad fits.
2. Only the central source changes: before the min-max, the residual (noise, neighbours) is
   untouched, the difference gains ``(k - 1) * PSF`` and the science ``(k - 1) * PSF / a``.
3. The features match the image: a refit of the rescaled difference has ``k`` times the flux,
   and ``magpsf`` / ``sigmapsf`` move by ``-2.5 log10 k`` and ``1 / k``.
4. The alert constraint holds; Bogus is never rescaled; the draws are seeded.
5. The plain engine's paper task (``tasks.paper``, through
   ``training.paper_augment.PaperAugment``), the settings and a run on each accelerator
   (split runs: ``test_split_runs.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from _training_support import ACCELERATORS

from astro_simclr.config import ConfigError, settings_from_mapping
from astro_simclr.data.m1_features import M1_FEATURE_NORM_STATS
from astro_simclr.data.schema import LABEL_NAMES
from astro_simclr.engine.checkpoint import load_best
from astro_simclr.presets import CARRASCO2021_PRESET, PRESETS, preset_report
from astro_simclr.tasks.paper import PaperTask, build_paper_task, paper_augment
from astro_simclr.tasks.plain import run_plain
from astro_simclr.training.class_augment import (
    KEPT_FEATURE_NAMES,
    augment_rng,
    renormalise,
    stamp_fits,
)
from astro_simclr.training.flux_rescale import (
    AUGMENT_COLUMNS,
    FLUX_STREAM,
    PSF_COLUMNS,
    fit_central_psf,
    flux_rescale,
    k_floor,
    psf_exclusions,
    psf_image,
    rescale_transient,
    rescaled_features,
)
from astro_simclr.training.paper_augment import PaperAugment, check_augment_context
from astro_simclr.training.tensor_splits import TensorSplit

_SIDE = 21
_BOGUS = LABEL_NAMES.index("Bogus")
_MAG, _SIG, _LIM = (KEPT_FEATURE_NAMES.index(n) for n in ("magpsf", "sigmapsf", "diffmaglim"))


def _gaussian(flux: float, x0: float, y0: float, sigma: float) -> np.ndarray:
    params = torch.tensor([[flux, x0, y0, sigma]], dtype=torch.float64)
    return psf_image(params, _SIDE)[0].numpy()


def _stamps(n: int = 6, seed: int = 0, offset: tuple[float, float] = (0.3, -0.2)) -> np.ndarray:
    """(n, 3, 21, 21) unnormalised crops: noise everywhere, a transient at ``offset`` (x, y).

    The difference is ``sci - 0.7 * tmpl`` (a = 1 for even rows; odd rows are negative
    alerts, ``-(sci - 0.7 tmpl)``, a = -1), so the stamp fit is exact.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((n, 3, _SIDE, _SIDE))
    for i in range(n):
        tmpl = rng.normal(1.0, 0.02, (_SIDE, _SIDE)) + _gaussian(3.0, 6.0, -5.0, 1.5)
        transient = _gaussian(4.0 + i, *offset, 0.9 + 0.05 * i)
        sci = 0.7 * tmpl + rng.normal(0.0, 0.03, (_SIDE, _SIDE))
        sci += transient if i % 2 == 0 else -transient
        sign = 1.0 if i % 2 == 0 else -1.0
        out[i] = sci, tmpl, sign * (sci - 0.7 * tmpl)
    return out


def _normalised(stamps: np.ndarray) -> np.ndarray:
    images = torch.from_numpy(stamps)
    return renormalise(images, torch.zeros_like(images, dtype=torch.bool)).float().numpy()


def _features(n: int, magpsf: float, diffmaglim: float, sigmapsf: float = 0.1) -> torch.Tensor:
    """(n, 23) standardised features with the given raw magpsf, diffmaglim and sigmapsf."""
    out = torch.randn(n, 23, generator=torch.Generator().manual_seed(1))
    for index, raw in ((_MAG, magpsf), (_LIM, diffmaglim), (_SIG, sigmapsf)):
        mean, std = M1_FEATURE_NORM_STATS[KEPT_FEATURE_NAMES[index]]
        out[:, index] = (raw - mean) / std
    return out


def _raw(features: torch.Tensor, index: int) -> torch.Tensor:
    mean, std = M1_FEATURE_NORM_STATS[KEPT_FEATURE_NAMES[index]]
    return features[:, index].double() * std + mean


def _table(images: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.concatenate([stamp_fits(images), fit_central_psf(images)], 1))


# --- 1. The PSF fit -----------------------------------------------------------------------


def test_fit_central_psf_recovers_a_gaussian_and_rejects_bad_fits() -> None:
    rng = np.random.default_rng(3)
    crops = np.zeros((4, 3, _SIDE, _SIDE))
    crops[:, 2] = rng.normal(0.0, 0.01, (4, _SIDE, _SIDE))
    crops[0, 2] += _gaussian(2.0, 0.4, -0.3, 1.1)
    crops[1, 2] += _gaussian(2.0, 4.5, 0.0, 1.1)  # off centre
    crops[3, 2] += _gaussian(2.0, 0.0, 0.0, 1.1)
    crops[3, 2, :3, :] = 0.0  # masked rows (several exact zeros): weight 0
    crops[3, 2, 0, 0] = 0.0
    fits = fit_central_psf(crops)
    assert fits.shape == (4, len(PSF_COLUMNS))
    assert fits.dtype == np.float32
    np.testing.assert_allclose(fits[0, :4], [2.0, 0.4, -0.3, 1.1], rtol=0.08, atol=0.1)
    assert fits[:, 4].tolist() == [1.0, 0.0, 0.0, 1.0]  # off centre and pure noise fail
    np.testing.assert_allclose(fits[3, :4], [2.0, 0.0, 0.0, 1.1], rtol=0.08, atol=0.1)
    with pytest.raises(ValueError, match="C >= 3"):
        fit_central_psf(crops[:, :2])
    with pytest.raises(ValueError, match="at least 11"):
        fit_central_psf(crops[:, :, :9, :9])


def test_psf_image_is_pixel_integrated() -> None:
    params = torch.tensor([[2.5, 0.2, -0.4, 1.3]], dtype=torch.float64)
    assert float(psf_image(params, 41).sum()) == pytest.approx(2.5, rel=1e-6)


def test_psf_exclusions_report_the_failed_fits_per_class() -> None:
    fits = np.zeros((4, len(AUGMENT_COLUMNS)), dtype=np.float32)
    fits[:, -1] = [1, 0, 1, 1]
    report = psf_exclusions(fits, np.array([0, 0, 1, 4]))
    assert report["skipped"]["AGN"] == 0.5
    assert report["skipped"]["SN"] == 0.0
    assert report["skipped"]["VS"] == 0.0


def test_tensor_split_takes_the_eleven_column_table() -> None:
    images = np.zeros((2, 3, _SIDE, _SIDE), dtype=np.float32)
    features = np.zeros((2, 23), dtype=np.float32)
    labels = np.zeros(2, dtype=np.int64)
    table = np.zeros((2, len(AUGMENT_COLUMNS)), dtype=np.float32)
    assert TensorSplit(images, features, labels, fits=table).fits is table
    with pytest.raises(ValueError, match=r"\(2, 6\) or \(2, 11\)"):
        TensorSplit(images, features, labels, fits=table[:, :8])


# --- 2. Only the central source changes ---------------------------------------------------


def test_only_the_central_source_changes_before_the_min_max() -> None:
    stamps = torch.from_numpy(_stamps())
    params = torch.tensor([[4.0 + i, 0.3, -0.2, 0.9 + 0.05 * i] for i in range(6)])
    params = params.double()
    a = torch.tensor([1.0, -1.0] * 3, dtype=torch.float64)
    k = torch.tensor([0.6, 1.4, 0.8, 1.2, 1.5, 0.5], dtype=torch.float64)
    out = rescale_transient(stamps, params, a, k)
    psf = psf_image(params, _SIDE)
    torch.testing.assert_close(out[:, 1], stamps[:, 1])  # the template
    torch.testing.assert_close(out[:, 2] - stamps[:, 2], (k - 1)[:, None, None] * psf)
    torch.testing.assert_close(
        out[:, 0] - stamps[:, 0], (k - 1)[:, None, None] * psf / a[:, None, None]
    )
    # The residual after removing the transient model is exactly the original's: the noise,
    # the template's neighbour and everything else are untouched.
    residual_new = out[:, 2] - k[:, None, None] * psf
    residual_old = stamps[:, 2] - psf
    torch.testing.assert_close(residual_new, residual_old)
    # The difference still follows the cross-channel relation diff = a*sci - 0.7a*tmpl.
    torch.testing.assert_close(
        out[:, 2], a[:, None, None] * (out[:, 0] - 0.7 * out[:, 1]), rtol=0, atol=1e-12
    )


# --- 3. The features match the image ------------------------------------------------------


def test_the_features_match_the_image_change() -> None:
    images = _normalised(_stamps())
    table = _table(images)
    assert (table[:, -1] == 1).all()
    targets = torch.nn.functional.one_hot(torch.tensor([0, 1, 2, 3, 0, 1]), 5).float()
    features = _features(6, magpsf=19.0, diffmaglim=20.5)
    rng = augment_rng(0, 0, FLUX_STREAM)
    tensor_images = torch.from_numpy(images)
    out, new_features, k = flux_rescale(
        tensor_images, features, targets, table, 1.0, (0.5, 1.5), rng
    )
    assert np.all(k != 1)
    # magpsf and sigmapsf, in raw units, move by -2.5 log10 k and 1/k.
    k_t = torch.from_numpy(k)
    torch.testing.assert_close(
        _raw(new_features, _MAG), _raw(features, _MAG) - 2.5 * torch.log10(k_t), atol=1e-4, rtol=0
    )
    torch.testing.assert_close(
        _raw(new_features, _SIG), _raw(features, _SIG) / k_t, atol=1e-5, rtol=0
    )
    others = [i for i in range(23) if i not in (_MAG, _SIG)]
    assert torch.equal(new_features[:, others], features[:, others])
    # The image: before the min-max the refitted flux of the difference is k times the old.
    crops = tensor_images.double()
    changed = rescale_transient(crops, table[:, 6:10].double(), table[:, 0].double(), k_t)
    refit = fit_central_psf(changed.numpy())
    np.testing.assert_allclose(refit[:, 0] / table[:, 6].numpy(), k, rtol=0.02)
    # And the min-max keeps the evaluation statistics.
    assert torch.equal(out[:, :3].amin(dim=(-2, -1)), torch.zeros(6, 3))
    torch.testing.assert_close(out[:, :3].amax(dim=(-2, -1)), torch.ones(6, 3))


def test_rescaled_features_round_trip() -> None:
    features = _features(3, magpsf=18.0, diffmaglim=20.0, sigmapsf=0.08)
    k = torch.tensor([0.5, 1.0, 2.0])
    out = rescaled_features(features, k)
    assert torch.allclose(out[1], features[1], atol=1e-6)
    np.testing.assert_allclose(_raw(out, _MAG).numpy(), [18.7526, 18.0, 17.2474], atol=1e-3)
    np.testing.assert_allclose(_raw(out, _SIG).numpy(), [0.16, 0.08, 0.04], atol=1e-6)


# --- 4. The alert constraint, Bogus, seeding ----------------------------------------------


def test_the_result_is_always_still_an_alert() -> None:
    images = torch.from_numpy(_normalised(_stamps()))
    table = _table(images.numpy())
    targets = torch.nn.functional.one_hot(torch.tensor([0, 1, 2, 3, 0, 1]), 5).float()
    # Headroom: 0.35, 0.96, 2.18, 0.92 mag (the classes' medians), 0 (at the limit), and a
    # stamp already 0.3 mag beyond its limit whose floor exceeds k_max (left alone).
    headroom = torch.tensor([0.35, 0.96, 2.18, 0.92, 0.0, -0.6])
    features = _features(6, magpsf=19.0, diffmaglim=19.0)
    mean, std = M1_FEATURE_NORM_STATS["diffmaglim"]
    features[:, _LIM] = (19.0 + headroom - mean) / std
    floor = k_floor(features.double())
    np.testing.assert_allclose(floor.numpy(), 10 ** (-0.4 * headroom.numpy()), rtol=1e-5)
    for step in range(30):
        rng = augment_rng(4, step, FLUX_STREAM)
        _, new, k = flux_rescale(images, features, targets, table, 1.0, (0.5, 1.5), rng)
        rescaled = k != 1
        assert not rescaled[5]
        assert np.all(k[rescaled] >= np.maximum(0.5, floor.numpy()[rescaled]) - 1e-9)
        assert np.all(k <= 1.5 + 1e-9)
        lim = _raw(new, _LIM).numpy()
        assert np.all(_raw(new, _MAG).numpy()[rescaled] <= lim[rescaled] + 1e-4)


def test_bogus_is_never_rescaled_and_draws_are_seeded() -> None:
    images = torch.from_numpy(_normalised(_stamps()))
    table = _table(images.numpy())
    labels = torch.tensor([_BOGUS, 0, _BOGUS, 3, 2, _BOGUS])
    targets = torch.nn.functional.one_hot(labels, 5).float()
    features = _features(6, magpsf=19.0, diffmaglim=21.0)
    for step in range(10):
        args = (images, features, targets, table, 1.0, (0.5, 1.5))
        out, feats, k = flux_rescale(*args, augment_rng(0, step, FLUX_STREAM))
        assert np.all(k[[0, 2, 5]] == 1)
        assert torch.equal(out[[0, 2, 5]], images[[0, 2, 5]])
        assert torch.equal(feats[[0, 2, 5]], features[[0, 2, 5]])
        again, _, k_again = flux_rescale(*args, augment_rng(0, step, FLUX_STREAM))
        assert torch.equal(out, again)
        np.testing.assert_array_equal(k, k_again)
    other = flux_rescale(*args, augment_rng(0, 99, FLUX_STREAM))[2]
    assert not np.array_equal(other, k)


def test_flux_rescale_edges() -> None:
    images = torch.from_numpy(_normalised(_stamps()))
    table = _table(images.numpy())
    targets = torch.nn.functional.one_hot(torch.zeros(6, dtype=torch.long), 5).float()
    features = _features(6, magpsf=19.0, diffmaglim=21.0)
    rng = augment_rng(0, 0, FLUX_STREAM)
    out, feats, k = flux_rescale(images, features, targets, table, 0.0, (0.5, 1.5), rng)
    assert out is images and feats is features and np.all(k == 1)
    skip = np.ones(6, dtype=bool)
    out, _, k = flux_rescale(images, features, targets, table, 1.0, (0.5, 1.5), rng, skip=skip)
    assert out is images and np.all(k == 1)
    failed = table.clone()
    failed[:, -1] = 0
    assert np.all(flux_rescale(images, features, targets, failed, 1.0, (0.5, 1.5), rng)[2] == 1)
    six = torch.cat([images, images], dim=1)
    out, _, k = flux_rescale(six, features, targets, table, 1.0, (0.5, 1.5), rng)
    assert torch.equal(out[:, 3:], images)  # channels beyond 3 are untouched
    with pytest.raises(ValueError, match=r"\(B, 11\)"):
        flux_rescale(images, features, targets, table[:, :6], 1.0, (0.5, 1.5), rng)
    with pytest.raises(ValueError, match="0 < low <= high"):
        flux_rescale(images, features, targets, table, 1.0, (1.5, 0.5), rng)


# --- 5. The task, settings and a run ------------------------------------------------------


def _task(preset: str = CARRASCO2021_PRESET, /, **changes: object) -> PaperTask:
    torch.manual_seed(0)
    return build_paper_task(settings_from_mapping({**PRESETS[preset], **changes}), {})


def _augment(**changes: object) -> PaperAugment:
    return paper_augment(settings_from_mapping({**PRESETS[CARRASCO2021_PRESET], **changes}))


def test_the_task_rescales_training_batches_only() -> None:
    images = torch.from_numpy(_normalised(_stamps()))
    table = _table(images.numpy())
    targets = torch.nn.functional.one_hot(torch.tensor([0, 1, 2, 3, 0, 1]), 5).float()
    features = _features(6, magpsf=19.0, diffmaglim=21.0)
    task = _task(paper_arch="p4", flux_rescale=1.0, seed=7)
    net = task.net.eval()
    plain = _task(paper_arch="p4")
    plain.net.eval()
    plain.net.load_state_dict(net.state_dict())
    loss = task.train_step((images, features, targets, table), 0)
    expected_images, expected_features, _ = flux_rescale(
        images, features, targets, table, 1.0, (0.5, 1.5), augment_rng(7, 0, FLUX_STREAM)
    )
    expected = net.criterion(net(expected_images, expected_features), targets)
    torch.testing.assert_close(loss, expected)
    assert not torch.equal(expected_images, images)
    batch = (images, features, targets)
    with torch.no_grad():
        rescaled, faithful = task.eval_step(batch), plain.eval_step(batch)
    torch.testing.assert_close(rescaled.loss, faithful.loss)
    torch.testing.assert_close(rescaled.probabilities, faithful.probabilities)
    with pytest.raises(ValueError, match="need the stamp fits"):
        task.train_step(batch, 0)


def _residual(target: np.ndarray, columns: list[np.ndarray]) -> float:
    """Max |residual| of the least-squares fit of ``target`` on ``columns`` and a constant."""
    design = np.stack([c.ravel() for c in columns] + [np.ones(target.size)], axis=1)
    coef, *_ = np.linalg.lstsq(design, target.ravel(), rcond=None)
    return float(np.abs(design @ coef - target.ravel()).max())


def test_rescaling_composes_with_rot90_at_the_rotated_transient() -> None:
    # Review of PR #22: with rot90_augment, the rescaled flux must land where the (rotated)
    # transient is. A transient 1.5 px right of the centre, k = 2 for every row (k_min =
    # k_max), every row rotated by its own k*90 degrees.
    images = torch.from_numpy(_normalised(_stamps(4, offset=(1.5, 0.0))))
    table = _table(images.numpy())
    assert (table[:, -1] == 1).all()
    np.testing.assert_allclose(table[:, 7], 1.5, atol=0.15)  # psf_x0, with noise
    targets = torch.nn.functional.one_hot(torch.zeros(4, dtype=torch.long), 5).float()
    features = _features(4, magpsf=19.0, diffmaglim=21.0)
    changes = {"flux_rescale": 1.0, "flux_k_min": 2.0, "flux_k_max": 2.0}
    augment = _augment(paper_arch="single", rot90_augment=True, seed=3, **changes)
    batch = (images, features, targets, table)
    out, _, _ = augment.apply(batch, augment.plan(batch, 0))
    turns = augment.rotation_steps(4, 0)
    assert set(turns.tolist()) - {0}  # some rows really are rotated
    # Exactly "rescale, then rotate".
    rescaled, _, k = flux_rescale(
        images, features, targets, table, 1.0, (2.0, 2.0), augment_rng(3, 0, FLUX_STREAM)
    )
    assert np.all(k == 2.0)
    expected = torch.stack([torch.rot90(rescaled[i], int(turns[i]), dims=(1, 2)) for i in range(4)])
    torch.testing.assert_close(out, expected)
    # Physically: each augmented difference is an affine function of the rotated original
    # and the PSF at the ROTATED transient position, not at the unrotated one.
    for i in range(4):
        turn = int(turns[i])
        original = torch.rot90(images[i, 2], turn).numpy()
        psf = psf_image(table[i : i + 1, 6:10].double(), _SIDE)[0]
        at_rotated = torch.rot90(psf, turn).numpy()
        diff = out[i, 2].numpy()
        assert _residual(diff, [original, at_rotated]) < 1e-5
        if turn % 4:
            assert _residual(diff, [original, psf.numpy()]) > 0.05


def test_the_augmentation_and_settings_reject_bad_values() -> None:
    cases: list[tuple[dict[str, Any], str]] = [
        ({"flux_rescale": 1.5}, r"\[0, 1\]"),
        ({"flux_k_min": 2.0}, "0 < min <= max"),
    ]
    for values, match in cases:
        with pytest.raises(ValueError, match=match):
            PaperAugment(**values)
    with pytest.raises(ValueError, match="context='none'"):
        check_augment_context(0.0, 0.5, "avgpool63", 5)
    with pytest.raises(ValueError, match="need the 5 CD21 classes"):
        check_augment_context(0.0, 0.5, "none", 3)
    preset = PRESETS[CARRASCO2021_PRESET]
    for bad, match in (
        ({"flux_rescale": -0.1}, r"flux_rescale must be in \[0, 1\]"),
        ({"flux_k_min": 0.0}, "0 < flux_k_min <= flux_k_max"),
        ({"flux_k_min": 2.0, "flux_k_max": 1.5}, "0 < flux_k_min <= flux_k_max"),
        ({"flux_rescale": 0.5, "paper_input": "uint8"}, "needs paper_input=float32"),
        ({"flux_rescale": 0.5, "context": "avgpool63"}, "needs context=none"),
    ):
        with pytest.raises(ConfigError, match=match):
            settings_from_mapping({**preset, **bad})
    assert preset["flux_rescale"] == 0.0
    report = preset_report(settings_from_mapping({**preset, "flux_rescale": 0.5}))
    assert report is not None
    assert set(report["deviations"]) == {"flux_rescale"}


@pytest.mark.parametrize("accelerator", ACCELERATORS)
def test_flux_rescaled_p4_run_trains_and_records_the_skips(
    accelerator: str, classifier_f32_dir: Path, tmp_path: Path
) -> None:
    changes = {"paper_arch": "p4", "flux_rescale": 0.5, "aug_noise": 1.0, "aug_min_r2": 0.0}
    settings = settings_from_mapping(
        {
            **PRESETS[CARRASCO2021_PRESET],
            "accelerator": accelerator,
            "deterministic": True,
            "data_dir": classifier_f32_dir.parent,
            "dataset": classifier_f32_dir.name,
            "output_dir": tmp_path,
            "batch_size": 8,
            "max_steps": 4,
            "val_every_n_steps": 2,
            "horizon_increment": 0,
            "checkpoint_every_n_steps": 2,
            **changes,
        }
    )
    metrics = run_plain("train-supervised", settings)
    assert metrics["last_step"] == 4
    assert set(metrics["paper_data"]["flux_rescale"]["skipped"]) == set(LABEL_NAMES)
    assert metrics["config"]["flux_rescale"] == 0.5
    # The best weights load back into the run's model.
    restored = _task(**changes)
    restored.net.load_state_dict(load_best(tmp_path / "ckpt" / "best.pt")["model"])
    assert restored.augment.flux_rescale == 0.5
