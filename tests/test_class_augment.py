"""Tests of the class-aware training augmentation (plan rows C5 and C6; ADR 0006).

``training.class_augment`` and its use in ``PaperClassifierModule.training_step``:

1. the cross-channel fit (``stamp_fits``) and the computed difference (``perturb``);
2. C5's noise keeps the evaluation inputs' min-max statistics and leaves the low-R^2 rows,
   the masked pixels and the context channels alone; the feature jitter never touches the
   discrete features;
3. C6's synthetic rows are Bogus, with a sharp spike at the centre (narrower than the
   smallest PSF in the data) and a real Bogus donor's features; real stamps keep their
   labels;
4. the module: defaults off are bit-identical, the draws are seeded by the step, and
   evaluation is never augmented; the settings, presets and a CPU run.

The split runs (``--splits --fold K``) with augmentation are in ``test_split_runs.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from _training_support import ACCELERATORS

from astro_simclr.config import ConfigError, Settings, settings_from_mapping
from astro_simclr.data.float_input import crop_then_normalise
from astro_simclr.data.schema import LABEL_NAMES
from astro_simclr.experiments import paper_module, run_command
from astro_simclr.presets import (
    CARRASCO2021_NO_FEATURES,
    CARRASCO2021_PRESET,
    PRESETS,
    preset_report,
)
from astro_simclr.training.class_augment import (
    DISCRETE_FEATURES,
    FIT_COLUMNS,
    KEPT_FEATURE_NAMES,
    MIN_PSF_FWHM_PX,
    NOISE_STREAM,
    SPIKE_LINE_RATIO,
    SYNTHETIC_STREAM,
    augment_rng,
    continuous_feature_mask,
    fit_exclusions,
    jitter_features,
    masked_pixels,
    moment_fwhm,
    noise_augment,
    perturb,
    renormalise,
    spike_patterns,
    stamp_fits,
    synthetic_bogus,
)
from astro_simclr.training.device_data import TensorSplit, without_train_indices
from astro_simclr.training.modules import PaperClassifierModule

_BOGUS = LABEL_NAMES.index("Bogus")
_ISDIFFPOS = KEPT_FEATURE_NAMES.index("isdiffpos")
_SIDE = 21


def _related_stamps(
    n: int, a: np.ndarray, b: np.ndarray, c: np.ndarray, seed: int = 0
) -> np.ndarray:
    """(n, 3, 21, 21) stamps: science and template noise, diff = a*sci + b*tmpl + c exactly."""
    rng = np.random.default_rng(seed)
    sci = rng.normal(0.4, 0.05, (n, _SIDE, _SIDE))
    tmpl = rng.normal(0.3, 0.02, (n, _SIDE, _SIDE))
    diff = a[:, None, None] * sci + b[:, None, None] * tmpl + c[:, None, None]
    return np.stack([sci, tmpl, diff], axis=1)


def _normalised(stamps: np.ndarray) -> np.ndarray:
    """The paper's per-channel min-max of (n, 3, S, S) stamps, float32."""
    hwc = stamps.transpose(0, 2, 3, 1).astype(np.float32)
    out = crop_then_normalise(hwc, _SIDE, nan_approximation=False)
    return np.ascontiguousarray(out.transpose(0, 3, 1, 2))


def _check_min_max(images: torch.Tensor, masked: torch.Tensor) -> None:
    """Every channel in [0, 1], minimum 0 and maximum 1 over its unmasked pixels."""
    assert images.min() >= 0
    assert images.max() <= 1
    inf = torch.tensor(float("inf"))
    low = torch.where(masked, inf, images).amin(dim=(-2, -1))
    high = torch.where(masked, -inf, images).amax(dim=(-2, -1))
    assert torch.equal(low, torch.zeros_like(low))
    torch.testing.assert_close(high, torch.ones_like(high), rtol=0, atol=1e-6)
    assert torch.equal(images[masked], torch.zeros_like(images[masked]))


# --- 1. The fit and the computed difference ---------------------------------------------


def test_stamp_fits_recover_the_relation_and_the_background_sigma() -> None:
    a = np.array([0.8, -1.2, 2.0])
    b = np.array([-0.5, 0.9, -1.5])
    c = np.array([0.1, 0.7, 0.0])
    stamps = _related_stamps(3, a, b, c)
    fits = stamp_fits(stamps.astype(np.float32))
    assert fits.dtype == np.float32
    assert fits.shape == (3, len(FIT_COLUMNS))
    np.testing.assert_allclose(fits[:, 0], a, rtol=1e-3)
    np.testing.assert_allclose(fits[:, 1], b, rtol=1e-3)
    np.testing.assert_allclose(fits[:, 3], 1.0, atol=1e-4)
    # The MAD of the 152 border pixels estimates the noise sigma (0.05 and 0.02).
    np.testing.assert_allclose(fits[:, 4], 0.05, rtol=0.25)
    np.testing.assert_allclose(fits[:, 5], 0.02, rtol=0.25)
    # The context channels are ignored; a constant difference has R^2 = 0.
    with_context = np.concatenate([stamps, np.zeros_like(stamps)], axis=1)
    np.testing.assert_array_equal(stamp_fits(with_context), stamp_fits(stamps))
    flat = stamps.copy()
    flat[:, 2] = 0.5
    assert np.all(stamp_fits(flat)[:, 3] == 0)
    assert np.all(np.isfinite(stamp_fits(flat)))
    with pytest.raises(ValueError, match="C >= 3"):
        stamp_fits(stamps[:, :2])


def test_background_sigma_ignores_masked_pixels_and_is_zero_when_all_are() -> None:
    stamps = _related_stamps(2, np.ones(2), -np.ones(2), np.zeros(2))
    crops = _normalised(stamps)
    crops[0, 0, :3] = 0  # a masked region (several zeros) on the science border
    crops[1, 1] = 0  # an all-masked template
    fits = stamp_fits(crops)
    assert fits[0, 4] > 0
    assert fits[1, 5] == 0


def test_the_difference_follows_the_relation_before_the_min_max() -> None:
    a = np.array([0.7, -1.1])
    b = np.array([-0.4, 0.6])
    c = np.array([0.2, -0.3])
    stamps = torch.from_numpy(_related_stamps(2, a, b, c))
    fits = torch.zeros(2, len(FIT_COLUMNS), dtype=torch.float64)
    fits[:, 0], fits[:, 1], fits[:, 2] = torch.from_numpy(a), torch.from_numpy(b), torch.tensor(c)
    gen = torch.Generator().manual_seed(0)
    d_sci = torch.randn(2, _SIDE, _SIDE, generator=gen, dtype=torch.float64)
    d_tmpl = torch.randn(2, _SIDE, _SIDE, generator=gen, dtype=torch.float64)
    out = perturb(stamps, fits, d_sci, d_tmpl)
    torch.testing.assert_close(out[:, 0], stamps[:, 0] + d_sci)
    torch.testing.assert_close(out[:, 1], stamps[:, 1] + d_tmpl)
    expected = fits[:, 0, None, None] * out[:, 0] + fits[:, 1, None, None] * out[:, 1]
    torch.testing.assert_close(out[:, 2], expected + fits[:, 2, None, None])


def test_renormalise_is_the_paper_min_max() -> None:
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(4, 3, _SIDE, _SIDE)).astype(np.float32)
    raw[1, 2] = 0.25  # constant channel -> all 0
    images = torch.from_numpy(raw)
    out = renormalise(images, torch.zeros_like(images, dtype=torch.bool))
    np.testing.assert_array_equal(out.numpy(), _normalised(raw))
    masked = torch.zeros_like(images, dtype=torch.bool)
    masked[0, 0, :5] = True
    masked[2, 1] = True  # all masked -> all 0
    images[0, 0, :5] = -100.0  # excluded from the minimum
    out = renormalise(images, masked)
    assert torch.equal(out[2, 1], torch.zeros(_SIDE, _SIDE))
    assert torch.equal(out[0, 0, :5], torch.zeros(5, _SIDE))
    assert out[0, 0, 5:].min() == 0
    assert out[0, 0, 5:].max() == 1


def test_fit_exclusions_report_the_low_r2_share_per_class() -> None:
    fits = np.zeros((4, len(FIT_COLUMNS)), dtype=np.float32)
    fits[:, 3] = [0.9, 0.2, 0.6, 0.4]
    labels = np.array([0, 0, 4, 4], dtype=np.int64)
    report = fit_exclusions(fits, labels, 0.5)
    assert report["excluded"]["AGN"] == 0.5
    assert report["excluded"]["Bogus"] == 0.5
    assert report["excluded"]["SN"] == 0.0
    assert report["count"] == {"AGN": 2, "SN": 0, "VS": 0, "Asteroid": 0, "Bogus": 2}


def test_tensor_split_checks_and_keeps_the_fits() -> None:
    images = np.zeros((3, 3, _SIDE, _SIDE), dtype=np.float32)
    features = np.zeros((3, 23), dtype=np.float32)
    labels = np.zeros(3, dtype=np.int64)
    fits = np.arange(18, dtype=np.float32).reshape(3, 6)
    split = TensorSplit(images, features, labels, fits=fits)
    with pytest.raises(ValueError, match=r"fits must be float32 \(3, 6\)"):
        TensorSplit(images, features, labels, fits=fits[:2])
    reduced = without_train_indices({"Train": split}, [1])["Train"]
    assert reduced.fits is not None
    np.testing.assert_array_equal(reduced.fits, fits[[0, 2]])


# --- 2. C5: noise and feature jitter -----------------------------------------------------


def _batch(n: int = 8, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    a = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    stamps = _related_stamps(n, a, -0.8 * a, np.zeros(n), seed=seed)
    crops = _normalised(stamps)
    return torch.from_numpy(crops), torch.from_numpy(stamp_fits(crops))


def test_noise_keeps_the_min_max_statistics_and_the_masked_pixels() -> None:
    images, fits = _batch()
    images[0, 0, :2] = 0  # a masked region
    masked = masked_pixels(images)
    out, rows = noise_augment(images, fits, 2.0, augment_rng(0, 0, NOISE_STREAM))
    assert rows.all()
    _check_min_max(out, masked)
    assert not torch.equal(out, images)
    assert torch.equal(masked_pixels(out)[0, 0, :2], torch.ones(2, _SIDE, dtype=torch.bool))
    again, _ = noise_augment(images, fits, 2.0, augment_rng(0, 0, NOISE_STREAM))
    assert torch.equal(out, again)
    other, _ = noise_augment(images, fits, 2.0, augment_rng(0, 1, NOISE_STREAM))
    assert not torch.equal(out, other)


def test_noise_leaves_low_r2_rows_skipped_rows_and_the_context_alone() -> None:
    images, fits = _batch()
    fits[1, 3] = 0.3  # below aug_min_r2
    context = torch.rand(8, 3, _SIDE, _SIDE)
    six = torch.cat([images, context], dim=1)
    skip = np.zeros(8, dtype=bool)
    skip[2] = True
    out, rows = noise_augment(six, fits, 1.0, augment_rng(0, 0, NOISE_STREAM), skip=skip)
    assert rows.tolist() == [True, False, False, True, True, True, True, True]
    assert torch.equal(out[1], six[1])
    assert torch.equal(out[2], six[2])
    assert torch.equal(out[:, 3:], context)
    assert not torch.equal(out[0], six[0])
    unchanged, none = noise_augment(images, fits, 0.0, augment_rng(0, 0, NOISE_STREAM))
    assert unchanged is images
    assert not none.any()


def test_noise_moves_the_difference_with_the_science_and_template() -> None:
    images, fits = _batch(4)
    out, _ = noise_augment(images, fits, 1.0, augment_rng(3, 7, NOISE_STREAM))
    # After the min-max, the difference is still an affine function of the other two.
    refit = stamp_fits(out.numpy())
    assert np.all(refit[:, 3] > 0.999)


def test_feature_jitter_leaves_the_discrete_features_untouched() -> None:
    assert set(DISCRETE_FEATURES) <= set(KEPT_FEATURE_NAMES)
    assert len(KEPT_FEATURE_NAMES) == 23
    mask = continuous_feature_mask()
    assert mask.sum() == 23 - len(DISCRETE_FEATURES)
    features = torch.randn(16, 23)
    out = jitter_features(features, 0.1, augment_rng(0, 0, 2))
    discrete = [KEPT_FEATURE_NAMES.index(name) for name in DISCRETE_FEATURES]
    assert torch.equal(out[:, discrete], features[:, discrete])
    changed = (out != features).all(dim=0)
    assert changed[torch.from_numpy(mask)].all()
    assert (out - features).abs().max() < 1.0


# --- 3. C6: synthetic hot-pixel Bogus ----------------------------------------------------


def test_spikes_are_centred_and_narrower_than_any_psf() -> None:
    spikes = spike_patterns(np.random.default_rng(0), 400)
    centre = _SIDE // 2
    assert np.all(spikes[:, centre, centre] == 1)
    assert np.all(spikes.max(axis=(1, 2)) == 1)
    nonzero = (spikes > 0).sum(axis=(1, 2))
    assert set(nonzero.tolist()) == {1, 2}
    for spike in spikes[nonzero == 2]:
        rows, cols = np.nonzero(spike)
        assert np.abs(rows - centre).sum() + np.abs(cols - centre).sum() == 1  # a 4-neighbour
    widths = [moment_fwhm(spike) for spike in spikes]
    assert max(widths) < MIN_PSF_FWHM_PX
    # The widest spike the generator can make: two equal pixels.
    worst = np.zeros((_SIDE, _SIDE))
    worst[centre, centre] = 1.0
    worst[centre, centre + 1] = SPIKE_LINE_RATIO[1]
    assert moment_fwhm(worst) == pytest.approx(0.481, abs=1e-3)
    assert moment_fwhm(worst) < MIN_PSF_FWHM_PX


def test_moment_fwhm_is_the_fwhm_of_a_pixel_integrated_gaussian() -> None:
    from math import erf, sqrt

    side, sigma = 41, 1.0
    edges = np.arange(side + 1) - side / 2
    cdf = np.array([0.5 * (1 + erf(x / (sigma * sqrt(2)))) for x in edges])
    profile = np.diff(cdf)
    assert moment_fwhm(np.outer(profile, profile)) == pytest.approx(2.3548 * sigma, rel=1e-3)
    single = np.zeros((5, 5))
    single[2, 2] = 3.0
    assert moment_fwhm(single) == 0.0
    with pytest.raises(ValueError, match="2-D"):
        moment_fwhm(np.zeros(3))
    with pytest.raises(ValueError, match="non-negative"):
        moment_fwhm(-single)


def _synthetic_inputs(
    n: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rows: a = +1 (even) or -1 (odd); AGN, SN, VS, Asteroid, Bogus, Bogus, AGN, SN."""
    images, fits = _batch(n)
    features = torch.randn(n, 23)
    labels = torch.tensor([0, 1, 2, 3, _BOGUS, _BOGUS, 0, 1])[:n]
    # Bogus row 4 is a positive alert (stored isdiffpos 0), row 5 a negative one.
    features[:, _ISDIFFPOS] = 0.0
    features[5, _ISDIFFPOS] = 2.1225
    targets = torch.nn.functional.one_hot(labels, 5).float()
    return images, features, targets, fits


def _centre_z(image: torch.Tensor) -> float:
    """The centre pixel's z against the border median and MAD (the integrator's measure)."""
    border = torch.from_numpy(~np.pad(np.ones((_SIDE - 4, _SIDE - 4), bool), 2))
    values = image[border]
    median = values.median()
    mad = (values - median).abs().median() * 1.4826
    return float((image[_SIDE // 2, _SIDE // 2] - median) / mad)


def test_synthetic_rows_are_bogus_with_a_sharp_central_spike() -> None:
    images, features, targets, fits = _synthetic_inputs()
    rng = augment_rng(0, 0, SYNTHETIC_STREAM)
    out, feats, tgts, replaced = synthetic_bogus(images, features, targets, fits, 0.5, rng)
    # Sources: positive alerts (a > 0: even rows) that are not SN: rows 0, 2, 4 and 6.
    assert replaced.tolist() == [True, False, True, False, True, False, True, False]
    rows = np.flatnonzero(replaced)
    assert torch.equal(tgts[rows].argmax(dim=1), torch.full((4,), _BOGUS))
    # Real stamps are never relabelled: the other rows are untouched.
    keep = np.flatnonzero(~replaced)
    assert torch.equal(tgts[keep], targets[keep])
    assert torch.equal(out[keep], images[keep])
    assert torch.equal(feats[keep], features[keep])
    _check_min_max(out[rows], torch.zeros_like(out[rows], dtype=torch.bool))
    centre = _SIDE // 2
    for row in rows:
        assert torch.equal(out[row, 1], images[row, 1])  # the template is unchanged
        # The donor is the batch's positive-alert real Bogus.
        assert torch.equal(feats[row], features[4])
        science, difference = out[row, 0], out[row, 2]
        assert science[centre, centre] == 1  # the spike is the science's maximum
        # Sharp: at most 2 pixels above half the maximum, at or next to the centre.
        above = torch.nonzero(science > 0.5)
        assert len(above) <= 2
        assert (above - centre).abs().sum(dim=1).max() <= 1
        # The difference has the same spike, a central peak as in every real alert.
        assert difference[centre, centre] == 1
        assert _centre_z(difference) > 5


def test_synthetic_differences_are_central_peaks_for_every_row() -> None:
    # Every replaced row, at any fraction and step: the difference's centre is a positive
    # peak against its border (no central dip, which no real alert has), and no negative
    # alert or SN row is ever a source.
    images, features, targets, fits = _synthetic_inputs()
    for step in range(20):
        rng = augment_rng(1, step, SYNTHETIC_STREAM)
        out, _, _, replaced = synthetic_bogus(images, features, targets, fits, 0.99, rng)
        assert replaced.any()
        assert not replaced[fits[:, 0].numpy() < 0].any()
        assert not replaced[targets.argmax(dim=1).numpy() == LABEL_NAMES.index("SN")].any()
        for row in np.flatnonzero(replaced):
            assert _centre_z(out[row, 2]) > 0


def test_synthetic_bogus_needs_sources_and_donors() -> None:
    images, features, targets, fits = _synthetic_inputs()
    rng = augment_rng(0, 0, SYNTHETIC_STREAM)
    no_bogus = targets.clone()
    no_bogus[4:6] = torch.nn.functional.one_hot(torch.tensor([0, 0]), 5).float()
    *_, replaced = synthetic_bogus(images, features, no_bogus, fits, 0.5, rng)
    assert not replaced.any()
    negative_donors = features.clone()
    negative_donors[:, _ISDIFFPOS] = 2.1225  # only negative-alert Bogus: no donor
    *_, replaced = synthetic_bogus(images, negative_donors, targets, fits, 0.5, rng)
    assert not replaced.any()
    low = fits.clone()
    low[:, 3] = 0.1
    *_, replaced = synthetic_bogus(images, features, targets, low, 0.5, rng)
    assert not replaced.any()
    masked = images.clone()
    masked[:, 0, :2] = 0
    *_, replaced = synthetic_bogus(masked, features, targets, fits, 0.5, rng)
    assert not replaced.any()
    out, *_ = synthetic_bogus(images, features, targets, fits, 0.0, rng)
    assert out is images
    with pytest.raises(ValueError, match=r"\(B, 3, S, S\)"):
        synthetic_bogus(torch.cat([images, images], 1), features, targets, fits, 0.5, rng)


# --- 4. The module, the settings and a run ----------------------------------------------


def _module(preset: str = CARRASCO2021_PRESET, /, **changes: object) -> PaperClassifierModule:
    torch.manual_seed(0)
    return paper_module(settings_from_mapping({**PRESETS[preset], **changes}))


def test_defaults_off_are_bit_identical() -> None:
    images, features, targets, fits = _synthetic_inputs()
    module = _module().train()
    assert (module.aug_noise, module.aug_feature_jitter, module.synthetic_bogus) == (0, 0, 0)
    torch.manual_seed(1)
    plain = module.training_step((images, features, targets), 0)
    torch.manual_seed(1)
    with_fits = module.training_step((images, features, targets, fits), 0)
    torch.manual_seed(1)
    expected = module.compute_loss(module(images, features), targets)
    assert torch.equal(plain, expected)
    assert torch.equal(with_fits, expected)


def test_training_is_augmented_and_evaluation_is_not() -> None:
    images, features, targets, fits = _synthetic_inputs()
    changes: dict[str, object] = {
        "aug_noise": 1.0,
        "aug_feature_jitter": 0.1,
        "synthetic_bogus": 0.25,
        "seed": 5,
    }
    module = _module(paper_arch="p4", **changes).eval()
    plain = _module(paper_arch="p4").eval()
    plain.load_state_dict(module.state_dict())
    batch = (images, features, targets)
    aug_images, aug_features, aug_targets = module.class_augment(images, features, targets, fits)
    assert not torch.equal(aug_images, images)
    assert (aug_targets.argmax(dim=1) == _BOGUS).sum() > (targets.argmax(dim=1) == _BOGUS).sum()
    again = module.class_augment(images, features, targets, fits)
    expected_batch = (aug_images, aug_features, aug_targets)
    assert all(torch.equal(x, y) for x, y in zip(again, expected_batch, strict=True))
    loss = module.training_step((*batch, fits), 0)
    rng = augment_rng(5, 0, 2)
    jittered = jitter_features(aug_features, 0.1, rng)
    expected = module.compute_loss(module(aug_images, jittered), aug_targets)
    torch.testing.assert_close(loss, expected)
    for step in ("validation_step", "test_step"):
        torch.testing.assert_close(getattr(module, step)(batch, 0), getattr(plain, step)(batch, 0))
    with pytest.raises(ValueError, match="need the stamp fits"):
        module.training_step(batch, 0)


def test_jitter_alone_needs_no_fits() -> None:
    images, features, targets, _ = _synthetic_inputs()
    module = _module(aug_feature_jitter=0.2).eval()
    loss = module.training_step((images, features, targets), 0)
    jittered = jitter_features(features, 0.2, augment_rng(0, 0, 2))
    torch.testing.assert_close(loss, module.compute_loss(module(images, jittered), targets))


def test_module_rejects_bad_augmentation_values() -> None:
    base = {"beta_loss": 0.5, "lr": 1e-3, "optimizer": "Adam"}
    for bad, match in (
        ({"aug_noise": -1.0}, ">= 0"),
        ({"synthetic_bogus": 1.0}, r"\[0, 1\)"),
        ({"aug_min_r2": 1.5}, r"\[0, 1\]"),
        ({"synthetic_bogus": 0.1, "context": "avgpool63"}, "context='none'"),
    ):
        with pytest.raises(ValueError, match=match):
            PaperClassifierModule(**base, **bad)  # type: ignore[arg-type]


def _preset(**changes: object) -> Settings:
    return settings_from_mapping({**PRESETS[CARRASCO2021_PRESET], **changes})


def test_settings_validate_the_augmentation() -> None:
    assert _preset(aug_noise=1.5).aug_noise == 1.5
    for bad, match in (
        ({"aug_noise": -0.1}, "aug_noise must be >= 0"),
        ({"aug_feature_jitter": -0.1}, "aug_feature_jitter must be >= 0"),
        ({"synthetic_bogus": 1.0}, r"synthetic_bogus must be in \[0, 1\)"),
        ({"aug_min_r2": -0.1}, r"aug_min_r2 must be in \[0, 1\]"),
        ({"aug_noise": 1.0, "paper_input": "uint8"}, "need paper_input=float32"),
        ({"synthetic_bogus": 0.1, "context": "avgpool63"}, "needs context=none"),
    ):
        with pytest.raises(ConfigError, match=match):
            _preset(**bad)
    with pytest.raises(ConfigError, match="needs with_features=true"):
        settings_from_mapping({**PRESETS[CARRASCO2021_NO_FEATURES], "aug_feature_jitter": 0.1})


def test_augmented_runs_are_preset_deviations() -> None:
    for name in (CARRASCO2021_PRESET, CARRASCO2021_NO_FEATURES):
        assert PRESETS[name]["aug_noise"] == 0.0
        assert PRESETS[name]["synthetic_bogus"] == 0.0
    report = preset_report(_preset(aug_noise=1.0, synthetic_bogus=0.1))
    assert report is not None
    assert not report["faithful"]
    assert set(report["deviations"]) == {"aug_noise", "synthetic_bogus"}


def test_augmentation_flags_need_the_paper_model(classifier_f32_dir: Path, tmp_path: Path) -> None:
    settings = Settings(
        accelerator="cpu",
        data_dir=classifier_f32_dir.parent,
        dataset=classifier_f32_dir.name,
        output_dir=tmp_path,
        aug_noise=1.0,
    )
    with pytest.raises(ConfigError, match="aug_noise apply only to"):
        run_command("train-supervised", settings, progress_bar=False)


@pytest.mark.parametrize("accelerator", ACCELERATORS)
def test_augmented_p4_run_trains_and_records_the_exclusions(
    accelerator: str, classifier_f32_dir: Path, tmp_path: Path
) -> None:
    changes = {
        "paper_arch": "p4",
        "aug_noise": 1.0,
        "aug_feature_jitter": 0.1,
        "synthetic_bogus": 0.25,
        "aug_min_r2": 0.0,
    }
    settings = _preset(
        accelerator=accelerator,
        deterministic=True,
        data_dir=classifier_f32_dir.parent,
        dataset=classifier_f32_dir.name,
        output_dir=tmp_path,
        batch_size=8,
        max_steps=4,
        val_every_n_steps=2,
        horizon_increment=0,
        checkpoint_every_n_steps=2,
        **changes,
    )
    metrics = run_command("train-supervised", settings, progress_bar=False)
    assert metrics["last_step"] == 4
    report = metrics["paper_data"]["class_augment"]
    assert report["min_r2"] == 0.0
    assert set(report["excluded"]) == set(LABEL_NAMES)
    assert set(metrics["preset"]["deviations"]) >= set(changes) - {"aug_min_r2"}
    checkpoint = torch.load(tmp_path / "checkpoints" / "best.ckpt", weights_only=True)
    restored = PaperClassifierModule(**checkpoint["hyper_parameters"])
    assert restored.synthetic_bogus == 0.25
