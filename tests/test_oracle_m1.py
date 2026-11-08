"""Tests of M1's weight mapping, the NumPy reference pass and the pinned download (task 2.2b).

Everything runs on synthetic weights: the real M1 is ALeRCE's and is never in the
repository (``docs/phases/02-stamp-classifier.md``, "Sources of truth").
"""

from __future__ import annotations

import hashlib
import json
from math import prod
from pathlib import Path

import numpy as np
import pytest
import torch
from _tf_writer import random_m1, synthetic_m1_dir, write_bundle

from astro_simclr.oracle import m1
from astro_simclr.oracle.m1 import (
    M1_FILES,
    RemoteFile,
    default_cache_dir,
    fetch_m1,
    load_m1_model,
    load_weights,
    numpy_forward,
    tf_shapes,
    to_state_dict,
)
from astro_simclr.paper_model import PaperStampClassifier


def test_pinned_files() -> None:
    assert m1.M1_COMMIT == "a3705acb58ce2e45c1f9b240b824c51260295a99"
    assert set(M1_FILES) == {"model.index", "model.data-00000-of-00001", "val.events"}
    for remote in M1_FILES.values():
        assert len(remote.sha256) == 64
        assert remote.size > 0
    assert m1.M1_BASE_URL.startswith("https://raw.githubusercontent.com/alercebroker/pipeline/")


@pytest.mark.parametrize(("with_features", "trainable"), [(True, 441_075), (False, 439_557)])
def test_tf_shapes_match_the_parameter_counts(with_features: bool, trainable: int) -> None:
    shapes = tf_shapes(with_features=with_features)
    moving = [n for n in shapes if "moving_" in n]
    assert sum(prod(s) for n, s in shapes.items() if n not in moving) == trainable
    assert len(moving) == (2 if with_features else 0)


@pytest.mark.parametrize("with_features", [True, False])
def test_state_dict_matches_the_model(with_features: bool) -> None:
    state = to_state_dict(random_m1(0, with_features=with_features))
    reference = PaperStampClassifier(with_features=with_features).state_dict()
    assert set(state) == set(reference)
    for name, tensor in reference.items():
        assert state[name].shape == tensor.shape, name
        assert state[name].dtype == tensor.dtype, name


def test_kernel_layouts() -> None:
    weights = random_m1(1)
    state = to_state_dict(weights)
    hwio = weights["network/conv_3_4/conv2d/kernel"]
    # torch OIHW [o, i, h, w] is TF HWIO [h, w, i, o].
    assert state["conv3.weight"][7, 5, 1, 3] == pytest.approx(float(hwio[1, 3, 5, 7]))
    dense = weights["network/dense_3_13/dense_3_13/kernel"]
    assert state["fc3.weight"][2, 9] == pytest.approx(float(dense[9, 2]))
    gamma = weights["network/batchnorm_features/gamma"]
    np.testing.assert_array_equal(state["bn_features.weight"].numpy(), gamma)
    assert int(state["bn_features.num_batches_tracked"]) == 0


def test_fc1_rows_are_permuted_from_hwc_to_chw() -> None:
    weights = random_m1(2)
    kernel = np.zeros((2304, 64), dtype=np.float32)
    # Row of TF's (h, w, c) flatten: h * 6 * 64 + w * 64 + c; encode (h, w, c) in the value.
    for h in range(6):
        for w in range(6):
            for c in range(64):
                kernel[h * 384 + w * 64 + c, :] = 10_000 * h + 100 * w + c
    weights["network/dense_1_10/dense_1_10/kernel"] = kernel
    fc1 = to_state_dict(weights)["fc1.weight"].numpy()
    for c, h, w in [(0, 0, 0), (5, 1, 2), (63, 5, 5), (17, 4, 0)]:
        assert fc1[3, c * 36 + h * 6 + w] == 10_000 * h + 100 * w + c


@pytest.mark.parametrize("with_features", [True, False])
def test_numpy_and_torch_agree_on_random_weights(with_features: bool) -> None:
    weights = random_m1(3, with_features=with_features)
    model = load_m1_model(weights)
    rng = np.random.default_rng(4)
    crops = rng.uniform(0, 1, size=(9, 21, 21, 3)).astype(np.float32)
    features = rng.normal(size=(9, 23)).astype(np.float32)
    with torch.no_grad():
        expected = model(
            torch.from_numpy(np.ascontiguousarray(crops.transpose(0, 3, 1, 2))),
            torch.from_numpy(features),
        ).numpy()
    got = numpy_forward(weights, crops, features if with_features else None, chunk=4)
    assert got.dtype == np.float64
    assert np.abs(got - expected).max() <= 1e-4
    assert (got.argmax(1) == expected.argmax(1)).all()
    assert np.abs(expected).max() > 0.1  # the check is not vacuous


def test_numpy_forward_on_27_pixel_crops() -> None:
    weights = random_m1(5)
    rng = np.random.default_rng(6)
    crops = rng.uniform(0, 1, size=(3, 21, 21, 3))
    features = rng.normal(size=(3, 23))
    padded = np.pad(crops, ((0, 0), (3, 3), (3, 3), (0, 0)))
    np.testing.assert_allclose(
        numpy_forward(weights, padded, features, pad=0), numpy_forward(weights, crops, features)
    )
    assert numpy_forward(weights, crops[:0], features[:0]).shape == (0, 5)


def test_numpy_forward_errors() -> None:
    weights = random_m1(7)
    crops, features = np.zeros((2, 21, 21, 3)), np.zeros((2, 23))
    with pytest.raises(ValueError, match="pad must be"):
        numpy_forward(weights, crops, features, pad=-1)
    with pytest.raises(ValueError, match=r"\(N, H, H, 3\)"):
        numpy_forward(weights, np.zeros((2, 21, 20, 3)), features)
    with pytest.raises(ValueError, match="not 27 pixels"):
        numpy_forward(weights, crops, features, pad=0)
    with pytest.raises(ValueError, match="needs features"):
        numpy_forward(weights, crops)
    with pytest.raises(ValueError, match="odd kernels"):
        m1._conv(np.zeros((1, 4, 4, 1)), np.zeros((2, 2, 1, 1)), np.zeros(1), same=True)


def test_weight_validation() -> None:
    weights = random_m1(8)
    del weights["network/conv_1_1/conv2d/bias"]
    weights["network/extra"] = np.zeros(1, dtype=np.float32)
    with pytest.raises(ValueError, match=r"missing \['network/conv_1_1/conv2d/bias'\]"):
        to_state_dict(weights)
    weights = random_m1(8)
    weights["network/output_logits/kernel"] = np.zeros((64, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="has shape"):
        load_m1_model(weights)


def test_default_cache_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert default_cache_dir() == tmp_path / "astro-simclr" / "m1" / "a3705ac"
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert default_cache_dir() == Path.home() / ".cache" / "astro-simclr" / "m1" / "a3705ac"


def test_fetch_downloads_verifies_and_converts(tmp_path: Path) -> None:
    served, files = synthetic_m1_dir(tmp_path)
    cache = tmp_path / "cache"
    out = fetch_m1(cache, base_url=served.as_uri(), files=files)
    assert out.weights == cache / "m1_weights.npz"
    assert out.val_events.read_bytes() == (served / "val.events").read_bytes()
    weights = load_weights(out.weights)
    expected = random_m1(9)
    assert set(weights) == set(expected)  # Adam slots and optimiser variables dropped
    for name, array in expected.items():
        np.testing.assert_array_equal(weights[name], array)
    with np.load(out.weights, allow_pickle=False) as archive:
        provenance = json.loads(str(archive["provenance"]))
    assert provenance["commit"] == m1.M1_COMMIT
    assert provenance["files"]["model.index"]["sha256"] == files["model.index"].sha256
    # A second call uses the verified cache: the source may be gone.
    for path in served.rglob("*.remote"):
        path.unlink()
    assert fetch_m1(cache, base_url=served.as_uri(), files=files).weights == out.weights


def test_fetch_from_a_source_dir(tmp_path: Path) -> None:
    served, files = synthetic_m1_dir(tmp_path)
    out = fetch_m1(tmp_path / "cache", source_dir=served, files=files)
    assert set(load_weights(out.weights)) == set(tf_shapes())


def test_fetch_uses_the_pinned_files_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    served, files = synthetic_m1_dir(tmp_path)
    monkeypatch.setattr(m1, "M1_FILES", files)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    out = fetch_m1(base_url=served.as_uri())
    assert out.weights.is_relative_to(tmp_path / "xdg")


def test_fetch_rejects_a_wrong_hash(tmp_path: Path) -> None:
    served, files = synthetic_m1_dir(tmp_path)
    files["val.events"] = RemoteFile(files["val.events"].path, files["val.events"].size, "0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        fetch_m1(tmp_path / "cache", base_url=served.as_uri(), files=files)
    assert not (tmp_path / "cache" / "val.events").exists()


def test_fetch_refuses_plain_http(tmp_path: Path) -> None:
    _, files = synthetic_m1_dir(tmp_path)
    with pytest.raises(ValueError, match="only https"):
        fetch_m1(tmp_path / "cache", base_url="http://example.invalid", files=files)


def test_fetch_rejects_a_checkpoint_without_m1(tmp_path: Path) -> None:
    served, files = synthetic_m1_dir(tmp_path)
    index, data = write_bundle({"network/other": np.zeros(3, dtype=np.float32)})
    for name, blob in {"model.index": index, "model.data-00000-of-00001": data}.items():
        (served / files[name].path).write_bytes(blob)
        files[name] = RemoteFile(files[name].path, len(blob), hashlib.sha256(blob).hexdigest())
    with pytest.raises(ValueError, match="missing"):
        fetch_m1(tmp_path / "cache", base_url=served.as_uri(), files=files)


def test_load_weights_checks_dtype(tmp_path: Path) -> None:
    weights = random_m1(10)
    weights["network/output_logits/bias"] = np.zeros(5, dtype=np.float64)
    path = tmp_path / "w.npz"
    np.savez(path, **weights)
    with pytest.raises(ValueError, match="float64"):
        load_weights(path)
