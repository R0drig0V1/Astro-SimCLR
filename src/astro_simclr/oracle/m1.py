"""M1, the published model of Carrasco-Davis et al. 2021: download, conversion and weight mapping.

Task 2.2b of phase 2 (``docs/phases/02-stamp-classifier.md``). M1 is ALeRCE's staging
model, a TF1 checkpoint vendored in ``alercebroker/pipeline`` at the pinned commit
:data:`M1_COMMIT`, under :data:`M1_DIR`. That repository has no LICENSE, so **nothing of
it is copied into this repository**: :func:`fetch_m1` downloads the files at use time into
a local cache (default :func:`default_cache_dir`, outside the repository), checks each
file's size and SHA-256 against :data:`M1_FILES`, and converts the checkpoint to a local
``.npz`` (:data:`WEIGHTS_NAME`) that must never be committed. Only the network's variables
are kept, in TF's layout and under their TF names; the optimiser's slots are dropped.

:func:`to_state_dict` is the weight mapping onto
:class:`~astro_simclr.paper_model.PaperStampClassifier`:

- convolution kernels HWIO -> OIHW (``transpose(3, 2, 0, 1)``);
- dense kernels ``(in, out)`` -> ``(out, in)``;
- fc1's input rows permuted from TF's flatten order of the NHWC map, ``(h, w, c)``, to
  PyTorch's flatten of NCHW, ``(c, h, w)``;
- the feature BatchNorm's gamma, beta and moving statistics (M1's are 0 and 1:
  ``docs/phases/02-stamp-classifier.md``, "Feature BatchNorm");
- the convolution and dense biases unchanged.

:func:`numpy_forward` is the independent reference of the port-equivalence check: M1's
forward pass in NumPy, NHWC, on the raw TF-layout weights (no transposes), float64, with
TF's symmetric "same" padding of the 5x5 convolutions. It follows ALeRCE's
``DeepHitsWithFeatures`` (``modules/networks/deep_hits_with_features.py`` at the pinned
commit) and ``augment_with_rotations_features``: the 4 rotations ``tf.image.rot90(k)`` are
stacked on the batch axis, padded by 3, through the trunk, then averaged
(``cyclic_avg_pool``).
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import torch

from astro_simclr.models import N_CLASSES, N_FEATURES
from astro_simclr.oracle.tf_files import read_bundle
from astro_simclr.paper_model import PaperStampClassifier

__all__ = [
    "M1_BASE_URL",
    "M1_COMMIT",
    "M1_DIR",
    "M1_FILES",
    "WEIGHTS_NAME",
    "M1Files",
    "RemoteFile",
    "default_cache_dir",
    "fetch_m1",
    "load_m1_model",
    "load_weights",
    "numpy_forward",
    "tf_shapes",
    "to_state_dict",
]

#: The pinned commit of ``alercebroker/pipeline`` that vendors M1.
M1_COMMIT: Final = "a3705acb58ce2e45c1f9b240b824c51260295a99"
#: M1's run directory in that repository.
M1_DIR: Final = (
    "stamp_classifier_step/model/results/staging_model/DeepHits_EntropyRegBeta0.5000_batch64_"
    "lr0.00100_droprate0.5000_inputsize21_filtersize5_0_20200708-160759"
)
#: Raw-file URL of :data:`M1_DIR` at :data:`M1_COMMIT`.
M1_BASE_URL: Final = f"https://raw.githubusercontent.com/alercebroker/pipeline/{M1_COMMIT}/{M1_DIR}"
#: The converted weights in the cache (never committed).
WEIGHTS_NAME: Final = "m1_weights.npz"
#: The variable scope of the network in M1's graph.
_PREFIX: Final = "network/"
#: Keras ``BatchNormalization``'s default epsilon, kept by M1's feature BN.
_BN_EPS: Final = 1e-3
#: Side of the last feature map (27 - 3 = 24, then two 2x2 max-pools) and its channels.
_LAST_SIDE: Final = 6
_LAST_CHANNELS: Final = 64
#: The input side of conv1 after the padding: 21 + 2 * 3.
_PADDED_SIDE: Final = 27
_N_ROTATIONS: Final = 4


@dataclass(frozen=True, slots=True)
class RemoteFile:
    """One pinned file of M1: its path under :data:`M1_DIR`, size and SHA-256."""

    path: str
    size: int
    sha256: str


#: Local name -> pinned remote file. The SHA-256 values were computed on the files served
#: at :data:`M1_COMMIT` (2026-11-08).
M1_FILES: Final[Mapping[str, RemoteFile]] = {
    "model.index": RemoteFile(
        "checkpoints/model.index",
        2416,
        "c9164a879c27793268ff5cebffd65a2d2b0ffbae744b8ad0b0e9d09bf018e2aa",
    ),
    "model.data-00000-of-00001": RemoteFile(
        "checkpoints/model.data-00000-of-00001",
        5293096,
        "aae98eba1e49b45b7ce7d49c4bf72dfe3272a137a25277856da81c88f49a9582",
    ),
    "val.events": RemoteFile(
        "tb_summaries/val/events.out.tfevents.1594238902.Peregrino",
        272732,
        "46338b25d4b6c42a4879a857ff2b350f44ea912d1df270614cc457ad712dcbfe",
    ),
}

# TF variable scope -> PaperStampClassifier attribute (the table of paper_model's docstring).
_CONVS: Final = {
    "conv_1_1/conv2d": "conv1",
    "conv_2_2/conv2d": "conv2",
    "conv_3_4/conv2d": "conv3",
    "conv_4_5/conv2d": "conv4",
    "conv_5_6/conv2d": "conv5",
}
_DENSES: Final = {
    "dense_1_10/dense_1_10": "fc1",
    "dense_2_12/dense_2_12": "fc2",
    "dense_3_13/dense_3_13": "fc3",
    "output_logits": "logits",
}
_BN: Final = "batchnorm_features"
_BN_TORCH: Final = {
    "gamma": "weight",
    "beta": "bias",
    "moving_mean": "running_mean",
    "moving_variance": "running_var",
}


def tf_shapes(*, with_features: bool = True) -> dict[str, tuple[int, ...]]:
    """TF name -> shape of every network variable of M1 (``DeepHitsExtraLayer`` without features).

    M1 itself has features. Kernels are HWIO and ``(in, out)``, as stored.
    """
    hidden = 64
    convs = [(4, 3, 32), (5, 32, 32), (5, 32, 64), (5, 64, 64), (5, 64, 64)]
    shapes: dict[str, tuple[int, ...]] = {}
    for scope, (side, c_in, c_out) in zip(_CONVS, convs, strict=True):
        shapes[f"{_PREFIX}{scope}/kernel"] = (side, side, c_in, c_out)
        shapes[f"{_PREFIX}{scope}/bias"] = (c_out,)
    fc2_in = hidden + (N_FEATURES if with_features else 0)
    denses = [(_LAST_SIDE * _LAST_SIDE * _LAST_CHANNELS, hidden), (fc2_in, hidden)]
    denses += [(hidden, hidden), (hidden, N_CLASSES)]
    for scope, (n_in, n_out) in zip(_DENSES, denses, strict=True):
        shapes[f"{_PREFIX}{scope}/kernel"] = (n_in, n_out)
        shapes[f"{_PREFIX}{scope}/bias"] = (n_out,)
    if with_features:
        for name in _BN_TORCH:
            shapes[f"{_PREFIX}{_BN}/{name}"] = (N_FEATURES,)
    return shapes


def _check_weights(weights: Mapping[str, npt.NDArray[Any]]) -> bool:
    """Validate the names and shapes of ``weights``; return whether it has features.

    Raises:
        ValueError: if a variable is missing, unexpected, or has the wrong shape.
    """
    with_features = f"{_PREFIX}{_BN}/gamma" in weights
    expected = tf_shapes(with_features=with_features)
    missing = sorted(set(expected) - set(weights))
    extra = sorted(set(weights) - set(expected))
    if missing or extra:
        raise ValueError(f"M1 weights: missing {missing}, unexpected {extra}")
    for name, shape in expected.items():
        if tuple(weights[name].shape) != shape:
            raise ValueError(f"M1 weights: {name} has shape {weights[name].shape}, not {shape}")
    return with_features


def _fc1_rows_to_torch(kernel: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """fc1's TF kernel ``(h*w*c, out)`` -> torch weight ``(out, c*h*w)``.

    TF flattens the NHWC map so that input ``h * 6 * 64 + w * 64 + c`` is pixel (h, w),
    channel c; torch flattens NCHW as ``c * 36 + h * 6 + w``.
    """
    n_out = kernel.shape[1]
    grid = kernel.reshape(_LAST_SIDE, _LAST_SIDE, _LAST_CHANNELS, n_out)  # (h, w, c, out)
    return grid.transpose(3, 2, 0, 1).reshape(n_out, -1)


def to_state_dict(weights: Mapping[str, npt.NDArray[Any]]) -> dict[str, torch.Tensor]:
    """The :class:`PaperStampClassifier` ``state_dict`` of TF-layout M1 ``weights``.

    ``weights`` maps TF names (``network/...``) to arrays in TF's layout, as
    :func:`load_weights` returns them. The mapping is in the module docstring; the result
    is float32, and ``bn_features.num_batches_tracked`` is 0 (with features).

    Raises:
        ValueError: if a variable is missing, unexpected, or has the wrong shape.
    """
    with_features = _check_weights(weights)

    def tensor(array: npt.NDArray[Any]) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))

    state: dict[str, torch.Tensor] = {}
    for scope, name in _CONVS.items():
        state[f"{name}.weight"] = tensor(weights[f"{_PREFIX}{scope}/kernel"].transpose(3, 2, 0, 1))
        state[f"{name}.bias"] = tensor(weights[f"{_PREFIX}{scope}/bias"])
    for scope, name in _DENSES.items():
        kernel = weights[f"{_PREFIX}{scope}/kernel"]
        state[f"{name}.weight"] = tensor(_fc1_rows_to_torch(kernel) if name == "fc1" else kernel.T)
        state[f"{name}.bias"] = tensor(weights[f"{_PREFIX}{scope}/bias"])
    if with_features:
        for tf_name, torch_name in _BN_TORCH.items():
            state[f"bn_features.{torch_name}"] = tensor(weights[f"{_PREFIX}{_BN}/{tf_name}"])
        state["bn_features.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)
    return state


def load_m1_model(weights: Mapping[str, npt.NDArray[Any]]) -> PaperStampClassifier:
    """A :class:`PaperStampClassifier` holding ``weights``, in eval mode.

    With or without features as ``weights`` has the feature BN. The ``state_dict`` is
    loaded strictly, so every parameter and buffer of the model is set.

    Raises:
        ValueError: as :func:`to_state_dict`.
    """
    state = to_state_dict(weights)
    model = PaperStampClassifier(with_features="bn_features.weight" in state)
    model.load_state_dict(state, strict=True)
    return model.eval()


def _conv(
    x: npt.NDArray[np.float64],
    kernel: npt.NDArray[np.float64],
    bias: npt.NDArray[np.float64],
    *,
    same: bool,
) -> npt.NDArray[np.float64]:
    """NHWC convolution, stride 1, of an HWIO ``kernel``: a sum of shifted matrix products.

    ``same`` pads ``k // 2`` on every side, which is TF's "same" for odd kernels.
    """
    k_h, k_w = kernel.shape[:2]
    if same:
        if k_h % 2 == 0 or k_w % 2 == 0:
            raise ValueError("'same' padding is symmetric only for odd kernels")
        x = np.pad(x, ((0, 0), (k_h // 2, k_h // 2), (k_w // 2, k_w // 2), (0, 0)))
    out_h, out_w = x.shape[1] - k_h + 1, x.shape[2] - k_w + 1
    out = np.broadcast_to(bias, (x.shape[0], out_h, out_w, kernel.shape[3])).copy()
    for i in range(k_h):
        for j in range(k_w):
            out += x[:, i : i + out_h, j : j + out_w, :] @ kernel[i, j]
    return out


def _relu(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.maximum(x, 0.0)


def _max_pool(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """2x2 max-pool, stride 2, "valid" (Keras ``MaxPooling2D``), NHWC."""
    n, h, w, c = x.shape
    trimmed = x[:, : h - h % 2, : w - w % 2, :]
    pooled: npt.NDArray[np.float64] = trimmed.reshape(n, h // 2, 2, w // 2, 2, c).max(axis=(2, 4))
    return pooled


def _forward_chunk(
    w: Mapping[str, npt.NDArray[np.float64]],
    images: npt.NDArray[np.float64],
    features: npt.NDArray[np.float64] | None,
    pad: int,
) -> npt.NDArray[np.float64]:
    n = images.shape[0]
    x = np.concatenate([np.rot90(images, k, axes=(1, 2)) for k in range(_N_ROTATIONS)])
    x = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    scopes = [f"{_PREFIX}{scope}" for scope in _CONVS]
    x = _relu(_conv(x, w[f"{scopes[0]}/kernel"], w[f"{scopes[0]}/bias"], same=False))
    x = _max_pool(_relu(_conv(x, w[f"{scopes[1]}/kernel"], w[f"{scopes[1]}/bias"], same=True)))
    for scope in scopes[2:4]:
        x = _relu(_conv(x, w[f"{scope}/kernel"], w[f"{scope}/bias"], same=True))
    x = _max_pool(_relu(_conv(x, w[f"{scopes[4]}/kernel"], w[f"{scopes[4]}/bias"], same=True)))
    dense = [f"{_PREFIX}{scope}" for scope in _DENSES]
    h = _relu(x.reshape(len(x), -1) @ w[f"{dense[0]}/kernel"] + w[f"{dense[0]}/bias"])
    h = h.reshape(_N_ROTATIONS, n, -1).mean(axis=0)
    if features is not None:
        bn = f"{_PREFIX}{_BN}"
        scale = w[f"{bn}/gamma"] / np.sqrt(w[f"{bn}/moving_variance"] + _BN_EPS)
        h = np.concatenate([h, (features - w[f"{bn}/moving_mean"]) * scale + w[f"{bn}/beta"]], 1)
    for scope in dense[1:3]:
        h = _relu(h @ w[f"{scope}/kernel"] + w[f"{scope}/bias"])
    logits: npt.NDArray[np.float64] = h @ w[f"{dense[3]}/kernel"] + w[f"{dense[3]}/bias"]
    return logits


def numpy_forward(
    weights: Mapping[str, npt.NDArray[Any]],
    images: npt.NDArray[np.floating[Any]],
    features: npt.NDArray[np.floating[Any]] | None = None,
    *,
    pad: int = 3,
    chunk: int = 64,
) -> npt.NDArray[np.float64]:
    """M1's logits in NumPy, NHWC, on TF-layout ``weights``: the reference of check (1).

    ``images`` is ``(N, H, W, 3)`` with ``H + 2 * pad == W + 2 * pad == 27`` (the paper's
    21x21 crops with ``pad=3``, or 27x27 crops with ``pad=0``); ``features`` is
    ``(N, 23)`` when ``weights`` has the feature BN, and ignored otherwise. The pass is
    ALeRCE's (module docstring), in float64, evaluated as in inference: no dropout, and the
    BN uses its moving statistics. Returns ``(N, 5)`` float64 logits. ``chunk`` stamps go
    through at a time (4 rotations each), which bounds the memory.

    Raises:
        ValueError: if a weight is missing or misshaped, the images do not reach 27 pixels
            after padding, ``pad`` or ``chunk`` is invalid, or the features are missing or
            misshaped.
    """
    with_features = _check_weights(weights)
    if pad < 0 or chunk < 1:
        raise ValueError(f"pad must be >= 0 and chunk >= 1, got {pad} and {chunk}")
    if images.ndim != 4 or images.shape[3] != 3 or images.shape[1] != images.shape[2]:
        raise ValueError(f"images must be (N, H, H, 3), got {images.shape}")
    if images.shape[1] + 2 * pad != _PADDED_SIDE:
        raise ValueError(f"images of side {images.shape[1]} with pad {pad} are not 27 pixels")
    if with_features and (features is None or features.shape != (len(images), N_FEATURES)):
        got = None if features is None else features.shape
        raise ValueError(f"M1 with features needs features of shape (N, {N_FEATURES}); got {got}")
    w = {name: np.asarray(array, dtype=np.float64) for name, array in weights.items()}
    parts = []
    for start in range(0, len(images), chunk):
        block = np.asarray(images[start : start + chunk], dtype=np.float64)
        feats = None
        if with_features and features is not None:
            feats = np.asarray(features[start : start + chunk], dtype=np.float64)
        parts.append(_forward_chunk(w, block, feats, pad))
    if not parts:
        return np.empty((0, N_CLASSES), dtype=np.float64)
    return np.concatenate(parts)


def default_cache_dir() -> Path:
    """``$XDG_CACHE_HOME/astro-simclr/m1/<commit>`` (``~/.cache`` by default).

    Outside the repository on purpose: the files are ALeRCE's and must never be committed.
    """
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "astro-simclr" / "m1" / M1_COMMIT[:7]


@dataclass(frozen=True, slots=True)
class M1Files:
    """The local M1 files :func:`fetch_m1` produced: the converted weights and the val log."""

    weights: Path
    val_events: Path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download(url: str) -> bytes:
    """GET ``url``; only ``https://`` and ``file://`` URLs are accepted.

    Raises:
        ValueError: for another scheme.
        OSError: if the request fails (``urllib.error.URLError`` is an ``OSError``).
    """
    if not url.startswith(("https://", "file://")):
        raise ValueError(f"refusing to download {url!r}: only https:// and file:// URLs")
    # The scheme is checked just above, so S310 (arbitrary schemes) does not apply.
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
        data: bytes = response.read()
    return data


def _fetch_file(
    name: str, remote: RemoteFile, cache_dir: Path, base_url: str, source_dir: Path | None
) -> Path:
    """Put ``remote`` in ``cache_dir/name`` (verified), from ``source_dir`` or the URL."""
    dest = cache_dir / name
    if dest.is_file() and _sha256(dest.read_bytes()) == remote.sha256:
        return dest
    if source_dir is not None:
        data = (source_dir / name).read_bytes()
        origin = str(source_dir / name)
    else:
        origin = f"{base_url}/{remote.path}"
        data = _download(origin)
    digest = _sha256(data)
    if len(data) != remote.size or digest != remote.sha256:
        msg = (
            f"{origin}: {len(data)} bytes with SHA-256 {digest}; expected {remote.size} "
            f"bytes with SHA-256 {remote.sha256}"
        )
        raise ValueError(msg)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return dest


def fetch_m1(
    cache_dir: Path | None = None,
    *,
    base_url: str = M1_BASE_URL,
    source_dir: Path | None = None,
    files: Mapping[str, RemoteFile] | None = None,
) -> M1Files:
    """Download M1 (pinned commit, SHA-256 checked) and convert it to a local ``.npz``.

    The files of ``files`` (default :data:`M1_FILES`, read at call time) go to
    ``cache_dir`` (default :func:`default_cache_dir`) under their local names; a cached
    file whose SHA-256 matches is not downloaded again. With
    ``source_dir``, they are copied from that directory (same local names) instead of
    downloaded, and verified the same way. The checkpoint is then converted to
    ``cache_dir/m1_weights.npz``: the ``network/`` variables of M1, TF names and layout,
    float32, plus a ``provenance`` entry (a JSON string: commit, directory, file hashes).
    Nothing is unpickled, and the ``.npz`` loads with ``allow_pickle=False``.

    Raises:
        ValueError: if a file has the wrong size or SHA-256, or the checkpoint does not hold
            M1's variables; ``TFFormatError`` (a ``ValueError``) if it is corrupt.
        OSError: if a download or a file operation fails.
    """
    cache = default_cache_dir() if cache_dir is None else cache_dir
    files = M1_FILES if files is None else files
    cache.mkdir(parents=True, exist_ok=True)
    local = {
        name: _fetch_file(name, remote, cache, base_url, source_dir)
        for name, remote in files.items()
    }
    weights_path = cache / WEIGHTS_NAME
    tensors = read_bundle(
        local["model.index"].read_bytes(),
        local["model.data-00000-of-00001"].read_bytes(),
        prefix=_PREFIX,
    )
    with_features = f"{_PREFIX}{_BN}/gamma" in tensors
    expected = tf_shapes(with_features=with_features)
    # The Adam slots share the prefix; a missing variable is reported by _check_weights.
    network = {name: array for name, array in tensors.items() if name in expected}
    _check_weights(network)
    provenance = {
        "repository": "alercebroker/pipeline",
        "commit": M1_COMMIT,
        "directory": M1_DIR,
        "files": {name: {"path": r.path, "sha256": r.sha256} for name, r in files.items()},
    }
    tmp = weights_path.with_name("m1_weights.part.npz")
    # Any: numpy's stubs would match the dict against savez's keyword-only parameters.
    arrays: dict[str, Any] = {"provenance": np.array(json.dumps(provenance, sort_keys=True))}
    arrays.update(network)
    np.savez(tmp, **arrays)
    tmp.replace(weights_path)
    return M1Files(weights=weights_path, val_events=local["val.events"])


def load_weights(path: Path) -> dict[str, npt.NDArray[np.float32]]:
    """The TF-layout M1 weights of a ``.npz`` written by :func:`fetch_m1`.

    Loaded with ``allow_pickle=False``; the ``provenance`` entry is dropped.

    Raises:
        OSError: if the file cannot be read.
        ValueError: if it is not such a ``.npz`` (names, shapes or dtype).
    """
    with np.load(path, allow_pickle=False) as archive:
        weights = {name: archive[name] for name in archive.files if name != "provenance"}
    for name, array in weights.items():
        if array.dtype != np.float32:
            raise ValueError(f"{path}: {name} is {array.dtype}, not float32")
    _check_weights(weights)
    return weights
