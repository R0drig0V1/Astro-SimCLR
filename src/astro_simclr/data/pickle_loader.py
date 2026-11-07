"""The lenient pickle loader: the only place in the package that unpickles.

The ZTF stamp datasets exist only as pickles (``docs/data.md``). :func:`load_pickle` reads
one, checks it against its schema (``classifier`` or ``pretraining``, see
:mod:`astro_simclr.data.schema`) and returns a :class:`StampData` with uint8 stamps. It is
lenient about containers: every key may hold a list or an ndarray, the features may be
object arrays (as in the real files), and the labels may be numpy or Python integers.
It is strict about content: wrong shapes, dtypes, label values, non-finite values, missing
or unexpected keys all raise :class:`SchemaError`, naming the file, split and key.

Stamps are converted with :func:`stamp_to_uint8` one by one, in the dtype they were
*stored* in. Converting float64 to float32 first changes the rounding of some pixels
(59 out of 10M uniform values), and stamp by stamp the peak memory stays at the unpickled
object plus the uint8 result.

Training never calls this module: ``prepare-data`` converts each pickle once into the
prepared ``.npy`` layout (:mod:`astro_simclr.data.prepared`).
"""

from __future__ import annotations

import importlib
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np
import numpy.typing as npt

from astro_simclr.data.preprocessing import stamp_to_uint8
from astro_simclr.data.schema import (
    FEATURE_NAMES,
    LABEL_NAMES,
    SPLITS,
    STAMP_SHAPE,
    DataError,
    Kind,
    SchemaError,
    SplitData,
    StampData,
    TruncatedFileError,
)

__all__ = ["load_pickle", "read_pickle", "validate"]

#: The last byte of every pickle (protocols 0-5): the STOP opcode.
_STOP = b"."

#: Keys of a labelled split.
_LABELLED_KEYS = frozenset({"images", "labels", "features"})

#: Globals a stamp pickle may reference: numpy arrays, dtypes and scalars, under both
#: the numpy 1.x (``numpy.core``) and 2.x (``numpy._core``) module names.
_ALLOWED_GLOBALS: dict[tuple[str, str], tuple[str, str]] = {
    ("numpy", "ndarray"): ("numpy", "ndarray"),
    ("numpy", "dtype"): ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"): ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"): ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"): ("numpy._core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"): ("numpy._core.multiarray", "scalar"),
}


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """An unpickler that resolves only the numpy globals of :data:`_ALLOWED_GLOBALS`."""

    def find_class(self, module: str, name: str) -> Any:
        target = _ALLOWED_GLOBALS.get((module, name))
        if target is None:
            raise pickle.UnpicklingError(f"global {module}.{name} is not allowed in a stamp pickle")
        return getattr(importlib.import_module(target[0]), target[1])


def _check_complete(path: Path, handle: BinaryIO) -> None:
    """Fail fast, before unpickling gigabytes, if the file lacks its final STOP opcode."""
    size = handle.seek(0, 2)
    if size == 0:
        raise TruncatedFileError(f"{path}: the file is empty")
    handle.seek(size - 1)
    if handle.read(1) != _STOP:
        raise TruncatedFileError(
            f"{path}: the pickle is truncated (it does not end with the STOP opcode; "
            f"{size} bytes). Download or copy it again and check its SHA-256."
        )
    handle.seek(0)


def read_pickle(path: Path) -> object:
    """Unpickle ``path`` and return the raw object, without schema checks.

    Trust assumption: unpickling can run arbitrary code, so call this only on files you
    trust (the project's own ZTF datasets, from a known source, with a verified SHA-256).
    As defence in depth, only the numpy globals of :data:`_ALLOWED_GLOBALS` are resolved;
    any other class in the file is an error. That narrows the risk, but does not make
    untrusted pickles safe.

    The whole object is held in memory: about 1.1x the file size.

    Raises:
        DataError: if the file does not exist or cannot be read.
        TruncatedFileError: if the file is empty or ends before its STOP opcode.
        DataError: if unpickling fails (corrupt data, a disallowed global).
    """
    try:
        handle = path.open("rb")
    except OSError as err:
        raise DataError(f"{path}: cannot open: {err.strerror or err}") from None
    with handle:
        _check_complete(path, handle)
        try:
            # The one unpickling call of the package; see the trust assumption above. (Ruff's
            # S301 does not flag an Unpickler subclass, so no noqa is needed: RUF100.)
            return _NumpyOnlyUnpickler(handle).load()
        except EOFError:
            raise TruncatedFileError(
                f"{path}: the pickle is truncated (it ends in the middle of the data). "
                "Download or copy it again and check its SHA-256."
            ) from None
        except Exception as err:  # corrupt pickles raise almost any exception type
            raise DataError(f"{path}: cannot unpickle ({type(err).__name__}: {err})") from None


def load_pickle(path: Path) -> StampData:
    """Read a ZTF stamp pickle and validate it (see :func:`validate`).

    Raises:
        DataError: if the file cannot be read or unpickled (see :func:`read_pickle`).
        SchemaError: if its content does not match the classifier or pretraining schema.
    """
    return validate(path, read_pickle(path))


def validate(path: Path, raw: object) -> StampData:
    """Check an unpickled object against its schema and convert it to :class:`StampData`.

    The kind is ``pretraining`` when the ``Train`` split holds only ``images``, otherwise
    ``classifier``. ``path`` is used only in error messages.

    Raises:
        SchemaError: on a missing or unexpected split or key, an empty split, a stamp that
            is not 63x63x3 floats in [0, 1], a label outside 0-4 or not an integer,
            features that are not 26 finite numbers per stamp, or keys of different
            lengths within a split.
    """
    top = _mapping(path, raw, None)
    _check_keys(path, top, frozenset(SPLITS), None, "splits")
    train = _mapping(path, top["Train"], "Train")
    kind: Kind = "pretraining" if set(train) == {"images"} else "classifier"
    splits: dict[str, SplitData] = {}
    for split in SPLITS:
        content = _mapping(path, top[split], split)
        unlabelled = kind == "pretraining" and split == "Train"
        _check_keys(
            path, content, frozenset({"images"}) if unlabelled else _LABELLED_KEYS, split, "keys"
        )
        images = _images(path, split, content["images"])
        if unlabelled:
            splits[split] = SplitData(images, None, None)
            continue
        labels = _labels(path, split, content["labels"], len(images))
        features = _features(path, split, content["features"], len(images))
        splits[split] = SplitData(images, labels, features)
    return StampData(kind, splits)


def _mapping(path: Path, value: object, split: str | None) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        what = "the top level" if split is None else "the split"
        raise SchemaError(path, f"{what} must be a dict, got {type(value).__name__}", split=split)
    return cast("Mapping[str, object]", value)


def _check_keys(
    path: Path,
    value: Mapping[str, object],
    expected: frozenset[str],
    split: str | None,
    what: str,
) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(map(str, set(value) - expected))
    if missing or unexpected:
        parts = [f"missing {what} {missing}"] if missing else []
        parts += [f"unexpected {what} {unexpected}"] if unexpected else []
        raise SchemaError(path, "; ".join(parts) + f" (expected {sorted(expected)})", split=split)


def _sequence(path: Path, split: str, key: str, value: object) -> Sequence[Any] | npt.NDArray[Any]:
    if not isinstance(value, list | tuple | np.ndarray):
        raise SchemaError(
            path, f"must be a list or an ndarray, got {type(value).__name__}", split=split, key=key
        )
    if len(value) == 0:
        raise SchemaError(path, "is empty", split=split, key=key)
    return value


def _images(path: Path, split: str, value: object) -> npt.NDArray[np.uint8]:
    items = _sequence(path, split, "images", value)
    out = np.empty((len(items), *STAMP_SHAPE), dtype=np.uint8)
    for index, item in enumerate(items):
        stamp = np.asarray(item)  # an ndarray item keeps its stored dtype
        if stamp.shape != STAMP_SHAPE:
            raise SchemaError(
                path,
                f"stamp {index} has shape {stamp.shape}, expected {STAMP_SHAPE}",
                split=split,
                key="images",
            )
        try:
            out[index] = stamp_to_uint8(stamp)
        except ValueError as err:
            raise SchemaError(path, f"stamp {index}: {err}", split=split, key="images") from None
    return out


def _labels(path: Path, split: str, value: object, n: int) -> npt.NDArray[np.int64]:
    labels = np.asarray(_sequence(path, split, "labels", value))
    if labels.ndim != 1 or len(labels) != n:
        raise SchemaError(
            path,
            f"must be {n} integers (one per stamp), got shape {labels.shape}",
            split=split,
            key="labels",
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise SchemaError(
            path, f"must be integers, got dtype {labels.dtype}", split=split, key="labels"
        )
    bad = (labels < 0) | (labels >= len(LABEL_NAMES))
    if bad.any():
        index = int(np.argmax(bad))
        raise SchemaError(
            path,
            f"label {int(labels[index])} at {index} is outside 0..{len(LABEL_NAMES) - 1}",
            split=split,
            key="labels",
        )
    return labels.astype(np.int64)


def _features(path: Path, split: str, value: object, n: int) -> npt.NDArray[np.float64]:
    items = _sequence(path, split, "features", value)
    try:
        features = np.asarray(items, dtype=np.float64)
    except (TypeError, ValueError) as err:
        raise SchemaError(
            path,
            f"must be numbers, {len(FEATURE_NAMES)} per stamp ({err})",
            split=split,
            key="features",
        ) from None
    expected = (n, len(FEATURE_NAMES))
    if features.shape != expected:
        raise SchemaError(
            path, f"has shape {features.shape}, expected {expected}", split=split, key="features"
        )
    if not np.isfinite(features).all():
        row = int(np.argmax(~np.isfinite(features).all(axis=1)))
        raise SchemaError(path, f"row {row} has NaN or inf values", split=split, key="features")
    return features
