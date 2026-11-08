"""The prepared layout: the ``.npy`` form of one source pickle, the only format training reads.

``prepare-data`` runs once per pickle, on a machine that can hold it in RAM, and writes::

    OUT/<source stem>/
        meta.json                 source name, size and SHA-256; kind; schema; per-file
                                  dtype, shape and SHA-256; class counts
        Train/images.npy          uint8 (N, 63, 63, 3)
        Train/labels.npy          int64 (N,)        (labelled splits only)
        Train/features.npy        float64 (N, 26)   (labelled splits only; all 26 columns)
        Validation/...  Test/...

:func:`load_prepared` opens it with ``np.load(allow_pickle=False, mmap_mode="r")``, so
repeated runs never unpickle anything and never hold the stamps in RAM. The writer is
deterministic (no timestamps), so the same pickle always gives the same bytes.
"""

from __future__ import annotations

import json
import shutil
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from astro_simclr.data.files import sha256_file
from astro_simclr.data.pickle_loader import load_pickle
from astro_simclr.data.schema import (
    FEATURE_NAMES,
    LABEL_NAMES,
    SPLITS,
    STAMP_SHAPE,
    DataError,
    SchemaError,
    SplitData,
    StampData,
)

__all__ = [
    "FORMAT",
    "FORMAT_VERSION",
    "META_NAME",
    "is_prepared",
    "load_prepared",
    "prepare",
    "read_meta",
    "verify_prepared",
    "write_prepared",
]

#: ``meta.json["format"]`` of a prepared directory.
FORMAT = "astro-simclr-prepared"
#: Bumped on any incompatible change of the layout.
FORMAT_VERSION = 1
#: Name of the metadata file at the root of a prepared directory.
META_NAME = "meta.json"

#: Array file -> (dtype, shape after N) of each key.
_ARRAYS: dict[str, tuple[np.dtype[Any], tuple[int, ...]]] = {
    "images": (np.dtype(np.uint8), STAMP_SHAPE),
    "labels": (np.dtype(np.int64), ()),
    "features": (np.dtype(np.float64), (len(FEATURE_NAMES),)),
}


def is_prepared(path: Path) -> bool:
    """True if ``path`` is a directory with a ``meta.json`` (a prepared layout)."""
    return path.is_dir() and (path / META_NAME).is_file()


def prepare(source: Path, out_dir: Path, *, overwrite: bool = False) -> Path:
    """Convert the pickle ``source`` into ``out_dir/<source stem>/`` and return that path.

    The pickle is hashed, loaded and validated (:func:`load_pickle`), then written.

    Raises:
        DataError: if the source cannot be loaded (see :func:`load_pickle`), or the
            destination exists and ``overwrite`` is false (or is not a prepared layout).
        OSError: if the destination cannot be written.
    """
    if not source.is_file():
        raise DataError(f"{source}: no such file")
    sha256 = sha256_file(source)
    data = load_pickle(source)
    info = {"name": source.name, "size_bytes": source.stat().st_size, "sha256": sha256}
    return write_prepared(data, out_dir / source.stem, info, overwrite=overwrite)


def write_prepared(
    data: StampData, dest: Path, source: dict[str, Any], *, overwrite: bool = False
) -> Path:
    """Write ``data`` as a prepared layout at ``dest`` and return ``dest``.

    ``source`` is recorded as ``meta.json["source"]`` (name, size and SHA-256 of the
    pickle). The layout is written to a sibling ``.partial`` directory and renamed at the
    end, so an interrupted run never leaves a directory that looks complete.

    Raises:
        DataError: if ``dest`` exists and ``overwrite`` is false, or if ``dest`` exists but
            is not a prepared layout (it is never deleted then).
        OSError: if the destination cannot be written.
    """
    if dest.exists():
        if not overwrite:
            raise DataError(f"{dest} already exists; pass --overwrite to replace it")
        if not is_prepared(dest):
            raise DataError(f"{dest} exists and is not a prepared layout; not replacing it")
    partial = dest.with_name(f".{dest.name}.partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)

    splits_meta: dict[str, Any] = {}
    for split, content in data.splits.items():
        (partial / split).mkdir()
        files: dict[str, Any] = {}
        for key in _ARRAYS:
            array = getattr(content, key)
            if array is None:
                continue
            file = partial / split / f"{key}.npy"
            np.save(file, np.ascontiguousarray(array), allow_pickle=False)
            files[file.name] = {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": sha256_file(file),
            }
        splits_meta[split] = {
            "n": len(content),
            "class_counts": content.class_counts(),
            "files": files,
        }

    meta = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "kind": data.kind,
        "source": source,
        "created_by": f"astro-simclr {version('astro-simclr')}",
        "schema": {
            "stamp_shape": list(STAMP_SHAPE),
            "label_names": list(LABEL_NAMES),
            "feature_names": list(FEATURE_NAMES),
        },
        "splits": splits_meta,
    }
    (partial / META_NAME).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if dest.exists():
        shutil.rmtree(dest)
    partial.rename(dest)
    return dest


def read_meta(path: Path) -> dict[str, Any]:
    """Read and check ``path/meta.json``.

    Raises:
        DataError: if it is missing or unreadable, not JSON, or of another format or
            format version.
    """
    file = path / META_NAME
    try:
        meta = json.loads(file.read_text(encoding="utf-8"))
    except OSError as err:
        raise DataError(
            f"{path}: not a prepared layout (cannot read {META_NAME}: {err.strerror or err})"
        ) from None
    except json.JSONDecodeError as err:
        raise DataError(f"{file}: invalid JSON ({err})") from None
    if not isinstance(meta, dict) or meta.get("format") != FORMAT:
        raise DataError(f"{file}: not an {FORMAT} file")
    if meta.get("format_version") != FORMAT_VERSION:
        raise DataError(
            f"{file}: format version {meta.get('format_version')!r}; this "
            f"version of astro-simclr reads version {FORMAT_VERSION}. Run "
            "prepare-data again."
        )
    return meta


def load_prepared(path: Path) -> StampData:
    """Open the prepared layout at ``path``, memory-mapped and read-only.

    Checks every array against the schema and against ``meta.json`` (dtype and shape),
    the labels' range and the features' finiteness. The stamps themselves are not read;
    :func:`verify_prepared` checks their hashes.

    Raises:
        DataError: if ``meta.json`` is missing or invalid (see :func:`read_meta`).
        SchemaError: if a file is missing, cannot be loaded without pickle, or does not
            match the schema or ``meta.json``.
    """
    meta = read_meta(path)
    kind = meta.get("kind")
    if kind not in ("classifier", "pretraining"):
        raise SchemaError(path, f"unknown kind {kind!r} in {META_NAME}")
    splits_meta = meta.get("splits")
    if not isinstance(splits_meta, dict) or sorted(splits_meta) != sorted(SPLITS):
        raise SchemaError(path, f"{META_NAME} must list the splits {list(SPLITS)}")
    splits: dict[str, SplitData] = {}
    for split in SPLITS:
        unlabelled = kind == "pretraining" and split == "Train"
        files = splits_meta[split].get("files", {})
        expected = ["images.npy"] if unlabelled else [f"{key}.npy" for key in _ARRAYS]
        if sorted(files) != sorted(expected):
            raise SchemaError(
                path, f"{META_NAME} lists {sorted(files)}, expected {sorted(expected)}", split=split
            )
        arrays = {
            key: _load_array(path, split, key, files[f"{key}.npy"])
            for key in _ARRAYS
            if f"{key}.npy" in files
        }
        content = SplitData(arrays["images"], arrays.get("labels"), arrays.get("features"))
        _check_split(path, split, content)
        splits[split] = content
    return StampData(kind, splits)


def verify_prepared(path: Path) -> None:
    """Recompute the SHA-256 of every array file of ``path`` and compare with ``meta.json``.

    Raises:
        DataError: if ``meta.json`` is invalid, a file is missing, or a hash differs.
    """
    meta = read_meta(path)
    for split, split_meta in meta["splits"].items():
        for name, info in split_meta["files"].items():
            file = path / split / name
            if not file.is_file():
                raise DataError(f"{file}: missing")
            if sha256_file(file) != info["sha256"]:
                raise DataError(
                    f"{file}: SHA-256 differs from {META_NAME}; the file was "
                    "modified or corrupted. Run prepare-data again."
                )


def _load_array(path: Path, split: str, key: str, info: dict[str, Any]) -> npt.NDArray[Any]:
    file = path / split / f"{key}.npy"
    try:
        array: npt.NDArray[Any] = np.load(file, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as err:
        raise SchemaError(path, f"cannot load {file.name} ({err})", split=split, key=key) from None
    dtype, tail = _ARRAYS[key]
    if array.dtype != dtype or array.shape[1:] != tail or array.ndim != 1 + len(tail):
        raise SchemaError(
            path,
            f"{file.name} is {array.dtype} {array.shape}, expected "
            f"{dtype} (N, {', '.join(map(str, tail))})".replace(", )", ")"),
            split=split,
            key=key,
        )
    if list(array.shape) != info.get("shape") or str(array.dtype) != info.get("dtype"):
        raise SchemaError(
            path,
            f"{file.name} is {array.dtype} {array.shape}, but "
            f"{META_NAME} says {info.get('dtype')} {info.get('shape')}",
            split=split,
            key=key,
        )
    return array


def _check_split(path: Path, split: str, content: SplitData) -> None:
    n = len(content)
    if n == 0:
        raise SchemaError(path, "is empty", split=split, key="images")
    for key in ("labels", "features"):
        array = getattr(content, key)
        if array is not None and len(array) != n:
            raise SchemaError(
                path, f"has {len(array)} rows, but images has {n}", split=split, key=key
            )
    if content.labels is not None and (
        content.labels.min() < 0 or content.labels.max() >= len(LABEL_NAMES)
    ):
        raise SchemaError(
            path, f"labels outside 0..{len(LABEL_NAMES) - 1}", split=split, key="labels"
        )
    if content.features is not None and not np.isfinite(content.features).all():
        raise SchemaError(path, "has NaN or inf values", split=split, key="features")
