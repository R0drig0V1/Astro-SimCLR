"""``inspect-data``: describe a stamp pickle or a prepared layout, and check it.

:func:`summarize` returns a JSON-ready summary: format, kind, size and SHA-256, and per
split the stored containers, dtypes and shapes of every key and the counts per class.
:func:`format_summary` renders it as text. Inspecting is also a full check: a pickle is
validated against its schema (:func:`load_pickle`), and a prepared layout is checked
against its ``meta.json``, including the SHA-256 of every array file. The data manifest of
the preflight (task 1.7) is built from these summaries.

A ``raw-alerts`` layout (task 3.1b) has no splits: its summary has ``format: raw-alerts``,
the row count, both files, the counts per class under the layout's own label names, and
the counts per ``label_source``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from astro_simclr.data.files import sha256_file
from astro_simclr.data.pickle_loader import read_pickle, validate
from astro_simclr.data.prepared import (
    RAW_ALERTS_KIND,
    is_prepared,
    layout_kind,
    load_prepared,
    read_meta,
    read_meta_file,
    verify_prepared,
)
from astro_simclr.data.raw_alerts import load_raw_alerts, verify_raw_alerts
from astro_simclr.data.schema import LABEL_NAMES, DataError, StampData

__all__ = ["format_summary", "summarize"]


def summarize(path: Path) -> dict[str, Any]:
    """Inspect and check the pickle or prepared directory at ``path``.

    Raises:
        DataError: if ``path`` does not exist, or the data is truncated, corrupt or does
            not match its schema (the subclasses of :class:`DataError`).
    """
    if is_prepared(path):
        if layout_kind(path) == RAW_ALERTS_KIND:
            return _summarize_raw_alerts(path)
        return _summarize_prepared(path)
    if path.is_dir():
        raise DataError(f"{path}: a directory without meta.json is not a prepared layout")
    if not path.exists():
        raise DataError(f"{path}: no such file or directory")
    return _summarize_pickle(path)


def _summarize_pickle(path: Path) -> dict[str, Any]:
    raw = read_pickle(path)
    data = validate(path, raw)
    assert isinstance(raw, dict)  # validate() checked it  # noqa: S101
    stored = {
        split: {str(key): _describe(value) for key, value in content.items()}
        for split, content in raw.items()
    }
    return {
        "path": str(path),
        "format": "pickle",
        "kind": data.kind,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "splits": _splits(data, stored),
    }


def _summarize_prepared(path: Path) -> dict[str, Any]:
    meta = read_meta(path)
    data = load_prepared(path)
    verify_prepared(path)
    stored = {
        split: {
            name: f"{info['dtype']} {tuple(info['shape'])}"
            for name, info in meta["splits"][split]["files"].items()
        }
        for split in data.splits
    }
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return {
        "path": str(path),
        "format": "prepared",
        "kind": data.kind,
        "size_bytes": size,
        "source": meta["source"],
        "splits": _splits(data, stored),
    }


def _summarize_raw_alerts(path: Path) -> dict[str, Any]:
    data = load_raw_alerts(path)
    verify_raw_alerts(path)
    meta = read_meta_file(path)
    sources = Counter(str(source) for source in data.index["label_source"])
    return {
        "path": str(path),
        "format": RAW_ALERTS_KIND,
        "kind": RAW_ALERTS_KIND,
        "size_bytes": sum(f.stat().st_size for f in path.rglob("*") if f.is_file()),
        "source": meta["source"],
        "n": len(data),
        "keys": {
            name: f"{info['dtype']} {tuple(info['shape'])}" for name, info in meta["files"].items()
        },
        "class_counts": dict(zip(data.label_names, data.class_counts(), strict=True)),
        "label_sources": dict(sorted(sources.items())),
    }


def _splits(data: StampData, stored: dict[str, dict[str, str]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split, content in data.splits.items():
        counts = content.class_counts()
        out[split] = {
            "n": len(content),
            "keys": stored[split],
            "class_counts": None if counts is None else dict(zip(LABEL_NAMES, counts, strict=True)),
        }
    return out


def _describe(value: object) -> str:
    """Describe a stored value: container, length, and the dtypes/shapes of its items."""
    if isinstance(value, np.ndarray):
        return f"ndarray {value.dtype} {value.shape}"
    if isinstance(value, list | tuple):
        kinds = Counter(_item_kind(item) for item in value)
        items = ", ".join(
            kind if len(kinds) == 1 else f"{kind} x{count}" for kind, count in sorted(kinds.items())
        )
        return f"{type(value).__name__}[{len(value)}] of {items}"
    return type(value).__name__


def _item_kind(item: object) -> str:
    if isinstance(item, np.ndarray):
        return f"ndarray {item.dtype} {item.shape}"
    if isinstance(item, np.generic):
        return str(item.dtype)
    return type(item).__name__


def format_summary(summary: dict[str, Any]) -> str:
    """Render a :func:`summarize` result as human-readable text (one block per split)."""
    lines = [summary["path"]]
    lines.append(
        f"  format: {summary['format']}   kind: {summary['kind']}   "
        f"size: {summary['size_bytes']:,} bytes"
    )
    if summary["format"] == "pickle":
        lines.append(f"  sha256: {summary['sha256']}")
    elif summary["format"] == RAW_ALERTS_KIND:
        return _format_raw_alerts(summary, lines)
    else:
        source = summary["source"]
        lines.append(f"  source: {source['name']} ({source['size_bytes']:,} bytes)")
        lines.append(f"  source sha256: {source['sha256']}")
        lines.append("  array files: SHA-256 match meta.json")
    for split, info in summary["splits"].items():
        lines.append(f"  {split}: {info['n']:,} stamps")
        width = max(len(key) for key in info["keys"])
        lines.extend(f"    {key:<{width}}  {desc}" for key, desc in info["keys"].items())
        counts = info["class_counts"]
        if counts is not None:
            lines.append("    classes: " + ", ".join(f"{k} {v:,}" for k, v in counts.items()))
    return "\n".join(lines) + "\n"


def _format_raw_alerts(summary: dict[str, Any], lines: list[str]) -> str:
    lines.append(f"  source: {json.dumps(summary['source'], sort_keys=True)}")
    lines.append("  files: SHA-256 match meta.json")
    lines.append(f"  {summary['n']:,} alerts")
    width = max(len(key) for key in summary["keys"])
    lines.extend(f"    {key:<{width}}  {desc}" for key, desc in summary["keys"].items())
    lines.append(
        "    classes: " + ", ".join(f"{k} {v:,}" for k, v in summary["class_counts"].items())
    )
    sources = summary["label_sources"]
    lines.append("    label sources: " + ", ".join(f"{k} {v:,}" for k, v in sources.items()))
    return "\n".join(lines) + "\n"
