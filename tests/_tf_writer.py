"""Minimal writers of TF1 checkpoints (tensor bundle V2) and event files, for the oracle tests.

They produce, from synthetic tensors, the formats that ``astro_simclr.oracle.tf_files``
reads (its module docstring describes them), with knobs to break one property at a time
(compression, shards, endianness, dtype, slices, CRCs). Written from the format's
description, not from TensorFlow's code; nothing of ALeRCE is involved.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from astro_simclr.oracle.m1 import RemoteFile
from astro_simclr.oracle.tf_files import masked_crc32c

TABLE_MAGIC = 0xDB4775248B80FB57


def varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def field_varint(number: int, value: int) -> bytes:
    return varint(number << 3) + varint(value)


def field_bytes(number: int, value: bytes) -> bytes:
    return varint((number << 3) | 2) + varint(len(value)) + value


def field_fixed32(number: int, value: bytes) -> bytes:
    return varint((number << 3) | 5) + value


def block(entries: list[tuple[bytes, bytes]], restart_interval: int = 2) -> bytes:
    """A table block with prefix-compressed keys and a restart every ``restart_interval``."""
    out, restarts, previous = bytearray(), [], b""
    for i, (key, value) in enumerate(entries):
        shared = 0
        if i % restart_interval == 0:
            restarts.append(len(out))
        else:
            while shared < min(len(key), len(previous)) and key[shared] == previous[shared]:
                shared += 1
        out += varint(shared) + varint(len(key) - shared) + varint(len(value))
        out += key[shared:] + value
        previous = key
    for offset in restarts or [0]:
        out += struct.pack("<I", offset)
    out += struct.pack("<I", len(restarts or [0]))
    return bytes(out)


@dataclass
class Knobs:
    """What to break in :func:`write_bundle` (defaults: a valid bundle)."""

    compression: int = 0
    shards: int = 1
    endianness: int = 0
    dtype: int = 1
    slices: bool = False
    corrupt_block_crc: bool = False
    corrupt_tensor_crc: bool = False
    omit_crc: bool = False
    size_delta: int = 0
    header: bool = True
    magic: int = TABLE_MAGIC
    extra_entries: dict[str, bytes] = field(default_factory=dict)


def _entry(array: npt.NDArray[Any], offset: int, knobs: Knobs) -> bytes:
    raw = array.astype("<f4").tobytes()
    shape = b"".join(field_bytes(2, field_varint(1, d)) for d in array.shape)
    out = field_varint(1, knobs.dtype) + field_bytes(2, shape)
    out += field_varint(4, offset) + field_varint(5, len(raw) + knobs.size_delta)
    if not knobs.omit_crc:
        crc = masked_crc32c(raw) ^ (1 if knobs.corrupt_tensor_crc else 0)
        out += field_fixed32(6, struct.pack("<I", crc))
    if knobs.slices:
        out += field_bytes(7, b"")
    return out


def _with_trailer(content: bytes, knobs: Knobs) -> bytes:
    kind = bytes([knobs.compression])
    crc = masked_crc32c(content + kind) ^ (1 if knobs.corrupt_block_crc else 0)
    return content + kind + struct.pack("<I", crc)


def write_bundle(
    tensors: dict[str, npt.NDArray[Any]], knobs: Knobs | None = None, *, per_block: int = 3
) -> tuple[bytes, bytes]:
    """``(index bytes, data bytes)`` of a single-shard bundle holding ``tensors``."""
    knobs = knobs or Knobs()
    data, entries = bytearray(), []
    if knobs.header:
        header = field_varint(1, knobs.shards) + field_varint(2, knobs.endianness)
        entries.append((b"", header))
    for name in sorted(tensors):
        entries.append((name.encode(), _entry(tensors[name], len(data), knobs)))
        data += tensors[name].astype("<f4").tobytes()
    for name, value in knobs.extra_entries.items():  # raw BundleEntryProto values
        entries.append((name.encode(), value))
    entries.sort()
    index, handles = bytearray(), []
    for start in range(0, len(entries), per_block):
        chunk = entries[start : start + per_block]
        content = block(chunk)
        handles.append((chunk[-1][0], varint(len(index)) + varint(len(content))))
        index += _with_trailer(content, knobs)
    meta_offset = len(index)
    meta = block([])
    index += _with_trailer(meta, knobs)
    index_offset = len(index)
    index_block = block(handles)
    index += _with_trailer(index_block, knobs)
    footer = varint(meta_offset) + varint(len(meta))
    footer += varint(index_offset) + varint(len(index_block))
    footer = footer.ljust(40, b"\0") + struct.pack("<Q", knobs.magic)
    return bytes(index + footer), bytes(data)


def _record(payload: bytes) -> bytes:
    header = struct.pack("<Q", len(payload))
    return (
        header
        + struct.pack("<I", masked_crc32c(header))
        + payload
        + struct.pack("<I", masked_crc32c(payload))
    )


def event_file(events: list[tuple[int, dict[str, float]]]) -> bytes:
    """A TF event file: a ``file_version`` event, then one event of scalars per step."""
    out = _record(field_fixed32(1, struct.pack("<d", 0.0)) + field_bytes(3, b"brain.Event:2"))
    for step, scalars in events:
        summary = b"".join(
            field_bytes(1, field_bytes(1, tag.encode()) + field_fixed32(2, struct.pack("<f", v)))
            for tag, v in scalars.items()
        )
        summary += field_bytes(1, field_bytes(1, b"image_only"))  # a non-scalar value
        out += _record(field_varint(2, step) + field_bytes(5, summary))
    return out


def random_m1(seed: int, *, with_features: bool = True) -> dict[str, Any]:
    """Random TF-layout weights with M1's names and shapes, activations of order 1.

    Kernels are He-scaled (std ``sqrt(2 / fan_in)``), biases and BN parameters are not
    trivial (moving statistics away from 0 and 1), so every mapped tensor matters.
    """
    from astro_simclr.oracle.m1 import tf_shapes

    rng = np.random.default_rng(seed)
    weights: dict[str, Any] = {}
    for name, shape in tf_shapes(with_features=with_features).items():
        if name.endswith("kernel"):
            array = rng.normal(0.0, np.sqrt(2.0 / np.prod(shape[:-1])), size=shape)
        elif name.endswith("moving_variance"):
            array = rng.uniform(0.5, 2.0, size=shape)
        elif name.endswith("gamma"):
            array = rng.uniform(0.5, 1.5, size=shape)
        else:
            array = rng.normal(0.0, 0.1, size=shape)
        weights[name] = array.astype(np.float32)
    return weights


def synthetic_m1_dir(tmp_path: Path, seed: int = 9) -> tuple[Path, dict[str, RemoteFile]]:
    """``(served dir, files)``: a synthetic M1 checkpoint (with optimiser slots) and val log.

    Each file is in the directory twice: under ``sub/<name>.remote`` (the "remote" path of
    ``files``, for a ``file://`` base URL) and under its local name (for ``source_dir``).
    """
    weights = random_m1(seed)
    tensors = dict(weights)
    tensors["network/conv_1_1/conv2d/kernel/Adam"] = np.ones((4, 4, 3, 32), dtype=np.float32)
    tensors["optimizer/beta2_power"] = np.array(0.5, dtype=np.float32)
    index, data = write_bundle(tensors, per_block=7)
    events = event_file([(0, {"loss": 1.0, "accuracy": 0.2}), (10, {"loss": 0.5, "accuracy": 0.4})])
    served = tmp_path / "served"
    contents = {"model.index": index, "model.data-00000-of-00001": data, "val.events": events}
    files = {}
    for name, blob in contents.items():
        remote = f"sub/{name}.remote"
        (served / "sub").mkdir(parents=True, exist_ok=True)
        (served / remote).write_bytes(blob)
        (served / name).write_bytes(blob)
        files[name] = RemoteFile(remote, len(blob), hashlib.sha256(blob).hexdigest())
    return served, files
