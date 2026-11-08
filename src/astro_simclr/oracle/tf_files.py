"""Pure-Python readers for the two TensorFlow 1 file formats of M1 (task 2.2b of phase 2).

The M1 oracle needs the published checkpoint's weights and its validation event log
without installing TensorFlow (``docs/phases/02-stamp-classifier.md``, task 2.2b). Both
formats are small and documented in TensorFlow's sources; this module reads exactly the
subset M1 uses and raises :class:`TFFormatError` on anything else, rather than guessing.

**Checkpoint (tensor bundle V2).** ``model.index`` is a LevelDB-style sorted table
(``tensorflow/core/lib/io/table``): data blocks of prefix-compressed ``key -> value``
entries followed by restart offsets, each block followed by a 5-byte trailer (compression
type, masked CRC32C); an index block of ``last key -> block handle``; and a 48-byte footer
(metaindex and index handles as varints, padded to 40 bytes, then the magic number). The
entry with the empty key is the ``BundleHeaderProto`` (number of shards, endianness); every
other key is a variable name whose value is a ``BundleEntryProto``: dtype, shape, shard,
offset and size in ``model.data-00000-of-00001``, and the masked CRC32C of those bytes.
:func:`read_bundle` checks the footer magic, that blocks are uncompressed, a single
little-endian shard, ``DT_FLOAT`` dtype, ``size == 4 * prod(shape)`` and every tensor's
CRC32C, so a misread offset cannot pass silently.

**Event log (TFRecord of ``Event`` protos).** Each record is a little-endian ``uint64``
length, its masked CRC32C, the payload and the payload's masked CRC32C.
:func:`read_scalar_events` keeps the ``simple_value`` scalars of each ``Summary``, by tag,
with the event's ``step``.

Protocol-buffer decoding is the minimal wire format (varint, 64-bit, length-delimited,
32-bit); unknown fields are skipped, as protobuf does.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from math import prod

import numpy as np
import numpy.typing as npt

__all__ = [
    "TFFormatError",
    "crc32c",
    "masked_crc32c",
    "read_bundle",
    "read_scalar_events",
]

#: ``kTableMagicNumber`` of TensorFlow's table format (the last 8 bytes of ``model.index``).
_TABLE_MAGIC = 0xDB4775248B80FB57
_FOOTER_SIZE = 48
#: Block trailer: 1 byte compression type, 4 bytes masked CRC32C.
_TRAILER_SIZE = 5
#: ``DataType.DT_FLOAT`` in ``types.proto``.
_DT_FLOAT = 1
#: ``crc32c::Mask``'s rotation delta (``tensorflow/core/lib/hash/crc32c.h``).
_MASK_DELTA = 0xA282EAD8

# Wire types of the protobuf encoding.
_VARINT, _I64, _LEN, _I32 = 0, 1, 2, 5


class TFFormatError(ValueError):
    """A TensorFlow file is truncated, corrupt, or uses a feature this reader does not support."""


def _crc32c_table() -> tuple[int, ...]:
    """The byte table of CRC-32C (Castagnoli, reflected polynomial 0x82F63B78)."""
    table = []
    for byte in range(256):
        crc = byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
        table.append(crc)
    return tuple(table)


_CRC_TABLE = _crc32c_table()


def crc32c(data: bytes) -> int:
    """CRC-32C of ``data`` (the checksum TensorFlow uses in tables, bundles and records)."""
    crc = 0xFFFFFFFF
    table = _CRC_TABLE
    for byte in data:
        crc = table[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def masked_crc32c(data: bytes) -> int:
    """``crc32c::Mask(crc32c(data))``: rotate right by 15 bits and add a constant.

    TensorFlow stores masked checksums so that a CRC of data containing CRCs stays useful.
    """
    crc = crc32c(data)
    return (((crc >> 15) | (crc << 17)) + _MASK_DELTA) & 0xFFFFFFFF


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    """Decode a base-128 varint at ``pos``; return ``(value, next position)``."""
    result = shift = 0
    while True:
        if pos >= len(buf):
            raise TFFormatError("truncated varint")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise TFFormatError("varint longer than 64 bits")


def _fields(message: bytes) -> Iterator[tuple[int, int, int | bytes]]:
    """``(field number, wire type, value)`` of a protobuf message, in order.

    Varints are returned as ``int``; the other wire types as their raw bytes.
    """
    pos = 0
    while pos < len(message):
        tag, pos = _varint(message, pos)
        number, wire = tag >> 3, tag & 7
        value: int | bytes
        if wire == _VARINT:
            value, pos = _varint(message, pos)
        elif wire in (_I64, _I32):
            width = 8 if wire == _I64 else 4
            value, pos = message[pos : pos + width], pos + width
        elif wire == _LEN:
            length, pos = _varint(message, pos)
            value, pos = message[pos : pos + length], pos + length
        else:
            raise TFFormatError(f"unsupported protobuf wire type {wire}")
        if pos > len(message):
            raise TFFormatError("truncated protobuf message")
        yield number, wire, value


def _int(value: int | bytes) -> int:
    """A decoded varint field; a field of the wrong wire type is a format error."""
    if not isinstance(value, int):
        raise TFFormatError("expected a varint field")
    return value


def _bytes(value: int | bytes) -> bytes:
    """A decoded length-delimited or fixed-width field."""
    if not isinstance(value, bytes):
        raise TFFormatError("expected a length-delimited or fixed-width field")
    return value


def _block(index: bytes, offset: int, size: int) -> list[tuple[bytes, bytes]]:
    """The ``(key, value)`` entries of the table block at ``offset`` (``size`` bytes).

    Checks the trailer: uncompressed, with a matching masked CRC32C over the block and the
    compression byte.
    """
    end = offset + size
    if end + _TRAILER_SIZE > len(index):
        raise TFFormatError("table block out of range")
    block = index[offset:end]
    compression = index[end]
    if compression != 0:
        raise TFFormatError(f"compressed table blocks are not supported (type {compression})")
    (stored,) = struct.unpack("<I", index[end + 1 : end + _TRAILER_SIZE])
    if masked_crc32c(index[offset : end + 1]) != stored:
        raise TFFormatError("table block CRC32C mismatch")
    if len(block) < 4:
        raise TFFormatError("table block too short")
    (n_restarts,) = struct.unpack("<I", block[-4:])
    limit = len(block) - 4 - 4 * n_restarts
    if limit < 0:
        raise TFFormatError("table block restart array out of range")
    entries: list[tuple[bytes, bytes]] = []
    key, pos = b"", 0
    while pos < limit:
        shared, pos = _varint(block, pos)
        unshared, pos = _varint(block, pos)
        value_size, pos = _varint(block, pos)
        if shared > len(key) or pos + unshared + value_size > limit:
            raise TFFormatError("corrupt table entry")
        key = key[:shared] + block[pos : pos + unshared]
        pos += unshared
        entries.append((key, block[pos : pos + value_size]))
        pos += value_size
    return entries


def _handle(value: bytes) -> tuple[int, int]:
    """A ``BlockHandle``: ``(offset, size)`` as two varints."""
    offset, pos = _varint(value, 0)
    size, _ = _varint(value, pos)
    return offset, size


def _shape(message: bytes) -> tuple[int, ...]:
    """The dimensions of a ``TensorShapeProto`` (``dim`` = 2, each ``Dim.size`` = 1)."""
    dims: list[int] = []
    for number, wire, value in _fields(message):
        if number == 3 and value:  # unknown_rank
            raise TFFormatError("tensor of unknown rank")
        if number == 2 and wire == _LEN:
            size = 0
            for sub_number, sub_wire, sub_value in _fields(_bytes(value)):
                if sub_number == 1 and sub_wire == _VARINT:
                    size = _int(sub_value)
            dims.append(size)
    return tuple(dims)


def _check_header(value: bytes) -> None:
    """``BundleHeaderProto``: one shard (field 1), little endian (field 2, ``LITTLE`` = 0)."""
    shards, endianness = 1, 0
    for number, wire, field in _fields(value):
        if number == 1 and wire == _VARINT:
            shards = _int(field)
        elif number == 2 and wire == _VARINT:
            endianness = _int(field)
    if shards != 1:
        raise TFFormatError(f"only single-shard bundles are supported, got {shards} shards")
    if endianness != 0:
        raise TFFormatError("only little-endian bundles are supported")


def _tensor(name: str, value: bytes, data: bytes) -> npt.NDArray[np.float32]:
    """Decode one ``BundleEntryProto`` and read its float32 tensor from ``data``."""
    entry: dict[int, int | bytes] = {}
    for number, _, field in _fields(value):
        if number == 7:
            raise TFFormatError(f"{name}: partitioned (sliced) variables are not supported")
        entry[number] = field
    dtype, shard = _int(entry.get(1, 0)), _int(entry.get(3, 0))
    offset, size = _int(entry.get(4, 0)), _int(entry.get(5, 0))
    raw_crc = entry.get(6)
    if dtype != _DT_FLOAT:
        raise TFFormatError(f"{name}: only DT_FLOAT tensors are supported, got dtype {dtype}")
    if shard != 0:
        raise TFFormatError(f"{name}: shard {shard}, expected 0")
    shape = _shape(_bytes(entry.get(2, b"")))
    if size != 4 * prod(shape):
        raise TFFormatError(f"{name}: {size} bytes for shape {shape}")
    if offset + size > len(data):
        raise TFFormatError(f"{name}: data out of range (truncated data file?)")
    chunk = data[offset : offset + size]
    if not isinstance(raw_crc, bytes) or len(raw_crc) != 4:
        raise TFFormatError(f"{name}: missing CRC32C")
    if masked_crc32c(chunk) != struct.unpack("<I", raw_crc)[0]:
        raise TFFormatError(f"{name}: CRC32C mismatch")
    return np.frombuffer(chunk, dtype="<f4").astype(np.float32).reshape(shape)


def read_bundle(
    index: bytes, data: bytes, *, prefix: str = ""
) -> dict[str, npt.NDArray[np.float32]]:
    """The float32 tensors of a TF1 checkpoint (``model.index`` and its single data shard).

    ``index`` and ``data`` are the files' bytes. Only variables whose name starts with
    ``prefix`` are decoded (for M1, ``"network/"``: the optimiser's slots are skipped); their
    names are kept whole. Arrays are new, writable and in TF's layout (convolution kernels
    HWIO, dense kernels ``(in, out)``).

    Raises:
        TFFormatError: if the files are truncated or corrupt (magic number, CRC32C, sizes),
            or the bundle uses something this reader does not support (compressed blocks,
            several shards, big endian, partitioned variables, a dtype other than float32
            among the selected variables).
    """
    if len(index) < _FOOTER_SIZE:
        raise TFFormatError("index file too short for a table footer")
    footer = index[-_FOOTER_SIZE:]
    (magic,) = struct.unpack("<Q", footer[-8:])
    if magic != _TABLE_MAGIC:
        raise TFFormatError("not a TensorFlow table (bad magic number)")
    _, pos = _varint(footer, 0)  # metaindex handle offset; the metaindex is not needed
    _, pos = _varint(footer, pos)
    index_offset, pos = _varint(footer, pos)
    index_size, _ = _varint(footer, pos)
    tensors: dict[str, npt.NDArray[np.float32]] = {}
    header_seen = False
    for _, block_handle in _block(index, index_offset, index_size):
        for key, value in _block(index, *_handle(block_handle)):
            if key == b"":
                _check_header(value)
                header_seen = True
                continue
            name = key.decode("utf-8")
            if name.startswith(prefix):
                tensors[name] = _tensor(name, value, data)
    if not header_seen:
        raise TFFormatError("bundle header (empty key) not found")
    return tensors


def _records(buf: bytes) -> Iterator[bytes]:
    """The payloads of a TFRecord file, with both CRCs checked."""
    pos = 0
    while pos < len(buf):
        if pos + 12 > len(buf):
            raise TFFormatError("truncated TFRecord header")
        header = buf[pos : pos + 8]
        (length,) = struct.unpack("<Q", header)
        (length_crc,) = struct.unpack("<I", buf[pos + 8 : pos + 12])
        if masked_crc32c(header) != length_crc:
            raise TFFormatError("TFRecord length CRC32C mismatch")
        start, end = pos + 12, pos + 12 + length
        if end + 4 > len(buf):
            raise TFFormatError("truncated TFRecord payload")
        payload = buf[start:end]
        if masked_crc32c(payload) != struct.unpack("<I", buf[end : end + 4])[0]:
            raise TFFormatError("TFRecord payload CRC32C mismatch")
        yield payload
        pos = end + 4


def _summary_value(value: bytes) -> tuple[str, float] | None:
    """``(tag, simple_value)`` of a ``Summary.Value``, or None if it is not a scalar."""
    tag, simple = None, None
    for number, wire, field in _fields(value):
        if number == 1 and wire == _LEN:
            tag = _bytes(field).decode("utf-8")
        elif number == 2 and wire == _I32:
            simple = struct.unpack("<f", _bytes(field))[0]
    if tag is None or simple is None:
        return None
    return tag, float(simple)


def read_scalar_events(buf: bytes) -> dict[str, list[tuple[int, float]]]:
    """Tag -> ``[(step, value), ...]`` of the scalar summaries in a TF event file.

    ``buf`` is the bytes of an ``events.out.tfevents.*`` file. An ``Event`` has ``step``
    (field 2) and ``summary`` (field 5); a ``Summary.Value`` (field 1) has ``tag`` (1) and
    ``simple_value`` (2, float32). Values keep file order; other events and summary kinds
    (images, histograms) are skipped.

    Raises:
        TFFormatError: if the file is truncated or a checksum does not match.
    """
    scalars: dict[str, list[tuple[int, float]]] = {}
    for record in _records(buf):
        step, summary = 0, b""
        for number, wire, value in _fields(record):
            if number == 2 and wire == _VARINT:
                step = _int(value)
            elif number == 5 and wire == _LEN:
                summary = _bytes(value)
        for number, wire, value in _fields(summary):
            scalar = _summary_value(_bytes(value)) if number == 1 and wire == _LEN else None
            if scalar is not None:
                scalars.setdefault(scalar[0], []).append((step, scalar[1]))
    return scalars
