"""Parse one ALeRCE avro alert into its raw cutouts and candidate record (task 3.1).

``get_avro`` returns an avro object container file (snappy codec on 2026-11-08) holding a
single ZTF alert: ``objectId``, ``candidate`` (the alert's own record: ``jd``, ``fid``,
``magpsf``, ``rb``, ``drb``, ``xpos``, ...), ``prv_candidates`` and the three cutouts
``cutoutScience``, ``cutoutTemplate`` and ``cutoutDifference``, each a record whose
``stampData`` is a gzipped FITS image (:mod:`~astro_simclr.alerts.fits`).

The file is read with fastavro using the writer schema embedded in it (ALeRCE re-serialises
the alerts, so its record names differ from ZTF's own schema and a fixed reader schema would
break on them). The record is then validated here: exactly one record, the expected
``objectId`` and ``candid``, the required candidate fields with their types, and three
cutouts of the same shape, at most 63x63. Only the scalar candidate fields are kept; the
previous candidates are not (``docs/data.md`` explains why).

**Size limits.** The download is capped (``alerce.MAX_AVRO_BYTES``), and before fastavro
decompresses anything, :func:`check_container` walks the container's blocks and bounds
their *decompressed* size (:data:`MAX_DECOMPRESSED_AVRO_BYTES`): the ``null`` codec is its
size, ``snappy`` declares its length in the block's preamble, ``deflate`` is inflated with
an output limit; other codecs are refused. The embedded writer schema is untrusted too: an
array or map whose items can encode in zero bytes (``null``, an empty record) would let a
tiny file declare billions of items, so such a schema is refused (:func:`check_schema`);
every other item takes at least one byte, so the decoding work stays bounded by the
decompressed size. Each cutout's gzip is capped by
:func:`~astro_simclr.alerts.fits.gunzip`. A decompression bomb is never expanded.

A cutout cut at a CCD edge (smaller than 63x63) is padded back with NaN by
:func:`~astro_simclr.data.preprocessing.repair_non_squared_stamp`, using the candidate's
``xpos``/``ypos``, as the stamps of the prepared datasets were built.
"""

from __future__ import annotations

import io
import json
import math
import zlib
from dataclasses import dataclass
from typing import Any

import fastavro
import numpy as np
import numpy.typing as npt

from astro_simclr.alerts.fits import FitsError, read_fits_image
from astro_simclr.data.preprocessing import repair_non_squared_stamp
from astro_simclr.data.schema import STAMP_SHAPE

__all__ = [
    "CUTOUT_FIELDS",
    "MAX_DECOMPRESSED_AVRO_BYTES",
    "REQUIRED_FIELDS",
    "Alert",
    "AlertFormatError",
    "Scalar",
    "check_container",
    "check_schema",
    "parse_alert",
]

#: Largest accepted decompressed alert: about 30x a real one (the cutouts stay gzipped
#: inside it, so a real alert inflates only its candidate records).
MAX_DECOMPRESSED_AVRO_BYTES = 4 * 1024 * 1024
#: The avro object container magic and sync marker size.
_MAGIC = b"Obj\x01"
_SYNC = 16

#: The cutout fields, in channel order (the order of the prepared stamps).
CUTOUT_FIELDS: tuple[str, str, str] = ("cutoutScience", "cutoutTemplate", "cutoutDifference")

#: A scalar candidate value, as kept in the store.
type Scalar = str | int | float | bool | None

#: Candidate fields that must be present, with their accepted types.
REQUIRED_FIELDS: dict[str, tuple[type, ...]] = {
    "candid": (int,),
    "jd": (float,),
    "fid": (int,),
    "magpsf": (float,),
    "sigmapsf": (float,),
    "isdiffpos": (str,),
    "ra": (float,),
    "dec": (float,),
}


class AlertFormatError(ValueError):
    """The bytes are not one valid alert for the requested ``oid``/``candid``."""


@dataclass(frozen=True, slots=True)
class Alert:
    """One parsed alert.

    Attributes:
        oid: the ZTF object id.
        candid: the alert's candidate id.
        candidate: the scalar fields of the candidate record (non-finite floats as
            ``None``, so the record is valid JSON).
        cutouts: float32 ``(63, 63, 3)``: science, template, difference, NaN kept.
        raw_shape: the cutouts' shape before any repair, ``(H, W)``.
    """

    oid: str
    candid: int
    candidate: dict[str, Scalar]
    cutouts: npt.NDArray[np.float32]
    raw_shape: tuple[int, int]

    @property
    def repaired(self) -> bool:
        """True if the cutouts were padded to 63x63 (cut at a CCD edge)."""
        return self.raw_shape != STAMP_SHAPE[:2]


def parse_alert(data: bytes, *, oid: str, candid: int) -> Alert:
    """Parse the avro bytes of alert (``oid``, ``candid``) (module docstring).

    Raises:
        AlertFormatError: if the bytes are not an avro file with exactly one alert, the
            alert is for another object or candidate, a required candidate field is
            missing or of the wrong type, or a cutout is missing, invalid, larger than
            63x63, or of a different shape than the others.
    """
    record = _single_record(data)
    if record.get("objectId") != oid:
        raise AlertFormatError(f"objectId {record.get('objectId')!r}, expected {oid!r}")
    raw_candidate = record.get("candidate")
    if not isinstance(raw_candidate, dict):
        raise AlertFormatError("no candidate record")
    candidate = _scalars(raw_candidate)
    for name, types in REQUIRED_FIELDS.items():
        value = candidate.get(name)
        if isinstance(value, bool) or not isinstance(value, types):
            raise AlertFormatError(f"candidate.{name} is {value!r}, expected {types[0].__name__}")
    if candidate["candid"] != candid:
        raise AlertFormatError(f"candid {candidate['candid']}, expected {candid}")
    channels = [_cutout(record, name) for name in CUTOUT_FIELDS]
    shapes = {channel.shape for channel in channels}
    if len(shapes) != 1:
        raise AlertFormatError(f"cutouts of different shapes {sorted(shapes)}")
    height, width = channels[0].shape
    stamp: npt.NDArray[np.floating[Any]] = np.stack(channels, axis=-1)
    if (height, width) != STAMP_SHAPE[:2]:
        stamp = _repair(stamp, candidate)
    cutouts = np.ascontiguousarray(stamp, dtype=np.float32)
    return Alert(oid, candid, candidate, cutouts, (height, width))


class _Cursor:
    """Reads avro's zig-zag varints and length-prefixed bytes from a buffer."""

    def __init__(self, data: bytes, pos: int = 0) -> None:
        self.data = data
        self.pos = pos

    def take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.data):
            raise AlertFormatError("truncated avro container")
        chunk = self.data[self.pos : self.pos + n]
        self.pos += n
        return chunk

    def long(self) -> int:
        value = shift = 0
        for _ in range(10):  # a 64-bit varint has at most 10 bytes
            byte = self.take(1)[0]
            value |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return (value >> 1) ^ -(value & 1)
            shift += 7
        raise AlertFormatError("invalid varint in the avro container")


def check_container(data: bytes, max_bytes: int = MAX_DECOMPRESSED_AVRO_BYTES) -> None:
    """Check an avro object container's codec and bound its decompressed size.

    Raises:
        AlertFormatError: if the container is malformed, uses a codec other than
            ``null``, ``deflate`` or ``snappy``, or decompresses to more than ``max_bytes``.
    """
    cursor = _Cursor(data)
    if cursor.take(len(_MAGIC)) != _MAGIC:
        raise AlertFormatError("not an avro object container (bad magic)")
    meta: dict[bytes, bytes] = {}
    while count := cursor.long():
        if count < 0:  # a negative count is followed by the block's byte size
            count = -count
            cursor.long()
        for _ in range(count):
            key = cursor.take(cursor.long())
            meta[key] = cursor.take(cursor.long())
    check_schema(meta.get(b"avro.schema"))
    codec = meta.get(b"avro.codec", b"null").decode("ascii", "replace")
    if codec not in ("null", "deflate", "snappy"):
        raise AlertFormatError(f"avro codec {codec!r} not accepted")
    sync = cursor.take(_SYNC)
    total = 0
    while cursor.pos < len(data):
        cursor.long()  # the block's object count
        block = cursor.take(cursor.long())
        if cursor.take(_SYNC) != sync:
            raise AlertFormatError("avro block without its sync marker")
        total += _block_size(codec, block, max_bytes - total)
        if total > max_bytes:
            raise AlertFormatError(f"avro decompresses to more than {max_bytes} bytes")


#: Primitive types that always take at least one byte (``null`` takes none).
_SIZED_PRIMITIVES = frozenset({"boolean", "int", "long", "float", "double", "bytes", "string"})


def check_schema(raw: bytes | None) -> None:
    """Refuse an avro writer schema whose decoding work is not bounded by its data size.

    The top level must be a record, and no array or map may have items that encode in
    zero bytes (module docstring).

    Raises:
        AlertFormatError: if the schema is missing, not JSON, not a record, or unbounded.
    """
    if raw is None:
        raise AlertFormatError("the avro file has no schema")
    try:
        schema = json.loads(raw)
        if not isinstance(schema, dict) or schema.get("type") != "record":
            raise AlertFormatError("the avro schema's top level is not a record")
        _walk_schema(schema, {})
    except (ValueError, RecursionError) as err:
        if isinstance(err, AlertFormatError):
            raise
        raise AlertFormatError(f"invalid avro schema ({type(err).__name__})") from None


def _walk_schema(schema: object, names: dict[str, object]) -> None:
    """Register the named types of ``schema`` and check every array and map in it."""
    if isinstance(schema, dict):
        kind = schema.get("type")
        if kind in ("record", "error", "enum", "fixed") and isinstance(schema.get("name"), str):
            names[schema["name"]] = schema
    for child in _children(schema):
        _walk_schema(child, names)
    if isinstance(schema, dict) and schema.get("type") in ("array", "map"):
        items = schema.get("items" if schema["type"] == "array" else "values")
        if _zero_size(items, names, set()):
            raise AlertFormatError(f"avro schema has an {schema['type']} of zero-size items")


def _children(schema: object) -> list[object]:
    """The sub-schemas of ``schema``: union branches, field types, items or values."""
    if isinstance(schema, list):
        return schema
    if not isinstance(schema, dict):
        return []
    kind = schema.get("type")
    if kind in ("record", "error"):
        return [f.get("type") for f in schema.get("fields") or [] if isinstance(f, dict)]
    if kind == "array":
        return [schema.get("items")]
    if kind == "map":
        return [schema.get("values")]
    return [kind] if isinstance(kind, dict | list) else []


def _zero_size(schema: object, names: dict[str, object], visiting: set[str]) -> bool:
    """True if a value of ``schema`` can encode in zero bytes."""
    if isinstance(schema, str):
        if schema == "null":
            return True
        if schema in _SIZED_PRIMITIVES or schema not in names or schema in visiting:
            return False  # unknown names fail in fastavro; a cycle must pass a union
        visiting.add(schema)
        try:
            return _zero_size(names[schema], names, visiting)
        finally:
            visiting.discard(schema)
    if not isinstance(schema, dict):
        return False  # a union always encodes its branch index
    kind = schema.get("type")
    if kind in ("record", "error"):
        fields = [f.get("type") for f in schema.get("fields") or [] if isinstance(f, dict)]
        return all(_zero_size(t, names, visiting) for t in fields)
    if kind == "fixed":
        return schema.get("size") == 0
    if kind in ("enum", "array", "map"):
        return False  # an index or a block count: at least one byte
    return _zero_size(kind, names, visiting)


def _block_size(codec: str, block: bytes, budget: int) -> int:
    """The decompressed size of one block, or ``budget + 1`` if it is larger."""
    if codec == "null":
        return len(block)
    if codec == "snappy":  # raw snappy (preamble: uncompressed length) + 4-byte CRC
        cursor, value, shift = _Cursor(block), 0, 0
        for _ in range(5):
            byte = cursor.take(1)[0]
            value |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return value
            shift += 7
        raise AlertFormatError("invalid snappy preamble")
    inflater = zlib.decompressobj(-15)  # deflate: raw stream, no header
    try:
        out = inflater.decompress(block, max(budget, 0) + 1)
    except zlib.error as err:
        raise AlertFormatError(f"corrupt deflate block ({err})") from None
    # No end-of-stream check: fastavro's writer ends its blocks with a sync flush.
    return len(out)


def _single_record(data: bytes) -> dict[str, Any]:
    check_container(data)
    try:
        records = iter(fastavro.reader(io.BytesIO(data)))
        first = next(records, None)
        extra = next(records, None)
    # fastavro raises many types on corrupt input (ValueError, EOFError, TypeError,
    # KeyError, IndexError, its own schema errors); all mean "not a valid alert file".
    except Exception as err:
        raise AlertFormatError(f"not a readable avro file ({type(err).__name__}: {err})") from None
    if not isinstance(first, dict):
        raise AlertFormatError("the avro file holds no record")
    if extra is not None:
        raise AlertFormatError("the avro file holds more than one record")
    return first


def _scalars(raw: dict[str, Any]) -> dict[str, Scalar]:
    """The scalar fields of ``raw``; non-finite floats become ``None``."""
    out: dict[str, Scalar] = {}
    for key, value in raw.items():
        if isinstance(value, float) and not math.isfinite(value):
            out[str(key)] = None
        elif value is None or isinstance(value, str | int | float | bool):
            out[str(key)] = value
    return out


def _cutout(record: dict[str, Any], name: str) -> npt.NDArray[np.float32]:
    field = record.get(name)
    stamp = field.get("stampData") if isinstance(field, dict) else None
    if not isinstance(stamp, bytes):
        raise AlertFormatError(f"{name}: no stampData")
    try:
        return read_fits_image(stamp)
    except FitsError as err:
        raise AlertFormatError(f"{name}: {err}") from None


def _repair(
    stamp: npt.NDArray[np.floating[Any]], candidate: dict[str, Scalar]
) -> npt.NDArray[np.floating[Any]]:
    xpos, ypos = candidate.get("xpos"), candidate.get("ypos")
    if not isinstance(xpos, float) or not isinstance(ypos, float):
        raise AlertFormatError("a cutout smaller than 63x63 needs candidate.xpos and ypos")
    try:
        return repair_non_squared_stamp(stamp, xpos, ypos)
    except ValueError as err:
        raise AlertFormatError(f"cutouts of shape {stamp.shape[:2]}: {err}") from None
