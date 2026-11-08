"""Tests of the pure-Python TF1 checkpoint and event readers (task 2.2b)."""

from __future__ import annotations

import struct

import numpy as np
import numpy.typing as npt
import pytest
from _tf_writer import Knobs, event_file, field_bytes, field_varint, write_bundle

from astro_simclr.oracle.tf_files import (
    TFFormatError,
    crc32c,
    masked_crc32c,
    read_bundle,
    read_scalar_events,
)

TENSORS: dict[str, npt.NDArray[np.float32]] = {
    "network/a/kernel": (np.arange(24).reshape(2, 3, 4) / 7).astype(np.float32),
    "network/a/bias": np.array([1.5, -2.0], dtype=np.float32),
    "network/b": np.array(3.25, dtype=np.float32),  # a scalar
    "optimizer/beta1_power": np.array(0.5, dtype=np.float32),
}


def test_crc32c_known_vector() -> None:
    # The CRC-32C check value of the standard test string (RFC 3720, appendix B.4).
    assert crc32c(b"123456789") == 0xE3069283
    assert crc32c(b"") == 0
    crc = crc32c(b"abc")
    assert masked_crc32c(b"abc") == (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


@pytest.mark.parametrize("per_block", [1, 2, 10])
def test_bundle_round_trip(per_block: int) -> None:
    index, data = write_bundle(TENSORS, per_block=per_block)
    out = read_bundle(index, data)
    assert set(out) == set(TENSORS)
    for name, array in TENSORS.items():
        assert out[name].dtype == np.float32
        assert out[name].shape == array.shape
        np.testing.assert_array_equal(out[name], array)
    assert out["network/a/kernel"].flags.writeable


def test_bundle_prefix_selects_variables() -> None:
    index, data = write_bundle(TENSORS)
    assert set(read_bundle(index, data, prefix="network/")) == {
        "network/a/kernel",
        "network/a/bias",
        "network/b",
    }


def test_unselected_variables_are_not_decoded() -> None:
    # A non-float variable outside the prefix is skipped, not an error.
    index, data = write_bundle(TENSORS, Knobs(extra_entries={"global_step": field_varint(1, 9)}))
    assert "global_step" not in read_bundle(index, data, prefix="network/")
    with pytest.raises(TFFormatError, match="DT_FLOAT"):
        read_bundle(index, data)


@pytest.mark.parametrize(
    ("knobs", "match"),
    [
        (Knobs(magic=123), "magic"),
        (Knobs(compression=1), "compressed"),
        (Knobs(corrupt_block_crc=True), "block CRC32C"),
        (Knobs(shards=2), "single-shard"),
        (Knobs(endianness=1), "little-endian"),
        (Knobs(dtype=2), "DT_FLOAT"),
        (Knobs(slices=True), "partitioned"),
        (Knobs(corrupt_tensor_crc=True), "CRC32C mismatch"),
        (Knobs(omit_crc=True), "missing CRC32C"),
        (Knobs(size_delta=4), "bytes for shape"),
        (Knobs(header=False), "header"),
    ],
)
def test_bundle_errors(knobs: Knobs, match: str) -> None:
    index, data = write_bundle(TENSORS, knobs)
    with pytest.raises(TFFormatError, match=match):
        read_bundle(index, data)


def test_truncated_files() -> None:
    index, data = write_bundle(TENSORS)
    with pytest.raises(TFFormatError, match="too short"):
        read_bundle(index[:10], data)
    with pytest.raises(TFFormatError, match="out of range"):
        read_bundle(index, data[:-4])
    # A footer pointing past the end of the file.
    footer = bytes([0x7F, 0x01, 0xFF, 0x7F, 0x10]).ljust(40, b"\0") + index[-8:]
    with pytest.raises(TFFormatError, match="out of range"):
        read_bundle(index[:-48] + footer, data)


def test_events_round_trip() -> None:
    buf = event_file([(0, {"loss": 1.0, "accuracy": 0.25}), (10, {"loss": -0.5})])
    out = read_scalar_events(buf)
    assert out == {"loss": [(0, 1.0), (10, -0.5)], "accuracy": [(0, 0.25)]}
    assert read_scalar_events(b"") == {}


def test_events_errors() -> None:
    buf = event_file([(0, {"loss": 1.0})])
    with pytest.raises(TFFormatError, match="truncated TFRecord header"):
        read_scalar_events(buf + b"\0" * 5)
    with pytest.raises(TFFormatError, match="truncated TFRecord payload"):
        read_scalar_events(buf[:-2])
    bad_length = bytearray(buf)
    bad_length[8] ^= 1
    with pytest.raises(TFFormatError, match="length CRC32C"):
        read_scalar_events(bytes(bad_length))
    bad_payload = bytearray(buf)
    bad_payload[-1] ^= 1
    with pytest.raises(TFFormatError, match="payload CRC32C"):
        read_scalar_events(bytes(bad_payload))


def _record(payload: bytes) -> bytes:
    header = struct.pack("<Q", len(payload))
    crcs = struct.pack("<I", masked_crc32c(header)), struct.pack("<I", masked_crc32c(payload))
    return header + crcs[0] + payload + crcs[1]


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (b"\x80", "truncated varint"),
        (b"\x80" * 11, "64 bits"),
        (b"\x0b", "wire type"),  # field 1, wire type 3 (groups)
        (b"\x2a\x05ab", "truncated protobuf"),  # field 5, 5 bytes announced, 2 given
    ],
)
def test_malformed_protobuf(payload: bytes, match: str) -> None:
    with pytest.raises(TFFormatError, match=match):
        read_scalar_events(_record(payload))


def test_fields_of_the_wrong_wire_type_are_skipped() -> None:
    # A Summary.Value sent as a varint and a step sent as bytes: ignored, as protobuf does.
    assert read_scalar_events(_record(b"\x2a\x02\x08\x01" + b"\x12\x01x")) == {}


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        (field_bytes(1, b"x"), "varint field"),  # dtype as bytes
        (field_varint(1, 1) + field_varint(2, 3), "length-delimited"),  # shape as a varint
        (field_varint(1, 1) + field_bytes(2, field_varint(3, 1)), "unknown rank"),
        (field_varint(1, 1) + field_varint(3, 1), "shard 1"),
    ],
)
def test_malformed_bundle_entries(entry: bytes, match: str) -> None:
    index, data = write_bundle(TENSORS, Knobs(extra_entries={"network/z": entry}))
    with pytest.raises(TFFormatError, match=match):
        read_bundle(index, data)
