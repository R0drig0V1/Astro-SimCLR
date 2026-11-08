"""A minimal, strict reader of the FITS cutouts inside ZTF alerts (task 3.1).

Each ZTF alert carries three cutouts (science, template, difference) as gzip-compressed
FITS files with a single 2D floating-point primary image (``BITPIX = -32``). Only that
case is needed, so this reader implements only it, instead of adding a FITS library as a
dependency: 2880-byte header blocks of 80-character ASCII cards ending with ``END``, then
the big-endian pixels, ``NAXIS2`` rows of ``NAXIS1`` values (row-major, as numpy reads
them; no flip, which the probe of 2026-11-08 confirmed against the stored stamps).

The input is untrusted, so every limit is explicit: the decompressed size, the number of
header blocks and the image side are bounded, the mandatory keywords are checked, and
anything else (integer images, scaling with ``BSCALE``/``BZERO``, extensions, more than two
axes) is rejected with a :class:`FitsError` rather than guessed.
"""

from __future__ import annotations

import zlib

import numpy as np
import numpy.typing as npt

__all__ = ["MAX_DECOMPRESSED_BYTES", "MAX_SIDE", "FitsError", "gunzip", "read_fits_image"]

#: Size of a FITS header card and block.
_CARD = 80
_BLOCK = 2880
#: Header blocks read before giving up (a cutout header has 1 or 2).
_MAX_HEADER_BLOCKS = 32
#: Largest accepted image side: ZTF cutouts are 63x63 or smaller (cut at a CCD edge).
MAX_SIDE = 256
#: Largest accepted decompressed cutout: a 256x256 float64 image plus its header.
MAX_DECOMPRESSED_BYTES = MAX_SIDE * MAX_SIDE * 8 + _MAX_HEADER_BLOCKS * _BLOCK
#: Accepted BITPIX values and their big-endian dtypes.
_DTYPES: dict[int, str] = {-32: ">f4", -64: ">f8"}
#: The two magic bytes of a gzip stream.
_GZIP_MAGIC = b"\x1f\x8b"


class FitsError(ValueError):
    """The bytes are not a FITS image this reader accepts; the message says why."""


def gunzip(data: bytes, max_bytes: int = MAX_DECOMPRESSED_BYTES) -> bytes:
    """Decompress a gzip stream of at most ``max_bytes``; return other bytes unchanged.

    Data that does not start with the gzip magic is returned as is (an uncompressed FITS
    file). Bytes after the end of the first gzip member are ignored.

    Raises:
        FitsError: if the stream is corrupt or truncated, or decompresses to more than
            ``max_bytes`` (a decompression bomb is never fully expanded).
    """
    if not data.startswith(_GZIP_MAGIC):
        if len(data) > max_bytes:
            raise FitsError(f"uncompressed cutout of {len(data)} bytes exceeds {max_bytes}")
        return data
    decompressor = zlib.decompressobj(wbits=31)  # 31: gzip header and trailer only
    try:
        out = decompressor.decompress(data, max_bytes)
    except zlib.error as err:
        raise FitsError(f"corrupt gzip stream ({err})") from None
    if decompressor.unconsumed_tail:
        raise FitsError(f"cutout decompresses to more than {max_bytes} bytes")
    if not decompressor.eof:
        raise FitsError("truncated gzip stream")
    return out


def read_fits_image(data: bytes) -> npt.NDArray[np.float32]:
    """Read the 2D floating-point primary image of a (possibly gzipped) FITS file.

    Returns a native-endian float32 ``(NAXIS2, NAXIS1)`` array (a new copy), NaN kept.
    A float64 image (``BITPIX = -64``) is rounded to float32.

    Raises:
        FitsError: if the file is corrupt, truncated or of another kind than a single
            2D float image without scaling (see the module docstring).
    """
    data = gunzip(data)
    header, offset = _read_header(data)
    if header.get("SIMPLE") != "T":
        raise FitsError("not a FITS file (SIMPLE = T missing)")
    bitpix = _int_keyword(header, "BITPIX")
    if bitpix not in _DTYPES:
        raise FitsError(f"BITPIX {bitpix} not supported (only -32 and -64)")
    if _int_keyword(header, "NAXIS") != 2:
        raise FitsError(f"NAXIS must be 2, got {header.get('NAXIS')}")
    width, height = _int_keyword(header, "NAXIS1"), _int_keyword(header, "NAXIS2")
    if not (1 <= width <= MAX_SIDE and 1 <= height <= MAX_SIDE):
        raise FitsError(f"image of {height}x{width} outside 1..{MAX_SIDE}")
    for key, neutral in (("BSCALE", 1.0), ("BZERO", 0.0)):
        if key in header and _float_keyword(header, key) != neutral:
            raise FitsError(f"{key} = {header[key]} not supported (scaled image)")
    dtype = np.dtype(_DTYPES[bitpix])
    count = width * height
    if len(data) < offset + count * dtype.itemsize:
        raise FitsError("truncated image data")
    pixels = np.frombuffer(data, dtype=dtype, count=count, offset=offset)
    return pixels.reshape(height, width).astype(np.float32)


def _read_header(data: bytes) -> tuple[dict[str, str], int]:
    """Parse the primary header: keyword -> raw value text, and the data offset."""
    header: dict[str, str] = {}
    for block in range(_MAX_HEADER_BLOCKS):
        start = block * _BLOCK
        chunk = data[start : start + _BLOCK]
        if len(chunk) < _BLOCK:
            raise FitsError("truncated header (no END card)")
        try:
            text = chunk.decode("ascii")
        except UnicodeDecodeError:
            raise FitsError("non-ASCII header") from None
        for i in range(0, _BLOCK, _CARD):
            card = text[i : i + _CARD]
            keyword = card[:8].rstrip()
            if keyword == "END":
                return header, start + _BLOCK
            if card[8:10] == "= ":
                header.setdefault(keyword, _card_value(card[10:]))
    raise FitsError(f"no END card in the first {_MAX_HEADER_BLOCKS} header blocks")


def _card_value(text: str) -> str:
    """The value of a card: a quoted string without its quotes, or the text before ``/``."""
    text = text.strip()
    if text.startswith("'"):
        end = text.find("'", 1)
        return text[1:end] if end > 0 else text[1:]
    return text.split("/", 1)[0].strip()


def _int_keyword(header: dict[str, str], key: str) -> int:
    try:
        return int(header[key])
    except KeyError:
        raise FitsError(f"missing keyword {key}") from None
    except ValueError:
        raise FitsError(f"keyword {key} is not an integer: {header[key]!r}") from None


def _float_keyword(header: dict[str, str], key: str) -> float:
    try:
        return float(header[key].replace("D", "E"))
    except ValueError:
        raise FitsError(f"keyword {key} is not a number: {header[key]!r}") from None
