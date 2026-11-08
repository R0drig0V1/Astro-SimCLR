"""Re-derive M1's feature normalisation statistics without unpickling (task 2.1).

Downloads M1's ``feature_norm_stats.pkl`` from ``alercebroker/pipeline`` at the pinned
commit into a temporary directory, checks its SHA-256, and reads it **statically** with
``pickletools``: the file is never unpickled, and nothing downloaded is kept. Then it
compares the values with ``astro_simclr.data.m1_features.M1_FEATURE_NORM_STATS`` and prints
them in the form recorded there. Exit 1 on any difference.

The file is a protocol-4 pickle of ``{feature: {"mean": scalar, "std": scalar}}``: each
scalar is a ``numpy.core.multiarray.scalar`` call on a dtype and 8 raw bytes
(``SHORT_BINBYTES``, little-endian float64, or int64 for ``isdiffpos``'s mean), except
that three means are plain Python floats (``BINFLOAT``, big-endian). The reader walks the
opcodes in order: a feature name is a string followed by an empty dict, and its two values
are the next two ``SHORT_BINBYTES``/``BINFLOAT`` arguments (mean, then std). The dtype of
each byte string is the most recent dtype object built (``f8``/``i8``) or re-used by
``BINGET``.

    uv run python scripts/characterize/data/m1_feature_stats.py

Needs network access to raw.githubusercontent.com. Excluded from ruff, mypy and pytest
(``scripts/characterize/``).
"""

import hashlib
import pickletools
import struct
import sys
import tempfile
import urllib.request
from pathlib import Path

from astro_simclr.data.m1_features import M1_FEATURE_NORM_STATS

COMMIT = "a3705acb58ce2e45c1f9b240b824c51260295a99"
PATH = (
    "stamp_classifier_step/model/results/staging_model/DeepHits_EntropyRegBeta0.5000_batch64_"
    "lr0.00100_droprate0.5000_inputsize21_filtersize5_0_20200708-160759/feature_norm_stats.pkl"
)
SHA256 = "65a5611c0ca6762622788b546d82097e0bc6a0c95798eb525c0da77253ff8064"
URL = f"https://raw.githubusercontent.com/alercebroker/pipeline/{COMMIT}/{PATH}"


def download(directory):
    path = Path(directory) / "feature_norm_stats.pkl"
    with urllib.request.urlopen(URL, timeout=60) as response:  # noqa: S310
        path.write_bytes(response.read())
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        sys.exit(f"{URL}: SHA-256 {digest}, expected {SHA256}")
    return data


def read_stats(data):
    """Feature -> (mean, std) from the opcodes alone (see the module docstring)."""
    # Resolve the memo: MEMOIZE stores the previous op, BINGET re-pushes a stored one. A
    # dtype object is built by ``dtype("f8", ...)`` REDUCE (its state is set later by
    # BUILD, in place); the MEMOIZE right after that REDUCE stores it, recorded here as
    # ("DTYPE", "f8"), so that a later BINGET of it says which dtype the next bytes have.
    resolved, memo, pending = [], [], None
    for op, arg, _ in pickletools.genops(data):
        if op.name == "MEMOIZE":
            if resolved[-1][0] == "REDUCE" and pending is not None:
                memo.append(("DTYPE", pending))
                pending = None
            else:
                memo.append(resolved[-1])
        elif op.name == "BINGET":
            resolved.append(memo[arg])
        elif op.name not in ("PROTO", "FRAME"):
            resolved.append((op.name, arg))
            if op.name == "SHORT_BINUNICODE" and arg in ("f8", "i8"):
                pending = arg
    stats, values, dtype = {}, None, None
    for i, (name, arg) in enumerate(resolved):
        if name == "DTYPE" or (name == "SHORT_BINUNICODE" and arg in ("f8", "i8")):
            dtype = arg
        elif name == "SHORT_BINUNICODE" and i > 1 and resolved[i + 1][0] == "EMPTY_DICT":
            values = stats.setdefault(arg, [])  # i > 1: the top-level dict has no key
        elif name == "SHORT_BINBYTES":
            assert len(arg) == 8 and dtype is not None, (i, arg, dtype)
            values.append(struct.unpack("<d" if dtype == "f8" else "<q", arg)[0])
        elif name == "BINFLOAT":
            values.append(arg)
    for feature, pair in stats.items():
        assert len(pair) == 2, (feature, pair)
    return {feature: (float(mean), float(std)) for feature, (mean, std) in stats.items()}


def main():
    with tempfile.TemporaryDirectory() as directory:
        data = download(directory)
    stats = read_stats(data)
    for feature, (mean, std) in stats.items():
        print(f"    {feature!r}: ({mean!r}, {std!r}),")
    if stats != M1_FEATURE_NORM_STATS or list(stats) != list(M1_FEATURE_NORM_STATS):
        sys.exit("differs from astro_simclr.data.m1_features.M1_FEATURE_NORM_STATS")
    print(f"OK: {len(stats)} features match M1_FEATURE_NORM_STATS")


if __name__ == "__main__":
    main()
