"""Generate the golden fixtures of the paper's float input path from ALeRCE's own code.

Task 2.1 (docs/phases/02-stamp-classifier.md). Runs ALeRCE's ``image_crop_at_center``,
``image_normalize_by_image`` and ``image_nan_to_num`` (plain numpy methods of
``ZTFDataPreprocessor``) on synthetic raw stamps with NaNs, constant channels and all-NaN
crops, and records inputs and outputs in ``tests/fixtures/data/alerce_crop.npz``.
``tests/test_data_float_input.py`` checks ``astro_simclr.data.float_input`` against it.

Licence: ALeRCE's pipeline repository has no LICENSE file, so its code is never copied
into this repository. This script downloads ``ztf_preprocessor.py`` at use time, pinned
by commit and SHA-256, into a temporary directory, extracts exactly the four methods it
needs with ``ast`` (the module imports ALeRCE's ``parameters`` and ``modules`` packages
at top level, which these methods do not use) and runs them verbatim in a namespace that
holds only ``np`` (and a stand-in for the ``DatasetObj`` annotation). Nothing downloaded is kept. Only the outputs (numbers) are committed.

Two pipelines are recorded, both as in ALeRCE:

- ``crop21``: crop 21 -> normalise by image -> NaN to 0, the input of the paper's model;
- ``stored``: the same with crop 63 (the identity), which is how our pickles were
  written; the float path then crops and re-normalises these stored stamps.

Run it with the project's numpy (numpy has no randomness here: the inputs come from a
seeded Generator, and min, max, subtraction and division are IEEE operations):

    uv run --isolated --no-project --python 3.12 --with numpy==2.5.4 python \
        scripts/characterize/data/alerce_crop.py

Needs network access to raw.githubusercontent.com. Excluded from ruff, mypy and pytest
(``scripts/characterize/``).
"""

import ast
import hashlib
import sys
import tempfile
import urllib.request
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "tests" / "fixtures" / "data" / "alerce_crop.npz"

ALERCE_COMMIT = "a3705acb58ce2e45c1f9b240b824c51260295a99"
ALERCE_PATH = "stamp_classifier_step/model/modules/data_loaders/ztf_preprocessor.py"
ALERCE_SHA256 = "f280659b03cc1fee1879bdbdbfa8110ab26116dd0215d351270469cd892f4fda"
ALERCE_URL = (
    f"https://raw.githubusercontent.com/alercebroker/pipeline/{ALERCE_COMMIT}/{ALERCE_PATH}"
)
METHODS = (
    "image_crop_at_center",
    "image_normalize_by_image",
    "image_nan_to_num",
    "_get_nans_samples_idx",
)
SEED = 20210231  # Carrasco-Davis et al. 2021, AJ 162, 231
SIDE = 63
N_RANDOM = 6  # few: float32 stamps do not compress, and the fixture is committed


def download(directory):
    """Download the pinned file into ``directory`` and check its SHA-256."""
    path = Path(directory) / "ztf_preprocessor.py"
    with urllib.request.urlopen(ALERCE_URL, timeout=60) as response:  # noqa: S310
        path.write_bytes(response.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != ALERCE_SHA256:
        sys.exit(f"{ALERCE_URL}: SHA-256 {digest}, expected {ALERCE_SHA256}")
    return path.read_text(encoding="utf-8")


def alerce_preprocessor(source, crop_size):
    """An object with ALeRCE's four methods, verbatim, and the attributes they read."""
    tree = ast.parse(source)
    (cls,) = [
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ZTFDataPreprocessor"
    ]
    defs = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
    assert sorted(d.name for d in defs) == sorted(METHODS), [d.name for d in defs]
    stub = ast.ClassDef(
        name="AlercePreprocessor", bases=[], keywords=[], body=defs, decorator_list=[]
    )
    module = ast.fix_missing_locations(ast.Module(body=[stub], type_ignores=[]))
    # The methods annotate their argument as ALeRCE's ``DatasetObj``; annotations are
    # evaluated at definition, so the name must exist. They only read ``.data_array``.
    namespace = {"np": np, "DatasetObj": SimpleNamespace}
    exec(compile(module, ALERCE_PATH, "exec"), namespace)  # noqa: S102
    pre = namespace["AlercePreprocessor"]()
    pre.crop_size = crop_size  # params[CROP_SIZE]
    pre.number_to_replace_nans = 0  # params[NANS_TO] of M1
    pre.verbose = False
    return pre


def run(pre, images):
    """ALeRCE's pipeline crop -> normalise by image -> NaN to num, on a copy."""
    dataset = SimpleNamespace(data_array=np.array(images, copy=True))
    for step in (pre.image_crop_at_center, pre.image_normalize_by_image, pre.image_nan_to_num):
        dataset = step(dataset)
    return dataset.data_array


def raw_stamp(rng):
    """A ZTF-like raw stamp: sky background plus a point source, per channel (float32)."""
    y, x = np.mgrid[:SIDE, :SIDE]
    stamp = np.empty((SIDE, SIDE, 3), dtype=np.float32)
    for c in range(3):
        sky = rng.normal(rng.uniform(-50, 200), rng.uniform(5, 30), (SIDE, SIDE))
        cy, cx = rng.uniform(26, 36, 2)
        source = rng.uniform(50, 2000) * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / 8)
        # Rounded to 1/8: real-looking values whose float32 mantissas compress well.
        stamp[..., c] = np.round(8 * (sky + (source if c != 1 or rng.random() < 0.5 else 0))) / 8
    return stamp


def cases(rng):
    """Named raw stamps (63x63x3 float32). The names say what each one exercises."""
    b0, b1 = 21, 42  # the crop of 21 at the centre of 63
    out = {}

    out["clean"] = raw_stamp(rng)

    stamp = raw_stamp(rng)
    stamp[:, :5, 0] = np.nan  # CCD edge, outside the crop
    stamp[60:, :, 2] = np.nan
    out["nan_outside_crop"] = stamp

    stamp = raw_stamp(rng)
    stamp[30:33, 25:28, 0] = np.nan  # bad pixels inside the crop
    stamp[b0, b0, 2] = np.nan
    out["nan_inside_crop"] = stamp

    stamp = raw_stamp(rng)
    stamp[..., 1] = 7.5  # a constant channel: 0 / 0 everywhere
    out["constant_channel"] = stamp

    stamp = raw_stamp(rng)
    stamp[b0:b1, b0:b1, 2] = -3.0  # varies outside the crop, constant inside
    out["constant_crop"] = stamp

    stamp = raw_stamp(rng)
    stamp[b0 - 2 : b1 + 2, b0 - 2 : b1 + 2, 1] = np.nan  # the whole crop is NaN
    out["all_nan_crop"] = stamp

    stamp = raw_stamp(rng)
    stamp[..., 0] = np.nan  # the whole channel is NaN
    out["all_nan_channel"] = stamp

    stamp = raw_stamp(rng)
    stamp[5, 5, 0] = np.nan
    stamp[31, 31, 0] = np.nanmin(stamp[..., 0]) - 100.0  # the minimum, inside the crop
    out["nan_and_min_inside_crop"] = stamp

    stamp = raw_stamp(rng)
    stamp[31, 31, 2] = np.nanmin(stamp[..., 2]) - 100.0  # the minimum inside, no NaN
    out["min_inside_crop"] = stamp

    stamp = raw_stamp(rng)
    stamp[3, 3, 0] = stamp[50, 50, 0] = np.nanmin(stamp[..., 0]) - 10.0  # tied minimum
    out["tied_min_outside_crop"] = stamp
    return out


def random_batch(rng, n):
    """``n`` raw stamps with random NaN blocks (none, outside, inside or across the crop)."""
    stamps = np.stack([raw_stamp(rng) for _ in range(n)])
    for stamp in stamps:
        for _ in range(rng.integers(0, 4)):
            c = rng.integers(0, 3)
            h, w = rng.integers(1, 12, 2)
            y, x = rng.integers(0, SIDE - h), rng.integers(0, SIDE - w)
            stamp[y : y + h, x : x + w, c] = np.nan
    return stamps


def main():
    rng = np.random.default_rng(SEED)
    named = cases(rng)
    names = sorted(named)
    raw = np.stack([named[name] for name in names] + list(random_batch(rng, N_RANDOM)))
    with tempfile.TemporaryDirectory() as directory:
        source = download(directory)
    crop21 = alerce_preprocessor(source, 21)
    full = alerce_preprocessor(source, SIDE)
    with np.errstate(invalid="ignore", divide="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # All-NaN slice, 0 / 0
            alerce_crop21 = run(crop21, raw)
            stored = run(full, raw)
            alerce_crop21_of_stored = run(crop21, stored)
    assert stored.dtype == np.float32 and alerce_crop21.shape == (len(raw), 21, 21, 3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        names=np.array(names),
        raw=raw,
        stored=stored,
        alerce_crop21=alerce_crop21,
        alerce_crop21_of_stored=alerce_crop21_of_stored,
        alerce_commit=np.array(ALERCE_COMMIT),
        alerce_path=np.array(ALERCE_PATH),
        alerce_sha256=np.array(ALERCE_SHA256),
        numpy_version=np.array(np.__version__),
        seed=np.array(SEED),
    )
    print(f"wrote {OUT} ({len(names)} named cases + {N_RANDOM} random stamps)")


if __name__ == "__main__":
    main()
