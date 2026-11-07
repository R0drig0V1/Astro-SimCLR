"""Generate the characterization fixtures of the data preprocessing from the legacy code.

Runs the legacy ``repair_non_squared_stamp``, ``preprocess_stamps`` and ``img_float2int``
(``utils/dataset.py``) on fixed inputs, and records inputs and outputs in
``tests/fixtures/data/*.npz``, with the git commit of the legacy code, the numpy version
and the torch version (``"not used"``: these functions need numpy only).
``tests/test_data_preprocessing.py`` and ``tests/test_data_prepared.py`` check the port
against them.

``utils/dataset.py`` imports torch, pandas, cv2 and albumentations at module level (through
``utils.transformations``), none of which the three functions use. So the generator reads
the file from git, extracts exactly those three function definitions with ``ast`` and runs
them verbatim in a namespace that holds only ``np``. It runs with the legacy numpy
(``env.yml``: numpy 1.23.5, which needs Python <= 3.11), without the project:

    uv run --isolated --no-project --python 3.11 --with numpy==1.23.5 python \
        scripts/characterize/data/generate.py

It refuses to run if ``utils/dataset.py`` has uncommitted changes (the recorded SHA would
lie). Excluded from ruff, mypy and pytest (``scripts/characterize/``).
"""

import ast
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "tests" / "fixtures" / "data"
LEGACY_FILE = "utils/dataset.py"
FUNCTIONS = ("repair_non_squared_stamp", "preprocess_stamps", "img_float2int")


def git(*args):
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def legacy_functions():
    if git("status", "--porcelain", "--", LEGACY_FILE):
        sys.exit(f"{LEGACY_FILE} has uncommitted changes; commit or revert them first")
    commit = git("rev-parse", "HEAD").strip()
    source = git("show", f"{commit}:{LEGACY_FILE}")
    tree = ast.parse(source)
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS]
    assert sorted(d.name for d in defs) == sorted(FUNCTIONS), [d.name for d in defs]
    namespace = {"np": np}
    exec(compile(ast.Module(body=defs, type_ignores=[]), LEGACY_FILE, "exec"), namespace)
    return commit, {name: namespace[name] for name in FUNCTIONS}


def meta(commit):
    return {
        "legacy_commit": np.array(commit),
        "numpy_version": np.array(np.__version__),
        "torch_version": np.array("not used"),
    }


def repair_cases(rng, repair):
    """Non-square stamps on each side and both, the 40 px boundary, NaNs in the input."""
    cases = {
        "square": ((63, 63), 10.0, 10.0),
        "short_rows_low": ((60, 63), 30.0, 10.0),
        "short_rows_high": ((60, 63), 30.0, 50.0),
        "short_rows_boundary": ((60, 63), 30.0, 40.0),
        "short_cols_low": ((63, 57), 10.0, 30.0),
        "short_cols_high": ((63, 57), 50.0, 30.0),
        "short_cols_boundary": ((63, 57), 40.0, 30.0),
        "short_both_low_high": ((58, 61), 5.0, 55.0),
        "short_both_high_low": ((58, 61), 55.0, 5.0),
        "one_row": ((1, 63), 30.0, 10.0),
    }
    out = {}
    for name, ((h, w), xpos, ypos) in cases.items():
        # Small integers: the padding position is what matters, and they compress well.
        stamp = rng.integers(0, 16, (h, w, 3)).astype(np.float32)
        stamp[0, 0, 1] = np.nan
        result = repair(stamp, xpos, ypos)
        out[f"repair__{name}__in"] = stamp
        out[f"repair__{name}__pos"] = np.array([xpos, ypos])
        out[f"repair__{name}__out"] = result
    return out


def preprocess_cases(rng, preprocess):
    """NaN, +inf and -inf pixels, a constant channel and an all-NaN channel, two dtypes.

    15x15 stamps: the function reduces over axes (0, 1) whatever their size.
    """
    shape = (15, 15, 3)
    base = rng.normal(50.0, 10.0, shape)
    base[3, 4, 0] = np.nan
    base[5, 6, 1] = np.inf
    base[7, 8, 2] = -np.inf
    constant = rng.normal(0.0, 1.0, shape)
    constant[..., 1] = 7.0
    all_nan = rng.normal(0.0, 1.0, shape)
    all_nan[..., 2] = np.nan
    out = {}
    for name, stamp in {"nan_inf": base, "constant": constant, "all_nan": all_nan}.items():
        for dtype in (np.float32, np.float64):
            key = f"preprocess__{name}_{np.dtype(dtype).name}"
            out[f"{key}__in"] = stamp.astype(dtype)
            for index, nan_val in enumerate((0.0, 0.5)):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # All-NaN slice
                    out[f"{key}__out{index}"] = preprocess(stamp.astype(dtype), nan_val)
    return out


def rounding_sensitive(rng, n):
    """float64 values where float32 coercion before img_float2int changes the result."""
    found = []
    while len(found) < n:
        k = rng.integers(0, 255, 4096)
        x = (k + 0.5) / 255 + rng.uniform(-3e-8, 3e-8, k.size)
        x = x[(x >= 0) & (x <= 1)]
        f64 = np.round(255 * x)
        f32 = np.round(np.float32(255) * x.astype(np.float32))
        found.extend(x[f64 != f32].tolist())
    return np.array(found[:n])


def float2int_cases(rng, float2int):
    """[0, 1] floats of both dtypes, including exact (k + 0.5) / 255 ties."""
    values = np.concatenate([
        rng.uniform(0, 1, 2000),
        (np.arange(255) + 0.5) / 255,
        np.array([0.0, 1.0, 1 / 255, 254.5 / 255]),
        rounding_sensitive(rng, 64),
    ])
    out = {}
    for dtype in (np.float32, np.float64):
        name = np.dtype(dtype).name
        out[f"float2int__{name}__in"] = values.astype(dtype)
        out[f"float2int__{name}__out"] = float2int(values.astype(dtype))
    return out


def prepare_case(rng, float2int):
    """Two float64 stamps (the pretraining pickle's dtype) and their legacy uint8 form.

    Mostly zeros, so the .npz stays small, plus a random block and rounding-sensitive
    pixels, so that a float32 detour would not give the same bytes.
    """
    stamps = np.zeros((2, 63, 63, 3))
    stamps[:, 20:36, 20:36, :] = rng.uniform(0, 1, (2, 16, 16, 3))
    sensitive = rounding_sensitive(rng, 300)
    flat = stamps.reshape(-1)
    flat[rng.choice(flat.size, sensitive.size, replace=False)] = sensitive
    uint8 = np.stack([float2int(stamp) for stamp in stamps])
    return {"prepare__images_float64": stamps, "prepare__legacy_uint8": uint8}


def main():
    commit, legacy = legacy_functions()
    rng = np.random.default_rng(20261107)
    OUT.mkdir(parents=True, exist_ok=True)
    preprocessing = {
        **repair_cases(rng, legacy["repair_non_squared_stamp"]),
        **preprocess_cases(rng, legacy["preprocess_stamps"]),
        **float2int_cases(rng, legacy["img_float2int"]),
        **meta(commit),
    }
    np.savez_compressed(OUT / "preprocessing.npz", **preprocessing)
    np.savez_compressed(
        OUT / "prepare_legacy.npz", **prepare_case(rng, legacy["img_float2int"]), **meta(commit)
    )
    print(f"wrote {OUT}/preprocessing.npz and prepare_legacy.npz (legacy {commit[:7]}, "
          f"numpy {np.__version__})")


if __name__ == "__main__":
    main()
