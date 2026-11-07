"""Characterization fixture for astro_simclr.augmentations (task 1.5).

Runs the LEGACY transforms (utils/transformations.py) on fixed uint8 stamps, with the global
Python / numpy / torch generators seeded the way ``lightning.seed_everything(seed)`` seeds
them, and records both views of each pair. The test replays the same seeds on the port.

Run from the repository root, in an isolated environment (legacy code only needs torch and
torchvision; cv2 and albumentations are imported at module level by the legacy file but only
used by the deferred variants, so they are stubbed):

    uv run --isolated --no-project --python 3.12 --with torch==2.14.1 \
        --with torchvision==0.29.2 --with numpy==2.5.4 \
        python scripts/characterize/augmentations/generate.py

Output: tests/fixtures/augmentations/legacy_views.npz. Views are stored as uint8 (every
legacy pipeline ends in ToTensor on a PIL image, so each value is exactly k/255).
"""

import random
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torchvision

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("cv2", types.ModuleType("cv2"))
sys.modules.setdefault("albumentations", types.ModuleType("albumentations"))

from utils import transformations as T  # noqa: E402

LEGACY = {
    "simclr": T.SimCLR_augmentation,
    "simclr2": T.SimCLR_augmentation_v2,
    "astro7": T.Astro_augmentation_v7,
    "astro8": T.Astro_augmentation_v8,
    "rotation_v3": T.Rotation_v3,
}
SIZES = (27, 63)
N_STAMPS = 2
SEED = 1234


def stamps() -> np.ndarray:
    """Smooth class-like blobs plus noise, so crops, blurs and jitter all change pixels."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:63, :63]
    out = np.empty((N_STAMPS, 63, 63, 3), dtype=np.uint8)
    for n in range(N_STAMPS):
        for c in range(3):
            cy, cx = rng.uniform(20, 43, size=2)
            blob = 200 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rng.uniform(3, 9) ** 2))
            noise = rng.normal(30, 10, size=(63, 63))
            out[n, :, :, c] = np.clip(blob + noise, 0, 255).astype(np.uint8)
    return out


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_uint8(t: torch.Tensor) -> np.ndarray:
    a = t.numpy() * 255
    r = np.rint(a)
    assert np.abs(a - r).max() < 1e-3, "legacy output is not k/255"
    return r.astype(np.uint8)


def main() -> None:
    imgs = stamps()
    to_pil = torchvision.transforms.ToPILImage()
    arrays: dict[str, np.ndarray] = {"stamps": imgs}
    for size in SIZES:
        factories = dict(LEGACY)
        for name in [*factories, "without_aug"]:
            first, second = [], []
            for n in range(N_STAMPS):
                # Build first, then seed: the legacy GaussianBlur creates two nn.Conv2d,
                # whose weight init draws from torch's generator. The port builds no module,
                # so seeding after construction gives both the same state at call time.
                if name == "without_aug":
                    resize = T.Resize_img(size=size)
                    aug = lambda pil: (resize(pil), resize(pil))  # noqa: E731
                else:
                    aug = factories[name](size=size)
                seed_all(SEED + n)
                a, b = aug(to_pil(imgs[n]))
                first.append(to_uint8(a))
                second.append(to_uint8(b))
            arrays[f"{name}_{size}_first"] = np.stack(first)
            arrays[f"{name}_{size}_second"] = np.stack(second)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    arrays["legacy_sha"] = np.array(sha)
    arrays["torch_version"] = np.array(torch.__version__)
    arrays["torchvision_version"] = np.array(torchvision.__version__)
    arrays["seed"] = np.array(SEED)
    out = ROOT / "tests/fixtures/augmentations/legacy_views.npz"
    np.savez_compressed(out, **arrays)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
