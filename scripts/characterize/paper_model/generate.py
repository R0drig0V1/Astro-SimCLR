"""Generate the golden fixture of the faithful paper model (task 2.5-models).

The 2.5 architecture flags (``paper_arch``, ``group_fields``, ``context``,
``rot90_augment``) must leave the faithful ``carrasco2021`` model bit-for-bit unchanged:
the same seeded initialisation (the same draws from the global torch generator, in the same
order) and the same forward output. This script runs ``PaperStampClassifier`` as it was
**before** those flags existed (``src/astro_simclr/paper_model.py`` at ``BASE_SHA``, read
with ``git show``, so the fixture does not depend on the checkout) and writes
``tests/fixtures/paper_model/faithful.npz``. ``tests/test_paper_variants.py`` compares the
current code with it, once per flag.

For each of ``with_features`` True and False the file records, after
``torch.manual_seed(0)``:

- ``<case>_params``: the trainable parameter count (441,075 / 439,557);
- ``<case>_init``: per-tensor sums (float64) of the ``state_dict``, in sorted key order;
- ``<case>_logits64``: the eval-mode logits of the seeded model on ``images`` and
  ``features``, computed in float64 (``model.double()``), so that a comparison can be tight
  without depending on the float32 convolution backend of the machine.

Run from the repository root (excluded from ruff, mypy and pytest)::

    uv run python scripts/characterize/paper_model/generate.py
"""

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "tests" / "fixtures" / "paper_model" / "faithful.npz"
#: The last commit of main before the 2.5-models flags (paper_model.py of task 2.2a).
BASE_SHA = "2697abb2d48c2fccb96d98d827a962b047f47e23"
BATCH = 6


def load_base_model(tmp):
    """Import ``paper_model.py`` at ``BASE_SHA`` as a standalone module."""
    source = subprocess.run(
        ["git", "show", f"{BASE_SHA}:src/astro_simclr/paper_model.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    path = Path(tmp) / "paper_model_base.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("paper_model_base", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    gen = torch.Generator().manual_seed(2025)
    images = torch.rand(BATCH, 3, 21, 21, generator=gen)
    features = torch.randn(BATCH, 23, generator=gen)
    arrays = {"images": images.numpy(), "features": features.numpy()}
    with tempfile.TemporaryDirectory() as tmp:
        base = load_base_model(tmp)
        for case, with_features in (("features", True), ("no_features", False)):
            torch.manual_seed(0)
            model = base.PaperStampClassifier(with_features=with_features).eval()
            state = model.state_dict()
            arrays[f"{case}_params"] = np.array(
                sum(p.numel() for p in model.parameters() if p.requires_grad)
            )
            arrays[f"{case}_init"] = np.array(
                [state[k].double().sum().item() for k in sorted(state)]
            )
            with torch.no_grad():
                logits = model.double()(images.double(), features.double())
            arrays[f"{case}_logits64"] = logits.numpy()
            print(case, int(arrays[f"{case}_params"]), logits[0].tolist())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, base_sha=np.array(BASE_SHA), torch_version=np.array(torch.__version__), **arrays)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
