"""Generate the characterization fixtures of src/astro_simclr/models.py from the legacy code.

Runs the legacy networks (``models/models.py`` at LEGACY_SHA, and ``P_stamps_net_SimCLR_v2``
from PSTAMPS_SHA, where it last existed) and writes one ``.npz`` per case to
``tests/fixtures/models/``. Each file records the legacy commit, the torch version, the
legacy parameter count, a fingerprint of the seeded default initialisation, the inputs and
the outputs of an eval-mode forward pass with the deterministic weights of
``tests/model_weights.py``.

The legacy sources are read with ``git show`` (not from the working tree), so the fixtures
do not depend on the checkout. Run from the repository root, in an isolated environment
pinned to the project's torch (excluded from ruff, mypy and pytest):

    uv run --isolated --no-project --python 3.12 \
        --with torch==2.14.1 --with torchvision==0.29.2 --with numpy \
        python scripts/characterize/models/generate.py
"""

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchvision

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "tests" / "fixtures" / "models"
#: Last commit that touched models/models.py (the legacy oracle at main).
LEGACY_SHA = "b357b4390d9f77e74bfdbbf88193299dd5edfc43"
#: b357b43^: the last tree with P_stamps_net_SimCLR_v2 (the 300k-alert encoder of 31b10c9).
PSTAMPS_SHA = "c5c7e6f3ccbcd9306267e4e307d7ac7aac8aeb5a"
BATCH = 4


def load_legacy(sha, tmp):
    """Import models/models.py at ``sha`` as a standalone module."""
    source = subprocess.run(
        ["git", "show", f"{sha}:models/models.py"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout
    path = Path(tmp) / f"legacy_models_{sha[:7]}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_weights_helper():
    spec = importlib.util.spec_from_file_location(
        "model_weights", ROOT / "tests" / "model_weights.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fill_deterministic


def legacy_resnet(name):
    """The legacy ResNet encoder: torchvision, no weights, ``fc`` replaced by Identity."""
    net = getattr(torchvision.models, name)()
    net.fc = torch.nn.Identity()
    return net


def init_fingerprint(factory):
    """Per-tensor sums of the default initialisation after ``torch.manual_seed(0)``."""
    torch.manual_seed(0)
    state = factory().state_dict()
    return np.array(
        [state[k].double().sum().item() for k in sorted(state) if state[k].is_floating_point()]
    )


def main():
    fill = load_weights_helper()
    gen = torch.Generator().manual_seed(1234)
    img27 = torch.rand(BATCH, 3, 27, 27, generator=gen)
    img27_j = torch.rand(BATCH, 3, 27, 27, generator=gen)
    img63 = torch.rand(BATCH, 3, 63, 63, generator=gen)
    feat = torch.randn(BATCH, 23, generator=gen)
    h151 = torch.randn(BATCH, 128 + 23, generator=gen)
    h512 = torch.randn(BATCH, 512, generator=gen)

    with tempfile.TemporaryDirectory() as tmp:
        legacy = load_legacy(LEGACY_SHA, tmp)
        historical = load_legacy(PSTAMPS_SHA, tmp)

        # name -> (sha, factory, inputs); the factory builds the legacy network.
        cases = {
            "stamp_encoder_k1": (LEGACY_SHA, lambda: legacy.Stamp_encoder(k=1.0), [img27]),
            "stamp_encoder_k0_5": (LEGACY_SHA, lambda: legacy.Stamp_encoder(k=0.5), [img27]),
            "stamp_encoder_k1_5": (LEGACY_SHA, lambda: legacy.Stamp_encoder(k=1.5), [img27]),
            "pstamps": (
                PSTAMPS_SHA, lambda: historical.P_stamps_net_SimCLR_v2(k=1.0), [img27]
            ),
            "stamp_classifier_features": (
                LEGACY_SHA, lambda: legacy.Stamp_classifier(0.2, True), [img27, feat]
            ),
            "stamp_classifier_no_features": (
                LEGACY_SHA, lambda: legacy.Stamp_classifier(0.2, False), [img27, feat]
            ),
            "linear_classifier": (
                LEGACY_SHA, lambda: legacy.Linear_classifier(512, 5), [h512]
            ),
            "three_layers_classifier": (
                LEGACY_SHA,
                lambda: legacy.Three_layers_classifier(151, 64, 64, 5, 0.2),
                [h151],
            ),
            "clr_pstamps": (
                PSTAMPS_SHA,
                lambda: historical.CLR(historical.P_stamps_net_SimCLR_v2(k=1.0), 128, 300),
                [img27, img27_j],
            ),
            "resnet18_27": (LEGACY_SHA, lambda: legacy_resnet("resnet18"), [img27]),
            "resnet18_63": (LEGACY_SHA, lambda: legacy_resnet("resnet18"), [img63]),
        }
        OUT.mkdir(parents=True, exist_ok=True)
        for name, (sha, factory, inputs) in cases.items():
            net = factory()
            fill(net)
            net.eval()
            with torch.no_grad():
                out = net(*inputs)
            outputs = out if isinstance(out, tuple) else (out,)
            arrays = {f"input_{i}": x.numpy() for i, x in enumerate(inputs)}
            arrays |= {f"output_{i}": y.numpy() for i, y in enumerate(outputs)}
            np.savez_compressed(
                OUT / f"{name}.npz",
                legacy_sha=np.array(sha),
                torch_version=np.array(torch.__version__),
                n_params=np.array(sum(p.numel() for p in net.parameters())),
                init_fingerprint=init_fingerprint(factory),
                **arrays,
            )
            print(f"{name}: {[tuple(y.shape) for y in outputs]}")

        # Parameter counts only (no forward): the other ResNets and the CLR of stamp1.
        counts = {
            name: sum(p.numel() for p in legacy_resnet(name).parameters())
            for name in ("resnet34", "resnet50", "resnet152")
        }
        counts["clr_stamp1"] = sum(
            p.numel() for p in legacy.CLR(legacy.Stamp_encoder(k=1.0), 128, 300).parameters()
        )
        np.savez_compressed(
            OUT / "param_counts.npz",
            legacy_sha=np.array(LEGACY_SHA),
            torch_version=np.array(torch.__version__),
            torchvision_version=np.array(torchvision.__version__),
            **{k: np.array(v) for k, v in counts.items()},
        )
        print(counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
