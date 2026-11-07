"""Generate the characterization fixtures of the losses from the legacy ``losses/losses.py``.

Runs the legacy ``NT_Xent`` and ``P_stamps_loss`` on fixed random inputs and records their
outputs and input gradients in ``tests/fixtures/losses/*.npz``, together with the git commit
of the legacy code and the torch version. ``tests/test_losses.py`` checks the port against
them.

The legacy package only needs torch and numpy, so it runs in an isolated environment, from
the repository root, without installing the project:

    uv run --isolated --no-project --with torch==2.14.1 --with numpy python \
        scripts/characterize/losses/generate.py

It refuses to run if ``losses/`` has uncommitted changes (the recorded SHA would lie).
Excluded from ruff, mypy and pytest (``scripts/characterize/``).
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from losses.losses import NT_Xent, P_stamps_loss  # noqa: E402  (legacy package)

OUT = ROOT / "tests" / "fixtures" / "losses"

#: Temperature of the reference SimCLR run R3 (31b10c9 settings), and a round one.
TEMPERATURES = (0.1791, 0.5)
#: Batch sizes the legacy class supports directly (its constructor batch size).
BATCH_SIZES = (1, 2, 7, 150)
DIM = 16


def legacy_commit():
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "losses/"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    if dirty:
        sys.exit("losses/ has uncommitted changes; commit or revert them first")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def nt_xent_case(loss_fn, z_i, z_j):
    z_i = z_i.clone().requires_grad_(True)
    z_j = z_j.clone().requires_grad_(True)
    loss, pos, neg = loss_fn(z_i, z_j)
    loss.backward()
    return {
        "z_i": z_i.detach().numpy(),
        "z_j": z_j.detach().numpy(),
        "loss": loss.detach().numpy(),
        "pos": pos.numpy(),
        "neg": neg.numpy(),
        "grad_z_i": z_i.grad.numpy(),
        "grad_z_j": z_j.grad.numpy(),
    }


def nt_xent_fixture(gen):
    arrays = {}
    for t_index, temperature in enumerate(TEMPERATURES):
        for batch in BATCH_SIZES:
            z_i = torch.randn(batch, DIM, generator=gen)
            z_j = torch.randn(batch, DIM, generator=gen)
            case = nt_xent_case(NT_Xent(batch, temperature), z_i, z_j)
            for key, value in case.items():
                arrays[f"t{t_index}/b{batch}/{key}"] = value
        # The legacy eval path: a batch of 100 through a loss built for another batch size.
        z_i = torch.randn(100, DIM, generator=gen)
        z_j = torch.randn(100, DIM, generator=gen)
        case = nt_xent_case(NT_Xent(150, temperature), z_i, z_j)
        for key, value in case.items():
            arrays[f"t{t_index}/b100/{key}"] = value
    arrays["temperatures"] = np.array(TEMPERATURES)
    arrays["batch_sizes"] = np.array((*BATCH_SIZES, 100))
    return arrays


def p_stamps_fixture(gen):
    arrays = {}
    betas = (0.0, 0.0161, 0.5)
    for b_index, beta in enumerate(betas):
        logits = (3 * torch.randn(9, 5, generator=gen)).requires_grad_(True)
        targets = torch.nn.functional.one_hot(torch.randint(0, 5, (9,), generator=gen), 5)
        targets = targets.float()
        loss = P_stamps_loss(9, beta)(logits, targets)
        loss.backward()
        arrays[f"beta{b_index}/logits"] = logits.detach().numpy()
        arrays[f"beta{b_index}/targets"] = targets.numpy()
        arrays[f"beta{b_index}/loss"] = loss.detach().numpy()
        arrays[f"beta{b_index}/grad_logits"] = logits.grad.numpy()
    arrays["betas"] = np.array(betas)
    return arrays


def main():
    meta = {"legacy_commit": np.array(legacy_commit()), "torch_version": np.array(torch.__version__)}
    gen = torch.Generator().manual_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "nt_xent.npz", **meta, **nt_xent_fixture(gen))
    np.savez_compressed(OUT / "p_stamps_loss.npz", **meta, **p_stamps_fixture(gen))
    print(f"wrote {OUT}/nt_xent.npz and p_stamps_loss.npz ({meta['legacy_commit']})")


if __name__ == "__main__":
    main()
