"""Characterization fixtures for astro_simclr.optim (task 1.5): LARS and the LR schedule.

1. LARS: the legacy ``utils/lars.py`` optimizer takes two steps on small fixed tensors, with
   and without Nesterov momentum. ``Tensor.get_device`` is monkeypatched to return "cpu",
   because the legacy code calls ``torch.Tensor([1.0]).to(g_norm.get_device())`` and
   ``get_device()`` is -1 on CPU.
2. LR schedule: the legacy SimCLR ``configure_optimizers`` (``utils/training.py``): timm 0.6.x
   ``CosineLRScheduler(warmup_t=5, warmup_lr_init=lr/3, t_initial=350, cycle_limit=1,
   lr_min=1e-4)`` wrapped in ``LambdaLR``; ``param_groups[0]["lr"]`` is recorded for epochs
   0-400 (one ``LambdaLR.step()`` per epoch, as Lightning does with interval "epoch").

Run from the repository root, in an isolated environment:

    uv run --isolated --no-project --python 3.12 --with timm==0.6.13 \
        --with torch==2.14.1 --with torchvision==0.29.2 --with numpy==2.5.4 \
        python scripts/characterize/optim/generate.py
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import timm
import torch
from timm.scheduler import CosineLRScheduler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
torch.Tensor.get_device = lambda self: "cpu"  # type: ignore[method-assign,assignment]

from utils.lars import LARS  # noqa: E402

LRS = (0.258, 1.258)
EPOCHS = 401


def params() -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return [
        torch.randn(4, 3, generator=g),
        torch.randn(5, generator=g),
        torch.randn(2, 3, 3, generator=g),
    ]


def grads(step: int) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(100 + step)
    return [
        torch.randn(4, 3, generator=g),
        torch.randn(5, generator=g),
        torch.randn(2, 3, 3, generator=g) * 1e-2,
    ]


def lars_case(nesterov: bool, weight_decay: float) -> list[np.ndarray]:
    ps = [torch.nn.Parameter(p.clone()) for p in params()]
    opt = LARS(ps, lr=0.258, momentum=0.9, use_nesterov=nesterov, weight_decay=weight_decay)
    out = []
    for step in range(2):
        for p, gr in zip(ps, grads(step)):
            p.grad = gr.clone()
        opt.step()
        out.extend(p.detach().numpy().copy() for p in ps)
    return out


def schedule(lr: float) -> np.ndarray:
    p = torch.nn.Parameter(torch.zeros(1))
    opt = LARS([p], lr=lr, weight_decay=0.0, momentum=0.9, use_nesterov=True)
    aux = CosineLRScheduler(
        opt, warmup_t=5, warmup_lr_init=lr / 3, t_initial=350, cycle_limit=1, lr_min=0.0001
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda epoch: aux.get_epoch_values(epoch)[0] / lr
    )
    values = []
    for _ in range(EPOCHS):
        values.append(opt.param_groups[0]["lr"])
        sched.step()
    return np.array(values, dtype=np.float64)


def main() -> None:
    arrays: dict[str, np.ndarray] = {}
    for i, p in enumerate(params()):
        arrays[f"lars_init_{i}"] = p.numpy()
    for step in range(2):
        for i, gr in enumerate(grads(step)):
            arrays[f"lars_grad_step{step}_{i}"] = gr.numpy()
    for nesterov in (False, True):
        for wd in (0.0, 1e-4):
            tag = f"nesterov{int(nesterov)}_wd{wd:g}"
            for k, a in enumerate(lars_case(nesterov, wd)):
                arrays[f"lars_{tag}_step{k // 3}_{k % 3}"] = a
    for lr in LRS:
        arrays[f"lr_{lr:g}"] = schedule(lr)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    arrays["legacy_sha"] = np.array(sha)
    arrays["torch_version"] = np.array(torch.__version__)
    arrays["timm_version"] = np.array(timm.__version__)
    out = ROOT / "tests/fixtures/optim/legacy_optim.npz"
    np.savez_compressed(out, **arrays)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
