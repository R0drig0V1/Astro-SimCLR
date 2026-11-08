<!-- ADR 0001: how torch and torchvision wheels are chosen per machine. -->
# 0001. Torch wheels and accelerators

- **Status:** accepted (2026-11-07, phase 1, task 1.1); rule 6 amended in task 1.7
  (compute capability, Blackwell, Windows)
- **Checked by:** `scripts/check-torch-wheels.sh` (run in CI)

## Context

The project must run on three kinds of machine with one `uv.lock`:

| Machine | Wanted torch build |
|---|---|
| macOS arm64 (development) | the PyPI wheel (MPS) |
| CI, `ubuntu-latest` x86_64, no GPU | CPU only: the default Linux wheel pulls ~3 GB of `nvidia-*` wheels |
| GPU machine (lab server, Colab, Kaggle, rented), linux x86_64 | CUDA, with a build the NVIDIA driver supports |

Facts that constrain the design (verified with uv 0.11 on toy projects; plan review
`docs/plan-reviews/2026-11-07-phase-1.md`, findings 2.1-2.3):

- On Linux, PyPI torch is the CUDA 13 build, which needs NVIDIA driver >= 580. Colab, Kaggle
  and many servers run older drivers.
- The last torch minor with CPU, CUDA 12 (`cu126`) *and* CUDA 13 builds is 2.14.
- **A single `cpu` extra cannot be locked usefully**: either uv refuses to lock it, or the
  index mapping leaks into the default resolution and every install silently gets CPU
  wheels (finding 2.1).
- `uv run` re-syncs the environment to the default extras before running, which silently
  undoes a `uv sync --extra ...` (finding 2.2).
- pip-audit skips packages with a local version label (`2.14.1+cpu`), so the audited export
  must hold PyPI torch.
- Platform markers alone cannot pick the build: CI and the GPU machine are both linux x86_64.

## Decision

1. **Default = PyPI torch.** `torch` and `torchvision` are plain dependencies, so
   `uv sync --locked` gives MPS on the Mac and CUDA 13 on Linux.
2. **Two conflicting extras, `cpu` and `cu126`**, declared in `[project.optional-dependencies]`
   and in `[tool.uv] conflicts`. `[tool.uv.sources]` maps **both** `torch` and
   `torchvision` to the `pytorch-cpu` / `pytorch-cu126` indexes per extra (mixing indexes
   breaks the torchvision ABI). The indexes are `explicit = true`, so nothing else is ever
   resolved from them. uv locks three forks: default, `cpu` and `cu126`.
3. **Pins:** `torch>=2.14,<2.15` and `torchvision==0.29.*`. Every fork resolves torch
   2.14.1; torchvision may differ by a patch between indexes (`0.29.1+cu126` vs `0.29.2`),
   which is accepted.
4. **Dependabot** (`.github/dependabot.yml`) ignores major and minor bumps of torch and
   torchvision; patches still flow. A minor bump is done by hand, only once the new minor
   has `cpu` and the chosen CUDA-12 builds, and it updates this ADR.
5. **Sync discipline.** Only the first command syncs; everything after it skips the sync:

   ```bash
   uv sync --locked --extra cpu            # or --extra cu126, or no extra
   UV_NO_SYNC=1 scripts/check.sh           # or: uv run --no-sync astro-simclr ...
   ```

6. **The driver and the GPU's compute capability pick the extra** on a GPU machine
   (`nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv`). *Amended in
   task 1.7:* the first version picked by driver only, which is wrong for pre-Turing GPUs
   with a new driver: CUDA 13 dropped compute capability < 7.5, so the default build has no
   kernels for them (the user's GTX 1050, sm_61, driver 582), and `cu126` has no Blackwell
   kernels (sm_100, sm_120).

   | GPU (compute capability) | Driver | Command |
   |---|---|---|
   | Maxwell, Pascal, Volta (< 7.5; e.g. GTX 1050, P100, V100) | >= 525 | `uv sync --locked --extra cu126` |
   | Turing to Hopper (7.5-9.x; e.g. T4, RTX 30/40, A100, H100) | >= 580 | `uv sync --locked` (PyPI, CUDA 13) |
   | Turing to Hopper | 525-579 | `uv sync --locked --extra cu126` |
   | Blackwell (>= 10.0; e.g. RTX 5090, sm_120) | >= 580 | `uv sync --locked` (`cu126` cannot run it) |
   | Blackwell | < 580 | none: update the driver |
   | any | < 525 | none: update the driver |
   | no NVIDIA GPU | | `uv sync --locked --extra cpu` |

   **Windows:** PyPI torch for native Windows is CPU-only; run the project inside WSL2
   (the Windows driver serves WSL2, and `nvidia-smi` inside WSL2 reports it).

   `astro-simclr preflight` (step 1; `scripts/gpu-preflight.sh`) applies this table
   (`preflight.choose_torch_build`), then checks the installed build against the machine:
   `torch.version.cuda` not None, a driver new enough for it (CUDA 13: >= 580; CUDA 12:
   >= 525), `torch.cuda.is_available()`, and the GPU's `sm_XY` in
   `torch.cuda.get_arch_list()` (same major, minor <= the GPU's). On a mismatch it prints
   the `uv sync` command that fixes it.

7. **CI** (on `ubuntu-latest` x86_64 and, since task 1.6, `macos-latest` arm64) runs
   `uv sync --locked --extra cpu`, checks that no `nvidia-*` package is
   installed, runs `scripts/check-torch-wheels.sh` and then `scripts/check.sh` with
   `UV_NO_SYNC=1`. The assertion script checks exact strings (dry-runs, no download):
   - the default `uv export --locked` pins `torch==X` with no `+` label (what pip-audit
     audits);
   - `uv sync --locked --dry-run --python-platform x86_64-manylinux_2_28` lists `nvidia-`
     packages and an unlabelled torch;
   - the same with `--extra cu126` lists `torch==...+cu126` and `torchvision==...+cu126`;
   - the same with `--extra cpu` lists `+cpu` builds and no `nvidia-` package.

### Rejected

- **One `cpu` extra:** cannot be locked without leaking (finding 2.1).
- **Platform markers only:** CI and the GPU machine share a platform.
- **Default = CPU, extra = CUDA:** the Mac would lose nothing, but a GPU user who forgets the
  extra would train on CPU without noticing, and pip-audit would lose torch.
- **The PyTorch index as the default source of torch:** every lock fork would carry a local
  label, so pip-audit could not audit torch.

## Consequences

- `uv.lock` is larger (three torch/torchvision forks and the CUDA-12 `nvidia-*` wheels).
- Anyone using an extra must remember `UV_NO_SYNC=1` / `--no-sync`; a plain `uv run` swaps
  torch back to the default build (correct, but a 3 GB download on Linux). The runbook and
  the reference configs (`accelerator: cuda`, which fails fast without a GPU) guard against it.
- torch stays on 2.14 until a newer minor has all three builds.
- pip-audit (`quality.yml`) audits the default fork, PyPI torch 2.14.1 included; the `+cpu`
  and `+cu126` builds are the same source release.
