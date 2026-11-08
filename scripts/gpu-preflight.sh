#!/usr/bin/env bash
# GPU preflight: a thin wrapper around `astro-simclr preflight` (src/astro_simclr/preflight.py),
# which holds all the logic. Run it after `uv sync --locked [--extra cu126]` (ADR 0001): it
# uses `uv run --no-sync`, so it never swaps the installed torch build.
#   scripts/gpu-preflight.sh --synthetic            # CUDA works (no real data needed)
#   scripts/gpu-preflight.sh --data-dir data/       # the prepared real data too
# Extra arguments are passed through (see `astro-simclr preflight --help`). The exit code is
# the preflight's: 0 if no step failed.
set -euo pipefail

cd "$(dirname "$0")/.."

exec uv run --no-sync astro-simclr preflight "$@"
