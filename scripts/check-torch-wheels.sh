#!/usr/bin/env bash
# Assert that uv.lock keeps the torch-wheel design of ADR 0001
# (docs/decisions/0001-torch-wheels-and-accelerators.md). CI runs it; run it after any
# change to the torch pins, extras, sources or indexes. It only resolves the lock: the
# dry-runs download no wheel. Exit code 0 means every assertion held; otherwise the failing
# assertion and the output it inspected are printed.
set -euo pipefail

cd "$(dirname "$0")/.."

linux=x86_64-manylinux_2_28
status=0

# expect DESCRIPTION REGEX TEXT: fail unless TEXT has a line matching the extended REGEX.
expect() {
  if grep -Eq -- "$2" <<<"$3"; then
    echo "ok: $1"
  else
    echo "FAIL: $1 (no line matches /$2/) in:" >&2
    echo "$3" >&2
    status=1
  fi
}

# reject DESCRIPTION REGEX TEXT: fail if TEXT has a line matching the extended REGEX.
reject() {
  if grep -Eq -- "$2" <<<"$3"; then
    echo "FAIL: $1 (a line matches /$2/) in:" >&2
    echo "$3" >&2
    status=1
  else
    echo "ok: $1"
  fi
}

# pip-audit reads this export; a local label (+cpu, +cu126) would make it skip torch.
export_default=$(uv export --locked --no-emit-project --format requirements-txt --no-hashes)
expect "default export pins PyPI torch" '^torch==[0-9][0-9.]*$' "$export_default"
reject "default export has no local-label torch" '^torch(vision)?==.*\+' "$export_default"

# Without an extra, Linux gets the PyPI CUDA build, which pulls the nvidia-* wheels.
default_linux=$(uv sync --locked --dry-run --python-platform "$linux" 2>&1)
expect "default Linux sync is the CUDA build" ' nvidia-' "$default_linux"
reject "default Linux sync has no local label" ' torch==.*\+' "$default_linux"

cu126_linux=$(uv sync --locked --dry-run --python-platform "$linux" --extra cu126 2>&1)
expect "cu126 Linux sync uses the CUDA 12.6 torch" ' torch==[0-9.]+\+cu126$' "$cu126_linux"
expect "cu126 Linux sync uses the CUDA 12.6 torchvision" \
  ' torchvision==[0-9.]+\+cu126$' "$cu126_linux"

# CI installs this one: it must not download any nvidia-* wheel.
cpu_linux=$(uv sync --locked --dry-run --python-platform "$linux" --extra cpu 2>&1)
expect "cpu Linux sync uses the CPU torch" ' torch==[0-9.]+\+cpu$' "$cpu_linux"
expect "cpu Linux sync uses the CPU torchvision" ' torchvision==[0-9.]+\+cpu$' "$cpu_linux"
reject "cpu Linux sync installs no nvidia wheel" ' nvidia-' "$cpu_linux"

exit "$status"
