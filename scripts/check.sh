#!/usr/bin/env bash
# All the checks that define "green" for this project: lint, formatting, types, tests.
# CI runs exactly this script; run it before every commit. Exit code 0 means every check
# passed. After `uv sync --locked --extra cpu|cu126`, run it with UV_NO_SYNC=1 (as CI does):
# `uv run` would otherwise re-sync the default torch build (ADR 0001). Formatters run in check mode (never rewrite files here): `uv run ruff format .`
# and `uv run ruff check --fix .` fix most findings.
#
# The tests run behind a machine-wide lock shared with the other project on the developer's
# Mac (scripts/check-lock.sh: one full check at a time, stale locks of dead pids broken;
# ASTRO_SIMCLR_CHECK_LOCK=0 disables it, GitHub Actions skips it) and with at most 4
# pytest-xdist workers (PYTEST_XDIST_AUTO_NUM_WORKERS overrides the cap).
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=scripts/check-lock.sh
source scripts/check-lock.sh

uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
# Tests run in parallel (pytest-xdist, check_pytest_workers processes). The suite is
# parallel-safe: every test writes under tmp_path, session fixtures are built per process,
# and it passes in reverse order. Coverage is combined across the processes. -rs prints the
# reason of every skipped test. A plain `uv run pytest` stays serial.
# The lock covers pytest only: it is the step that spawns the workers and fills the memory.
check_lock_install_traps
check_lock_acquire
PYTEST_XDIST_AUTO_NUM_WORKERS=$(check_pytest_workers)
export PYTEST_XDIST_AUTO_NUM_WORKERS
# The gate checks the golden regression set; a stale ASTRO_SIMCLR_RECORD_GOLDEN=1 in the
# shell would make it rewrite the goldens instead (tests/test_engine_regression.py).
unset ASTRO_SIMCLR_RECORD_GOLDEN
uv run --locked pytest -n auto -rs
