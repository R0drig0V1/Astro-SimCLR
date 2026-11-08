#!/usr/bin/env bash
# All the checks that define "green" for this project: lint, formatting, types, tests.
# CI runs exactly this script; run it before every commit. Exit code 0 means every check
# passed. After `uv sync --locked --extra cpu|cu126`, run it with UV_NO_SYNC=1 (as CI does):
# `uv run` would otherwise re-sync the default torch build (ADR 0001). Formatters run in check mode (never rewrite files here): `uv run ruff format .`
# and `uv run ruff check --fix .` fix most findings.
set -euo pipefail

cd "$(dirname "$0")/.."

uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
# Tests run in parallel (pytest-xdist, one process per CPU; PYTEST_XDIST_AUTO_NUM_WORKERS
# caps it). The suite is parallel-safe: every test writes under tmp_path, session fixtures
# are built per process, and it passes in reverse order. Coverage is combined across the
# processes. -rs prints the reason of every skipped test. A plain `uv run pytest` stays serial.
uv run --locked pytest -n auto -rs
