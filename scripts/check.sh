#!/usr/bin/env bash
# All the checks that define "green" for this project: lint, formatting, types, tests.
# CI runs exactly this script; run it before every commit. Exit code 0 means every check
# passed. Formatters run in check mode (never rewrite files here): `uv run ruff format .`
# and `uv run ruff check --fix .` fix most findings.
set -euo pipefail

cd "$(dirname "$0")/.."

uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
uv run --locked pytest
