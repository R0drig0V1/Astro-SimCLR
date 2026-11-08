"""The argument parser of ``astro-simclr runs`` (task A2): argparse only.

Kept apart from :mod:`~astro_simclr.registry.commands` (the handlers) so that building the
top-level parser, and ``astro-simclr --help``, imports nothing heavy: the handlers, and
with them the run modules (numpy), are imported when a ``runs`` subcommand runs. The two
choice lists below are copies of ``registry.compare.MIXABLE`` and
``runs.manifest.STATUSES``; a test keeps them equal.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

__all__ = ["MIXABLE", "STATUSES", "add_runs_command"]

#: ``registry.compare.MIXABLE``.
MIXABLE: Final = ("machines", "engines", "numerics")
#: ``runs.manifest.STATUSES``.
STATUSES: Final = ("queued", "running", "finished", "failed", "cancelled")


def _dispatch(args: argparse.Namespace) -> int:
    from astro_simclr.registry.commands import run

    return run(args)


def add_runs_command(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``runs`` and its subcommands (module docstring) on ``commands``."""
    runs = commands.add_parser(
        "runs",
        help="the local run registry: reindex, ls, show, compare",
        description="A derived SQLite index of run directories at "
        "$ASTRO_SIMCLR_HOME/registry.sqlite (default ~/.astro-simclr), rebuildable from "
        "them with `runs reindex`.",
    )
    sub = runs.add_subparsers(dest="runs_command", metavar="SUBCOMMAND", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--registry",
        type=Path,
        default=None,
        metavar="FILE",
        help="the registry file (default: $ASTRO_SIMCLR_HOME/registry.sqlite)",
    )
    as_json = argparse.ArgumentParser(add_help=False)
    as_json.add_argument("--json", action="store_true", help="print JSON")

    index = sub.add_parser(
        "reindex",
        parents=[common],
        help="scan run roots into the registry",
        description="Scan each ROOT (default: every root scanned before) for run "
        "directories (manifest.json, or a legacy metrics.json) and the specs directory. "
        "Exit 0 when the scan ends, even if some directories were skipped.",
    )
    index.add_argument("roots", type=Path, nargs="*", metavar="ROOT")
    index.add_argument(
        "--rebuild", action="store_true", help="forget every indexed run first and rehash"
    )
    index.set_defaults(handler=_dispatch)

    ls = sub.add_parser("ls", parents=[common, as_json], help="list the indexed runs")
    ls.add_argument("--command", dest="command_filter", default=None, metavar="COMMAND")
    ls.add_argument("--config", default=None, metavar="SHA", help="a config hash or its prefix")
    ls.add_argument("--seed", type=int, default=None, metavar="N")
    ls.add_argument("--status", choices=STATUSES, default=None)
    ls.add_argument(
        "--legacy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="only backfilled runs (--legacy) or only runs with a manifest (--no-legacy)",
    )
    ls.set_defaults(handler=_dispatch)

    show = sub.add_parser("show", parents=[common, as_json], help="one indexed run in full")
    show.add_argument("key", metavar="KEY", help="an attempt key, run id prefix or run directory")
    show.set_defaults(handler=_dispatch)

    compare = sub.add_parser(
        "compare",
        parents=[common, as_json],
        help="resolve an experiment spec to runs and report them",
        description="Resolve every cell (arm x seed x split) of the experiment spec to "
        "one finished attempt and report the metrics per cell and per arm (mean, std with "
        "ddof=1). Fails (exit 1) if a cell is missing, a cell has several attempts and no "
        "--use/--attempts choice, or the runs mix machines, engines or numerics epochs. "
        "A report, never a decision.",
    )
    compare.add_argument(
        "spec", metavar="SPEC", help="a spec file, or an indexed spec's id or name"
    )
    compare.add_argument(
        "--metric",
        action="append",
        default=[],
        metavar="NAME",
        help="a metric to report, e.g. val.balanced_accuracy (repeatable; default: the "
        "balanced accuracy and macro-F1 of every stage)",
    )
    compare.add_argument(
        "--use",
        action="append",
        default=[],
        metavar="KEY",
        help="the attempt to take for its cell (repeatable)",
    )
    compare.add_argument(
        "--attempts",
        choices=["error", "latest"],
        default="error",
        help="a cell with several attempts: fail (default) or take the latest-ended one",
    )
    compare.add_argument(
        "--allow-mixed",
        action="append",
        choices=MIXABLE,
        default=[],
        help="allow runs that differ in this (repeatable); data never mixes",
    )
    compare.set_defaults(handler=_dispatch)
