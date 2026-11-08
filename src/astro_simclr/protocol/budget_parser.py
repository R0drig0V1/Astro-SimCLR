"""``make-label-budgets`` and ``verify-label-budgets`` (phase 5, task 5.4): parser and handlers.

The parser is argparse only, so building the top-level parser imports nothing heavy; each
handler imports :mod:`astro_simclr.protocol.label_budgets` (numpy) when it runs. The parent
of a budgets file is either the stored CD21 CV splits (``--splits FILE --data
PREPARED_DIR``) or R5's stored object splits (``--object-splits FILE --alerts
ALERTS_CSV``); both are verified against the data before anything is drawn or checked.
Errors (malformed, infeasible or foreign files) are a
:class:`~astro_simclr.data.DataError` and exit 1 through the CLI's handler.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astro_simclr.protocol.label_budgets import PartsSource

__all__ = ["add_budget_commands"]


def _parts(args: argparse.Namespace) -> PartsSource:
    from astro_simclr.config import ConfigError
    from astro_simclr.protocol.label_budgets import (
        cv_parts_from_files,
        object_split_parts_from_files,
    )

    if args.splits is not None:
        if args.data is None or args.object_splits is not None or args.alerts is not None:
            raise ConfigError("--splits needs --data PREPARED_DIR (and no --object-splits)")
        return cv_parts_from_files(args.splits, args.data)
    if args.object_splits is None or args.alerts is None or args.data is not None:
        raise ConfigError(
            "give --splits FILE --data PREPARED_DIR, or --object-splits FILE --alerts CSV"
        )
    names = [n.strip() for n in args.labels.split(",") if n.strip()]
    return object_split_parts_from_files(args.object_splits, args.alerts, names)


def _make(args: argparse.Namespace) -> int:
    from astro_simclr.config import ConfigError
    from astro_simclr.protocol.label_budgets import make_label_budgets, write_label_budgets

    if args.out.exists() and not args.overwrite:
        raise ConfigError(f"{args.out} already exists; pass --overwrite to replace it")
    found = _parts(args)
    budgets = make_label_budgets(
        found.parts,
        source=found.source,
        label_names=found.label_names,
        object_id=found.object_id,
        budgets=args.budgets,
        n_repeats=args.repeats,
        selection_fraction=args.selection_fraction,
        include_full=not args.no_full,
        seed=args.seed,
    )
    write_label_budgets(budgets, args.out)
    sizes = ", ".join(f"{b}: {n}" for b, n in budgets.selection_per_class.items())
    sys.stdout.write(
        f"parts: {', '.join(budgets.part_names)}; {budgets.n_repeats} repeats of budgets "
        f"{list(budgets.budgets)} objects per class (nested), 100 %: "
        f"{'yes' if budgets.include_full else 'no'}\n"
        f"selection objects per class ({sizes})\nwrote {args.out}\n"
    )
    return 0


def _verify(args: argparse.Namespace) -> int:
    from astro_simclr.protocol.label_budgets import read_label_budgets, verify_label_budgets

    budgets = read_label_budgets(args.file)
    found = _parts(args)
    verify_label_budgets(budgets, found.parts, source=found.source)
    sys.stdout.write(f"{args.file}: OK ({len(budgets.part_names)} parts)\n")
    return 0


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    from astro_simclr.protocol.alerts import BENCHMARK_LABEL_NAMES

    parser.add_argument(
        "--splits", type=Path, metavar="FILE", help="stored CV splits (make-splits)"
    )
    parser.add_argument("--data", type=Path, metavar="PREPARED_DIR")
    parser.add_argument(
        "--object-splits", type=Path, metavar="FILE", help="stored object splits (R5)"
    )
    parser.add_argument("--alerts", type=Path, metavar="ALERTS_CSV")
    parser.add_argument(
        "--labels",
        default=",".join(BENCHMARK_LABEL_NAMES),
        metavar="NAMES",
        help="the alert table's class names in label order (default: %(default)s)",
    )


def add_budget_commands(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``make-label-budgets`` and ``verify-label-budgets`` on ``commands``."""
    make = commands.add_parser(
        "make-label-budgets",
        help="write the stored label-budget subsets of phase 5 (objects per class)",
        description="Verify the parent splits on the data, then draw, from each training "
        "part (every CV fold's, or R5 Train), REPEATS nested subsets of BUDGET labelled "
        "objects per class and a selection set of at most 20 %% of each budget, and write "
        "them as JSON (docs/evaluation-protocol.md, Phase 5 label budgets).",
    )
    make.add_argument("--out", type=Path, required=True, metavar="FILE")
    _add_source_arguments(make)
    make.add_argument("--budgets", type=int, nargs="+", default=[10, 100, 1000], metavar="BUDGET")
    make.add_argument("--repeats", type=int, default=20, metavar="R")
    make.add_argument(
        "--selection-fraction",
        type=float,
        default=0.2,
        metavar="F",
        help="selection objects per class as a fraction of the budget, at most 0.2; 0 "
        "stores none (the fixed-schedule alternative)",
    )
    make.add_argument("--no-full", action="store_true", help="do not include 100 %%")
    make.add_argument("--seed", type=int, default=0, metavar="N")
    make.add_argument("--overwrite", action="store_true")
    make.set_defaults(handler=_make)

    verify = commands.add_parser(
        "verify-label-budgets",
        help="check a label-budgets file against its parent splits and the data",
        description="Verify the parent splits on the data, then check that the file was "
        "drawn from that parent (SHA-256 and data fingerprint) and is well formed: exact "
        "objects per class, all of that class in the training part, selection sets "
        "outside the subset. Exit 0 if it is, 1 otherwise.",
    )
    verify.add_argument("file", type=Path, metavar="FILE")
    _add_source_arguments(verify)
    verify.set_defaults(handler=_verify)
