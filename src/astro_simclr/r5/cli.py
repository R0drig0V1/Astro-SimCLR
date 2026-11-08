"""The R5 commands of the ``astro-simclr`` CLI (task 3.5), registered by ``cli.build_parser``.

- ``fetch-alerts --oids FILE --out DIR`` (the by-oid mode of task 3.1's command,
  :mod:`astro_simclr.alerts.by_oid`): fetch each listed object's detection list (``OID``
  lines) and alert (``OID CANDID`` lines), resumable with ``--resume``;
- ``r5-sources --out DIR``: snapshot the Zenodo labelled set and the BTS SN table
  (:mod:`astro_simclr.r5.sources`);
- ``build-r5``: the R5 build (:mod:`astro_simclr.r5.build`), online by default (it fetches
  what the snapshot lacks into ``--fetched``), offline with ``--from-snapshot`` (no network;
  ``--expect build.json`` checks every output's SHA-256 against an earlier build), or
  ``--dry-run`` (one round, nothing fetched or written: the next round's request count).

SIGTERM, SIGHUP and Ctrl-C stop a fetch after the current item and commit its batch (exit
128 + the signal); a second signal forces the exit. The modules are imported inside the
handlers, so ``--help`` stays fast.
"""

from __future__ import annotations

import argparse
import signal
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from astro_simclr.config import ConfigError

if TYPE_CHECKING:
    from astro_simclr.alerts.by_oid import ItemLine

__all__ = ["add_commands", "fetch_by_oid_command"]

#: Exit code of a run whose result is a failure (a check failed, the build is incomplete).
EXIT_FAIL = 1


@contextmanager
def stop_on_signals() -> Iterator[list[int]]:
    """Install the stop-after-this-item handlers; yields the list of signals received."""
    stopping: list[int] = []

    def on_signal(signum: int, _frame: object) -> None:
        if stopping:
            raise KeyboardInterrupt
        stopping.append(signum)
        sys.stderr.write(f"astro-simclr: signal {signum}: stopping after the current item\n")

    handled = [getattr(signal, s) for s in ("SIGTERM", "SIGHUP", "SIGINT") if hasattr(signal, s)]
    previous = {sig: signal.signal(sig, on_signal) for sig in handled}
    try:
        yield stopping
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, signal.SIG_DFL if handler is None else handler)


def _check_ranges(args: argparse.Namespace) -> None:
    for flag, value, ok in (
        ("--rate", args.rate, args.rate > 0),
        ("--batch-size", args.batch_size, args.batch_size >= 1),
        ("--cooldown", args.cooldown, args.cooldown >= 0),
    ):
        if not ok:
            raise ConfigError(f"{flag} is out of range: {value}")


def _progress(quiet: bool) -> Callable[[int, int, ItemLine], None] | None:
    if quiet:
        return None

    def report(position: int, total: int, line: ItemLine) -> None:
        what = line.oid if line.candid is None else f"{line.oid}:{line.candid}"
        sys.stderr.write(
            f"[{position}/{total}] {line.kind} {what} {line.status} {line.requests} req "
            f"{line.seconds:.1f} s\n"
        )

    return report


def fetch_by_oid_command(args: argparse.Namespace) -> int:
    """``fetch-alerts --oids FILE --out DIR``."""
    from astro_simclr.alerts.alerce import make_client
    from astro_simclr.alerts.by_oid import fetch_by_oid, read_items

    _check_ranges(args)
    if args.out is None:
        raise ConfigError("--oids needs --out")
    for flag in ("dataset", "split", "from_snapshot", "limit"):
        if getattr(args, flag) is not None:
            raise ConfigError(f"--{flag.replace('_', '-')} does not apply with --oids")
    items = read_items(args.oids)
    with stop_on_signals() as stopping:
        run = fetch_by_oid(
            items,
            args.out,
            client=make_client(args.rate),
            resume=args.resume,
            batch_size=args.batch_size,
            cooldown=args.cooldown,
            progress=_progress(args.quiet),
            stop=lambda: bool(stopping),
        )
    counts = ", ".join(f"{k} {v}" for k, v in run.counts.items() if v)
    sys.stdout.write(
        f"attempted {run.attempted} items ({run.skipped} already done): {counts or 'none'}; "
        f"{run.requests} requests, {run.seconds:.1f} s, {run.cooldowns} cooldowns\n"
    )
    return 128 + stopping[0] if stopping else 0


def _r5_sources(args: argparse.Namespace) -> int:
    from astro_simclr.alerts.transport import Fetcher
    from astro_simclr.r5.sources import download_bts, download_zenodo

    if not args.rate > 0:
        raise ConfigError(f"--rate is out of range: {args.rate}")
    if args.only in (None, "bts"):
        fetcher = Fetcher(rate=args.rate)
        path = download_bts(args.out, get=lambda url, n: fetcher.get(url, max_bytes=n))
        sys.stdout.write(f"wrote {path}\n")
    if args.only in (None, "zenodo"):
        path = download_zenodo(args.out, keep_archive=args.keep_archive)
        sys.stdout.write(f"wrote {path}\n")
    return 0


def _log(text: str) -> None:
    sys.stderr.write(text + "\n")


def _build_r5(args: argparse.Namespace) -> int:
    from astro_simclr.alerts.alerce import make_client
    from astro_simclr.r5.build import build_r5, check_expected

    _check_ranges(args)
    if args.seed < 0:
        raise ConfigError(f"--seed must be non-negative, got {args.seed}")
    if args.max_rounds < 1:
        raise ConfigError(f"--max-rounds must be at least 1, got {args.max_rounds}")
    if args.dry_run and args.expect is not None:
        raise ConfigError("--expect needs a build, not --dry-run")
    if not args.dry_run and args.out is None:
        raise ConfigError("build-r5 needs --out (or --dry-run)")
    offline = args.from_snapshot or args.dry_run
    with stop_on_signals() as stopping:
        result = build_r5(
            args.sources,
            args.cd21_raw,
            args.fetched,
            None if args.dry_run else args.out,
            seed=args.seed,
            client=None if offline else make_client(args.rate),
            max_rounds=args.max_rounds,
            batch_size=args.batch_size,
            cooldown=args.cooldown,
            progress=_progress(args.quiet),
            log=_log,
            stop=lambda: bool(stopping),
        )
    requests = sum(r.requests for r in result.fetched)
    if args.dry_run:
        kinds = {k: sum(1 for i in result.missing if i.kind == k) for k in ("detections", "avro")}
        sys.stdout.write(
            f"dry run: the next round fetches {kinds['detections']} detection lists and "
            f"{kinds['avro']} alerts (about {len(result.missing) / args.rate / 3600:.2f} h at "
            f"{args.rate:g} req/s, retries excluded)\n"
        )
        return 0
    if not result.complete:
        kinds = {k: sum(1 for i in result.missing if i.kind == k) for k in ("detections", "avro")}
        sys.stdout.write(
            f"R5 not complete after {result.rounds} rounds ({requests} requests): "
            f"{kinds['detections']} detection lists and {kinds['avro']} alerts missing "
            f"(about {len(result.missing) / args.rate / 3600:.2f} h at {args.rate:g} req/s "
            "for the next round)\n"
        )
        return 128 + stopping[0] if stopping else EXIT_FAIL
    assert result.record is not None  # noqa: S101 - a complete build has its record
    counts = result.record["counts"]
    by_class = ", ".join(f"{k} {v}" for k, v in counts["by_class"].items())
    sys.stdout.write(
        f"wrote {args.out}: {counts['objects']} objects ({by_class}), "
        f"{counts['first_stamps']} first stamps; {result.rounds} rounds, {requests} requests\n"
    )
    if args.expect is not None:
        differ = check_expected(result.record, args.expect)
        if differ:
            sys.stdout.write("differs from --expect: " + ", ".join(differ) + "\n")
            return EXIT_FAIL
        sys.stdout.write("every output matches --expect byte for byte\n")
    return 0


def add_fetch_by_oid_arguments(fetch: argparse.ArgumentParser) -> None:
    """The ``--oids`` option of ``fetch-alerts``."""
    fetch.add_argument(
        "--oids",
        type=Path,
        default=None,
        metavar="FILE",
        help="by-oid mode (task 3.5): fetch the detection list of each 'OID' line and the "
        "alert of each 'OID CANDID' line into --out (no --dataset/--split)",
    )


def _add_fetch_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rate", type=float, default=1.5, metavar="R", help="requests per second (default 1.5)"
    )
    parser.add_argument("--batch-size", type=int, default=100, metavar="N")
    parser.add_argument(
        "--cooldown",
        type=float,
        default=300.0,
        metavar="S",
        help="pause after 3 consecutive transient failures (default 300 s; 0 disables)",
    )
    parser.add_argument("--quiet", action="store_true", help="no per-item progress on stderr")


def add_commands(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``r5-sources`` and ``build-r5``."""
    sources = commands.add_parser(
        "r5-sources",
        help="snapshot R5's label sources: the Zenodo labelled set and the BTS SN table",
        description="Download the ALeRCE light-curve labelled set (Zenodo 4279623, CC BY "
        "4.0; the 1.12 GB archive is checked against Zenodo's size and MD5, the CSV read "
        "out of it without extracting anything) and the ZTF Bright Transient Survey SN "
        "table (public explorer CSV) into OUT, each with a JSON record (URL, UTC time, "
        "size, SHA-256, licence or terms, citation). Never commit them.",
    )
    sources.add_argument("--out", type=Path, required=True, metavar="DIR")
    sources.add_argument("--only", choices=["zenodo", "bts"], default=None)
    sources.add_argument("--keep-archive", action="store_true", help="keep the Zenodo .tar.gz")
    sources.add_argument("--rate", type=float, default=1.5, metavar="R")
    sources.set_defaults(handler=_r5_sources)

    build = commands.add_parser(
        "build-r5",
        help="build the R5 benchmark from the sources and the response snapshots (task 3.5)",
        description="Select R5 (RJ23's recipe on public sources: one seeded alert per "
        "object before JD_MAX = 2022-12-31), fetching what the snapshots lack into "
        '--fetched (resumable; rerun to continue), group it with CD21 by oid and 1.5" '
        "positions, draw the constrained 60/20/20 object splits, add R5 Test's first "
        "stamps, and write the raw-alerts layouts, the alert tables, the splits, the "
        "frozen index, the deviation, drop, era and split tables and build.json into --out. "
        "--from-snapshot: no network, refuse an incomplete snapshot. Exit 0 when written "
        "(and matching --expect), 1 when incomplete or different.",
    )
    build.add_argument("--sources", type=Path, required=True, metavar="DIR")
    build.add_argument(
        "--cd21-raw",
        type=Path,
        required=True,
        metavar="RAW_DIR",
        help="task 3.1's fetch-alerts store of CD21 (Train, Validation and Test)",
    )
    build.add_argument(
        "--fetched",
        type=Path,
        required=True,
        metavar="DIR",
        help="the by-oid store (created on the first online run)",
    )
    build.add_argument("--out", type=Path, default=None, metavar="DIR")
    build.add_argument("--seed", type=int, default=0, metavar="N")
    build.add_argument("--from-snapshot", action="store_true", help="offline rebuild")
    build.add_argument("--dry-run", action="store_true", help="one offline round, no output")
    build.add_argument("--expect", type=Path, default=None, metavar="BUILD_JSON")
    build.add_argument(
        "--max-rounds", type=int, default=20, metavar="N", help="fetch rounds (default 20)"
    )
    _add_fetch_options(build)
    build.set_defaults(handler=_build_r5)
