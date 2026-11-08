"""``astro-simclr runs reindex | ls | show | compare``: the registry's command line.

Every subcommand takes ``--registry FILE`` (default ``$ASTRO_SIMCLR_HOME/registry.sqlite``;
refused inside a git worktree). Exit codes: 0 on success, 1 when the registry cannot be
used or a comparison does not resolve (one ``astro-simclr: error:`` message, listing every
problem), 2 on a usage error.

- ``runs reindex [ROOT ...] [--rebuild]``: scan the roots (default: every remembered
  root) and ``$ASTRO_SIMCLR_HOME/specs``. Prints one line per root and per skipped
  directory; exit 0 when the scan ends, even with skipped directories. This argv is
  stable: the queue's ``sync`` (task A3) runs ``astro-simclr runs reindex ROOT``.
- ``runs ls [--command C] [--config SHA] [--seed N] [--status S] [--legacy |
  --no-legacy] [--json]``: one line per run directory.
- ``runs show KEY [--json]``: one attempt in full, with its artefacts and its other
  copies. ``KEY`` is an attempt key (``runs ls``), a prefix of it or of the run id (at
  least 6 characters), or a run directory.
- ``runs compare SPEC [--metric M ...] [--use KEY ...] [--attempts error|latest]
  [--allow-mixed machines|engines|numerics ...] [--json]``: resolve an experiment spec
  (:mod:`~astro_simclr.registry.compare`) and report its cells and the mean ± std per arm.
  ``SPEC`` is a spec file, or the id (or a prefix of at least 6 characters) or the name
  of a spec indexed from ``$ASTRO_SIMCLR_HOME/specs``. A report, never a decision: the
  protocol's decisions are ``compare-variants``'.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final, cast

from astro_simclr.registry.compare import (
    AttemptPolicy,
    Mixable,
    default_metrics,
    format_report,
    resolution_to_plain,
    resolve,
)
from astro_simclr.registry.home import RegistryError, registry_path, specs_dir
from astro_simclr.registry.index import reindex
from astro_simclr.registry.readers import RunRecord
from astro_simclr.registry.spec_view import (
    ComparisonPlan,
    SpecUnavailableError,
    load_spec_file,
    plan_from_spec,
)
from astro_simclr.registry.store import load_records, open_registry

__all__ = ["HANDLERS", "record_to_plain", "run"]

_EXIT_ERROR: Final = 1
_MIN_PREFIX: Final = 6


def _connect(args: argparse.Namespace) -> sqlite3.Connection:
    return open_registry(registry_path(args.registry))


def _reindex(args: argparse.Namespace) -> int:
    conn = _connect(args)
    try:
        report = reindex(conn, args.roots, specs=specs_dir(), rebuild=args.rebuild)
    finally:
        conn.close()
    out = sys.stdout
    for root, count in report.indexed.items():
        out.write(f"{root}: {count} runs\n")
    for root in report.missing_roots:
        sys.stderr.write(f"astro-simclr: warning: remembered root {root} is missing; kept\n")
    for skipped in report.skipped:
        sys.stderr.write(f"astro-simclr: skipped {skipped.path}: {skipped.reason}\n")
    out.write(f"specs: {report.specs}\n")
    return 0


def _split_text(record: RunRecord) -> str:
    return "-" if record.split is None else str(record.split)


def record_to_plain(record: RunRecord) -> dict[str, object]:
    """``record`` as JSON values (``runs ls --json``, ``runs show --json``)."""
    return {
        "key": record.key,
        "run_dir": str(record.run_dir),
        "source": record.source,
        "legacy": record.legacy,
        "config_format": record.config_format,
        "run_id": record.run_id,
        "attempt": record.attempt,
        "command": record.command,
        "config_sha256": record.config_sha256,
        "seed": record.seed,
        "split": record.split,
        "data_sha256": record.data_sha256,
        "splits_sha256": record.splits_sha256,
        "pretrained_sha256": record.pretrained_sha256,
        "data_source_sha256": record.data_source_sha256,
        "machine_class": record.machine_class,
        "engine": record.engine,
        "numerics_epoch": record.numerics_epoch,
        "git": {"sha": record.git_sha, "dirty": record.git_dirty},
        "uv_lock_sha256": record.uv_lock_sha256,
        "status": record.status,
        "timestamps": {
            "created_at": record.created_at,
            "started_at": record.started_at,
            "ended_at": record.ended_at,
        },
        "preset": record.preset,
        "metrics_sha256": record.metrics_sha256,
        "machine": None if record.machine is None else dict(record.machine),
        "metrics": dict(record.metrics),
        "config": dict(record.config),
    }


def _records(args: argparse.Namespace) -> list[RunRecord]:
    conn = _connect(args)
    try:
        return load_records(conn)
    finally:
        conn.close()


def _ls(args: argparse.Namespace) -> int:
    records = [
        r
        for r in _records(args)
        if (args.command_filter is None or r.command == args.command_filter)
        and (args.config is None or r.config_sha256.startswith(args.config))
        and (args.seed is None or r.seed == args.seed)
        and (args.status is None or r.status == args.status)
        and (args.legacy is None or r.legacy == args.legacy)
    ]
    records.sort(key=lambda r: (r.command, r.config_sha256, r.seed, _split_text(r), r.key))
    if args.json:
        json.dump([record_to_plain(r) for r in records], sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    rows = [["key", "command", "config", "seed", "split", "status", "machine", "engine", "run_dir"]]
    for r in records:
        rows.append(
            [
                r.key,
                r.command,
                r.config_sha256[:12],
                str(r.seed),
                _split_text(r),
                r.status,
                r.machine_class or "-",
                r.engine,
                str(r.run_dir),
            ]
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        line = "  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True))
        sys.stdout.write(line.rstrip() + "\n")
    sys.stdout.write(f"{len(records)} runs\n")
    return 0


def _find(records: Sequence[RunRecord], token: str) -> list[RunRecord]:
    path = Path(token).expanduser()
    if path.is_dir():
        resolved = path.resolve()
        return [r for r in records if r.run_dir == resolved]
    exact = [r for r in records if r.key == token]
    if exact or len(token) < _MIN_PREFIX:
        return exact
    return [
        r
        for r in records
        if r.key.startswith(token) or (r.run_id is not None and r.run_id.startswith(token))
    ]


def _artefact_rows(args: argparse.Namespace, run_dir: Path) -> list[dict[str, object]]:
    conn = _connect(args)
    try:
        rows = conn.execute(
            "SELECT path, kind, size, sha256 FROM artefacts WHERE run_dir = ? ORDER BY path",
            (str(run_dir),),
        ).fetchall()
    finally:
        conn.close()
    return [{"path": p, "kind": k, "size": s, "sha256": h} for p, k, s, h in rows]


def _show(args: argparse.Namespace) -> int:
    records = _records(args)
    hits = _find(records, args.key)
    keys = sorted({r.key for r in hits})
    if not hits:
        raise RegistryError(f"no indexed run matches {args.key!r} (run `runs reindex`?)")
    if len(keys) > 1:
        raise RegistryError(f"{args.key!r} is ambiguous: {', '.join(keys)}")
    record = min(hits, key=lambda r: str(r.run_dir))
    copies = [r for r in records if r.key == record.key and r.run_dir != record.run_dir]
    plain = record_to_plain(record)
    plain["artefacts"] = _artefact_rows(args, record.run_dir)
    plain["copies"] = [str(copy.run_dir) for copy in copies]
    if args.json:
        json.dump(plain, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    out = sys.stdout
    for name in (
        "key",
        "run_dir",
        "source",
        "legacy",
        "config_format",
        "run_id",
        "attempt",
        "command",
        "config_sha256",
        "seed",
        "split",
        "status",
        "preset",
        "engine",
        "machine_class",
        "numerics_epoch",
        "data_sha256",
        "data_source_sha256",
        "splits_sha256",
        "pretrained_sha256",
        "uv_lock_sha256",
        "metrics_sha256",
    ):
        out.write(f"{name}: {plain[name]}\n")
    out.write(f"git: {record.git_sha} (dirty: {record.git_dirty})\n")
    out.write(f"timestamps: {json.dumps(plain['timestamps'])}\n")
    out.write(f"machine: {json.dumps(plain['machine'])}\n")
    out.write("metrics:\n")
    for metric, value in record.metrics.items():
        out.write(f"  {metric}: {value:.6g}\n")
    out.write("artefacts:\n")
    for row in cast(list[dict[str, object]], plain["artefacts"]):
        out.write(f"  {row['path']}  {row['kind']}  {row['size']} B  {str(row['sha256'])[:12]}\n")
    for copy in copies:
        out.write(f"copy: {copy.run_dir}\n")
    return 0


def _indexed_spec(args: argparse.Namespace, token: str) -> Path:
    conn = _connect(args)
    try:
        rows = conn.execute("SELECT spec_id, name, path FROM specs ORDER BY spec_id").fetchall()
    finally:
        conn.close()
    hits = [
        (spec_id, path)
        for spec_id, name, path in rows
        if spec_id == token
        or name == token
        or (len(token) >= _MIN_PREFIX and spec_id.startswith(token))
    ]
    if not hits:
        msg = (
            f"no spec file and no indexed spec matches {token!r} (indexed: {len(rows)}; "
            f"specs are read from {specs_dir()} by `runs reindex`)"
        )
        raise RegistryError(msg)
    if len({spec_id for spec_id, _ in hits}) > 1:
        ids = ", ".join(spec_id[:16] for spec_id, _ in hits)
        raise RegistryError(f"spec {token!r} is ambiguous: {ids}")
    return Path(hits[0][1])


def _plan(args: argparse.Namespace) -> ComparisonPlan:
    path = Path(args.spec).expanduser()
    if not path.is_file():
        path = _indexed_spec(args, args.spec)
    try:
        return plan_from_spec(load_spec_file(path))
    except SpecUnavailableError as err:
        raise RegistryError(str(err)) from None
    except (ValueError, OSError) as err:
        raise RegistryError(f"{path}: {err}") from None


def _compare(args: argparse.Namespace) -> int:
    plan = _plan(args)
    resolution = resolve(
        plan,
        _records(args),
        use=args.use,
        attempts=cast(AttemptPolicy, args.attempts),
        allow_mixed=cast(list[Mixable], args.allow_mixed),
    )
    metrics = args.metric or default_metrics(resolution.chosen.values())
    if args.json:
        json.dump(resolution_to_plain(resolution, metrics), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write("\n".join(format_report(resolution, metrics)) + "\n")
    return 0


#: The handler of each ``runs`` subcommand.
HANDLERS: Final = {"reindex": _reindex, "ls": _ls, "show": _show, "compare": _compare}


def run(args: argparse.Namespace) -> int:
    """Run the ``runs`` subcommand ``args.runs_command``; a :class:`RegistryError` is exit 1.

    The parser is :func:`astro_simclr.registry.parser.add_runs_command`.
    """
    try:
        code: int = HANDLERS[args.runs_command](args)
    except RegistryError as err:
        sys.stderr.write(f"astro-simclr: error: {err}\n")
        return _EXIT_ERROR
    return code
