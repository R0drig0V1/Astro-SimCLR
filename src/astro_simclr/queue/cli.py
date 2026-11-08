"""The ``spec`` and ``queue`` subcommands of ``astro-simclr`` (task A3; ``docs/queue.md``).

``spec`` (the per-commit entry point, :mod:`~astro_simclr.queue.job`):

- ``spec check FILE [--json]``: validate a spec; print its id and cells;
- ``spec expand FILE --data-dir D --repo-root R``: the cells with their run ids, as JSON;
- ``spec run-cell FILE --cell KEY --attempt N --run-dir DIR --data-dir D --repo-root R``:
  run one cell (the argv every queued job runs). Its exit code is the job's.

``queue`` (the runner and the laptop):

- runner: ``queue init --repo R --data-dir D [--runs-root P] [--uv PATH]``, ``queue
  enqueue --sha SHA FILE|- [--new-attempt]``, ``queue paths``, ``queue worker``,
  ``queue status``, ``queue cancel``, ``queue service`` (the systemd unit) and ``queue tmux
  --session NAME`` (a worker in an exactly named tmux session);
- laptop: ``queue submit FILE --host H``, ``queue status --host H``, ``queue sync --host H
  --dest DIR``.

Every runner command reads ``$ASTRO_SIMCLR_HOME/runner.yaml`` (``queue init`` writes it).
Errors print one ``astro-simclr: error:`` line and exit with 2 (refused) or 1.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from astro_simclr.queue.admission import Slot
    from astro_simclr.queue.runner import RunnerConfig
    from astro_simclr.queue.store import QueueStore

__all__ = ["EXIT_ERROR", "add_queue_commands", "add_spec_commands"]

#: Exit code of a refused or failed queue operation.
EXIT_ERROR: Final = 1


def _print_json(data: object) -> None:
    sys.stdout.write(json.dumps(data, indent=2) + "\n")


def _error(message: str, code: int = EXIT_ERROR) -> int:
    sys.stderr.write(f"astro-simclr: error: {message}\n")
    return code


type _Handler = Callable[[argparse.Namespace], int]


def _reported(handler: _Handler) -> _Handler:
    """Turn the queue's own errors into one ``error:`` line and exit 1 (no traceback).

    ``ConfigError`` (a ``SpecError``) and ``DataError`` are left to ``cli.main`` (exit 2
    and 1).
    """

    @wraps(handler)
    def run(args: argparse.Namespace) -> int:
        from astro_simclr.queue.location import LocationError
        from astro_simclr.queue.remote import SubmitError
        from astro_simclr.queue.runner import RunnerError
        from astro_simclr.queue.store import QueueError
        from astro_simclr.runs.manifest import ManifestError

        try:
            return handler(args)
        except (LocationError, SubmitError, RunnerError, QueueError, ManifestError) as err:
            return _error(str(err))
        except OSError as err:
            return _error(f"{type(err).__name__}: {err}")

    return run


# --- spec ------------------------------------------------------------------------------------


@_reported
def _spec_check(args: argparse.Namespace) -> int:
    from astro_simclr.runs.spec import load_spec

    spec = load_spec(args.file)
    cells = [cell.key for cell in spec.cells()]
    if args.json:
        _print_json({"spec_id": spec.spec_id, "name": spec.name, "cells": cells})
    else:
        sys.stdout.write(f"{spec.name}: spec id {spec.spec_id}, {len(cells)} cells\n")
        sys.stdout.writelines(f"  {cell}\n" for cell in cells)
    return 0


@_reported
def _spec_expand(args: argparse.Namespace) -> int:
    from astro_simclr.queue.job import expand

    _print_json(expand(args.file, data_dir=args.data_dir, repo_root=args.repo_root))
    return 0


@_reported
def _spec_run_cell(args: argparse.Namespace) -> int:
    from astro_simclr.queue.job import run_cell

    return run_cell(
        args.file,
        args.cell,
        attempt=args.attempt,
        run_dir=args.run_dir,
        data_dir=args.data_dir,
        repo_root=args.repo_root,
    )


def add_spec_commands(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``spec check|expand|run-cell``."""
    spec = commands.add_parser("spec", help="experiment specs (docs/queue.md)")
    sub = spec.add_subparsers(dest="spec_command", metavar="SPEC_COMMAND", required=True)
    check = sub.add_parser("check", help="validate a spec; print its id and cells")
    check.add_argument("file", type=Path, metavar="FILE")
    check.add_argument("--json", action="store_true")
    check.set_defaults(handler=_spec_check)
    for name, handler, help_text in (
        ("expand", _spec_expand, "the cells of a spec with their run ids (JSON)"),
        ("run-cell", _spec_run_cell, "run one cell of a spec (what a queued job runs)"),
    ):
        parser = sub.add_parser(name, help=help_text)
        parser.add_argument("file", type=Path, metavar="FILE")
        parser.add_argument("--data-dir", type=Path, required=True, metavar="DIR")
        parser.add_argument(
            "--repo-root", type=Path, required=True, metavar="DIR", help="the commit's checkout"
        )
        if name == "run-cell":
            parser.add_argument("--cell", required=True, metavar="KEY")
            parser.add_argument("--attempt", type=int, required=True, metavar="N")
            parser.add_argument("--run-dir", type=Path, required=True, metavar="DIR")
        parser.set_defaults(handler=handler)


# --- queue: runner side ----------------------------------------------------------------------


def _open_store() -> tuple[RunnerConfig, QueueStore]:
    from astro_simclr.queue.runner import RunnerConfig
    from astro_simclr.queue.store import QueueStore

    config = RunnerConfig.load()
    return config, QueueStore.open(config.queue_file)


@_reported
def _queue_init(args: argparse.Namespace) -> int:
    from astro_simclr.queue.location import check_queue_location
    from astro_simclr.queue.runner import RunnerConfig, default_home
    from astro_simclr.queue.store import QueueStore

    home = default_home()
    check_queue_location(home)
    uv = args.uv or shutil.which("uv") or "uv"
    config = RunnerConfig(
        home=home,
        repo=args.repo.expanduser().resolve(),
        data_dir=args.data_dir.expanduser().resolve(),
        runs_root=(args.runs_root or home / "runs").expanduser().resolve(),
        uv=uv,
    )
    path = config.save()
    QueueStore.open(config.queue_file).close()
    sys.stdout.write(f"wrote {path}; queue at {config.queue_file}\n")
    return 0


@_reported
def _queue_enqueue(args: argparse.Namespace) -> int:
    from astro_simclr.queue.runner import enqueue

    text = sys.stdin.buffer.read() if str(args.file) == "-" else Path(args.file).read_bytes()
    config, store = _open_store()
    try:
        result = enqueue(
            config, store, text, sha=args.sha, now=time.time(), new_attempt=args.new_attempt
        )
    finally:
        store.close()
    _print_json(result.to_plain())
    return 0


@_reported
def _queue_paths(_args: argparse.Namespace) -> int:
    from astro_simclr.queue.runner import RunnerConfig

    config = RunnerConfig.load()
    _print_json({"runs_root": str(config.runs_root), "specs_dir": str(config.specs_dir)})
    return 0


def _slots(args: argparse.Namespace) -> list[Slot]:
    from astro_simclr.queue.admission import Slot

    if not args.gpu:
        return [Slot("cpu", None, None, 1)]
    if args.slots_per_gpu > 1 and args.gpu_memory_mib is None:
        raise ValueError("--slots-per-gpu above 1 needs --gpu-memory-mib (admission by memory)")
    capacity = None if args.gpu_memory_mib is None else args.gpu_memory_mib * 2**20
    return [Slot(f"cuda:{gpu}", gpu, capacity, args.slots_per_gpu) for gpu in args.gpu]


@_reported
def _queue_worker(args: argparse.Namespace) -> int:
    from astro_simclr.queue.worker import Worker
    from astro_simclr.runs.identity import default_machine_class

    try:
        slots = _slots(args)
    except ValueError as err:
        return _error(str(err), 2)
    if args.machine_class is None and args.gpu:
        return _error("a GPU worker needs --machine-class (e.g. cuda-nvidia-geforce-gtx-1050)", 2)
    machine_class = args.machine_class or default_machine_class("cpu", None, platform.machine())
    config, store = _open_store()
    worker = Worker(
        store,
        machine_class,
        slots,
        config.log_dir,
        lease_seconds=args.lease_seconds,
        grace_seconds=args.grace_seconds,
    )
    sys.stderr.write(
        f"astro-simclr: worker {worker.worker_id}: {machine_class} on "
        f"{', '.join(slot.name for slot in slots)}\n"
    )
    try:
        worker.run(poll_seconds=args.poll_seconds, idle_exit=args.idle_exit)
    finally:
        store.close()
    return 0


def _status_line(job: dict[str, object]) -> str:
    extra = f" ({job['error']})" if job.get("error") else ""
    return (
        f"{job['job_id']:>5}  {job['status']:<9}  {job['cell']:<28}  a{job['attempt']}"
        f"  resumes {job['resumes']}  {job['run_dir']}{extra}\n"
    )


@_reported
def _queue_status(args: argparse.Namespace) -> int:
    if args.host is not None:
        from astro_simclr.queue.remote import SshTransport, remote_json

        argv = [args.remote_cli, "queue", "status", "--json"]
        if args.spec is not None:
            argv += ["--spec", args.spec]
        data = remote_json(SshTransport(args.host), argv)
        jobs = data if isinstance(data, list) else []
    else:
        _config, store = _open_store()
        try:
            jobs = [job.to_plain() for job in store.jobs(spec_id=args.spec)]
        finally:
            store.close()
    if args.json:
        _print_json(jobs)
    else:
        sys.stdout.writelines(_status_line(job) for job in jobs if isinstance(job, dict))
    return 0


@_reported
def _queue_cancel(args: argparse.Namespace) -> int:
    from astro_simclr.queue.worker import cancel_manifest

    _config, store = _open_store()
    try:
        ids = list(args.job_ids)
        if args.spec is not None:
            ids += [job.job_id for job in store.jobs(spec_id=args.spec)]
        result = store.cancel(ids, now=time.time())
        for job_id, outcome in result.items():
            job = store.job(job_id)
            if outcome == "cancelled" and job is not None:
                cancel_manifest(job.run_dir)  # a requeued attempt may have one
    finally:
        store.close()
    sys.stdout.writelines(f"{job_id}: {outcome}\n" for job_id, outcome in result.items())
    return 0


def _worker_argv(args: argparse.Namespace) -> list[str]:
    argv = [sys.executable, "-m", "astro_simclr.cli", "queue", "worker"]
    for gpu in args.gpu:
        argv += ["--gpu", str(gpu)]
    if args.machine_class is not None:
        argv += ["--machine-class", args.machine_class]
    return argv


@_reported
def _queue_service(args: argparse.Namespace) -> int:
    from astro_simclr.queue.runner import RunnerConfig
    from astro_simclr.queue.service import service_commands, systemd_unit, unit_path

    config = RunnerConfig.load()
    unit = systemd_unit(_worker_argv(args), home=config.home, working_dir=config.home)
    if not args.write:
        sys.stdout.write(unit)
        return 0
    xdg = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    path = unit_path(xdg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(unit, encoding="utf-8")
    sys.stdout.write(f"wrote {path}; now run:\n")
    sys.stdout.writelines(f"  {' '.join(argv)}\n" for argv in service_commands(getpass.getuser()))
    return 0


@_reported
def _queue_tmux(args: argparse.Namespace) -> int:
    from astro_simclr.queue.runner import RunnerConfig
    from astro_simclr.queue.service import tmux_has_session_argv, tmux_start_argv

    config = RunnerConfig.load()
    # Fixed argvs, no shell; `tmux` from PATH.
    if subprocess.run(tmux_has_session_argv(args.session), check=False).returncode == 0:  # noqa: S603
        sys.stdout.write(f"tmux session {args.session} already runs\n")
        return 0
    start = tmux_start_argv(args.session, _worker_argv(args), home=config.home)
    return subprocess.run(start, check=False).returncode  # noqa: S603


# --- queue: laptop side ----------------------------------------------------------------------


@_reported
def _queue_submit(args: argparse.Namespace) -> int:
    from astro_simclr.queue.remote import SshTransport, submit

    report = submit(
        args.file,
        SshTransport(args.host),
        repo_root=args.repo_root,
        remote_cli=(args.remote_cli,),
        new_attempt=args.new_attempt,
    )
    _print_json(report)
    return 0


@_reported
def _queue_sync(args: argparse.Namespace) -> int:
    from astro_simclr.queue.remote import SshTransport, sync
    from astro_simclr.queue.runner import default_home

    report = sync(
        SshTransport(args.host),
        args.dest,
        specs_dest=default_home() / "specs",
        remote_cli=(args.remote_cli,),
        reindex=not args.no_reindex,
    )
    sys.stdout.write(
        f"synced {report.runs} run directories into {args.dest} "
        f"({len(report.checkpoints)} retained checkpoints); reindexed: {report.reindexed}\n"
    )
    return 0


def _add_remote(parser: argparse.ArgumentParser, *, required: bool) -> None:
    from astro_simclr.queue.remote import DEFAULT_REMOTE_CLI

    parser.add_argument("--host", required=required, metavar="HOST", help="ssh host (user@host)")
    parser.add_argument(
        "--remote-cli",
        default=DEFAULT_REMOTE_CLI,
        metavar="PATH",
        help=f"the runner's astro-simclr, relative to its home (default {DEFAULT_REMOTE_CLI})",
    )


def _add_worker_target(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gpu", type=int, action="append", default=[], metavar="N", help="a GPU index (repeat)"
    )
    parser.add_argument(
        "--machine-class", metavar="SLUG", help="the jobs' machine class (default cpu-<isa>)"
    )


def add_queue_commands(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``queue init|enqueue|paths|worker|status|cancel|service|tmux|submit|sync``."""
    from astro_simclr.queue.worker import (
        DEFAULT_GRACE_SECONDS,
        DEFAULT_LEASE_SECONDS,
        DEFAULT_POLL_SECONDS,
    )

    queue = commands.add_parser("queue", help="the durable run queue (docs/queue.md)")
    sub = queue.add_subparsers(dest="queue_command", metavar="QUEUE_COMMAND", required=True)

    init = sub.add_parser("init", help="runner: write $ASTRO_SIMCLR_HOME/runner.yaml")
    init.add_argument("--repo", type=Path, required=True, metavar="DIR", help="the git clone")
    init.add_argument("--data-dir", type=Path, required=True, metavar="DIR")
    init.add_argument("--runs-root", type=Path, metavar="DIR", help="default HOME/runs")
    init.add_argument("--uv", metavar="PATH", help="the uv binary (default: uv on PATH)")
    init.set_defaults(handler=_queue_init)

    enq = sub.add_parser("enqueue", help="runner: accept a spec at a commit (stdin with -)")
    enq.add_argument("file", metavar="FILE|-")
    enq.add_argument("--sha", required=True, metavar="SHA")
    enq.add_argument("--new-attempt", action="store_true", help="rerun cells already done")
    enq.set_defaults(handler=_queue_enqueue)

    paths = sub.add_parser("paths", help="runner: the runs root and specs directory (JSON)")
    paths.set_defaults(handler=_queue_paths)

    worker = sub.add_parser("worker", help="runner: run queued jobs")
    _add_worker_target(worker)
    worker.add_argument("--slots-per-gpu", type=int, default=1, metavar="N")
    worker.add_argument("--gpu-memory-mib", type=int, metavar="MIB")
    worker.add_argument("--lease-seconds", type=float, default=DEFAULT_LEASE_SECONDS)
    worker.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    worker.add_argument("--grace-seconds", type=float, default=DEFAULT_GRACE_SECONDS)
    worker.add_argument("--idle-exit", action="store_true", help="exit when nothing is left")
    worker.set_defaults(handler=_queue_worker)

    status = sub.add_parser("status", help="the jobs (on the runner, or on --host)")
    status.add_argument("--spec", metavar="SPEC_ID", help="one spec (an id or a prefix)")
    status.add_argument("--json", action="store_true")
    _add_remote(status, required=False)
    status.set_defaults(handler=_queue_status)

    cancel = sub.add_parser("cancel", help="runner: cancel jobs (running ones via the worker)")
    cancel.add_argument("job_ids", type=int, nargs="*", metavar="JOB")
    cancel.add_argument("--spec", metavar="SPEC_ID")
    cancel.set_defaults(handler=_queue_cancel)

    service = sub.add_parser("service", help="runner: the worker's systemd user unit")
    _add_worker_target(service)
    service.add_argument("--write", action="store_true", help="install it under ~/.config")
    service.set_defaults(handler=_queue_service)

    tmux = sub.add_parser("tmux", help="runner: start a worker in an exactly named tmux session")
    tmux.add_argument("--session", default="astro-simclr-worker", metavar="NAME")
    _add_worker_target(tmux)
    tmux.set_defaults(handler=_queue_tmux)

    submit = sub.add_parser("submit", help="laptop: submit a spec at HEAD to a runner")
    submit.add_argument("file", type=Path, metavar="FILE")
    _add_remote(submit, required=True)
    submit.add_argument("--repo-root", type=Path, default=Path(), metavar="DIR")
    submit.add_argument("--new-attempt", action="store_true")
    submit.set_defaults(handler=_queue_submit)

    sync = sub.add_parser("sync", help="laptop: rsync the run directories, then runs reindex")
    _add_remote(sync, required=True)
    sync.add_argument("--dest", type=Path, required=True, metavar="DIR")
    sync.add_argument("--no-reindex", action="store_true")
    sync.set_defaults(handler=_queue_sync)
