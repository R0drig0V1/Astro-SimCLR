"""A fake queued job for the worker's crash-injection tests (``_queue_support.FAKE_JOB``).

``fake_job.py --run-dir DIR --mode MODE``. Every start appends a line to ``DIR/starts``;
``DIR/last`` is its "checkpoint" (the last step reached), read back on a resume. Modes:

- ``ok``: write ``job_result.json`` with a peak of 1 MiB, exit 0;
- ``fail``: exit 3;
- ``busy``: exit 75 (``job.EXIT_BUSY``);
- ``sleep``: count steps into ``DIR/last`` every 20 ms forever; SIGTERM exits 143;
- ``resume``: the first start behaves as ``sleep``; any later one resumes from ``DIR/last``
  (recorded in ``DIR/resumed_from``) and finishes as ``ok``;
- ``stubborn``: like ``sleep`` but ignores SIGTERM (only SIGKILL stops it).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path


def _ok(run_dir: Path) -> int:
    result = {"format": "astro-simclr-job-result", "peak_memory_bytes": 2**20}
    (run_dir / "job_result.json").write_text(json.dumps(result))
    return 0


def _sleep(run_dir: Path, start: int) -> int:
    step = start
    while True:
        step += 1
        tmp = run_dir / "last.tmp"
        tmp.write_text(str(step))
        os.replace(tmp, run_dir / "last")  # atomic: the test never reads a half-written file
        time.sleep(0.02)


def main() -> int:
    """Run the fake job; return its exit code."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()
    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "starts").open("a") as starts:
        starts.write(f"{args.mode}\n")
    n_starts = len((run_dir / "starts").read_text().splitlines())
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if args.mode == "ok":
        return _ok(run_dir)
    if args.mode == "fail":
        return 3
    if args.mode == "busy":
        return 75
    if args.mode == "stubborn":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        return _sleep(run_dir, 0)
    if args.mode == "resume" and n_starts > 1:
        (run_dir / "resumed_from").write_text((run_dir / "last").read_text())
        return _ok(run_dir)
    return _sleep(run_dir, 0)


if __name__ == "__main__":
    sys.exit(main())
