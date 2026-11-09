"""A fake ``ssh`` for the transport tests (``_queue_support.FAKE_SSH``).

``fake_ssh.py [-o OPT ...] [--] HOST COMMAND...``: skips the options and the host, joins
the command words with spaces and runs the result with ``sh -c`` in ``$FAKE_SSH_HOME``
(the "remote" home), as sshd hands a command to the login shell. ``ASTRO_SIMCLR_HOME``
becomes ``$FAKE_SSH_ASTRO_HOME`` when set. If the file ``$FAKE_SSH_DROP`` exists, the
connection "drops": exit 255 with ssh's message, as a real dropped connection does.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    """Run the remote command of ``argv``; return its exit code."""
    args = list(argv)
    while args and args[0].startswith("-"):
        option = args.pop(0)
        if option == "--":
            break
        if option in ("-o", "-p", "-i", "-l"):
            args.pop(0)
    if not args:
        sys.stderr.write("fake ssh: no host\n")
        return 255
    args.pop(0)  # the host
    drop = os.environ.get("FAKE_SSH_DROP")
    if drop and Path(drop).exists():
        sys.stderr.write("ssh: connect to host runner port 22: Connection timed out\n")
        return 255
    env = dict(os.environ)
    if os.environ.get("FAKE_SSH_ASTRO_HOME"):
        env["ASTRO_SIMCLR_HOME"] = os.environ["FAKE_SSH_ASTRO_HOME"]
    cwd = os.environ.get("FAKE_SSH_HOME") or None
    # The point of the fake: the command line goes through a shell, as with sshd.
    return subprocess.run(["sh", "-c", " ".join(args)], cwd=cwd, env=env, check=False).returncode  # noqa: S603, S607


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
