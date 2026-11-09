"""Where the queue's SQLite file may live: a local disk, never ``/mnt/c`` or a synced folder.

SQLite's locking needs POSIX ``fcntl`` locks and ``fsync`` that tell the truth. On WSL2 the
Windows drives (``/mnt/c``, a ``9p``/``drvfs`` mount) do neither reliably, network file
systems do not either, and a sync client (Dropbox, OneDrive, iCloud, Syncthing, ...) may
copy a half-written database or its ``-wal`` file to another machine. So
:func:`check_queue_location` refuses:

- a path on a mount whose file system type is in :data:`UNSAFE_FSTYPES` (from
  ``/proc/mounts`` on Linux; elsewhere there is no mount table to read);
- a WSL Windows-drive path (``/mnt/<letter>/...``), even if the mount table is unreadable;
- a path with a component in :data:`SYNCED_FOLDERS`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Final

__all__ = [
    "SYNCED_FOLDERS",
    "UNSAFE_FSTYPES",
    "LocationError",
    "check_queue_location",
    "read_mounts",
]

#: File system types the queue refuses (WSL Windows drives, network and FUSE mounts).
UNSAFE_FSTYPES: Final = frozenset(
    {"9p", "drvfs", "v9fs", "virtiofs", "nfs", "nfs4", "cifs", "smb3", "smbfs", "afs"}
    | {"fuse.sshfs", "fuse.rclone", "fuse.gcsfuse", "fuse.s3fs", "davfs", "fuse.davfs"}
)
#: Folder names of sync clients; a path through one of them is refused.
SYNCED_FOLDERS: Final = frozenset(
    {"Dropbox", "OneDrive", "Google Drive", "GoogleDrive", "iCloud Drive", "iCloudDrive"}
    | {"Mobile Documents", "CloudStorage", "Syncthing", "Resilio Sync", "pCloud Drive", "MEGA"}
)
_WSL_DRIVE = re.compile(r"/mnt/[a-zA-Z](/|$)")


class LocationError(ValueError):
    """The queue's SQLite file would live on an unsafe file system (module docstring)."""


def read_mounts(path: Path = Path("/proc/mounts")) -> list[tuple[str, str]]:
    """The ``(mount point, file system type)`` pairs of ``/proc/mounts``; empty elsewhere.

    The mount points are unescaped (``\\040`` is a space, as the kernel writes it).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    mounts: list[tuple[str, str]] = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 3:
            point = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[1])
            mounts.append((point, fields[2]))
    return mounts


def _mount_of(path: PurePosixPath, mounts: Iterable[tuple[str, str]]) -> tuple[str, str] | None:
    best: tuple[str, str] | None = None
    for point, fstype in mounts:
        mount = PurePosixPath(point)
        if (path == mount or mount in path.parents) and (best is None or len(point) > len(best[0])):
            best = (point, fstype)
    return best


def check_queue_location(path: Path, *, mounts: Sequence[tuple[str, str]] | None = None) -> Path:
    """``path`` resolved, if a queue database may live there (module docstring).

    ``mounts`` defaults to :func:`read_mounts` (tests pass their own table).

    Raises:
        LocationError: if the path is on an unsafe file system, a WSL Windows drive or in a
            synced folder.
    """
    resolved = path.expanduser().resolve()
    posix = PurePosixPath(resolved.as_posix())
    if _WSL_DRIVE.match(str(posix)):
        msg = (
            f"{resolved} is on a Windows drive: SQLite locking is not reliable there. Put the "
            "queue on the Linux side (ext4), for example ~/.astro-simclr"
        )
        raise LocationError(msg)
    synced = sorted(set(posix.parts) & SYNCED_FOLDERS)
    if synced:
        msg = (
            f"{resolved} is inside a synced folder ({', '.join(synced)}): a sync client can "
            "copy a half-written database. Put the queue on a local, unsynced disk"
        )
        raise LocationError(msg)
    mount = _mount_of(posix, read_mounts() if mounts is None else mounts)
    if mount is not None and mount[1] in UNSAFE_FSTYPES:
        msg = (
            f"{resolved} is on a {mount[1]} file system (mounted at {mount[0]}): SQLite needs "
            "a local disk with POSIX locks (ext4 on WSL2)"
        )
        raise LocationError(msg)
    return resolved
