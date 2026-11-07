"""The ``astro-simclr`` command line (``[project.scripts]``).

Phase 1 grows this one subcommand per task (``inspect-data``, ``pretrain``, ...). Each
subcommand registers itself in :func:`build_parser` and reads its settings through
:func:`add_settings_arguments` / :func:`settings_from_args`, so that every command accepts
the same ``--config FILE`` and ``--set key=value`` options. For now the only subcommand is
``show-config``, which prints the resolved settings.

:func:`main` takes ``argv`` explicitly so tests can call it in-process; it returns the exit
code instead of calling ``sys.exit``.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

import yaml

from astro_simclr.config import ConfigError, Settings, load_settings

__all__ = ["add_settings_arguments", "build_parser", "main", "settings_from_args"]

#: Exit code for invalid settings or arguments (argparse also uses 2 for usage errors).
EXIT_USAGE = 2


def add_settings_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--config`` and ``--set`` options to a subcommand parser."""
    parser.add_argument(
        "--config", type=Path, metavar="FILE", help="YAML settings file (see config.Settings)"
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one setting; repeatable; VALUE is parsed as YAML",
    )


def settings_from_args(args: argparse.Namespace) -> Settings:
    """Resolve the settings of a parsed command line (``--config``, then ``--set``).

    Raises:
        ConfigError: if the file or any override is invalid.
    """
    config: Path | None = args.config
    overrides: list[str] = args.overrides
    return load_settings(config, overrides)


def _show_config(args: argparse.Namespace) -> int:
    settings = settings_from_args(args)
    # Paths are printed as strings so the output is itself a valid settings file.
    plain = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in dataclasses.asdict(settings).items()
    }
    sys.stdout.write(yaml.safe_dump(plain, sort_keys=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser with every subcommand."""
    parser = argparse.ArgumentParser(
        prog="astro-simclr",
        description="SimCLR and supervised contrastive learning on ZTF stamps.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('astro-simclr')}"
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    show = commands.add_parser("show-config", help="print the resolved settings as YAML")
    add_settings_arguments(show)
    show.set_defaults(handler=_show_config)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI on ``argv`` (default: ``sys.argv[1:]``) and return the exit code.

    Without a subcommand it prints the help and returns 2. Invalid settings print one
    ``error:`` line on stderr and return 2; argparse usage errors exit with 2 as well.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        return EXIT_USAGE
    try:
        code: int = args.handler(args)
    except ConfigError as err:
        sys.stderr.write(f"astro-simclr: error: {err}\n")
        return EXIT_USAGE
    return code


if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    sys.exit(main())
