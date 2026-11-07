"""The ``astro-simclr`` command line (``[project.scripts]``).

Phase 1 grows this one subcommand per task (``inspect-data``, ``pretrain``, ...). Each
subcommand registers itself in :func:`build_parser` and reads its settings through
:func:`add_settings_arguments` / :func:`settings_from_args`, so that every command accepts
the same ``--config FILE`` and ``--set key=value`` options. Subcommands: ``show-config``
(the resolved settings), and the data commands of task 1.2: ``inspect-data``,
``make-synthetic`` and ``prepare-data`` (see ``docs/data.md``).

The data modules are imported inside the handlers, so ``show-config`` and ``--help`` stay
fast. A :class:`~astro_simclr.data.DataError` (bad, truncated or missing data) prints one
``astro-simclr: error: ...`` line and exits with 1.

:func:`main` takes ``argv`` explicitly so tests can call it in-process; it returns the exit
code instead of calling ``sys.exit``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

import yaml

from astro_simclr.config import ConfigError, Settings, load_settings

__all__ = ["add_settings_arguments", "build_parser", "main", "settings_from_args"]

#: Exit code for invalid settings or arguments (argparse also uses 2 for usage errors).
EXIT_USAGE = 2
#: Exit code for unusable data: missing, truncated, corrupt or of the wrong schema.
EXIT_DATA = 1


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


def _inspect_data(args: argparse.Namespace) -> int:
    from astro_simclr.data.inspection import format_summary, summarize

    settings_from_args(args)  # validates --config/--set, which inspect-data does not use
    path: Path = args.path
    summary = summarize(path)
    if args.json:
        sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    else:
        sys.stdout.write(format_summary(summary))
    return 0


def _make_synthetic(args: argparse.Namespace) -> int:
    from astro_simclr.data.synthetic import make_synthetic

    settings = settings_from_args(args)
    seed: int = settings.seed if args.seed is None else args.seed
    out: Path = args.out
    try:
        paths = make_synthetic(out, args.n_per_split, seed)
    except ValueError as err:
        raise ConfigError(str(err)) from None
    for path in paths:
        sys.stdout.write(f"wrote {path}\n")
    return 0


def _prepare_data(args: argparse.Namespace) -> int:
    from astro_simclr.data.prepared import prepare

    settings = settings_from_args(args)
    out: Path = settings.data_dir if args.out is None else args.out
    source: Path = args.pickle
    dest = prepare(source, out, overwrite=args.overwrite)
    sys.stdout.write(f"wrote {dest}\n")
    return 0


def _add_data_commands(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    inspect = commands.add_parser(
        "inspect-data",
        help="describe and check a stamp pickle or a prepared directory",
        description="Print keys, dtypes, shapes, counts per class, size and SHA-256 of a "
        "stamp pickle or a prepared directory, and check it against its schema. Exits "
        "with 1 if the data is truncated, corrupt or does not match the schema.",
    )
    inspect.add_argument("path", type=Path, metavar="PATH", help="a .pkl or a prepared dir")
    inspect.add_argument("--json", action="store_true", help="print the summary as JSON")
    add_settings_arguments(inspect)
    inspect.set_defaults(handler=_inspect_data)

    synthetic = commands.add_parser(
        "make-synthetic",
        help="write the four synthetic stand-ins of the ZTF pickles",
        description="Write synthetic versions of the three classifier pickles and the "
        "pretraining pickle, with the real schemas and a strong class pattern. The same "
        "seed gives byte-identical files.",
    )
    synthetic.add_argument("--out", type=Path, required=True, metavar="DIR")
    synthetic.add_argument(
        "--n-per-split",
        type=int,
        default=50,
        metavar="N",
        help="stamps per split of every file (default: 50; >= 5)",
    )
    synthetic.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="S",
        help="random seed (default: the seed setting)",
    )
    add_settings_arguments(synthetic)
    synthetic.set_defaults(handler=_make_synthetic)

    prepare = commands.add_parser(
        "prepare-data",
        help="convert a stamp pickle into the prepared .npy layout",
        description="Unpickle a trusted stamp pickle (it needs about its size in RAM), "
        "check it, and write OUT/<pickle stem>/ with uint8 images, labels, features and "
        "meta.json. Training reads only this layout.",
    )
    prepare.add_argument("pickle", type=Path, metavar="PKL")
    prepare.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="DIR",
        help="parent directory (default: the data_dir setting)",
    )
    prepare.add_argument(
        "--overwrite", action="store_true", help="replace an existing prepared directory"
    )
    add_settings_arguments(prepare)
    prepare.set_defaults(handler=_prepare_data)


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
    _add_data_commands(commands)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI on ``argv`` (default: ``sys.argv[1:]``) and return the exit code.

    Without a subcommand it prints the help and returns 2. Invalid settings print one
    ``error:`` line on stderr and return 2; argparse usage errors exit with 2 as well.
    Unusable data (a ``DataError``) prints one ``error:`` line and returns 1.
    """
    from astro_simclr.data.schema import DataError

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
    except DataError as err:
        sys.stderr.write(f"astro-simclr: error: {err}\n")
        return EXIT_DATA
    return code


if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    sys.exit(main())
