"""The ``build-corpus-index`` and ``build-corpus-fits`` subcommands (task 5.1; ADR 0019).

Registered by :func:`add_corpus_index_command` from :func:`astro_simclr.cli.build_parser`;
the handlers import the builders lazily, so ``--help`` stays fast.

``build-corpus-index ARCHIVE_DIR --out INDEX_FILE --cd21 LAYOUT [--exclude-oids FILE]...
[--exclude-candids FILE]... [--radius-arcsec R] [--duplicates first|random] [--seed N]
[--near-cosine C] [--near-max-diff D] [--no-near-duplicates] [--overwrite] [--json]``
applies the exclusion union to every complete night of ``ARCHIVE_DIR`` and writes the index
(:mod:`~astro_simclr.archive.corpus_index`); it prints the counts per rule (or the whole
record with ``--json``). Exit 0 when the index is written, 1 for unusable data, 2 for
unusable arguments.

``build-corpus-fits INDEX_FILE --out FITS_FILE [--archive ARCHIVE_DIR] [--chunk-rows N]
[--overwrite] [--json] [--quiet]`` computes the stamp fits of every row of the index (the
inputs of the C5 noise view) and writes them beside it, bound to the index's SHA-256
(:mod:`~astro_simclr.archive.corpus_fits`); ``simclr-pilot INDEX_FILE --noise-scale S``
reads them with ``--corpus-fits FITS_FILE``. Same exit codes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from astro_simclr.config import ConfigError

__all__ = ["add_corpus_index_command"]


def _build(args: argparse.Namespace) -> int:
    from astro_simclr.archive.corpus_index import (
        DEFAULT_NEAR,
        build_corpus_index,
        write_corpus_index,
    )
    from astro_simclr.archive.near_duplicates import NearDuplicateRules
    from astro_simclr.data.schema import DataError

    if args.out.exists() and not args.overwrite:
        raise ConfigError(f"{args.out} already exists; pass --overwrite to replace it")
    try:
        near = None
        if not args.no_near_duplicates:
            near = NearDuplicateRules(args.near_cosine, args.near_max_diff, DEFAULT_NEAR.candidates)

        def progress(line: str) -> None:
            if not args.quiet:
                sys.stderr.write(f"{line}\n")

        index = build_corpus_index(
            args.archive,
            layout=args.cd21,
            oid_files=args.exclude_oids or (),
            candid_files=args.exclude_candids or (),
            radius_arcsec=args.radius_arcsec,
            duplicates=args.duplicates,
            seed=args.seed,
            near=near,
            progress=progress,
        )
    except ValueError as err:
        if isinstance(err, DataError):
            raise
        raise ConfigError(str(err)) from None
    write_corpus_index(index, args.out, overwrite=args.overwrite)
    if args.json:
        sys.stdout.write(json.dumps(index.meta, indent=2, sort_keys=True) + "\n")
        return 0
    counts = index.meta["counts"]
    lines = [f"build-corpus-index: {counts['rows']} archive rows"]
    lines += [f"  excluded by {rule:<16} {n:>10}" for rule, n in counts["excluded"].items()]
    lines.append(f"  kept {counts['kept']:>26} -> {args.out}")
    lines.append(f"  exclusion SHA-256 {index.meta['exclusion_sha256']}")
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def _build_fits(args: argparse.Namespace) -> int:
    from astro_simclr.archive.corpus_fits import compute_corpus_fits, write_corpus_fits

    if args.out.exists() and not args.overwrite:
        raise ConfigError(f"{args.out} already exists; pass --overwrite to replace it")
    if args.chunk_rows < 1:
        raise ConfigError(f"--chunk-rows must be >= 1, got {args.chunk_rows}")

    def progress(line: str) -> None:
        if not args.quiet:
            sys.stderr.write(f"{line}\n")

    fits = compute_corpus_fits(
        args.index, archive=args.archive, chunk_rows=args.chunk_rows, progress=progress
    )
    write_corpus_fits(fits, args.out, overwrite=args.overwrite)
    if args.json:
        sys.stdout.write(json.dumps(fits.meta, indent=2, sort_keys=True) + "\n")
        return 0
    sys.stdout.write(
        f"build-corpus-fits: {len(fits)} stamps -> {args.out}\n"
        f"  index SHA-256 {fits.index_sha256}\n"
    )
    return 0


def add_corpus_index_command(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register ``build-corpus-index`` and ``build-corpus-fits``."""
    build = commands.add_parser(
        "build-corpus-index",
        help="the archive rows pretraining may read: the exclusion union and one alert per oid",
        description="Index every complete night of an archive corpus, dropping the rows "
        "whose candid or oid is listed, every row within --radius-arcsec of any CD21 alert "
        "(Train, Validation, Test), the objects with a near-duplicate stamp in CD21, and all "
        "but one alert per oid. Records the counts per rule and the SHA-256 of every "
        "exclusion input (docs/data.md, 'Corpus index'; ADR 0019).",
    )
    build.add_argument("archive", type=Path, metavar="ARCHIVE_DIR", help="ingest-archive --out")
    build.add_argument("--out", type=Path, required=True, metavar="INDEX_FILE")
    build.add_argument(
        "--cd21", type=Path, required=True, metavar="LAYOUT", help="the prepared CD21 layout"
    )
    build.add_argument(
        "--exclude-oids",
        type=Path,
        action="append",
        metavar="FILE",
        help="oids to drop, one per line (repeatable)",
    )
    build.add_argument(
        "--exclude-candids",
        type=Path,
        action="append",
        metavar="FILE",
        help="candids to drop, one per line (repeatable)",
    )
    build.add_argument("--radius-arcsec", type=float, default=1.5)
    build.add_argument(
        "--duplicates",
        choices=("first", "random"),
        default="first",
        help="which alert of an oid to keep: the earliest (default) or a seeded draw",
    )
    build.add_argument("--seed", type=int, default=0, help="the draw of --duplicates random")
    build.add_argument(
        "--near-cosine",
        type=float,
        default=0.99,
        help="stage-1 cosine of the pooled stamps (default 0.99)",
    )
    build.add_argument(
        "--near-max-diff",
        type=float,
        default=1e-3,
        help="stage-2 largest pixel difference of a near-duplicate (default 1e-3)",
    )
    build.add_argument(
        "--no-near-duplicates",
        action="store_true",
        help="skip the pixel check (recorded; for a layout without float32 stamps)",
    )
    build.add_argument("--overwrite", action="store_true")
    build.add_argument("--json", action="store_true", help="print the index's whole record")
    build.add_argument("--quiet", action="store_true", help="no progress lines on stderr")
    build.set_defaults(handler=_build)
    _add_fits_command(commands)


def _add_fits_command(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    fits = commands.add_parser(
        "build-corpus-fits",
        help="the stamp fits of a corpus index (the C5 noise view's inputs), beside it",
        description="Compute stamp_fits (diff ~ a*sci + b*tmpl + c, R^2, background sigmas) "
        "of every row of a corpus index, on the stamps normalised as the pilot reads them, "
        "and write them to FITS_FILE bound to the index's SHA-256. simclr-pilot INDEX_FILE "
        "--noise-scale S --corpus-fits FITS_FILE reads them (docs/data.md, 'Corpus index'; "
        "ADR 0019).",
    )
    fits.add_argument("index", type=Path, metavar="INDEX_FILE", help="build-corpus-index --out")
    fits.add_argument("--out", type=Path, required=True, metavar="FITS_FILE")
    fits.add_argument(
        "--archive",
        type=Path,
        metavar="ARCHIVE_DIR",
        help="the archive corpus of the index (default: the recorded one)",
    )
    fits.add_argument(
        "--chunk-rows", type=int, default=4096, help="stamps fitted at a time (default 4096)"
    )
    fits.add_argument("--overwrite", action="store_true")
    fits.add_argument("--json", action="store_true", help="print the sidecar's record")
    fits.add_argument("--quiet", action="store_true", help="no progress lines on stderr")
    fits.set_defaults(handler=_build_fits)
