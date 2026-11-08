<!-- ADR 0018: the local run registry (task A2): a derived SQLite index outside every worktree,
     the legacy backfill with config_format readers, and comparisons by experiment-spec id. -->
# 0018: The local run registry, the legacy backfill and comparisons by spec

- **Status:** accepted (2026-11-08, phase 4, task A2)
- **Implemented by:** `src/astro_simclr/registry/` (`home.py`, `readers.py`, `store.py`,
  `index.py`, `spec_view.py`, `compare.py`, `parser.py`, `commands.py`); the `runs`
  subcommand of `cli.py`. Its record of a run directory is `IndexedRun`, not to be confused
  with task A1's writer `runs.record.RunRecord`
- **Checked by:** `tests/test_registry_readers.py` (on the archived
  `results/phase2/replication/**/metrics.json`), `tests/test_registry_store.py`,
  `tests/test_registry_index.py`, `tests/test_registry_compare.py`,
  `tests/test_registry_cli.py`, with the builders of `tests/_registry_support.py`;
  `tests/test_cli_training.py` (`test_the_registry_reads_what_runs_write`: the directories
  task A1's `run_command` writes index as manifest runs)

## Context

Results sit in run directories: `results/phase2/**` in the repository, `~/runs` on the
GPU machine, and from task A3 the queue's run root. Nothing queries them by config, seed,
split, machine or commit, and a comparison is a hand-typed list of directories, which can
drift from what was pre-registered (plan review F10). Task A0 (ADR 0015) froze the run
identity and `manifest.json`; task A1 makes every new run write one. The archived runs
have no manifest, and their `metrics.json` records a config with fewer fields than today's
`Settings` (13 were added since: C5-C7, p4/p4m, `fp32_precision`, ...). Task A3 owns the
experiment-spec format, in parallel.

## Decision

1. **A derived SQLite index, outside every worktree.** The registry is
   `$ASTRO_SIMCLR_HOME/registry.sqlite` (default `~/.astro-simclr`; `--registry FILE`
   overrides). It is a cache of the run directories, which stay the source of truth:
   deleting it loses nothing that `runs reindex` cannot rebuild, and it is never copied
   between machines (the queue rsyncs run directories, then reindexes). A path inside a git
   working tree (a directory above it holds `.git`) is refused, so that no checkout
   commits, copies or deletes it.
2. **Schema version and migrations.** The version is the file's `PRAGMA user_version`.
   `MIGRATIONS` is an append-only tuple (entry `i` takes version `i` to `i + 1`), each
   applied in one transaction with its version bump. A file of a newer version is refused
   rather than misread. Migrating instead of always rebuilding keeps the remembered roots
   and the artefact hashes, which are slow to recompute.
3. **Reindex.** `runs reindex [ROOT ...] [--rebuild]` walks the roots (no symbolic links,
   no hidden directories). A directory with `manifest.json` or `metrics.json` is a run
   directory and is not descended into. Each root's rows are replaced in one transaction.
   An unreadable directory is skipped with its reason, never half-indexed. Roots are
   remembered; a remembered root that has gone is reported and keeps its rows (an
   unmounted disk is not a deletion). Every file of a run directory is recorded with its
   kind (`runs.layout.classify_artefact`), size, mtime and SHA-256, the "artefact SHA and
   presence" of ADR 0015's retention; an unchanged size and mtime reuse the recorded hash.
   This argv is the interface of A3's `queue sync`.
4. **Backfill: missing provenance is null, flagged `legacy`.** A run directory without a
   manifest is read from its `metrics.json`. The run id, attempt, data SHA-256 (ADR 0015
   rule 6 needs the array hashes; the file has the source pickle's SHA-256 only, kept as
   `data_source_sha256`), pretrained SHA-256, git state, `uv.lock` and `numerics_epoch`
   are null. Three values are derived from facts of the code, not guessed:
   - `status: finished`: `run_command` writes `metrics.json` only after the fit;
   - `engine: lightning`: the only engine before A1, and every run since writes a
     manifest;
   - `machine_class`: `default_machine_class` of the recorded machine.
5. **Readers versioned on `config_format`.** The recorded config is read by the reader of
   its format; a `metrics.json` without a `config_format` key is `flat-1`, the flat
   `Settings` view, the only format so far. The `flat-1` reader lowers a config to today's
   flat view before hashing: each field added after the config was recorded is filled with
   the value that reproduces the code from before the field existed (`FLAT_1_FILLS`, one
   reason per field). For example, `fp32_precision: ieee` follows ADR 0010 rule 5. A
   test pins each value to today's default, and a later default change must not edit the
   table. A key that today's `Settings` lacks, or a missing field the table does not fill,
   makes the run unreadable (skipped, with the reason), never hashed. So an archived run
   gets the config hash today's code gives the same experiment: the five archived
   with-features runs hash as `train-supervised --preset carrasco2021`, and the five
   without as `carrasco2021-no-features` (tested). Fields added after A0 need no entry
   (ADR 0015 rule 9). The nested config of task 4.2 will be `nested-1`, with a reader that
   lowers to the same view.
6. **One attempt, one key; copies are one attempt.** A record's key is
   `<run id[:16]>-a<attempt>`, or `legacy-<metrics.json SHA-256[:16]>` for a legacy run.
   `results/phase2/replication` is a byte-exact copy of `~/runs/replication`, so a machine
   that indexes both has two directories per attempt; they share a key and count once. Two
   copies of one key with different `metrics.json` fail a comparison.
7. **Comparisons take an experiment spec, never a directory list.** `runs compare SPEC`
   takes a spec file, or the id (a prefix of at least 6 characters) or name of a spec
   indexed from `$ASTRO_SIMCLR_HOME/specs/` (the queue writes each accepted spec there as
   `<spec_id>.yaml`). The comparison runs on `ComparisonPlan`, a minimal frozen dataclass
   of what it needs: the spec id and name, the command, arm name → config hash, the cells
   (arm, seed, split part) and the optional machine class and splits SHA-256. The spec
   itself is task A3's `runs.spec.ExperimentSpec`. The registry reads it only through a
   structural `Protocol` agreed with A3 (`name`, `spec_id`, `command`, `machine_class`,
   `splits.file`, `arms[].name`, `cells()` with `arm` as the arm's name,
   `arm_config_sha256(arm)`), and imports `runs.spec` lazily. Without A3 the package works,
   and a comparison says that specs need A3. A run fills a cell when it is finished and
   its command, config hash, seed and split part are the cell's, with the spec's machine
   class and splits file.
8. **What makes a comparison fail.** It fails, listing every problem, when:
   - a cell has no run (the resolved set must be the spec's arms × seeds × splits);
   - a cell has more than one attempt without an explicit choice: `--use KEY` names the
     attempt, or `--attempts latest` takes the one that ended last (legacy runs have no end
     time, so they need `--use`);
   - the chosen runs mix machine signatures (`reference.machine_signature` without its
     engine: device, GPU, torch, CUDA, effective float32 precision), engines (plan A1;
     checked once, on the record's engine) or numerics epochs (a
     legacy run's is unknown), unless `--allow-mixed machines|engines|numerics` says so;
   - two data SHA-256s or two source-pickle SHA-256s appear among the chosen runs. This
     is never allowed.

   Matching finished runs outside the cells (another seed, split, machine class or splits
   file) are counted in the report. A spec run on 5 of 10 seeds is visible that way.
9. **A report, not a decision.** `runs compare` prints each cell and the mean ± standard
   deviation (ddof=1) per arm of the requested metrics (default: the balanced accuracy and
   macro-F1 of every stage). The protocol's decisions remain `compare-variants`'
   (ADR 0004), on the run directories the report lists.

## Consequences

- A legacy run never matches a spec that names a data SHA-256, and mixing legacy and
  manifest runs takes `--allow-mixed numerics`. Comparing a 2.5 variant with the archived
  replication is possible, and the report says how the data was checked (the source
  pickle only).
- The registry costs one scan: about 3 s for `results/phase2` on the Mac, mostly the
  hashing of the 50 files. Rescans hash only new or changed files.
- A config that the `flat-1` reader cannot lower (an intermediate commit with a field that
  was later removed, say) is skipped, not guessed. Supporting it means a new
  `config_format` reader, which is the point of versioning them.
- When A3's `runs.spec` lands, the lazy import can become a direct one. The `Protocol` stays
  the contract, and A3's `spec_id` is the experiment-spec id.
- The registry deletes nothing. Applying `runs.layout.retention` is left to the runners
  (A3), which can read the artefact table.
