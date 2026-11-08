<!-- ADR 0017: the nested run configuration of task 4.2: eight sections with tagged unions,
     lowering to the flat view that defines the config hash, the tag rule of the merge, the
     lossless flat shim, and `astro-simclr run`. -->
# 0017: A nested config of tagged unions that lowers to the flat view

- **Status:** accepted (2026-11-08, phase 4, task 4.2)
- **Implemented by:** `src/astro_simclr/config_tree/` (`schema.py`, `presets.py`,
  `resolve.py`); `astro-simclr run` in `src/astro_simclr/cli.py`; `configs/runs/`
- **Checked by:** `tests/test_config_tree_presets.py` (the hash migration test),
  `tests/test_config_tree.py`, `tests/test_cli_run.py`

## Context

The flat `Settings` has about 60 fields; each training subcommand reads some of them, and
nothing says which. The plan (phase 4, B0 and 4.2) asks for nested frozen dataclasses with
tagged unions, `--set a.b=c` that fails on an unknown path, presets migrated, a shim for
the flat YAML, and `astro-simclr run CONFIG` with the old subcommands as aliases until
4.8. ADR 0015 (rule 10) fixed the contract: the config hash stays defined on the flat
view, and a nested config proves hash equality by lowering to it.

## Decision

1. **Eight sections.** `run` (seed, output directory), `runtime` (engine, device,
   workers, determinism, precision, progress bar), `data` (data directory, dataset, crop,
   training batch, features), `model`, `optim`, `schedule`, `augment`, `eval` (evaluation
   batch and batch limit). The four with a choice are tagged unions on `kind`:
   - `model`: `stamp_classifier`, `resnet` (`name`), `paper_stamp_classifier`
     (`train-supervised`); `simclr` (`pretrain`); `linear_eval`; `fine_tune`;
   - `optim`: `Adam` / `AdamW` (`betas`), `SGD`, `LARS` (`legacy_trust_ratio`);
   - `schedule`: `epochs` (EarlyStopping fields), `carrasco2021` (steps, grid, horizon);
   - `augment`: `transforms` (a registry name), `paper` (rot90, C5, C6, C7).

   A variant holds only the fields that choice reads: the paper model's input flags are in
   its variant, not in `data`, because no other model reads them, and `pretrained` is in
   `linear_eval` and `fine_tune`. Loss terms sit with the model that has them
   (`beta_loss` in the classifiers, the temperature in `simclr`); a ninth `loss` section
   would hold one field per model.
2. **The model decides the command.** There is no `run.command`: each model variant
   belongs to one training subcommand, so a second field could only disagree with it.
3. **Defaults are the flat defaults.** Every field's default is the default of the
   `Settings` field it lowers to (read from `Settings` itself), so `Config()` lowers to
   `Settings()` exactly. Named starting points are presets, not per-command defaults.
4. **Lowering.** Each field names its flat field in its metadata; a variant can fix flat
   fields (`carrasco2021` lowers `early_stopping: false`). The flat view is the `Settings`
   defaults overwritten by those values: exactly the `Settings` fields, which
   `semantic_config_from_plain(complete=True)` checks. `runtime.engine` is not a flat
   field: it is the run id's `engine` component (ADR 0015, rule 5).
5. **Validation by construction.** A `Config` checks its tags, the cross-section rules
   (`paper_stamp_classifier` goes with `schedule.kind carrasco2021` and `augment.kind
   paper`, and only with them; today `run_command` enforces the first) and then builds
   the lowered `Settings`, whose range checks run with the flat names in their message
   translated to nested paths. Only `lightning` is a valid engine until 4.1 and 4.6.
6. **Presets are written out.** `carrasco2021`, `carrasco2021-no-features` and
   `<command>-legacy` (the four `COMMAND_DEFAULTS` protocols) are `Config` objects in
   code, not raised from their flat forms, so the hash migration test compares two
   definitions. A nested file without `preset` starts from the legacy preset of its
   model's command, as the flat subcommands start from their legacy protocol.
7. **The tag rule.** Merging the file, then the dotted `--set` patch, onto the base is key
   by key, except that a `kind` that differs from the base's replaces the section: the
   base's fields belong to another variant. The dotted overrides form one patch first, so
   their order does not matter. A path is `section.key`: an unknown section, a deeper
   path, or a key the selected variant does not have is an error that lists the valid keys.
8. **The flat shim is lossless or refuses.** `--set key=value` without a dot applies to
   the flat view after the dotted ones, which is then raised back; a flat file needs
   `--command` and resolves as `<command> --config FILE`. Raising picks the variants from
   `supervised_model` (and the command), `optimizer` and `schedule`, and is refused when a
   flat field the variants do not hold is not at its default. Dropping it would move the
   config hash; keeping it would need a field nobody reads. The old subcommands accept
   such files unchanged, and every committed flat config (`configs/reference/`,
   `configs/phase2/`) raises with its old hash (tested).
9. **`astro-simclr run CONFIG`** resolves, lowers, and calls today's `run_command` with
   `--resume` and `--splits FILE --fold K | --screen` as the old subcommands; `--show`
   prints the resolved config and its config hash without training. The old subcommands
   keep their flags and their resolution order, as aliases, until 4.8.
10. **Canonical JSON of the nested form** (`config_json`) uses ADR 0015's
    `canonical_json` on the nested plain values (defaults materialised). It is for
    recording and comparing configs; the run identity still hashes the flat view.

## Consequences

- A new `Settings` field needs a nested home: `test_every_flat_field_has_one_nested_home`
  fails until it has exactly one. The checklist in CLAUDE.md says so.
- Until 4.8 a value lives in two places (flat preset and nested preset), kept equal by the
  migration test and the fixture pinned by A0. After 4.8 the flat `Settings` can be
  retired; the flat view (field names, types, ADR 0015's rules) stays the hash definition.
- `adam_betas` is read by the paper model only, but sits in the Adam variant of `optim`:
  a phase-1 model with `betas: paper` hashes differently and computes the same, the safe
  direction of ADR 0015. The paper model's `augmentation` name (unused) is fixed at its
  default by the `paper` augment variant.
- The plain engine (4.1, 4.6) reads the `Config` directly; `runtime.engine: plain` is
  added then, and `run` dispatches on it.
- Revisit if a phase needs a section nested more than one level (SimCLR view pipelines in
  phase 5 may): paths become `a.b.c` and the tag rule applies at each level.
