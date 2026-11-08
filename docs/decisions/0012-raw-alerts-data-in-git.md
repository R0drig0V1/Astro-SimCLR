<!-- ADR 0012: what of the phase-3 raw-alert data is committed to git (task 3.1b). -->
# 0012: Raw-alert data in git: hashes yes, data and per-row indexes no (until governance)

- **Status:** accepted (2026-11-08, phase 3, task 3.1b)
- **Implemented by:** `src/astro_simclr/data/raw_alerts.py` (the `raw-alerts` layout, its
  SHA-256 per file in `meta.json`), `.gitignore` (`*.npy`, `dataset/*`)
- **Checked by:** `tests/test_data_raw_alerts.py` (hashes, determinism of the writer)

## Context

Phase 3 builds new labelled datasets from raw ZTF alerts (`docs/phases/03-newer-benchmarks.md`,
tasks 3.1, 3.1b and 3.5):

- task 3.1 downloads the cutouts of each labelled alert and stores every raw response with
  its timestamp and SHA-256;
- task 3.1b (this one) defines the `raw-alerts` prepared layout: `cutouts.npy`, a per-row
  `index.npy` (`oid`, `candid`, `jd`, `fid`, `ra`, `dec`, `label`, `label_source`,
  `source_dataset`, `source_row`), optional CD21 features and `meta.json`;
- task 3.2 (ADR 0008) reads a per-alert table (`oid`, `candid`, `jd`, `label`) and stores
  object-grouped splits **by group id, the `oid`**;
- task 3.5 freezes an index (`oid`, `candid`, `label`, `label_source`, split, seed) as a
  dated version, with `rebuild --from-snapshot` reproducing it offline.

Three things push toward committing more than code:

1. **Reproducibility.** A dataset that cannot be rebuilt bit for bit cannot back a
   replication claim; a frozen index in git is the obvious anchor.
2. **Reviewability.** A list in git can be diffed and audited.
3. **Convenience.** Agents and machines would get the data with `git clone`.

Against it:

- **The labels are not ours to publish.** CD21's pickles are not public, and phase 1 left
  their redistribution as an open data-governance question
  (`docs/phases/01-runnable-package.md`, "Data governance"). A per-row index of
  `oid`/`candid` → label for CD21's alerts *is* CD21's label set in another form, even with
  no pixel in it. The same applies to any list ALeRCE shares on request (task 3.0a).
- **Size.** About 3 GB of raw cutouts for the labelled set (plan, "Risks"); git is the wrong
  store.
- **Downloaded data is untrusted.** Whatever lands in git is pulled by every clone and
  every agent; a mistake there propagates.
- **Hard to reverse.** Once pushed to GitHub, data is cached and forked even if deleted.

## Decision

1. **Never committed:** cutouts (raw or normalised), prepared arrays (`*.npy` of any
   layout), the raw API responses of task 3.1, and any derived per-row array.
   `.gitignore` ignores `*.npy` (no `.npy` file is tracked) in addition to `dataset/*`.
2. **Nothing that lists the label set's objects or alerts is committed either, until the
   user decides on governance.** That covers `index.npy`, the alert-table CSV of task 3.2,
   the frozen index of task 3.5, and the object-splits files of task 3.2 (they list the
   `oid`s of each split, so they reveal which objects CD21 labelled, and their class
   counts). They stay with the data, out of git, like `SHA256SUMS` today. ADR 0008 already
   commits no object-splits file until 3.1 gives the real table; this makes the gate a
   governance decision, not only a data dependency.
3. **What is committed instead are hashes and provenance:**
   - the SHA-256 of `meta.json`, `cutouts.npy`, `index.npy` and `features.npy` of each
     frozen dataset version, with its name, date, row count and class counts, in a small
     JSON record under `configs/` (written by task 3.5; `meta.json` already holds the
     file hashes, so the record pins it);
   - the fingerprint of its alert table (`AlertTable.sha256`) and the SHA-256 of its
     splits file, which the benchmark runs already record (`benchmark.splits_sha256`);
   - the code and configs that rebuild it, and the seeds.

   A rebuild is checked against the recorded hashes, and the splits refuse another table.
4. **If the user approves publishing** (a governance decision, recorded as an amendment
   here), the frozen index is committed as a **CSV export** of `index.npy` (plus split and
   seed), and the splits files with it: text, diffable and reviewable, with their SHA-256
   in the record.
   The `.npy` stays the format the code reads (no pickle, a typed header, exact floats,
   memory-mapped; the reasons are in the docstring of `src/astro_simclr/data/raw_alerts.py`
   and in `docs/data.md`); the CSV is a derived view of it.
5. **Results stay as in phase 2:** metrics, figures and tables may be committed under
   `results/`; per-stamp predictions only when they carry no alert id (see Consequences).

## Consequences

- A clone has no phase-3 data. Every machine gets the raw responses or the layout out of
  band (as with `dataset/` today), and checks them against the committed hashes.
- Exact reproduction across machines relies on the hashes and on
  `rebuild --from-snapshot`, not on git. Losing the out-of-git snapshot loses the exact
  dataset; it must be backed up with the data.
- Reviewers cannot diff label changes in a PR; a rebuild reports the changed hashes and
  class counts instead, and task 3.5's tables (class × year, counts per `label_source`)
  are committed as results.
- Publishing later is cheap (item 4); un-publishing would not be, which is why the default
  is closed.
- Per-stamp predictions: phase 2's (`index` = position in a split) stay committable. The
  phase-3 benchmark predictions use `index` = `candid` (ADR 0008), so they list the label
  set's alerts: they fall under item 2 and stay out of git until the decision; their
  metrics, tables and figures do not.
