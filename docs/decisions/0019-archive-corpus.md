<!-- ADR 0019: how ingest-archive turns ZTF archive tarballs into the unlabelled corpus, and why. -->
# 0019: The archive corpus: first alerts by oid, no candidate cuts, night shards, a seeded plan

- **Status:** accepted (2026-11-08, phase 5, task 5.1a; code only, nothing downloaded)
- **Reviewed by:** a `code-reviewer` pass on 2026-11-08. Its findings are fixed and tested:
  an out-of-range `jd` that overflowed the date arithmetic, pax and GNU long-name headers
  read whole into memory, the gzip trailer never checked, `corpus.json` written in place,
  no lock between concurrent ingests, a window in which a replaced night was gone, and a
  missing tarball raised as a traceback.
- **Implemented by:** `src/astro_simclr/archive/` (`stream`, `selection`, `layout`,
  `ingest`, `plan`), `parse_alert(oid=None, candid=None, repair=False)` and
  `Alert.previous_detections` in `src/astro_simclr/alerts/parse.py`, and the
  `ingest-archive`, `archive-summary` and `plan-archive-nights` commands in
  `src/astro_simclr/cli.py`
- **Checked by:** `tests/test_archive_stream.py`, `tests/test_archive_ingest.py`,
  `tests/test_archive_plan.py` (offline: synthetic tarballs and avro made by the tests)

## Context

Phase 5 pretrains SimCLR on unlabelled ZTF stamps. The source is the ZTF public alert
archive (`https://ztf.uw.edu/alerts/public/`): one `ztf_public_YYYYMMDD.tar.gz` per night,
one avro alert per member. Task 5.1a measured 3,087 nightly files on 2026-11-08: 758 are
empty, a non-empty night is 6.5–9.8 GB (8.0 GB in 2020) with about 175k alerts, and a night
takes about 30 min to download, so about 20 TB in all. The corpus must:

- match the stamp classifier's input (Carrasco-Davis et al. 2021, CD21): the first alert of
  each object, as a 63×63 science / reference / difference triplet;
- remove no target class (task 5.1a: no `rb`/`drb` threshold, which would drop Bogus);
- exclude every object of every evaluation set (the phase-3 cross-benchmark rule);
- treat the tarballs as untrusted, store none of them, and resume night by night;
- sample nights by a seeded rule over 2018–2026, and state its disk and transfer budget
  before the user is asked to approve the download.

The user has not approved the bulk download, so this ADR covers the code only. The
thresholds below are taken from the public papers and documentation cited, never from
ALeRCE's code.

## Decision

1. **Stream, never extract.** A tarball is read in one sequential pass: the raw bytes go
   through a SHA-256, gzip decompresses them under a 64 GiB cap, and `tarfile` reads the tar
   in stream mode (`r|`). Only regular `.avro` members of at most 1 MiB are read, into
   memory; links, directories, devices, other files and oversized members are skipped and
   counted. At most 5M members are accepted. Member names are never used as paths.
   - tarfile reads a pax or GNU long-name header's body whole, before the member reaches
     the caller, so such headers are refused above 1 MiB, and GNU sparse members are
     refused outright. The check hooks `TarInfo._proc_member`, which every supported
     Python calls on each header; a test pins it.
   - After the last member the gzip stream is read to its end, so its CRC-32 and length
     trailer is checked. A truncated download is refused, never committed.
   - A `jd` outside 2017-09-04 to 2050-01-01 is `invalid_values`. The
   source can be a file or stdin (`ingest-archive - --night N`), so a download is piped in
   and no tarball is stored. Every alert then goes through the size-limited
   `alerts.parse` (container walk, schema check, decompression caps, strict FITS reader).
2. **First alert by `oid`.** CD21 uses the first alert of each object, ALeRCE's first
   detection of the `oid` (CD21 Sec. 1; its App. E counts the alerts "that have 1
   detection", ALeRCE's `ndet`). An archive alert carries 30 days of history, not the
   whole history of its object, so the rule combines what it carries:
   - **`previous`:** no detection (an entry with a `candid`) in `prv_candidates`;
   - **`oid_year`:** the year in the `oid` (`ZTF18…`) equals the UT year of the alert's
     `jd`, since ZTF names an object after the year it was first seen. This drops objects
     that return after a gap of more than 30 days from an earlier year (variable stars
     after their seasonal gap). It is an *assumption* that the pilot measures;
   - **one alert per `oid`** in a night (the ingest) and over all the nights
     (`corpus_index` keeps the earliest by `jd`).

   **`ndethist` is not used by default.** ZTF's schema counts in it the detections within
   1.5″ on the **same field and readout channel** only, down to S/N ≈ 3, so it includes
   detections that never made an alert. CD21 clips `ndethist` at 20 (Table A1), so its
   first alerts do reach such values. A bound would remove the first alerts of AGN and
   variable stars. `--max-ndethist N` adds it for an ablation.
3. **No candidate cuts.** CD21 states no cut on `rb`, `drb`, `nbad`, `fwhm`, `elong` or
   `magdiff`. Its alerts are the ZTF stream's: 5σ detections that passed ZTF's real-bogus
   filter (CD21 Sec. 2). ZTF's documentation suggests a purity filter (rb ≥ 0.65, nbad = 0,
   fwhm ≤ 5, elong ≤ 1.2, |magdiff| ≤ 0.1), but that filter removes Bogus, so it is not
   applied. The index keeps `rb`, `drb`, `ndethist` and `isdiffpos`, so a cut can be
   applied when the corpus is read.
4. **63×63 only.** CD21 removed its 650 non-square stamps (Sec. 3.1). `parse_alert(...,
   repair=False)` raises `CutoutShapeError` for them, and they are counted as
   `not_63x63`. `fetch-alerts` pads such stamps instead, because it must reproduce CD21's
   stored rows.
5. **Exclusions by `oid` at ingest, and again at read time.** `--exclude FILE` (one `oid`
   per line) is required unless `--no-exclusions` is given explicitly. The file's SHA-256
   is recorded in `corpus.json` and in every night. A resume with another list is refused,
   so no corpus mixes two lists. A list that grows later (a new benchmark) is applied by
   `corpus_index(exclude=...)` without re-ingesting the tarballs. The `candid` and 1.5″
   position-group exclusion of task 5.1a (ghosts included), and the near-duplicate check,
   belong to the corpus index build. They are not done here.
6. **Night shards, written atomically.** Each night is one directory: `index.npy`
   (structured, little-endian: `oid`, `candid`, `jd`, `fid`, `ra`, `dec`, `night`,
   `ndethist`, `isdiffpos`, `rb`, `drb`, `member`), `cutouts-NNNNN.npy` files (float32
   (≤ 4096, 63, 63, 3), raw, NaN kept), and `meta.json`. The meta holds:
   - the source: tarball name, compressed and decompressed bytes, SHA-256, member counts;
   - the selection rules, the exclusions, and the count per reason;
   - every file's dtype, shape and SHA-256.

   A night is written under `.<night>.partial`, fsynced, and renamed. A complete night is
   skipped on resume without reading its tarball, after checking that the tarball has the
   recorded name and size. A replaced night (`--overwrite`) is moved aside before the new
   one is renamed in, and is put back if a crash came in between. `corpus.json` is written
   through a temporary file. One ingest at a time holds a corpus: an exclusive `flock` on
   `OUT/.lock`. The format (`astro-simclr-archive`, version 1) is not the
   prepared layout's, so the phase-2 readers refuse it. Nothing is pickled. It is the
   unlabelled sibling of ADR 0012's `raw-alerts` kind; that kind needs at least two labels
   and one pool, so it was not reused.
7. **The sampling plan is seeded per year.** Each year from 2018-06-01 is cut into `k`
   equal strata: 24 per year for the labelled era 2018–2020, and 8 per year after it. In
   each stratum, `default_rng([seed, year])` draws a primary night and 2 spares, in order.
   The spares replace an empty or missing night: with 758 of 3,087 nights empty, about
   1.5 % of the strata would still have no night. Seeding per year keeps the years
   independent. Each night is tagged `h1` (2018–2022) or `ablation` (later). The pilot is
   one primary in 2018, 2020 and 2024, drawn after the strata so that it moves no night.
   Seed 0 gives the pilot nights 2018-10-20, 2020-09-25 and 2024-10-22.

## Consequences

- **Budget of the default plan** (seed 0: 120 primary nights, `plan-archive-nights`):
  - About 2.1M stamps: 1.54M for H1 and 0.56M for the ablation.
  - Storage: 100 GB in float32 (73.6 GB for H1), or 50 GB in float16.
  - Download: 960 GB in about 60 h.
  - The stamps per night (17,500 = 10 % of 175k alerts) are an **assumption**.
    `--measured CORPUS_DIR` replaces it once the 3 pilot nights are ingested. Storage is
    exact per stamp: 47,773 bytes with its index row.
  - With the download streamed, no tarball needs disk space.
- **What the first-alert rule misses:** an object whose earlier alert of the same year
  is more than 30 days older and outside the sampled nights. The pilot should measure the
  `oid_year` and `previous` counts per era, and the composition report with our own
  classifier's pseudo-labels (task 5.1a, not here) checks that no class was removed.
- **Not reproducible from the code alone:** the corpus also depends on the tarballs. Each
  night records their SHA-256, and the writer is deterministic, so a re-ingest of the same
  tarball gives the same bytes.
- **Re-ingest cost:** a change of the selection rules needs a new corpus directory and
  every tarball again, about 30 min per night. This is why the index keeps the fields a
  later cut would read.
- **Schema eras:** the tests cover a 2018-like schema without `drb`, a 2020 one, and a
  2024 one with `fp_hists`. The real schema versions are checked by the pilot.

## Amendment (2026-11-08, task 5.1): the corpus index

Point 5 left the `candid` and position exclusions and the near-duplicate check to the
corpus index. `build-corpus-index` (`archive/corpus_index.py`) builds it from the complete
nights. It applies five rules in order, and counts each row under the first rule that
drops it:

1. listed `candid`s;
2. listed `oid`s;
3. every alert of an `oid` that has a row within 1.5″ of **any** CD21 alert. Train is included, since CD21 CV
   evaluates it. The CD21 positions are the layout's features de-standardised with M1's
   statistics, matched with `protocol.groups.match_any_within` (haversine, RA wrap);
4. near-duplicates of a CD21 stamp, which drop their whole `oid`;
5. one alert per `oid`.

Decisions and their reasons:

- **The position rule is direct, not transitive.** A row goes if it is within 1.5″ of a
  CD21 alert itself. A chain of corpus rows, each within 1.5″ of the next, does not carry
  the exclusion further. The CD21 groups are already transitive among CD21's own alerts,
  so this removes every row within 1.5″ of any member of a group. Chaining through the
  corpus would remove crowded fields wholesale and prevent no CD21 leak. A matched row's
  `oid`, however, goes whole, as for near-duplicates: the object is a CD21 object even
  where another of its alerts carries a bad coordinate.
- **One alert per `oid` keeps the earliest by default.** The earliest (`jd`, then
  `candid`) is CD21's first alert (point 2). The seeded choice that the task brief asked
  for is available as `--duplicates random --seed N`, drawn from a new reserved stream,
  `corpus_duplicates` (11). It is an ablation, not the default.
- **The near-duplicate check has two stages, and drops only confirmed pixel copies.**
  - Stage 1 takes, for each corpus row, the CD21 stamps whose 9×9×3 pooled descriptor
    reaches a cosine of 0.99.
  - Stage 2 confirms a pair only if the full 63×63×3 normalised stamps agree within 1e-3 at
    every pixel.
  - Measured on 4 real nights (2026-11-08, 144,981 rows against the 51,582 CD21 stamps):
    stage 1 flags 29,041 candidates (about 20 % of the rows), because pooled point
    sources look alike. Dropping on the cosine alone would remove a fifth of the corpus.
  - The 142 rows within 1.5″ of a CD21 alert calibrate stage 2. 131 are pixel copies of
    their CD21 stamp (largest difference 1.2e-7: the normalisation reproduces the stored
    stamps to float32 rounding). The other 11 differ by at least 0.39.
  - The check uses all three channels, not only the difference. On the difference
    channel alone, isolated point sources look alike.
- **The index file** is a pickle-free `.npz`: `rows`, plus `meta` as a JSON string. It
  records the SHA-256 of every exclusion input: the layout's `meta.json`, and each
  `features.npy` and `images_f32.npy` it read, re-hashed against that `meta.json`, and
  each list file. It also records one `exclusion_sha256`, which pretraining manifests
  carry. It covers those inputs, labelled, and the rules, so a 1.5″ index and a 0.1″ one
  never share it. The index also records the SHA-256 of `corpus.json` and of each
  night's `meta.json`. A
  reader refuses an archive whose indexed nights changed. The index names local paths, so
  it is not committed.
- **Reading** (`archive/reader.py`) memory-maps the night shards and reads each batch
  sorted by file offset. This is ADR 0018's tier 3, over the shards instead of one file.
  The raw cutouts are normalised exactly as the stored stamps are (`alerce_normalise` in
  float64, then cast to float32). ADR 0025 records how the pilot reads them.

Measured costs on the 5090 (shared host, load average about 78): the index of 145k rows in
76 s with the near-duplicate check, so about 3.5 min for the 390k-row corpus; one batch of
256 stamps read and normalised in about 66 ms.
