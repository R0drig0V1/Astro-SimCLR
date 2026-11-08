<!-- ADR 0018: how R5 is rebuilt from public sources under RJ23's recipe (task 3.5), and why. -->
# 0018: The R5 reconstruction (task 3.5)

- **Status:** accepted (2026-11-08, phase 3, task 3.5)
- **Implemented by:** `src/astro_simclr/r5/` (`recipe`, `sources`, `select`, `build`,
  `cli`), `src/astro_simclr/alerts/by_oid.py` (`fetch-alerts --oids`), the constraints of
  `src/astro_simclr/protocol/grouping.py` and `object_splits.py` (object-splits format 2)
- **Checked by:** `tests/test_r5_*.py`, `tests/test_alerts_by_oid.py`,
  `tests/test_split_constraints.py` (offline: tiny CSVs, detection JSON and avro made by the
  tests)

## Context

RJ23 (Reyes-Jainaga et al. 2023) cannot be rebuilt exactly from public data
(`docs/phases/03-newer-benchmarks.md`, "Task 3.0 results"): their alert per object was an
unseeded random draw, their SN/VS/AGN labels come from an internal ALeRCE table, and their
satellites were tagged by hand. The plan (task 3.5, after review round 2) asks for **R5**:
RJ23's five non-satellite classes under RJ23's recipe, from public sources only, with a
pre-registered deviation table instead of a count tolerance. Several details of that recipe
are interpretations; they are fixed here before any data is seen.

## Decision

1. **Time cutoff and draw** (`r5/recipe.py`).
   - `JD_MAX` is JD 2459945.49998843, 2022-12-31T23:59:59 UTC. A detection counts only if
     `jd <= JD_MAX`, computed as `mjd + 2400000.5` from ALeRCE's detection list.
   - The eligible detections have `has_stamp` true. They are sorted by `candid`, and one
     is drawn with `default_rng([seed, oid_hash]).integers(n)`, where `oid_hash` is the
     first 8 bytes of SHA-256(oid), big-endian. The draw is per object, independent of the
     visiting order, and the same on every machine.
   - `ndet` is the number of distinct candids in the stored list with `jd <= JD_MAX`, with
     or without a stamp (RJ23's `ndet` counted detections, not alerts).
   - A drawn alert whose raw cutouts are not 63×63 drops the object, with no redraw
     (RJ23 discarded non-square stamps). The drop is counted.
   - The default seed is 0 for every stream (RJ23's `random_state=0`). The streams are the
     draw (`oid_hash`), the subclass permutations (`stable_hash("sample:<name>")`) and
     the splits (`make_object_splits(seed)`).
2. **Targets and caps.**
   - Each AGN and VS subclass is walked in a seeded permutation of its Zenodo objects. The
     walk accepts objects until it reaches **RJ23's published count**, the target. RJ23's
     cap (`sample(min(n, cap))`: 4,000 for AGN, 1,200 for VS, 2,400 for `E`) is recorded
     in the deviation table, and a target never exceeds it.
   - So a loss (no stamp, a 404, a CCD edge) is replaced by the next object in the
     permutation, rather than lowering the count as in RJ23. The plan asks for RJ23's
     counts; replacing losses is the way to reach them that is honest about the sampling.
   - Zenodo's `E` stands for RJ23's `EB` (`EB/EW` + `EA`): target 2,339, cap 2,400. `CEP`
     is RJ23's `Ceph`. `RSCVn` and `ZZ` are absent from Zenodo and recorded as such.
     `CV/Nova` is not an R5 subclass.
3. **SN = Zenodo first, then BTS.**
   - Zenodo's SN are walked first in a seeded permutation; all of them fit under 3,615.
     The BTS SN that Zenodo lacks follow, in their own seeded permutation, until 3,615 SN
     in total.
   - Zenodo first, because RJ23's SN table is the successor of Zenodo's, so Zenodo's SN
     are the most RJ23-like. A plain union in one permutation would discard a random part
     of them.
   - BTS `type` maps to RJ23's SN subclasses (`SN Ia*` and `SN Iax` → SNIa; `SN II`, `IIP`,
     `II-pec` → SNII; `IIn`; `IIb`; `Ib*`, `Ic*`, `Ib/c`, `Ibn`, `Icn` → SNIbc; `SLSN-*`
     → SLSN). Any other type is not a SN.
   - "First detection ≤ `JD_MAX`" follows from having an eligible detection.
   - BTS's magnitude limit (peak brighter than about 18.5–19) and its 2018+ footprint are
     a deviation, read in the era table.
4. **Asteroid and Bogus come from CD21, through task 3.1's store.**
   - Each CD21 row that 3.1 resolved (`ok`) gives its `oid`, and its detection list is
     already in 3.1's snapshot.
   - Asteroid: `ndet == 1`. Bogus: `ndet < 10`, and no `oid` in Zenodo or BTS (RJ23's
     cross-match cut, done with the public tables, by oid).
   - Ghost rows are dropped and counted. So are rows 3.1 could not resolve (`mismatch`,
     not found, ambiguous): a stamp with no verified object has no detection list to draw
     from.
   - RJ23's 48 pearl necklaces and 12 verified long bogus are not public. They are not
     applied; the deviation table says so.
   - All objects that pass are kept, with no sampling: RJ23 kept them all. CD21's ceiling
     is accepted.
5. **Label agreement.**
   - An object whose sources give it two R5 classes is dropped (`label_conflict`):
     Zenodo, BTS, or CD21 as any of its five labels. One exception: a CD21 *Bogus* label
     never claims against a known object, so RJ23's cross-match keeps the object's class
     and drops the bogus.
   - A position group (below) whose R5 objects carry two classes is dropped as well
     (`group_label_conflict`).
   - ALeRCE classifier outputs are never a source.
6. **Grouping and the cross-benchmark rule.**
   - R5's accepted objects and every CD21 row (ghosts included, at their de-standardised
     positions) are linked by equal `oid` and by any pair within 1.5″ (union-find across
     both benchmarks). The R5 group id is the smallest R5 `oid` in the component.
   - A group with a CD21 Train row may only go to R5 Train. A group with a CD21 Validation
     or Test row may only go to R5 Validation or Test.
   - A group with both cannot satisfy the matrix, because CD21 itself leaks there. It is
     dropped (`cd21_train_and_eval`).
   - `assert_no_shared_groups` then checks the 3×3 matrix, and the counts are recorded in
     `build.json`.
7. **Constrained splits** (`grouping.assign_groups(allowed=...)`).
   - Within a class, the most constrained groups are placed first, then the free ones.
     Each tier keeps the class's random order, and each group goes to the allowed part
     furthest below its target. So the free groups make up for the constrained ones as far
     as they can.
   - Without constraints the draw is unchanged, number for number.
   - Object-splits files are now format version 2: a `constraints` list, one entry per set
     of allowed splits. Version 1 is still read.
   - Verification checks every constraint. It checks the stratification bound only for the
     classes with no constrained group, because the bound does not hold under constraints.
     In R5 nearly every class has one (any object that shares an oid or position with a CD21
     row), so **R5's per-class balance is reported, not verified**: `split_counts.csv` gives
     the achieved fractions, and a class that cannot fill a split under its constraints
     stops the build with a `SplitsError` naming the class and the empty part.
8. **First-stamp view.**
   - For R5 Test objects, the earliest eligible detection (`jd`, then `candid`) whose alert
     is a full 63×63 stamp, as RJ23's `strategy='first'` (RJ23 did this for Test only).
   - It is a second `raw-alerts` layout (`r5-first/`, `source_dataset` `r5-first-stamp`),
     with its own alert table.
9. **Everything from snapshots; byte-identical rebuild.**
   - The selection reads every byte from the response snapshots: 3.1's store and the
     by-oid store. Its final failures come from the by-oid index. It is a pure function of
     the sources, the snapshots and the seed.
   - Online, the build alternates the selection and a fetch of exactly what is missing. A
     walk stops once accepted + pending reaches the target, so each round asks for just
     enough. Offline (`--from-snapshot`), it runs the selection once and refuses an
     incomplete snapshot.
   - The outputs carry no wall time or request count, and record the SHA-256 of every
     file. `--expect build.json` compares a rebuild with a recorded build.
10. **Sources are dated snapshots, never committed.**
    - Zenodo's 1.12 GB archive is streamed to disk and checked against Zenodo's pinned size
      and MD5. Only the labelled set is read out of it, by base name, as a regular file,
      without extracting anything; then the archive is deleted.
    - The BTS explorer CSV (the task-3.0 query: spectroscopic SN, quality and purity cuts)
      is fetched through the polite `Fetcher`.
    - Each source gets a record: URL, UTC time, size, SHA-256, licence or terms, citation.
    - The BTS pages state no terms of use (checked 2026-11-08). We cite Fremling et al.
      2020 (ApJ 895, 32) and Perley et al. 2020 (ApJ 904, 35), as the explorer asks.
      Zenodo 4279623 is CC BY 4.0: cite Sánchez-Sáez et al. 2021.
    - ADR 0012 applies to the outputs. The layouts, alert tables, splits and the frozen
      index list the label set's objects and stay out of git. The deviation, drop, era and
      split tables and `build.json` hold only counts and hashes.

## Consequences

- **Known shortfalls**, pre-registered (the deviation table will show them):
  - Blazar (Zenodo 1,267 against 1,932);
  - Ceph (618 against 973);
  - RSCVn (1,171) and ZZ (9), absent;
  - VS therefore about 8.7k against 10,211;
  - Asteroid and Bogus limited to CD21's ceiling after the cuts (CD21 has about 9.9k
    Asteroid and 10.8k Bogus rows; RJ23 had 9,180 and 15,950).
- **The cross-benchmark rule shapes R5's Asteroid and Bogus splits.**
  - Nearly every R5 Asteroid and Bogus object is a CD21 object, and CD21's Validation and
    Test hold only 100 per class.
  - So R5's Asteroid and Bogus Validation and Test hold at most CD21's Validation and Test
    objects that survive the cuts: about 100 each per split, against about 1,800 in RJ23's
    Test. Train holds the rest.
  - `split_counts.csv` reports it. The per-class recall of those classes has a wide
    minimum detectable effect, and the prior-reweighted headline weights them by RJ23's
    proportions.
  - This follows from the plan's matrix as written. Relaxing it would be a plan decision:
    for example, letting CD21 Train objects into R5 Test and excluding them only when a
    CD21-trained model is evaluated on R5.
- **Cost.** The detection lists and alerts of the Zenodo and BTS objects, the Bogus
  alerts whose draw is not CD21's own, and R5 Test's first stamps add about 50–55k requests
  to 3.1's (`docs/data.md`, "R5"). That is about 10 h at 1.5 req/s, or about 19 h at the
  1.29 s per request that 3.1's smoke run measured. The build is resumable, and
  `build-r5 --dry-run` prints the next round's count.
- A rebuild from the same snapshots and code reproduces every output byte for byte. A
  rebuild after the sources move (a new Zenodo version, BTS growth) is a new dated version
  with new hashes: `JD_MAX` keeps the BTS growth out of the selection, but not the sources'
  own edits.
