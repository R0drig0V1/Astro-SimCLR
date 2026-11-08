<!-- Phase 3 of the roadmap (docs/phases/README.md); owned by the integrator. Draft amended
after review round 1 and task 3.0's research (docs/plan-reviews/2026-11-08-phase-3.md). -->
# Phase 3: the newer benchmarks and the raw-alert dataset (draft, review round 1 applied)

## Goal

The user's request (2026-11-08) is to prepare the jump to the post-2021 work while phase 2
runs: replicate the newer ALeRCE stamp classifiers and rebuild the data they need.

1. **RJ23**: Reyes-Jainaga et al. 2023 (ApJL, arXiv:2304.13080).
   - **The data:** 6 classes (Satellite added). **One randomly chosen alert per object**, not
     necessarily the first; an imbalanced, stratified 60/20/20 split.
   - **AGN and VS** are sampled from ALeRCE's light-curve labelled set:
     - AGN 9,774, from AGN 3,916, Blazar 1,932 and QSO 3,926;
     - VS from nine periodic subclasses.
   - **Asteroids** are single-detection objects.
   - **Bogus:** alerts of objects with ≥ 10 detections are removed (except 12 confirmed), as
     are those cross-matched to SN, VS or AGN; 48 Bogus are relabelled Satellite.
   - **Satellites:** 580, by visual inspection.
   - **The input:**
     - scenarios full 63², cropped 16², low-resolution 16², and **multi-scale 4×8²**
       (1″/2″/4″/8″ per pixel, 256 values, the same size as the 16² views);
     - normalisation |x|-min-p99, clip ±2, plus a binary invalid-pixel mask (NaN, inf,
       |x| > 1e10);
     - the position as a unit-sphere (x, y, z), not CD21's 23 features.
   - **The training:** cross-entropy loss, batch 256 class-balanced, validation F1 every
     500 iterations with patience 5, 8 rotation and flip variants, per-scenario learning
     rates.
   - **The result:** macro-F1 about 87, with full and multi-scale on a par; also evaluated
     on each object's first stamp.
2. **C25**: Carvajal et al. 2025 (A&A 697, A119, arXiv:2411.03258).
   - **A different label set:** RJ23's alerts with about 3 % of the Bogus relabelled
     Satellite (Bogus 15,326, Satellite 1,204); test 9,888 alerts; F1 about 91.
   - **The FFT branch:** the normalised 63×63 difference, median-padded to 128, amplitude
     only, the central 16×16 kept.
   - **The result:** the satellite accuracy of the 16″ model rises from 72.0 % to 87.8 %.
3. **N24** (Neira et al. 2024; sequences, 3 classes) is out of this phase: it needs another
   dataset and input. It is a later note.

The phase-2 rules carry over: the protocol of task 2.3, no test-based decisions, and the
portability rule. Never copy ALeRCE code or weights. **ALeRCE classifier outputs are never a
label source** (circular). Never unpickle downloads, and treat downloaded data as untrusted.

## Facts so far (2026-11-08)

- **Raw stamps.** De-standardised `ra`, `dec`, `magpsf` and `sigmapsf`, matched on ALeRCE's
  API to an `oid` and `candid`, then downloaded from `avro.alerce.online`. In a probe, 9 of
  10 came back, equal to our stored stamps to ≤ 1e-7 after ALeRCE's min-max.
- **ALeRCE's light-curve labelled set is public:** Zenodo 10.5281/zenodo.4279623, CC BY 4.0,
  1.12 GB. Which version RJ23 used is to be confirmed (3.0).
- **CD21 leaks at the object level.** Within 1.5″ of a Train alert, verified by the
  integrator:
  - Validation: 24 of 500 rows (Bogus 20 of 100);
  - Test: 27 of 500 (Bogus 21 of 100).

  Phase 2 handles it: task "grouped decision splits", PR pending. Any phase-3 split must
  be grouped by object.
- **Unlabelled corpus source:** the ZTF public alert archive (`ztf.uw.edu/alerts/public/`),
  for the next phase.

## Task 3.0 results (research, 2026-11-08)

Each claim is marked verified, inferred or unknown in the research report. The main points:

- **RJ23 code is public under MIT** (`github.com/alercebroker/multiscale_stamps`), but
  incomplete: `models.py` and `config.py` are missing. No data or weights are published. The
  rule stands: we reimplement and cite, we do not copy.
- **C25 and N24:** no code, data or weights found.
- **The light-curve labelled set** (Zenodo 4279623, CC BY 4.0) has 123,496 rows, `oid`
  and class but **no `candid`**, and 1,718 SN (RJ23 has 3,615).
- **No public stamp-classifier labelled set with `candid`s exists** (not CD21, RJ23 or C25).
- **The RJ23 input, exactly:**
  - normalisation `(x − min|x|) / (p99|x| − min|x| + 1e-8)`, clipped to [−2, 2], per
    stamp and channel on the full 63×63, before any crop or pooling;
  - a 3-channel invalid mask, 1 where a pixel is NaN or ±inf;
  - channel order: sci, ref, diff, then their 3 masks;
  - the multi-scale view built by iterated 2×2 average pooling ('same' padding) after
    normalisation, masks pooled too, ordered by scale 1″/2″/4″/8″, giving
    **256×8×8×24**;
  - crop offsets 24:40 (16 px) and 28:36 (8 px).

  RJ23's code leaves values with |x| > 1e10 out of the statistics but does not zero
  them, unlike the paper.
- **RJ23 alerts:** one random detection per object from ALeRCE's API, with an unseeded
  `sample()`. Only the "ghost bogus" reuse CD21's cutouts. Recovering CD21's stamps (3.1)
  therefore does **not** reproduce RJ23's inputs.
- **RJ23 training:**
  - Table 2 filters, dropout and learning rates per scenario;
  - 8 rotation and flip views averaged;
  - position (x, y, z) concatenated after dropout, then BN, FC and FC-6;
  - Adam, unweighted cross-entropy, batch 256 class-balanced;
  - validation macro-F1 every 500 iterations with patience 5;
  - **the model saved at the stopping iteration** (not the best);
  - 5 runs on one split (60/20/20, `random_state=0`).
- **RJ23 unknowns:** the activation, the conv padding, the flatten point and the BN
  placement (lost with `models.py`).
- **RJ23 Table 1, test macro-F1:** Full 86.68, Cropped 86.19, Low-res 82.69, Multi-scale
  87.39. First-stamp view: 86.65 / 86.10 / 82.65 / 87.30.
- **C25:**
  - quote the v2/A&A numbers: satellite accuracy 72.0 → 87.8 % (v1 said 66.9 → 79.7 %);
  - labels: RJ23's 49,310 alerts with 624 Bogus moved to Satellite (Bogus 15,326,
    Satellite 1,204);
  - Table 2 F1 per model (A–F): 91.0, 91.1, 87.9, 91.0, 90.5, 90.9; test 9,888; no
    first-stamp view;
  - FFT details not stated (the median of what, the placement, fftshift, log or
    normalisation), so they need an ADR.
- **RJ23 label sources:**
  - SN, VS and AGN from ALeRCE's internal cross-match table, VS capped at 1,200 and AGN at
    4,000 per subclass;
  - Bogus from CD21 after cuts;
  - Asteroids are CD21's with ndet == 1;
  - **satellites tagged by hand in SN Hunter, whose reports are not public**, so this class
    cannot be rebuilt from public data and needs a documented substitute.
- **N24:** Part III uses CD21's data, 52,244 first alerts (a cross-check for our counts).
  The rest is out of scope.
- **TNS:** the daily `tns_public_objects.csv.zip` needs a `tns_marker` User-Agent, and an
  anonymous request returned 401. **A free TNS account and bot API key are needed (a user
  action)**, kept out of the repo.

**Consequence:** an exact RJ23 replication needs ALeRCE's list (3.0a). Without it, 3.5
rebuilds from Zenodo (VS, AGN, part of SN), TNS (more SN), CD21 (Bogus, Asteroids, with
RJ23's cuts), one random detection per object (seeded) and a satellite substitute. The
result is reported only as "on our reconstruction".

## Tasks

Running now: 3.0, 3.1, 3.2, and phase 2's grouped splits. 3.0a is the user's decision.
After 3.0 and review round 2: 3.1b, then 3.3/3.4, then 3.5 and 3.6.

### 3.0 Research: availability and exact specifications — done (2026-11-08; results above)
- For RJ23 and C25: a **per-scenario specification table**, each value marked as
  published, inferred or unknown. It covers:
  - layers and filters, dense sizes, dropout and BatchNorm;
  - the pooling of the coarse scales;
  - the learning rates and selection.
- The labelled-set version, its columns and subclass names.
- How to access TNS (dated public CSV via a bot key from an environment variable).

### 3.0a Ask ALeRCE for the RJ23/C25 object lists (user decision)
Day 0, time-boxed to two weeks: ask the authors for the `oid`/`candid`/label lists of RJ23
and C25. With them, 3.5 rebuilds exactly; without them, it follows the recipe below. 3.5
waits for the answer or the deadline. It is an outward contact, so it is the user's call.

### 3.1 Raw alert retrieval, `fetch-alerts` (running, PR pending)
Acceptance:
- per-class recovery of at least 98 %;
- misses classified as not-found, ambiguous, timeout or mismatch;
- **zero stored rows failing the stamp check** (≤ 1e-6 after ALeRCE's min-max);
- ties resolved by downloading every candidate and letting the stamp check decide;
- many-to-one row→candid mappings reported;
- the `/detections` list cached per `oid`;
- every raw response stored with its timestamp and SHA-256 (an offline rebuild must be
  possible);
- non-63×63 cutouts reported.

Load: about 155k requests; about 21 h at 2 req/s, resumable, one throttle across all
hosts. No network in CI.

### 3.1b Raw / N-class prepared data kind (new; before 3.3)
- **Format:** a `meta.json` kind `raw-alerts` (format version 3) with:
  - configurable label names;
  - float32 raw cutouts with NaN kept, plus a mask;
  - per-row `oid`, `candid`, `jd`, `fid`, `label_source` and `source_row`.
- **Code:** `N_CLASSES` comes from the dataset (today `models.py:60` hard-codes 5).
- **ADR:** whether derived indexes are committed.
- **Acceptance:** the phase-2 path is byte-identical, and a synthetic 6-class layout trains
  end to end.

### 3.2 Benchmark metrics and object-grouped splits (running, PR pending)
- Macro-F1, macro precision and macro recall, plus per class.
- A **macro-F1 bootstrap that recomputes the confusion matrix per resample** (seed-ensemble
  argmax, paired, class-stratified), golden-tested against brute force.
- A prior-reweighted macro-F1 next to the raw one.
- Object-grouped, class-stratified splits, and selection/decision halves.
- A first-stamp view.
- The MDE per class.
- A cross-benchmark check: no object shared across the 3×3 split pairs of two benchmarks.

### 3.3 RJ23 / C25 input path (after 3.0 and 3.1b)
From raw cutouts:
- the four RJ23 views (full 63², cropped 16², low-resolution 16², multi-scale 4×8²);
- the |x|-min-p99 normalisation with clip ±2;
- the invalid-pixel mask channels;
- the unit-sphere position;
- C25's FFT cutout.

### 3.4 RJ23 / C25 models (after 3.0 and 3.3)
The specification table of 3.0. Where values are unknown, a bounded search on the phase-3
selection half only (for example 20 trials per scenario, with a documented budget). Our
p4/p4m variants are not part of this phase: they belong to a later improvement phase under
the same protocol.

### 3.5 Dataset reconstruction (after 3.0a's answer or deadline, 3.0 and 3.1)
If no list is shared, follow RJ23's recipe per class:
- **AGN/VS:** `oid`s from the light-curve labelled set, sampled to RJ23's subclass counts
  with a stored seed.
- **SN:** CD21's SN plus TNS spectroscopic SN, from a dated TNS snapshot.
- **Asteroids:** CD21's Asteroids that are single-detection objects.
- **Bogus:** CD21's Bogus after RJ23's ≥ 10-detection and cross-match removals.
- **Satellites:** candidates from a heuristic (elongation or multiple point sources in the
  63″ difference; TLE cross-match), **each confirmed by a blinded visual pass** with a
  written rubric and a second-reviewer agreement rate. The same pass runs on CD21's Bogus,
  which relabels or drops.

Alert choice: one alert per object, uniformly at random from its detection list (stored
seed), plus the first alert for the first-stamp view. Drop non-63×63 cutouts.

Splits:
- grouped by `oid`;
- any object with a CD21 Validation or Test alert goes to the phase-3 Test or Validation,
  never to Train;
- zero objects shared across the 3×3 split pairs of both benchmarks (the 3.2 check).

Era:
- record `jd` per row and publish a class × year table;
- add a train-before-T / test-after-T diagnostic view;
- report the share of TNS SN that ALeRCE itself reported.

Freeze the index (oid, candid, label, label_source, split, seed) as a dated version;
`rebuild --from-snapshot` reproduces it offline byte for byte.

Acceptance: class counts within ±5 % of RJ23's, or the deviation tabulated.

### 3.6 Replication runs (after 3.2–3.5; 5090; after phase 2.5's CV, or on a stated share)
- RJ23's four scenarios first, then C25's FFT models, the latter only if the satellites
  of 3.5 exist. 5 seeds each.
- **Pre-registered gate:**
  - (a) the ordering multi-scale ≈ full > cropped > low-res holds, with paired-bootstrap
    CIs;
  - (b) FFT raises the cropped-16 satellite recall, with a CI that excludes 0;
  - (c) absolute macro-F1 is advisory: it is "on our reconstruction".
- Compute line: runs × the measured s/step on the 5090, about 35 runs.

## Risks
- **Comparability:** the reconstruction is not RJ23's unless 3.0a succeeds.
- **Satellites:** visual labels, so inter-reviewer agreement must be reported.
- **Moving sources:** TNS and ALeRCE re-process their data, so everything works from dated
  snapshots.
- **Era confounds:** new classes come from other years than CD21's.
- **Load and compute:** the external load on ALeRCE, and GPU contention with phase 2.5.
- **Storage:** about 3 GB for the raw labelled set; the unlabelled corpus is the next phase.
