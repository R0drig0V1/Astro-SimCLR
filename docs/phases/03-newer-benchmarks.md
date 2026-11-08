<!-- Phase 3 of the roadmap (docs/phases/README.md); owned by the integrator. Reviewed in 2
rounds plus task 3.0's research (docs/plan-reviews/2026-11-08-phase-3.md). -->
# Phase 3: raw-alert infrastructure and the R5 benchmark (RJ23 protocol, public sources)

## Goal (after review round 2, 2026-11-08)

The user asked (2026-11-08) to prepare the jump to the post-2021 work while phase 2 runs.
The research (3.0) showed that **no exact replication of RJ23 or C25 is possible from public
data**:
- RJ23's alert choice was random and unseeded;
- RJ23's labels come from internal tables;
- the satellites were tagged by hand in SN Hunter, and those reports are not public;
- C25 relabelled 624 Bogus by hand.

So the phase delivers, in order:

1. **Raw-alert infrastructure**, which phase 2 (row C1, fake SN) and the SimCLR phase (raw
   cutouts, exclusion lists, N-class evaluation) also need: 3.1, 3.1b, 3.2 and 3.3 (without
   C25's FFT).
2. **R5**: an object-level, 5-class benchmark (AGN, Asteroid, Bogus, SN, VS) under RJ23's
   protocol, built from **public sources only**. Its test set has about 8–9k objects, far
   more statistical power than CD21's 500 stamps. It also serves as the evaluation set of the
   SimCLR phase.
3. **RJ23's four scenarios on R5** (full 63², cropped 16², low-resolution 16², multi-scale
   4×8²), with the gate of 3.6.

**Out of scope, as later notes:**
- the Satellite class, C25 and its FFT branch (only if ALeRCE shares C25's list);
- N24;
- our p4/p4m variants on R5 (a later improvement phase);
- manual labelling.

**Reporting rule:** every number is "on R5", never "RJ23's result", unless 3.0a delivers
RJ23's exact lists. That would add an "RJ23-exact" benchmark next to R5, not replace it.

The phase-2 rules carry over: the protocol of task 2.3, no test-based decisions, and the
portability rule. Never copy ALeRCE code or weights (RJ23's repository is MIT; we
reimplement and cite). ALeRCE classifier outputs are never a label source. Never unpickle
downloads, and treat downloaded data as untrusted.

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

Running now: 3.1, 3.1b, 3.2, 3.3, and phase 2's grouped splits. 3.5 starts right after 3.1,
without waiting for 3.0a.

### 3.0 Research — done (2026-11-08; results above)

### 3.0a Ask ALeRCE for the RJ23/C25 lists (user decision; does not block anything)

### 3.1 Raw alert retrieval, `fetch-alerts` (running)
Acceptance:
- per-class recovery of at least 98 % over **non-ghost** rows. Ghost rows are CD21 Bogus
  with no ALeRCE object within the radius; they are counted per class, not retried;
- misses classified;
- zero stored rows failing the stamp check (≤ 1e-6);
- ties resolved by the stamp check;
- `/detections` cached per `oid`;
- raw responses snapshotted with their SHA-256 (offline rebuild);
- non-63×63 cutouts reported.

### 3.1b Raw / N-class prepared kind (running)
`raw-alerts`, format version 3, with `N_CLASSES` taken from the data and the phase-2 path
byte-identical. It also stores the CD21 23-feature candidate fields of each drawn alert, so
phase-2 models can be evaluated on R5.

### 3.2 Benchmark metrics and grouped splits (running)
Macro-F1 with a per-resample confusion bootstrap, a prior-reweighted macro-F1, grouped
splits, the first-stamp view and the MDE per class.

**The cross-benchmark matrix:**
- allowed: (Train, Train), (Val, Val), (Val, Test), (Test, Val), (Test, Test);
- forbidden: Train ↔ Val and Train ↔ Test, in both directions;
- group id: the `oid` where it is matched, otherwise a 1.5″ position group (union-find
  across both benchmarks; ghosts included).

### 3.3 RJ23 input functions (running; C25's FFT removed)

### 3.4 RJ23 models (after 3.3)
**No hyperparameter search.** An ADR pins the details the research marked unknown. Each
follows from RJ23's stated 8×8 final maps or from C25's description:
- 'same' conv padding and ceil pooling;
- flatten, then average over the 8 views;
- dropout → concat(x, y, z) → BN → FC → FC-5;
- ReLU.

Table 2's per-scenario hyperparameters are used as published. One pre-registered
sensitivity arm: multi-scale with BN before versus after the concat, 5 seeds, decided on the
decision half.

### 3.4b RJ23 training regime (new)
- `schedule: rj23`:
  - evaluation every 500 iterations, iteration 0 included;
  - patience 5 on strict improvement of the validation macro-F1;
  - `keep: stop|best`, both reported from one run; **stop** is primary, as RJ23 did;
  - a hard cap of 100k iterations, with the runs that hit it reported.
- A per-sample class-uniform sampler (not CD21's tiling).
- TF-default parity, each with a hand-computed golden test and no TF dependency:
  - Adam eps 1e-7;
  - BN momentum 0.99 and eps 1e-3;
  - glorot-uniform initialisation;
  - 'same' pooling semantics.
- **Protocol:** the 3.6 replication stops on the whole validation set (the paper protocol,
  as 2.4 did). Any phase-3 *decision* stops on the selection half and decides on the
  decision half.

### 3.5 R5 reconstruction (after 3.1; public sources only)
- **Time cutoff `JD_MAX`,** pre-registered as 2022-12-31: an inferred stand-in for RJ23's
  build date.
  - Each object's random alert comes from its detections with a stamp and `jd ≤ JD_MAX`,
    sorted by `candid` and drawn with `default_rng([seed, oid_hash])`.
  - The ndet counts of the cuts come from that stored list, not from today's API `ndet`.
  - The first-stamp view is the first such detection.
  - A non-63×63 draw drops the object, and the drops are counted.
- **Per-class sources:**

| Class | Source and rule |
|---|---|
| AGN | Zenodo 4279623, sampled to RJ23's subclass counts with a stored seed |
| VS | Zenodo 4279623, sampled to RJ23's subclass counts with a stored seed; the `E` subclass capped at 2,400 (two of RJ23's caps); RSCVn and ZZ are absent from Zenodo and recorded as such |
| SN | Zenodo SN ∪ **ZTF Bright Transient Survey** spectroscopic SN (public CSV, no login; 6,758 SN with ZTF IDs, checked 2026-11-08), first detection ≤ `JD_MAX`, up to 3,615, seeded. BTS's magnitude limit (peak below about 18.5–19) is recorded as a deviation. Cite Fremling et al. 2020 and Perley et al. 2020, and check BTS's terms. TNS is an optional cross-check |
| Asteroid | CD21's, with ndet == 1 in the stored list |
| Bogus | CD21's, after RJ23's ≥ 10-detection and cross-match cuts. CD21's ceiling is accepted: no hunting for other Bogus sources. Ghost rows are dropped and counted |

- **Acceptance:** a pre-registered deviation table, target versus achieved per class and
  subclass with the cause. There is no ±5 % rule, which was infeasible by construction.
- **Headline metric:** the macro-F1 reweighted to RJ23's 5-class counts, with the raw
  macro-F1 next to it.
- **Splits:**
  - grouped by object, following the 3.2 matrix (a CD21 Train object goes only to R5
    Train; CD21 Validation and Test objects never to R5 Train);
  - an era table (class × year) and the temporal diagnostic view;
  - a frozen index rebuilt offline from the snapshots.
- **Load:** about 50–60k requests beyond 3.1 (about 8 h at 2 req/s), resumable, with the
  same snapshot format.

**R5 size finding (r5-builder, PR #44, 2026-11-08).** Under the 3.2 cross-benchmark
matrix, R5's Asteroid and Bogus Validation/Test can only hold CD21 Validation/Test objects,
about 100 per class per split. R5 Test is therefore **about 4.5k objects, not 8–9k**. The
high-power claim holds for AGN, SN and VS only; Asteroid and Bogus keep CD21's power.

The builder's expected real load is about 52–54k requests: about 10 h at 1.5 req/s, or
about 19 h at fetch-alerts' measured latency.

### 3.6 RJ23's four scenarios on R5 (after 3.2–3.5; the 5090, or on a stated share with 2.5)
About 20 runs: 4 scenarios × 5 seeds, with the 3.4b regime. The pre-registered gate runs on
R5's test set, seed-ensembled, with 3.2's paired bootstrap:
- **(a1)** low-resolution is below each of full, cropped and multi-scale, with the CI
  excluding 0. RJ23's gap is at least 3.5 points.
  - On failure: treat it as a pipeline bug. Re-run the 3.3 and 3.4b golden tests before any
    write-up.
- **(a2)** multi-scale is non-inferior to full, with the lower CI bound of (MS − Full)
  above −1.0 point. This is RJ23's headline claim.
  - On failure: report that it did not replicate on R5, with no tuning to rescue it.
- **(a3)** full versus cropped and the per-class recalls are reported, not gated. RJ23's
  full-over-cropped gap was carried by satellites.

## Risks
- **Comparability:** the reconstruction is not RJ23's unless 3.0a succeeds.
- **Satellites:** visual labels, so inter-reviewer agreement must be reported.
- **Moving sources:** TNS and ALeRCE re-process their data, so everything works from dated
  snapshots.
- **Era confounds:** new classes come from other years than CD21's.
- **Load and compute:** the external load on ALeRCE, and GPU contention with phase 2.5.
- **Storage:** about 3 GB for the raw labelled set; the unlabelled corpus is the next phase.
