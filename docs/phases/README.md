<!-- Roadmap index; owned by the integrator. -->
# Development plan

The roadmap, one phase per file in this folder. Owned by the integrator: development
sessions list completed tasks in their PR body instead of ticking them here.

| Phase | File | Status |
|-------|------|--------|
| 1. Runnable, tested package (refactor legacy code into `src/astro_simclr/`) | [01-runnable-package.md](01-runnable-package.md) | in progress (1.1–1.7 done; 1.9a remains, 1.9b optional) |
| 2. Replicate and improve the stamp classifier (Carrasco-Davis 2021) | [02-stamp-classifier.md](02-stamp-classifier.md) | in progress: replication **passed** (CD21 faithful, test 0.943 ± 0.004); 2.5 starting |

**Priority (user decision, 2026-11-07):** the goal is to replicate the paper (Carrasco-Davis
et al. 2021), not the 2022 legacy numbers. Phase 2 is the main line. In phase 1, task 1.7
(CLI, preflight) is done; 1.9a (runbook) is needed only for cloud GPUs. The legacy reference runs R1–R4
(task 1.9b) are optional: they are run only if a legacy comparison is ever needed.

## Next phase (not started): self-supervised pretraining data

Findings of 2026-11-08 (integrator probes). Nothing is downloaded yet: the user decided to
wait.

- **Raw stamps of the labelled set can be recovered.**
  1. De-standardise `ra`, `dec`, `magpsf` and `sigmapsf` with M1's statistics.
  2. Search ALeRCE's public API: `api.alerce.online/ztf/v1/objects/?ra&dec&radius=1.5`, then
     `/objects/{oid}/detections`, matched on `magpsf` and `sigmapsf` to give the `candid`.
  3. Download the stamps from `avro.alerce.online/get_stamp` (FITS) or `get_avro` (the whole
     alert).

  Result: 9 of 10 probe stamps (2 per class, 2018–2020) were recovered, and the tenth timed
  out. After ALeRCE's per-channel min-max they equal our stored stamps to ≤ 1e-7. The raw
  stamps keep the real NaN and saturation masks: one probe Asteroid had 5,142 NaN pixels,
  which the stored stamps hold as 0.
  - Cost: about 52.6k alerts, about 3 requests each, 1–2 requests/s, about 20–24 h.
  - Uses:
    - Poisson noise;
    - fake SN injection (row C5/C6 follow-ups);
    - other normalisations (row C1);
    - the `candid` exclusion list for any pretraining corpus.
- **The unlabelled corpus comes from the ZTF public alert archive** (`ztf.uw.edu/alerts/public/`):
  - 3,065 nightly `.tar.gz` files from 2018-06-01 to 2026-11-07, about 2.4 GB per 2020
    night;
  - every alert is an avro with the three raw 63×63 cutouts;
  - bulk files, so no load on ALeRCE's services.

  Today's `simclr_300` has about 300k normalised stamps; the archive allows 10–100× more,
  raw. Design notes:
  1. Take first detections (low `ndethist`) with ALeRCE's quality cuts, matching the
     stamp classifier's input.
  2. Exclude every `oid`/`candid` of our Validation and Test.
  3. Sample nights from 2018–2020 (the labelled era), and later years to measure the
     domain shift (useful for Rubin).
  4. Storage: about 48 GB float32 (24 GB float16) per million stamps, under
     `~/Astro-SimCLR/dataset` on the GPU machine (user decision).
