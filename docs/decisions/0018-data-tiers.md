<!-- ADR 0018: the engine's data layer (task 4.3): one dataset API over the prepared kinds, the
     three tiers and how they stay exact, the stream definition and the shuffle granularity. -->
# 0018: The data tiers: one dataset API, exact tiers, and the shuffle granularity

- **Status:** accepted (2026-11-08, phase 4, task 4.3). Implements and amends ADR 0014,
  section 3 ("The data tiers").
- **Implemented by:** `src/astro_simclr/engine/` (`dataset.py`, `stream.py`, `tiers.py`,
  `bench.py`); `training/tiling.py` now re-exports `engine.stream`
- **Checked by:** `tests/test_engine_tiers.py` (the tiers `torch.equal` to each other on
  every accelerator present, and the `device` tier to `DeviceTensorDataModule`; exact
  resume on every tier, also onto another tier; the storage-dtype deviations),
  `tests/test_engine_stream.py`, `tests/test_engine_dataset.py`, `tests/test_engine_bench.py`

## Context

ADR 0014 froze the batch format, the `BatchSource` / `EvalSource` protocols and the three
tiers in one table: `device` (today's `DeviceTensorDataModule`), `pinned` (float32 in pinned
host memory, a per-step gather and an asynchronous upload, exact) and `memmap` (a disk
memory map read in shards, for phase 5's corpus, larger than host memory). Task 4.3 builds
them. Four questions were left open: what a tier reads (three prepared kinds exist:
classifier layouts, pretraining layouts with an unlabelled `Train`, and the `raw-alerts`
pool of phase 3); what exactly is pinned; how a disk tier reads a shuffled stream without
making each batch dozens of random reads, and what that does to the shuffle; and what a
checkpoint must record of the stream.

## Decision

1. **One dataset API.** A `Dataset` is row-aligned arrays: float32 `images` of any per-row
   shape (possibly a read-only memory map), optional in-memory `features`, `labels` (with
   `n_classes`) and `fits`, and an optional `rows` selection of the stored rows. Adapters
   build one per kind: `prepared_split` (classifier, or `unlabelled` for a pretraining
   `Train`), `raw_alerts_dataset` (with a `candid` split as a selection) and the
   constructor itself (`arrays`: the paper's crops, a `TensorSplit`'s arrays). Only float32
   stamps are served: a layout without `images_f32.npy` is refused (the uint8 file is the
   quantised input of plan finding F6). The model's input transform (channels first, the
   crop, the min-max) belongs to the task and runs on the device batch, so it is the same on
   every tier. A selection is kept, not applied, by the `memmap` tier: a split of a large
   pool copies nothing.

2. **The tiers serve identical values, so the tier is not part of the stream.** The batch
   of step `s` is the rows `TrainOrder.rows(s)` of the dataset, wherever they are:
   - `device`: the materialised dataset on the device, gathered by device indices. Each
     pass's row order is uploaded once and sliced (as `DeviceTensorDataModule` does), so a
     step makes no host-to-device copy, not even of 64 indices: on CUDA a pageable copy
     would block the host until the queue drains.
   - `pinned`: the materialised dataset in **pageable** host memory; the per-step gather
     writes into one of two **pinned staging buffers**, uploaded with `non_blocking=True` on
     the device's current stream, one step ahead (the next batch is gathered and enqueued
     when the current one is handed out). A buffer is refilled only after the event
     recorded behind its previous upload has completed. **Amendment of ADR 0014's table:**
     the dataset itself is not pinned. The gather is a CPU copy that pinning does not speed
     up, pinning is what makes the *upload* asynchronous, and pinning the 2.4 GB of `red`
     would only lock it out of the page cache.
   - `memmap`: the images stay in their file; a batch's rows are read sorted by file offset
     (one forward sweep through the blocks it touches) and scattered into the pinned
     staging buffer at their batch positions, then uploaded as in `pinned`; the small
     columns are in memory.

   The copy is ordered before the compute that reads it on the same stream, and every tier
   copies the same float32 bytes, so the batches are `torch.equal` (tested on CPU and MPS;
   the CUDA parametrisation runs where CUDA is present). On a CPU device nothing is staged,
   pinned or prefetched: the host gather is the batch. A run may therefore resume on another
   tier and stay exact (tested).

3. **The stream definition and the resume.** `TrainSource.state()` is the definition a
   checkpoint stores in its `stream` key (ADR 0014): the stream's `n`, `batch_size` and
   `seed` (the keys of the 2.2a Lightning loader), the shuffle's parameters when not `row`,
   `rows_sha256` (the SHA-256 of the position → row map, which catches a resume on other
   labels with the same tiled size, a case the Lightning loader's size check let through)
   and the `storage_dtype`. The only mutable state is the position, the engine's
   `steps_done`: `resume(state, steps_done)` checks the definition (`DataError` otherwise)
   and expects that step next. Any step can still be asked for out of order; it is then
   fetched synchronously, with the same values.

4. **The shuffle granularity is a stream choice, recorded, never a tier side effect.**
   - `row` (the default, the paper's, and the only one a class-balanced stream accepts):
     each pass a uniform permutation of every position (ALeRCE's TF shuffle with a buffer
     larger than the data). On the `memmap` tier a batch is `batch_size` scattered reads.
   - `block`: positions cut into blocks of `block_rows`; each pass permutes the blocks,
     then shuffles positions within each window of `window_blocks` consecutive blocks of
     that order, so a batch reads at most two windows of contiguous file ranges. Items of
     one block always share a window of the pass, so neighbours in the file (alerts of one
     night) share batches far more often than under `row`. Refused with `balance`, whose
     tiled order is sorted by class. Its generator is `default_rng([seed, pass, 5])`, the new
     `RESERVED_STREAMS["block_shuffle"]`. Phase 5 decides whether its corpus needs it (a
     cold-disk measurement on the GPU machine), and records the choice.

   Shuffling is applied on the stream for every tier: a `block` run on the `device` tier
   has the `block` batches. Exactness across tiers holds for either granularity.

5. **Quantised storage stays an explicit, recorded flag.** `TierChoice.storage_dtype`
   `float16` or `uint8` quantises the images once, when the tier is built (`uint8` stores
   `rint(255 x)` and needs finite images in [0, 1], else `DataError`), and dequantises them
   on the device after each gather (`float16 → float32`; `uint8 → k / 255`, correctly
   rounded, gathered from a host-computed table so the bits do not depend on the device);
   features and fits stay float32. It is part of the stream definition (a resume under another dtype
   is refused) and of `TrainSource.describe()`, the manifest record (`deviation: true`). The
   `memmap` tier serves the file's float32 only (`ConfigError` otherwise): quantising a file
   larger than memory is a conversion of the file, not a load-time flag.

6. **The engine owns the stream.** `tile_indices` and `PermutationStream` moved from
   `training/tiling.py` (which re-exports them) to `engine/stream.py`, because importing
   `astro_simclr.training` imports Lightning and the engine must not.

## Consequences

- One data path for the three kinds, and the tier becomes a deployment choice
  (memory versus upload bandwidth) that cannot change a result.
- The throughput microbenchmark (`python -m astro_simclr.engine.bench`) measures the
  tiers' own overhead from the page cache, interleaved across tiers (the development Mac
  is shared); the measured numbers are in `docs/architecture.md`. The cold-disk number
  for phase 5's corpus is a GPU-machine measurement, not taken here.
- **Seed coupling found, not changed:** the `row` stream's pass `j` and the `rot90`
  augmentation of step `j` both draw from `default_rng([seed, j])` (`training/modules.py`,
  task 2.5-models), so their numbers are not independent. Changing either changes the
  replication's numbers (a `numerics_epoch` bump, ADR 0015), so it is left to an
  explicit decision; new streams take reserved third words.
- The tiers do not wire into a loop yet: 4.1's loop and 4.6's tasks consume `TrainSource`
  and `EvalBatches`, and 4.6 writes `describe()` into the manifest.
- C5-C7 (the class-aware augmentation and the flux rescaling) are not ported here; their
  prefetch is task `perf/augment-prefetch`'s.
