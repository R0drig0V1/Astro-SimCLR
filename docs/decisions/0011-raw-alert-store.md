<!-- ADR 0011: how fetch-alerts retrieves, verifies and stores the raw alerts of task 3.1, and why. -->
# 0011: Raw alert retrieval and its store (task 3.1)

- **Status:** accepted (2026-11-08, phase 3, task 3.1; includes the plan-review amendments
  of rounds 1 and 2)
- **Reviewed by:** a `code-reviewer` pass on 2026-11-08. Its findings (commit
  idempotency, hostile JSON numbers, unbounded schemas, the cache bound, `Retry-After`, the
  read deadline, directory fsync, final 4xx) are fixed and tested.
- **Implemented by:** `src/astro_simclr/alerts/` (`transport`, `snapshot`, `alerce`,
  `parse`, `fits`, `fetch`, `store`), `fetch-alerts` and `alerts-summary` in
  `src/astro_simclr/cli.py`
- **Checked by:** `tests/test_alerts_fetch.py`, `tests/test_alerts_io.py`,
  `tests/test_alerts_store.py`, `tests/test_alerts_transport.py` (offline); the live smoke
  run in `docs/data.md`, "Raw alerts"

## Context

Phase 3 and several phase-2 rows need the raw alerts behind the normalised stamps:
- the real NaN and saturation masks;
- other normalisations;
- the `oid`/`candid` of every labelled row, for object-level splits and leakage exclusion.

The integrator's probe showed they can be recovered from ALeRCE's public API and avro
service, and verified exactly against the stored stamps. The full `red` set is 51,582 rows
and about 160k requests on a shared public service, so the run lasts 1–2 days unattended
on the GPU machine (WSL2), and it will be interrupted.

## Decision

1. **Download the whole avro, parsed with `fastavro[snappy]`**, not the three `get_stamp`
   FITS files.
   - A row costs about 3 requests (cone search, detections, avro) instead of about 5, which
     saves roughly 40 % of a day-long run against a shared service.
   - The avro carries the candidate record (`jd`, `drb`, `xpos`/`ypos` for the CCD-edge
     repair) that the API's detection list lacks.
   - fastavro is a maintained, widely used reader. ALeRCE serves the files with the snappy
     codec, so its `snappy` extra (cramjam) is needed too.
   - The container is checked before fastavro decompresses anything: the codec must be
     `null`, `deflate` or `snappy`, and the decompressed size is capped.
   - The record is validated by hand rather than with a fixed reader schema, because
     ALeRCE's record names differ from ZTF's.
2. **A minimal, strict FITS reader** instead of astropy. It reads one 2D float image, with
   explicit limits; ZTF cutouts use only that layout. A FITS library would be a large
   dependency for one layout.
3. **Ties are decided by the stamp check.** Every matching `candid` is downloaded, up to
   16, and ALeRCE's min-max must reproduce the stored stamp to ≤ 1e-6. One pass is `ok`,
   several are `ambiguous`, none is `mismatch`.
   - Only `ok` rows are stored, so no stored row fails the check.
   - The tolerances are never widened.
4. **Ghosts:** a row whose cone search is empty is `ghost_candidate` (RJ23's "ghost bogus":
   objects that are not in ALeRCE's database).
   - It is final: no retry beyond the transport's backoff.
   - It is left out of the 98 %-per-class recovery target and counted apart.
   - Its de-standardised position stays in the index.
5. **The store is npz batch shards plus an append-only CSV index.**
   - A batch of 100 rows is committed as one shard: temporary file, fsync, rename. Only
     then are its index lines appended and fsynced. The last line of a row wins.
   - Resume skips final statuses and retries only `http_error`.
   - Shards load with `allow_pickle=False`; the records are JSON text in uint8 arrays.
   - Parquet was rejected: pyarrow is a large dependency for 52k small rows. One file per
     response was rejected: about 200k files.
6. **Every raw response is kept** (bytes, URL, kind, UTC time, SHA-256) in the shard of
   its batch.
   - This makes the index rebuildable offline (`--from-snapshot`), every line traceable to
     the bytes it was decided on, and it stores the full detection list of each object for
     the phase-3 reconstruction.
   - Cone searches and detection lists are cached in memory, because rows share objects.
7. **Throttling is ours.** The API sends no rate-limit headers.
   - One rate limiter covers both hosts. Timeouts, 5xx and 429 get exponential backoff,
     and `Retry-After` is honoured up to 10 minutes; other statuses are final.
   - The run pauses for 300 s after 3 consecutive transient failures.
   - Each response has a 300 s total deadline, and redirects stay on https.
   - The `User-Agent` is generic.
8. **Signals stop the run between rows.** SIGTERM, SIGHUP and Ctrl-C set a flag that the
   run polls between rows, so a commit is never cut. A full batch is detached before its
   commit, so a failed commit (ENOSPC) is never written a second time. Its responses stay
   pending until a commit makes them durable, so the snapshot keeps every body that an
   index line references.
9. **Untrusted avro schemas are bounded.** Besides the decompressed-size cap, a writer
   schema with an array or map of zero-size items is refused. With that rule, a 177-byte
   file can no longer declare 10⁹ `null` items, and decoding work stays bounded by the
   decompressed size.

## Consequences

- Storage is about 94 KB per row, about 4.9 GB for `red`. The plan estimated about 3 GB
  for the cutouts alone; the snapshot (avro files and detection lists) is the difference.
  An option to skip the snapshot can be added if space matters.
- The run is sequential and latency-bound: 1.29 s per request on the smoke run, so about
  58 h for `red` at any `--rate` near 1. Concurrency, within the same global limit, would
  be the next step if that is too slow; it is not needed for correctness.
- A crash loses at most one batch. A crash between a shard's rename and its index lines
  leaves an orphan shard, which costs space only (its rows are fetched again).
- The in-memory cache is lost on restart, so shared objects are fetched again after a
  resume.
- New runtime dependencies: `fastavro` and `cramjam` (through `fastavro[snappy]`).
