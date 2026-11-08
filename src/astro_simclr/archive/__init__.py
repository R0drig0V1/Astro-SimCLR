"""The unlabelled pretraining corpus from the ZTF public alert archive (``ingest-archive``).

The archive (``https://ztf.uw.edu/alerts/public/``) holds one ``ztf_public_YYYYMMDD.tar.gz``
per night with one avro alert per member. This package turns tarballs (files, or one piped
in) into the corpus that self-supervised pretraining reads (phase 5, task 5.1a): the first
alerts, at 63x63, of the objects that no evaluation set holds, as raw float32 cutouts (NaN
kept), one shard per night. It never downloads anything: the tarballs are fetched
separately, after the user approves the bulk download.

Submodules (``docs/data.md`` "Archive corpus" has the layout and the commands,
``docs/architecture.md`` the design, ADR 0019 the decisions):

- :mod:`~astro_simclr.archive.stream`: one tarball read as a stream, hashed and bounded;
- :mod:`~astro_simclr.archive.selection`: the first-alert rules and the exclusions;
- :mod:`~astro_simclr.archive.layout`: the per-night shards, written atomically, and the
  corpus index (one alert per object);
- :mod:`~astro_simclr.archive.ingest`: the resumable ingest;
- :mod:`~astro_simclr.archive.plan`: the seeded night sampling plan and its size estimate;
- :mod:`~astro_simclr.archive.corpus_index`: the corpus index of task 5.1, the exclusion
  union (listed candids and oids, 1.5" of any CD21 alert, near-duplicate stamps) and one
  alert per object, with the SHA-256 of every exclusion input;
- :mod:`~astro_simclr.archive.near_duplicates`: the pixel check against the CD21 stamps;
- :mod:`~astro_simclr.archive.reader`: the indexed stamps, memory-mapped from the night
  shards and normalised like the stored stamps (what ``simclr-pilot`` reads);
- :mod:`~astro_simclr.archive.commands`: the ``build-corpus-index`` subcommand.

The tarballs are untrusted: nothing is extracted to disk, every alert goes through the
size-limited parser of :mod:`astro_simclr.alerts.parse`, and nothing is unpickled.
"""
