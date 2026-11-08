"""The unlabelled pretraining corpus from the ZTF public alert archive (``ingest-archive``).

The archive (``https://ztf.uw.edu/alerts/public/``) holds one ``ztf_public_YYYYMMDD.tar.gz``
per night with one avro alert per member. This package turns tarballs (files, or one piped
in) into the corpus that self-supervised pretraining reads (phase 5, task 5.1a): the first
alerts, at 63x63, of the objects that no evaluation set holds, as raw float32 cutouts (NaN
kept), one shard per night. It never downloads anything: the tarballs are fetched
separately, after the user approves the bulk download.

Submodules (``docs/data.md`` "Archive corpus" has the layout and the commands,
``docs/architecture.md`` the design, ADR 0016 the decisions):

- :mod:`~astro_simclr.archive.stream`: one tarball read as a stream, hashed and bounded;
- :mod:`~astro_simclr.archive.selection`: the first-alert rules and the exclusions;
- :mod:`~astro_simclr.archive.layout`: the per-night shards, written atomically, and the
  corpus index (one alert per object);
- :mod:`~astro_simclr.archive.ingest`: the resumable ingest;
- :mod:`~astro_simclr.archive.plan`: the seeded night sampling plan and its size estimate.

The tarballs are untrusted: nothing is extracted to disk, every alert goes through the
size-limited parser of :mod:`astro_simclr.alerts.parse`, and nothing is unpickled.
"""
