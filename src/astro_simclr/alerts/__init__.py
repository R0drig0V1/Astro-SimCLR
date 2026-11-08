"""Raw alert retrieval from ALeRCE (task 3.1, ``fetch-alerts``).

The prepared stamps are already normalised (per-channel min-max over 63x63, NaN -> 0). This
package recovers the raw alerts behind them: for each row of a prepared split it finds the
alert on ALeRCE's public services from the row's de-standardised features, downloads it,
checks that ALeRCE's normalisation of its raw cutouts reproduces the stored stamp, and
keeps the raw cutouts (NaN included) and the candidate record in a sharded store.

Submodules (``docs/data.md`` "Raw alerts" has the store layout and the provenance,
``docs/architecture.md`` the design):

- :mod:`~astro_simclr.alerts.fits`: a minimal, strict reader of the gzipped FITS cutouts;
- :mod:`~astro_simclr.alerts.transport`: the rate limiter and the retrying HTTP client;
- :mod:`~astro_simclr.alerts.alerce`: ALeRCE's endpoints and the validation of their JSON;
- :mod:`~astro_simclr.alerts.parse`: the avro alert (fastavro) to an :class:`Alert`;
- :mod:`~astro_simclr.alerts.store`: the index, the shards and the summary;
- :mod:`~astro_simclr.alerts.fetch`: matching, verification and the resumable run.

Downloaded bytes are untrusted: they are parsed as JSON, avro and FITS with explicit size
limits and type checks, never executed and never unpickled.
"""
