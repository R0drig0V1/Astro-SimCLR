"""Data: schemas, preprocessing, the pickle loader, the prepared layout and the datasets.

Submodules (``docs/data.md`` has the file schemas, ``docs/architecture.md`` the design):

- :mod:`~astro_simclr.data.schema`: constants, :class:`StampData`, the errors;
- :mod:`~astro_simclr.data.preprocessing`: the legacy stamp preprocessing functions;
- :mod:`~astro_simclr.data.pickle_loader`: the only place that unpickles;
- :mod:`~astro_simclr.data.prepared`: the ``.npy`` layout that training reads;
- :mod:`~astro_simclr.data.raw_alerts`: the ``raw-alerts`` layout (raw cutouts, a per-row
  index, any label set; phase 3);
- :mod:`~astro_simclr.data.float_input`: the paper's float input (crop 21, per-crop
  min-max, the NaN approximation);
- :mod:`~astro_simclr.data.rj23_input`: RJ23's input on raw cutouts (normalisation, mask,
  the four views, the unit-sphere position);
- :mod:`~astro_simclr.data.m1_features`: M1's feature statistics and the
  de-standardisation check;
- :mod:`~astro_simclr.data.datasets`: ``StampDataset`` and ``ContrastiveDataset`` (torch);
- :mod:`~astro_simclr.data.synthetic`: the synthetic stand-ins of the four pickles;
- :mod:`~astro_simclr.data.inspection`: what ``inspect-data`` prints.

The torch datasets are not re-exported here, so that the CLI's data commands do not pay
for importing torch; import them from :mod:`astro_simclr.data.datasets`.
"""

from astro_simclr.data.pickle_loader import load_pickle
from astro_simclr.data.prepared import load_prepared, prepare
from astro_simclr.data.schema import (
    DISCARDED_FEATURES,
    FEATURE_NAMES,
    LABEL_NAMES,
    SPLITS,
    STAMP_SHAPE,
    DataError,
    SchemaError,
    SplitData,
    StampData,
    TruncatedFileError,
)

__all__ = [
    "DISCARDED_FEATURES",
    "FEATURE_NAMES",
    "LABEL_NAMES",
    "SPLITS",
    "STAMP_SHAPE",
    "DataError",
    "SchemaError",
    "SplitData",
    "StampData",
    "TruncatedFileError",
    "load_pickle",
    "load_prepared",
    "prepare",
]
