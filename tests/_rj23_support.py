"""Synthetic 5-class ``raw-alerts`` layouts and object splits for the RJ23 regime tests."""

from __future__ import annotations

from pathlib import Path

from astro_simclr.data.raw_alerts import load_raw_alerts, write_raw_alerts
from astro_simclr.data.schema import LABEL_NAMES
from astro_simclr.data.synthetic import synthetic_raw_alerts
from astro_simclr.protocol.alerts import alert_table_from_raw
from astro_simclr.protocol.object_splits import make_object_splits, write_object_splits

#: The dataset name of the layout under ``data_dir``.
DATASET = "r5-synthetic"


def rj23_layout(root: Path, *, n_per_class: int = 20, seed: int = 0) -> tuple[Path, Path]:
    """Write a 5-class layout ``root/data/DATASET`` and its object splits ``root/splits.json``.

    The classes are R5's (CD21's five names); each alert is its own object (synthetic
    ``oid``s are unique), so 20 per class give 12/4/4 alerts per class in
    Train/Validation/Test and 2/2 in the selection and decision halves. Returns the data
    directory and the splits file.
    """
    cutouts, columns = synthetic_raw_alerts(LABEL_NAMES, n_per_class=n_per_class, seed=seed)
    data_dir = root / "data"
    path = write_raw_alerts(data_dir / DATASET, cutouts, columns, LABEL_NAMES, {"name": "syn"})
    table = alert_table_from_raw(load_raw_alerts(path))
    splits_file = root / "splits.json"
    write_object_splits(make_object_splits(table, seed=seed, name="synthetic"), splits_file)
    return data_dir, splits_file
