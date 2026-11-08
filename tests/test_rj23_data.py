"""Tests of the parts of an RJ23 run (astro_simclr.rj23_data, task 3.4b)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from _rj23_support import DATASET, rj23_layout

from astro_simclr.data.raw_alerts import RawAlerts, load_raw_alerts
from astro_simclr.data.rj23_input import VIEW_NAMES, VIEW_SHAPES, ViewName
from astro_simclr.data.schema import DataError
from astro_simclr.protocol.alerts import AlertTable, alert_table_from_raw
from astro_simclr.protocol.object_splits import ObjectSplits, make_object_splits, read_object_splits
from astro_simclr.protocol.splits import SplitsError
from astro_simclr.rj23_data import MODE_PARTS, RJ23Part, RunPlan, rj23_parts, rj23_rows


@pytest.fixture(scope="module")
def layout(tmp_path_factory: pytest.TempPathFactory) -> tuple[RawAlerts, ObjectSplits]:
    data_dir, splits_file = rj23_layout(tmp_path_factory.mktemp("rj23-data"))
    return load_raw_alerts(data_dir / DATASET), read_object_splits(splits_file)


@pytest.mark.parametrize("scenario", VIEW_NAMES)
def test_replication_parts(layout: tuple[RawAlerts, ObjectSplits], scenario: ViewName) -> None:
    data, splits = layout
    parts = rj23_parts(data, splits, scenario)
    assert list(parts) == ["Train", "Validation", "Test"]
    height, width, channels = VIEW_SHAPES[scenario]
    table = alert_table_from_raw(data)
    for name, part in parts.items():
        assert part.name == name
        assert part.images.dtype == np.float32
        assert part.images.shape == (len(part), channels, height, width)
        assert part.positions.shape == (len(part), 3)
        np.testing.assert_allclose(np.linalg.norm(part.positions, axis=1), 1.0, atol=1e-6)
        assert np.all(np.diff(part.candids) > 0)
        counts = np.bincount(part.labels, minlength=5).tolist()
        assert counts == splits.counts[name]["alerts"]
        np.testing.assert_array_equal(part.candids, table.candid[splits.rows(table, name)])
    every = np.concatenate([p.candids for p in parts.values()])
    assert np.unique(every).size == len(data)  # every alert in exactly one split


def test_decision_parts_never_read_test(layout: tuple[RawAlerts, ObjectSplits]) -> None:
    data, splits = layout
    parts = rj23_parts(data, splits, "lowres", mode="decision")
    assert list(parts) == ["Train", "selection", "decision"]
    validation = rj23_parts(data, splits, "lowres")["Validation"].candids
    halves = np.concatenate([parts["selection"].candids, parts["decision"].candids])
    np.testing.assert_array_equal(np.sort(halves), validation)


def test_plans() -> None:
    assert MODE_PARTS["replication"].selection == "Validation"
    assert MODE_PARTS["replication"].evaluated == (("Validation", "val"), ("Test", "test"))
    assert MODE_PARTS["decision"].selection == "selection"
    assert MODE_PARTS["decision"].parts == ("Train", "selection", "decision")
    assert RunPlan("A", "A", (("A", "a"),)).parts == ("A",)


def test_views_follow_large_values(layout: tuple[RawAlerts, ObjectSplits]) -> None:
    data, splits = layout
    paper = rj23_parts(data, splits, "full")["Train"]
    code = rj23_parts(data, splits, "full", large_values="code")["Train"]
    # The synthetic stamps have no |x| > 1e10 pixel: both rules agree on them.
    np.testing.assert_array_equal(paper.images, code.images)


def _relabelled(data: RawAlerts) -> AlertTable:
    table = alert_table_from_raw(data)
    labels = table.label.copy()
    labels[0] = (labels[0] + 1) % 5
    return dataclasses.replace(table, label=labels)


def test_a_csv_table_must_fit_the_layout(layout: tuple[RawAlerts, ObjectSplits]) -> None:
    data, _ = layout
    table = _relabelled(data)
    splits = make_object_splits(table, seed=0)
    with pytest.raises(DataError, match="in the layout but"):
        rj23_parts(data, splits, "cropped", table=table)


def test_an_alert_missing_from_the_layout(
    layout: tuple[RawAlerts, ObjectSplits], tmp_path: Path
) -> None:
    data, _ = layout
    table = alert_table_from_raw(data)
    bigger = AlertTable.from_rows(
        [*table.oid.tolist(), "ZTF99extra"],
        [*table.candid.tolist(), int(table.candid.max()) + 7],
        [*table.jd.tolist(), 2459000.5],
        [*table.label.tolist(), 0],
        table.label_names,
    )
    splits = make_object_splits(bigger, seed=0)
    with pytest.raises(DataError, match="is not in the layout"):
        for part in ("Train", "Validation", "Test"):
            rj23_rows(data, bigger, splits, part)


def test_splits_of_another_table_are_refused(layout: tuple[RawAlerts, ObjectSplits]) -> None:
    data, _ = layout
    other = make_object_splits(_relabelled(data), seed=0)
    with pytest.raises(SplitsError, match="another alert table"):
        rj23_parts(data, other, "full")


def test_splits_of_other_classes_are_refused(layout: tuple[RawAlerts, ObjectSplits]) -> None:
    data, splits = layout
    renamed = dataclasses.replace(splits, label_names=("a", "b", "c", "d", "e"))
    with pytest.raises(DataError, match="classes"):
        rj23_parts(data, renamed, "full")


def test_unknown_scenario_or_mode(layout: tuple[RawAlerts, ObjectSplits]) -> None:
    data, splits = layout
    with pytest.raises(ValueError, match="scenario"):
        rj23_parts(data, splits, "huge")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mode"):
        rj23_parts(data, splits, "full", mode="cv")  # type: ignore[arg-type]


def test_part_checks_its_lengths() -> None:
    images = np.zeros((2, 6, 16, 16), dtype=np.float32)
    positions = np.zeros((2, 3), dtype=np.float32)
    labels = np.zeros(2, dtype=np.int64)
    with pytest.raises(ValueError, match="N > 0"):
        RJ23Part("x", images, positions, labels, np.zeros(3, dtype=np.int64))
    with pytest.raises(ValueError, match="N > 0"):
        RJ23Part("x", images[:0], positions[:0], labels[:0], labels[:0])
