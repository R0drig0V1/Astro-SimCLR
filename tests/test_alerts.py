"""Tests of the alert table: the CSV schema (untrusted input), the fingerprint, first stamps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _bench_support import synthetic_table

from astro_simclr.data.schema import DataError
from astro_simclr.protocol.alerts import (
    BENCHMARK_LABEL_NAMES,
    AlertTable,
    LabelConflict,
    first_stamps,
    label_conflicts,
    read_alert_table,
    write_alert_table,
)

HEADER = "oid,candid,jd,label\n"


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "alerts.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_round_trip_and_order_independent_fingerprint(tmp_path: Path) -> None:
    table = synthetic_table(50, seed=1)
    path = tmp_path / "a.csv"
    write_alert_table(table, path)
    again = read_alert_table(path)
    assert again.sha256 == table.sha256 and len(again) == len(table)
    np.testing.assert_array_equal(again.jd, table.jd)
    assert again.group_key == "oid" and np.array_equal(again.group, again.oid)
    # The same rows in another order, with extra columns (3.1's index), are the same table.
    lines = path.read_text().splitlines()
    shuffled = ["fid,split," + lines[0], *(f"1,Test,{line}" for line in reversed(lines[1:]))]
    other = _write(tmp_path, "\n".join(shuffled) + "\n")
    assert read_alert_table(other).sha256 == table.sha256
    # A changed jd changes the fingerprint.
    changed = AlertTable.from_rows(
        table.oid.tolist(),
        table.candid.tolist(),
        (table.jd + 1e-6).tolist(),
        table.label.tolist(),
        BENCHMARK_LABEL_NAMES,
    )
    assert changed.sha256 != table.sha256


def test_group_column(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "oid,candid,jd,label,group\nZTF1a,2,2458000.5,SN,pos7\nZTF1b,1,2458001.5,SN,pos7\n",
    )
    table = read_alert_table(path)
    assert table.group_key == "group" and table.group.tolist() == ["pos7", "pos7"]
    assert table.candid.tolist() == [1, 2]
    out = tmp_path / "out.csv"
    write_alert_table(table, out)
    assert out.read_text().splitlines()[0] == "oid,candid,jd,label,group"
    assert read_alert_table(out).sha256 == table.sha256
    plain = AlertTable.from_rows(
        ["ZTF1b", "ZTF1a"], [1, 2], [2458001.5, 2458000.5], [1, 1], BENCHMARK_LABEL_NAMES
    )
    assert plain.sha256 != table.sha256  # the group is part of the fingerprint
    with pytest.raises(DataError, match=":3: missing group"):
        read_alert_table(_write(tmp_path, "oid,candid,jd,label,group\nA,1,2.0,SN,g\nB,2,2.0,SN\n"))
    with pytest.raises(DataError, match="invalid group"):
        read_alert_table(_write(tmp_path, "oid,candid,jd,label,group\nA,1,2.0,SN,a b\n"))


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("ZTF 1,1,2458000.5,SN", "invalid oid"),
        (",1,2458000.5,SN", "invalid oid"),
        ("ZTF1,-1,2458000.5,SN", "candid must be a positive integer"),
        ("ZTF1,1e5,2458000.5,SN", "candid must be a positive integer"),
        ("ZTF1,0,2458000.5,SN", "candid must be a positive integer"),
        ("ZTF1,99999999999999999999,2458000.5,SN", "candid must be a positive integer"),
        ("ZTF1,1,abc,SN", "is not a number"),
        ("ZTF1,1,nan,SN", "finite and positive"),
        ("ZTF1,1,-3,SN", "finite and positive"),
        ("ZTF1,1,2458000.5,Galaxy", "unknown label 'Galaxy'"),
        ("ZTF1,1,2458000.5", "missing label"),
    ],
)
def test_schema_errors_name_the_line(tmp_path: Path, row: str, message: str) -> None:
    path = _write(tmp_path, HEADER + "ZTF0,7,2458000.5,AGN\n" + row + "\n")
    with pytest.raises(DataError, match=f"alerts.csv:3: .*{message}"):
        read_alert_table(path)


def test_file_level_errors(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="missing columns"):
        read_alert_table(_write(tmp_path, "oid,candid,label\nA,1,SN\n"))
    with pytest.raises(DataError, match="appears more than once"):
        read_alert_table(_write(tmp_path, HEADER + "A,1,2.0,SN\nB,1,3.0,SN\n"))
    with pytest.raises(DataError, match="N > 0"):
        read_alert_table(_write(tmp_path, HEADER))
    with pytest.raises(DataError, match="cannot read"):
        read_alert_table(tmp_path / "missing.csv")
    binary = tmp_path / "binary.csv"
    binary.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(DataError, match="cannot read"):
        read_alert_table(binary)


def test_from_rows_checks() -> None:
    names = ("A", "B")
    with pytest.raises(DataError, match="2 distinct"):
        AlertTable.from_rows(["x"], [1], [1.0], [0], ("A", "A"))
    with pytest.raises(DataError, match="N > 0"):
        AlertTable.from_rows(["x"], [1, 2], [1.0], [0], names)
    with pytest.raises(DataError, match="candid must be positive"):
        AlertTable.from_rows(["x"], [0], [1.0], [0], names)
    with pytest.raises(DataError, match="jd must be"):
        AlertTable.from_rows(["x"], [1], [float("inf")], [0], names)
    with pytest.raises(DataError, match="labels must be"):
        AlertTable.from_rows(["x"], [1], [1.0], [2], names)
    with pytest.raises(DataError, match="one value per row"):
        AlertTable.from_rows(["x"], [1], [1.0], [0], names, group=["a", "b"])


def test_rows_of() -> None:
    table = AlertTable.from_rows(
        ["a", "b", "c"], [30, 10, 20], [1.0, 2.0, 3.0], [0, 1, 0], ("A", "B")
    )
    assert table.rows_of([20, 30, 10]).tolist() == [1, 2, 0]
    with pytest.raises(DataError, match="candid 40"):
        table.rows_of([40])
    with pytest.raises(DataError, match="candid 5"):
        table.rows_of([5])


def test_label_conflicts() -> None:
    table = AlertTable.from_rows(
        ["o2", "o1", "o1", "o2", "o3"], [1, 2, 3, 4, 5], [1.0] * 5, [2, 0, 0, 0, 1], ("A", "B", "C")
    )
    assert label_conflicts(table) == (LabelConflict("o2", ("A", "C")),)


def test_first_stamps_earliest_jd_then_candid() -> None:
    table = AlertTable.from_rows(
        ["o1", "o1", "o1", "o2", "o2", "o3"],
        [10, 11, 12, 20, 21, 30],
        [5.0, 3.0, 3.0, 9.0, 8.0, 1.0],
        [0, 0, 0, 1, 1, 0],
        ("A", "B"),
    )
    # o1: jd 3.0 twice -> smaller candid 11 (row 1); o2: row 4; o3: row 5.
    assert first_stamps(table).tolist() == [1, 4, 5]
    # Restricted to some rows: the earliest among them.
    assert first_stamps(table, [0, 2, 3]).tolist() == [2, 3]
    assert first_stamps(table, []).size == 0
