"""Synthetic ZTF archive tarballs for the archive-corpus tests (offline, made here).

:func:`archive_alert` writes one avro alert with a ZTF-like schema (top-level ``candid``,
the candidate fields the selection reads, ``prv_candidates`` as records whose ``candid``
is null for upper limits) and the three FITS cutouts of ``_alerts_support``. The schema
follows the alert's era (:data:`SCHEMA_ERAS`, the "schema fixtures per year" of task
5.1a): 2018 alerts have no ``drb`` (ZTF added it in schema 3.3), 2024 ones carry the
forced-photometry history ``fp_hists`` of schema 4.x, which the parser must skip.
:func:`write_tarball` packs members into ``ztf_public_YYYYMMDD.tar.gz``, including the
hostile kinds a downloaded tar may hold (links, directories, absolute and ``..`` names).
Nothing is downloaded.
"""

from __future__ import annotations

import io
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fastavro
import numpy as np
import numpy.typing as npt
from _alerts_support import fits_bytes, raw_cutouts

_CUTOUT = {
    "type": "record",
    "name": "cutout",
    "fields": [
        {"name": "fileName", "type": "string"},
        {"name": "stampData", "type": "bytes"},
    ],
}
_CANDIDATE_FIELDS = [
    ("candid", "long"),
    ("jd", "double"),
    ("fid", "int"),
    ("magpsf", "float"),
    ("sigmapsf", "float"),
    ("isdiffpos", "string"),
    ("ra", "double"),
    ("dec", "double"),
    ("ndethist", "int"),
    ("rb", "float"),
    ("drb", "float"),
    ("nbad", "int"),
    ("fwhm", "float"),
    ("elong", "float"),
    ("magdiff", "float"),
    ("xpos", "float"),
    ("ypos", "float"),
]
#: The schema eras of the fixtures: version, candidate fields dropped, extra top-level field.
SCHEMA_ERAS: dict[int, tuple[str, tuple[str, ...], bool]] = {
    2018: ("3.1", ("drb",), False),
    2020: ("3.3", (), False),
    2024: ("4.02", (), True),
}
_FP_HIST = {
    "name": "fp_hists",
    "type": [
        "null",
        {
            "type": "array",
            "items": {
                "type": "record",
                "name": "fp_hist",
                "fields": [
                    {"name": "jd", "type": "double"},
                    {"name": "forcediffimflux", "type": ["null", "float"]},
                ],
            },
        },
    ],
}


def _schema(era: int) -> dict[str, Any]:
    """A ZTF-like alert schema of ``era`` (field names of ZTF's ``ztf.alert``, a subset)."""
    _, dropped, with_fp = SCHEMA_ERAS[era]
    schema = _base_schema()
    fields = [(n, t) for n, t in _CANDIDATE_FIELDS if n not in dropped]
    schema["fields"][3]["type"]["fields"] = [{"name": n, "type": ["null", t]} for n, t in fields]
    if with_fp:
        schema["fields"].append(_FP_HIST)
    return schema


def _base_schema() -> dict[str, Any]:
    return {
        "type": "record",
        "name": "alert",
        "namespace": "ztf",
        "fields": [
            {"name": "schemavsn", "type": "string"},
            {"name": "objectId", "type": "string"},
            {"name": "candid", "type": "long"},
            {
                "name": "candidate",
                "type": {
                    "type": "record",
                    "name": "candidate",
                    "fields": [{"name": n, "type": ["null", t]} for n, t in _CANDIDATE_FIELDS],
                },
            },
            {
                "name": "prv_candidates",
                "type": [
                    "null",
                    {
                        "type": "array",
                        "items": {
                            "type": "record",
                            "name": "prv_candidate",
                            "fields": [
                                {"name": "jd", "type": "double"},
                                {"name": "candid", "type": ["null", "long"]},
                                {"name": "magpsf", "type": ["null", "float"]},
                            ],
                        },
                    },
                ],
            },
            {"name": "cutoutScience", "type": ["null", _CUTOUT]},
            {"name": "cutoutTemplate", "type": ["null", "cutout"]},
            {"name": "cutoutDifference", "type": ["null", "cutout"]},
        ],
    }


_PARSED = {era: fastavro.parse_schema(_schema(era)) for era in SCHEMA_ERAS}


@dataclass
class Spec:
    """One synthetic alert: identity, candidate overrides, history and cutouts."""

    oid: str
    candid: int
    seed: int = 0
    candidate: dict[str, Any] = field(default_factory=dict)
    previous: list[int | None] = field(default_factory=list)  # candids; None = upper limit
    shape: tuple[int, int] = (63, 63)
    nans: int = 0
    era: int = 2020

    def cutouts(self) -> npt.NDArray[np.float32]:
        return raw_cutouts(self.seed, self.shape, self.nans)


#: A Julian date of 2020-01-15 UT (the oids of the fixtures are ``ZTF20...``).
JD_2020: float = 2458863.5


def candidate(candid: int, **overrides: Any) -> dict[str, Any]:
    """A first-detection candidate of 2020; ``overrides`` replace fields."""
    record: dict[str, Any] = {
        "candid": candid,
        "jd": JD_2020 + candid * 1e-6,
        "fid": 1,
        "magpsf": 18.5,
        "sigmapsf": 0.05,
        "isdiffpos": "t",
        "ra": 150.0 + candid % 100 * 0.01,
        "dec": 2.0,
        "ndethist": 1,
        "rb": 0.9,
        "drb": 0.99,
        "nbad": 0,
        "fwhm": 2.5,
        "elong": 1.05,
        "magdiff": 0.02,
        "xpos": 1500.0,
        "ypos": 1500.0,
    }
    record.update(overrides)
    return record


def archive_alert(spec: Spec) -> bytes:
    """The avro bytes of ``spec`` (null codec, as ZTF writes them)."""
    cutouts = spec.cutouts()
    stamps = [
        {"fileName": f"{spec.candid}_{name}.fits.gz", "stampData": fits_bytes(cutouts[:, :, k])}
        for k, name in enumerate(("scimref", "ref", "diff"))
    ]
    version, dropped, with_fp = SCHEMA_ERAS[spec.era]
    record = candidate(spec.candid, **spec.candidate)
    alert: dict[str, Any] = {
        "schemavsn": version,
        "objectId": spec.oid,
        "candid": spec.candid,
        "candidate": {k: v for k, v in record.items() if k not in dropped},
        "prv_candidates": [
            {"jd": 2458840.5 + i, "candid": c, "magpsf": None if c is None else 19.0}
            for i, c in enumerate(spec.previous)
        ]
        or None,
        "cutoutScience": stamps[0],
        "cutoutTemplate": stamps[1],
        "cutoutDifference": stamps[2],
    }
    if with_fp:
        alert["fp_hists"] = [{"jd": 2460000.5, "forcediffimflux": 12.5}]
    out = io.BytesIO()
    fastavro.writer(out, _PARSED[spec.era], [alert], codec="null")
    return out.getvalue()


#: A tar member: (name, bytes) for a regular file, or (name, kind) for a hostile kind
#: (``"dir"``, ``"symlink"``, ``"hardlink"``, ``"fifo"``).
type TarMember = tuple[str, bytes | str]


def write_tarball(path: Path, members: list[TarMember]) -> Path:
    """Write ``members`` as a gzip-compressed tar at ``path`` and return ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        for name, content in members:
            info = tarfile.TarInfo(name)
            info.mtime = 0
            if isinstance(content, bytes):
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
                continue
            info.type = {
                "dir": tarfile.DIRTYPE,
                "symlink": tarfile.SYMTYPE,
                "hardlink": tarfile.LNKTYPE,
                "fifo": tarfile.FIFOTYPE,
            }[content]
            info.linkname = "/etc/passwd" if "link" in content else ""
            tar.addfile(info)
    return path


def night_tarball(directory: Path, night: int, specs: list[Spec]) -> Path:
    """A tarball of ``night`` holding one ``<candid>.avro`` member per spec."""
    members: list[TarMember] = [(f"{s.candid}.avro", archive_alert(s)) for s in specs]
    return write_tarball(directory / f"ztf_public_{night}.tar.gz", members)
