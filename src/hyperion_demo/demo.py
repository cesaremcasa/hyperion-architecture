from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from hyperion_demo.leakage_inspector import assert_dataset_ok, inspect_dataset
from hyperion_demo.science_metrics import fit_f1_threshold, operational_metrics
from hyperion_demo.split_contracts import temporal_split


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "cc0_synthetic_backtest.csv"
MANIFEST = ROOT / "tests" / "fixtures" / "manifest.json"


def load_fixture() -> list[dict[str, Any]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    if digest != manifest["expected"]["backtest_sha256"]:
        raise ValueError("CC0 fixture hash does not match manifest")
    if manifest["license"]["spdx"] != "CC0-1.0":
        raise ValueError("fixture must be CC0-1.0")
    with FIXTURE.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for column in ("i", "j", "label"):
            row[column] = int(row[column])
        for column in ("lon", "lat", "score"):
            row[column] = float(row[column])
    return rows


def run_demo() -> dict[str, Any]:
    rows = load_fixture()
    ids = [row["sample_id"] for row in rows]
    leakage = inspect_dataset(
        rows,
        ["score"],
        labels=[row["label"] for row in rows],
        label_ids=ids,
        row_ids=ids,
        label_columns=("label",),
        time_column="time_gmt",
        location_columns=("lon", "lat", "i", "j"),
    )
    assert_dataset_ok(leakage)
    split = temporal_split(
        rows,
        time_column="time_gmt",
        test_fraction=0.5,
        identity_columns=("sample_id",),
        location_columns=("lon", "lat"),
    )
    calibration = split["train"]
    evaluation = split["test"]
    threshold = fit_f1_threshold(
        [row["label"] for row in calibration],
        [row["score"] for row in calibration],
    )
    return {
        "evidence_mode": "historical_synthetic_demo",
        "live": False,
        "fixture_license": "CC0-1.0",
        "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        "split": split["metadata"],
        "leakage_ok": leakage["ok_prod"],
        "calibration_threshold": threshold,
        "evaluation": operational_metrics(
            [row["label"] for row in evaluation],
            [row["score"] for row in evaluation],
            threshold=threshold,
        ),
        "interpretation": "Synthetic historical evaluation only; not a forecast or flood truth.",
    }


def calibration_threshold(rows: list[dict[str, Any]]) -> float:
    split = temporal_split(
        rows,
        time_column="time_gmt",
        test_fraction=0.5,
        identity_columns=("sample_id",),
        location_columns=("lon", "lat"),
    )
    calibration = split["train"]
    return fit_f1_threshold(
        [row["label"] for row in calibration],
        [row["score"] for row in calibration],
    )


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2, sort_keys=True))
