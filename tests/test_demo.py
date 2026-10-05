from __future__ import annotations

from hyperion_demo.demo import calibration_threshold, load_fixture, run_demo
from hyperion_demo.leakage_inspector import inspect_dataset


def test_demo_uses_verified_synthetic_fixture_and_safe_split():
    result = run_demo()
    assert result["evidence_mode"] == "historical_synthetic_demo"
    assert result["live"] is False
    assert result["fixture_license"] == "CC0-1.0"
    assert result["leakage_ok"] is True
    assert result["split"]["safety"]["time_overlap_n"] == 0
    assert result["split"]["train_n"] > 0
    assert result["split"]["test_n"] > 0


def test_leakage_inspector_rejects_label_as_feature():
    rows = load_fixture()
    ids = [row["sample_id"] for row in rows]
    report = inspect_dataset(
        rows,
        ["label"],
        labels=[row["label"] for row in rows],
        label_ids=ids,
        row_ids=ids,
        label_columns=("label",),
        time_column="time_gmt",
        location_columns=("lon", "lat", "i", "j"),
    )
    assert report["ok_prod"] is False


def test_evaluation_labels_do_not_change_calibration_threshold():
    rows = load_fixture()
    first = calibration_threshold(rows)
    split = run_demo()["split"]
    for row in rows:
        if row["time_gmt"] >= split["cutoff_time"]:
            row["label"] = 1 - row["label"]
    assert calibration_threshold(rows) == first
