"""Leakage-safe deterministic temporal and spatial split contracts."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from math import ceil
from typing import Any, Iterable


def _time(value: Any) -> datetime:
    raw = str(value or "").strip().replace("Z", "+00:00")
    if not raw:
        raise ValueError("split rows require a non-empty time value")
    parsed = datetime.fromisoformat(raw)
    parsed = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _canonical_value(value: Any) -> str | float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        text = str(value or "").strip()
        if not text:
            raise ValueError("split identity/location values must not be empty")
        return text.casefold()
    if not math.isfinite(number):
        raise ValueError("split identity/location values must be finite")
    return round(number, 9)


def _identity_value(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if not text:
        raise ValueError("split identity values must not be empty")
    return text


def _location(row: dict[str, Any], columns: tuple[str, ...]) -> tuple[str | float, ...]:
    if not columns:
        raise ValueError("location columns must not be empty")
    if any(column not in row for column in columns):
        raise ValueError(f"split row missing location columns: {columns}")
    return tuple(_canonical_value(row[column]) for column in columns)


def split_overlap_report(
    train_rows: Iterable[dict[str, Any]],
    test_rows: Iterable[dict[str, Any]],
    *,
    identity_columns: tuple[str, ...] = ("sample_id",),
    time_column: str = "time_gmt",
    location_columns: tuple[str, ...] = ("lon", "lat"),
) -> dict[str, Any]:
    """Report identity, timestamp, and location intersections without policy."""
    train = list(train_rows)
    test = list(test_rows)

    def keys(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> set[tuple[str | float, ...]]:
        if any(column not in row for row in rows for column in columns):
            raise ValueError(f"split row missing identity columns: {columns}")
        return {tuple(_identity_value(row[column]) for column in columns) for row in rows}

    def duplicate_keys(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> list[list[str | float]]:
        counts: dict[tuple[str | float, ...], int] = {}
        for row in rows:
            key = tuple(_identity_value(row[column]) for column in columns)
            counts[key] = counts.get(key, 0) + 1
        return [list(key) for key, count in sorted(counts.items(), key=lambda item: repr(item[0])) if count > 1]

    identity_overlap = sorted(
        keys(train, identity_columns) & keys(test, identity_columns), key=repr
    )
    train_duplicates = duplicate_keys(train, identity_columns)
    test_duplicates = duplicate_keys(test, identity_columns)
    time_overlap = sorted(
        {_time(row[time_column]).isoformat() for row in train}
        & {_time(row[time_column]).isoformat() for row in test}
    )
    location_overlap = sorted(
        {_location(row, location_columns) for row in train}
        & {_location(row, location_columns) for row in test},
        key=repr,
    )
    return {
        "identity_overlap": [list(item) for item in identity_overlap],
        "train_duplicate_identity": train_duplicates,
        "test_duplicate_identity": test_duplicates,
        "time_overlap": time_overlap,
        "location_overlap": [list(item) for item in location_overlap],
        "identity_overlap_n": len(identity_overlap),
        "train_duplicate_identity_n": len(train_duplicates),
        "test_duplicate_identity_n": len(test_duplicates),
        "time_overlap_n": len(time_overlap),
        "location_overlap_n": len(location_overlap),
    }


def assert_split_safe(
    train_rows: Iterable[dict[str, Any]],
    test_rows: Iterable[dict[str, Any]],
    *,
    mode: str,
    identity_columns: tuple[str, ...] = ("sample_id",),
    time_column: str = "time_gmt",
    location_columns: tuple[str, ...] = ("lon", "lat"),
) -> dict[str, Any]:
    """Enforce no identity overlap and the no-leakage rule for a split mode."""
    if mode not in {"temporal", "spatial"}:
        raise ValueError("mode must be temporal or spatial")
    report = split_overlap_report(
        train_rows,
        test_rows,
        identity_columns=identity_columns,
        time_column=time_column,
        location_columns=location_columns,
    )
    violations = []
    if report["identity_overlap_n"]:
        violations.append("identity_overlap")
    if report["train_duplicate_identity_n"]:
        violations.append("train_duplicate_identity")
    if report["test_duplicate_identity_n"]:
        violations.append("test_duplicate_identity")
    if mode == "temporal" and report["time_overlap_n"]:
        violations.append("time_overlap")
    if mode == "spatial" and report["location_overlap_n"]:
        violations.append("location_overlap")
    if violations:
        raise ValueError("unsafe split: " + ", ".join(violations))
    return {"mode": mode, **report, "ok": True}


def temporal_split(
    rows: Iterable[dict[str, Any]],
    *,
    time_column: str = "time_gmt",
    test_fraction: float = 0.25,
    embargo_hours: float = 0.0,
    identity_columns: tuple[str, ...] = ("sample_id",),
    location_columns: tuple[str, ...] = ("lon", "lat"),
) -> dict[str, Any]:
    """Split whole timestamps in chronological order with an optional embargo."""
    rows_list = list(rows)
    if not rows_list:
        raise ValueError("cannot split an empty dataset")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    if embargo_hours < 0:
        raise ValueError("embargo_hours must be non-negative")
    times = sorted({_time(row[time_column]) for row in rows_list})
    if len(times) < 2:
        raise ValueError("temporal split requires at least two distinct timestamps")
    cutoff_index = max(1, min(len(times) - 1, ceil(len(times) * (1.0 - test_fraction))))
    cutoff = times[cutoff_index]
    test_start = cutoff + timedelta(hours=embargo_hours)
    train = [row for row in rows_list if _time(row[time_column]) < cutoff]
    test = [row for row in rows_list if _time(row[time_column]) >= test_start]
    embargoed = [
        row
        for row in rows_list
        if cutoff <= _time(row[time_column]) < test_start
    ]
    if not train or not test:
        raise ValueError("temporal split produced an empty train or test set")
    safety = assert_split_safe(
        train,
        test,
        mode="temporal",
        identity_columns=identity_columns,
        time_column=time_column,
        location_columns=location_columns,
    )
    return {
        "train": train,
        "test": test,
        "embargoed": embargoed,
        "metadata": {
            "method": "temporal_ordered",
            "test_fraction": test_fraction,
            "embargo_hours": embargo_hours,
            "cutoff_time": cutoff.isoformat(),
            "test_start": test_start.isoformat(),
            "train_n": len(train),
            "test_n": len(test),
            "embargoed_n": len(embargoed),
            "safety": safety,
        },
    }


def spatial_split(
    rows: Iterable[dict[str, Any]],
    *,
    axis: str = "lon",
    test_fraction: float = 0.25,
    identity_columns: tuple[str, ...] = ("sample_id",),
    time_column: str = "time_gmt",
    location_columns: tuple[str, ...] = ("lon", "lat"),
) -> dict[str, Any]:
    """Split whole coordinate bands along a deterministic longitude/latitude axis."""
    rows_list = list(rows)
    if not rows_list:
        raise ValueError("cannot split an empty dataset")
    if axis not in location_columns:
        raise ValueError(f"axis {axis!r} must be in location_columns")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    coordinates = sorted({float(row[axis]) for row in rows_list})
    if len(coordinates) < 2:
        raise ValueError("spatial split requires at least two coordinate bands")
    cutoff_index = max(1, min(len(coordinates) - 1, ceil(len(coordinates) * (1.0 - test_fraction))))
    cutoff = coordinates[cutoff_index]
    train = [row for row in rows_list if float(row[axis]) < cutoff]
    test = [row for row in rows_list if float(row[axis]) >= cutoff]
    if not train or not test:
        raise ValueError("spatial split produced an empty train or test set")
    safety = assert_split_safe(
        train,
        test,
        mode="spatial",
        identity_columns=identity_columns,
        time_column=time_column,
        location_columns=location_columns,
    )
    return {
        "train": train,
        "test": test,
        "metadata": {
            "method": "spatial_band",
            "axis": axis,
            "test_fraction": test_fraction,
            "cutoff": cutoff,
            "train_n": len(train),
            "test_n": len(test),
            "safety": safety,
        },
    }
