from __future__ import annotations

"""LeakageInspector — feature, label, time, and location safety gates."""

import math
import re
from datetime import datetime, timezone
from typing import Any, Iterable

# Features banned in prod susceptibility model (spatial leakage / identity)
GEO_LEAKAGE = frozenset({"lon", "lat", "i", "j"})

# Label generators — circular if used as features for the same y
LABEL_CIRCULAR = frozenset(
    {
        "soft_score",
        "soft_score_nfip",
        "nfip_soft_score",
        "y_nfip",
        "y_fused",
        "y_lsr",
        "y_storm",
        "n_claims_hard",
        "n_lsr",
        "n_events",
    }
)

PROD_STATIC = ("elev_m", "hand_lite_m", "hand_dem_m", "elev_z")
PROD_DYNAMIC = (
    "rain_max_mm",
    "rain_at_residual_peak_mm",
    "rain_at_rain_peak_mm",
    "rain_sum_mm",
    "peak_rain_mm",
)
MIRROR_ABS_TOLERANCE = 1e-6
MIRROR_REL_TOLERANCE = 1e-6


def normalize_name(name: str) -> str:
    """Canonicalize case and punctuation before applying leakage rules."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _geo_proxy(name: str) -> bool:
    canonical = normalize_name(name)
    tokens = set(canonical.split("_"))
    return (
        canonical in GEO_LEAKAGE
        or bool(tokens & {"lon", "longitude", "lat", "latitude"})
        or canonical in {"x", "y", "row", "col", "column"}
    )


def _circular_proxy(name: str) -> bool:
    canonical = normalize_name(name)
    tokens = set(canonical.split("_"))
    return (
        canonical in LABEL_CIRCULAR
        or ("soft" in tokens and "score" in tokens)
        or "label" in tokens
        or "target" in tokens
        or "circular" in tokens
        or canonical.startswith("y_")
    )


def _time_proxy(name: str) -> bool:
    tokens = set(normalize_name(name).split("_"))
    return bool(tokens & {"time", "timestamp", "datetime", "date"})


def inspect_features(
    feature_names: list[str],
    *,
    importances: dict[str, float] | None = None,
    mode: str = "prod",
) -> dict[str, Any]:
    """Inspect feature names; debug mode reports geo importance but never hides it."""
    names = list(feature_names)
    geo = [name for name in names if _geo_proxy(name)]
    circular = [name for name in names if _circular_proxy(name)]
    time = [name for name in names if _time_proxy(name)]
    imp = importances or {}
    geo_imp = sum(float(imp.get(name, 0.0)) for name in geo)
    total_imp = sum(float(value) for value in imp.values()) or 1.0
    geo_share = geo_imp / total_imp
    flags: list[str] = []
    if geo:
        flags.append(f"geo_features={geo}")
    if circular:
        flags.append(f"circular_label_features={circular}")
    if time:
        flags.append(f"time_features={time}")
    if geo_share >= 0.35:
        flags.append(f"geo_importance_share={geo_share:.2f}>=0.35")
    return {
        "mode": mode,
        "feature_names": names,
        "normalized_feature_names": {name: normalize_name(name) for name in names},
        "geo_features": geo,
        "circular_features": circular,
        "geo_importance_share": round(geo_share, 4),
        "time_features": time,
        "ok_prod": not geo and not circular and not time,
        "flags": flags,
        "note": (
            "Prod forbids location proxies and label generators. "
            "High geo share under debug mode indicates spatial memorization."
        ),
    }


def filter_prod_features(feature_names: list[str]) -> list[str]:
    """Drop normalized geo and circular proxies."""
    return [name for name in feature_names if not _geo_proxy(name) and not _circular_proxy(name)]


def _canonical_time(value: Any) -> str | None:
    raw = str(value or "").strip().replace("Z", "+00:00")
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _numeric_mirror(
    rows: list[dict[str, Any]], feature: str, source: str, tolerance: float
) -> bool:
    if not rows:
        return False
    for row in rows:
        lhs = _number(row.get(feature))
        rhs = _number(row.get(source))
        if lhs is None or rhs is None or abs(lhs - rhs) > tolerance:
            return False
    return True


def _strict_targets(labels: Iterable[Any]) -> list[int]:
    values = list(labels)
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) not in (0.0, 1.0)
        ):
            raise ValueError("labels must contain only exact numeric 0/1 values")
    return [int(value) for value in values]


def inspect_dataset(
    rows: Iterable[dict[str, Any]],
    feature_names: list[str],
    *,
    labels: Iterable[Any],
    label_ids: Iterable[str],
    row_ids: Iterable[str],
    label_columns: tuple[str, ...] = ("label", "y", "y_flood", "y_fused"),
    time_column: str | None = "time_gmt",
    location_columns: tuple[str, ...] = ("lon", "lat", "i", "j"),
    mode: str = "prod",
) -> dict[str, Any]:
    """Inspect names and exact/near value mirrors for leakage.

    Time values are canonicalized to UTC. Numeric label/location mirrors use a
    tight tolerance so harmless string formatting cannot bypass the gate.
    """
    materialized = list(rows)
    target_labels = _strict_targets(labels)
    if len(target_labels) != len(materialized):
        raise ValueError("labels length must match rows length")

    def canonical_ids(values: Iterable[str], name: str) -> list[str]:
        raw = list(values)
        if len(raw) != len(materialized):
            raise ValueError(f"{name} length must match rows and labels length")
        if any(not isinstance(value, str) or not value.strip() for value in raw):
            raise ValueError(f"{name} must contain non-empty string IDs")
        canonical = [value.strip().casefold() for value in raw]
        if len(canonical) != len(set(canonical)):
            raise ValueError(f"{name} IDs must be unique")
        return canonical

    canonical_label_ids = canonical_ids(label_ids, "label_ids")
    canonical_row_ids = canonical_ids(row_ids, "row_ids")
    if canonical_row_ids != canonical_label_ids:
        raise ValueError("label_ids order does not match row_ids order")
    report = inspect_features(feature_names, mode=mode)
    normalized_features = {normalize_name(name): name for name in feature_names}
    normalized_labels = {normalize_name(name) for name in label_columns}
    label_features = sorted(
        normalized_features[name] for name in normalized_features if name in normalized_labels
    )
    time_features: list[str] = []
    if time_column and normalize_name(time_column) in normalized_features:
        time_features.append(normalized_features[normalize_name(time_column)])
    source_times = (
        [_canonical_time(row.get(time_column)) for row in materialized]
        if time_column
        else []
    )
    mirror_features: list[str] = []
    for feature in feature_names:
        feature_times = [_canonical_time(row.get(feature)) for row in materialized]
        if materialized and feature_times == source_times and all(value is not None for value in feature_times):
            time_features.append(feature)
            mirror_features.append(feature)
        for source in label_columns:
            if _numeric_mirror(materialized, feature, source, 1e-9):
                mirror_features.append(feature)
        target_mirror = True
        for row, target in zip(materialized, target_labels):
            feature_value = _number(row.get(feature))
            tolerance = max(
                MIRROR_ABS_TOLERANCE,
                MIRROR_REL_TOLERANCE * max(1.0, abs(float(target))),
            )
            if feature_value is None or abs(feature_value - target) > tolerance:
                target_mirror = False
                break
        if target_mirror and materialized:
            mirror_features.append(feature)
        for source in location_columns:
            if _numeric_mirror(materialized, feature, source, 1e-6):
                mirror_features.append(feature)

    location_features = sorted(
        set(report["geo_features"])
        | {
            normalized_features[name]
            for name in normalized_features
            if any(token in name.split("_") for token in ("lon", "longitude", "lat", "latitude"))
        }
    )
    flags = list(report["flags"])
    if label_features:
        flags.append(f"label_features={label_features}")
    if time_features:
        flags.append(f"time_leakage_features={sorted(set(time_features))}")
    if mirror_features:
        flags.append(f"value_mirror_features={sorted(set(mirror_features))}")
    if location_features and mode == "prod":
        flags.append(f"location_features={location_features}")
    ok_prod = report["ok_prod"] and not label_features and not time_features and not mirror_features
    return {
        **report,
        "label_features": label_features,
        "time_leakage_features": sorted(set(time_features)),
        "location_features": location_features,
        "value_mirror_features": sorted(set(mirror_features)),
        "labels_aligned": True,
        "flags": flags,
        "ok_prod": ok_prod,
        "note": (
            "Feature contract excludes labels, split timestamps, location identifiers, "
            "and exact/near value mirrors. Live feeds and historical evidence remain separate."
        ),
    }


def assert_dataset_ok(report: dict[str, Any]) -> None:
    if not report.get("ok_prod"):
        raise RuntimeError(
            "LeakageInspector dataset contract failed: "
            + "; ".join(report.get("flags") or ["unknown"])
        )


def assert_prod_ok(report: dict[str, Any]) -> None:
    if not report.get("ok_prod"):
        raise RuntimeError(
            "LeakageInspector prod failed: " + "; ".join(report.get("flags") or ["unknown"])
        )
