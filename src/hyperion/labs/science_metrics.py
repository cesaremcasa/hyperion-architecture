"""Deterministic probability, calibration, and operational metrics.

These helpers are deliberately stdlib-only. They accept a frozen evaluation
array and never fetch live data, fit on the evaluation split, or silently
coerce invalid probabilities.
"""

from __future__ import annotations

import math
from typing import Any, Iterable


def _validated(y_true: Iterable[int], probabilities: Iterable[float]) -> tuple[list[int], list[float]]:
    raw_y = list(y_true)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) not in (0.0, 1.0)
        for value in raw_y
    ):
        raise ValueError("y_true must contain only exact numeric 0/1 labels")
    y = [int(value) for value in raw_y]
    p = [float(v) for v in probabilities]
    if not y or len(y) != len(p):
        raise ValueError("y_true and probabilities must be non-empty and equal length")
    if any(v not in (0, 1) for v in y):
        raise ValueError("y_true must contain only binary 0/1 labels")
    if any(not math.isfinite(v) or v < 0.0 or v > 1.0 for v in p):
        raise ValueError("probabilities must be finite values in [0, 1]")
    return y, p


def brier_score(y_true: Iterable[int], probabilities: Iterable[float]) -> float:
    """Mean squared probability error for binary outcomes."""
    y, p = _validated(y_true, probabilities)
    return sum((score - label) ** 2 for label, score in zip(y, p)) / len(y)


def calibration_bins(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    *,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Return uniform reliability bins, including empty bins."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    y, p = _validated(y_true, probabilities)
    buckets: list[list[tuple[int, float]]] = [[] for _ in range(n_bins)]
    for label, score in zip(y, p):
        index = min(n_bins - 1, int(score * n_bins))
        buckets[index].append((label, score))
    out: list[dict[str, Any]] = []
    for index, bucket in enumerate(buckets):
        lower = index / n_bins
        upper = (index + 1) / n_bins
        if bucket:
            mean_predicted = sum(score for _, score in bucket) / len(bucket)
            observed_rate = sum(label for label, _ in bucket) / len(bucket)
            gap = abs(mean_predicted - observed_rate)
        else:
            mean_predicted = None
            observed_rate = None
            gap = None
        out.append(
            {
                "bin": index,
                "lower": round(lower, 6),
                "upper": round(upper, 6),
                "n": len(bucket),
                "mean_predicted": None if mean_predicted is None else round(mean_predicted, 6),
                "observed_rate": None if observed_rate is None else round(observed_rate, 6),
                "abs_gap": None if gap is None else round(gap, 6),
            }
        )
    return out


def expected_calibration_error(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    *,
    n_bins: int = 10,
) -> float:
    """Sample-weighted expected calibration error (ECE)."""
    y, p = _validated(y_true, probabilities)
    bins = calibration_bins(y, p, n_bins=n_bins)
    n = len(y)
    return sum((row["n"] / n) * (row["abs_gap"] or 0.0) for row in bins)


def operational_metrics(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    *,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Return Brier/calibration plus thresholded alert metrics."""
    if not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be finite and in [0, 1]")
    y, p = _validated(y_true, probabilities)
    predicted = [1 if score >= threshold else 0 for score in p]
    tp = sum(label == 1 and pred == 1 for label, pred in zip(y, predicted))
    fp = sum(label == 0 and pred == 1 for label, pred in zip(y, predicted))
    fn = sum(label == 1 and pred == 0 for label, pred in zip(y, predicted))
    tn = sum(label == 0 and pred == 0 for label, pred in zip(y, predicted))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else 0.0
    )
    return {
        "n": len(y),
        "pos": sum(y),
        "pred_pos": sum(predicted),
        "alert_rate": sum(predicted) / len(y),
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": None if precision is None else round(precision, 6),
        "recall": None if recall is None else round(recall, 6),
        "specificity": None if specificity is None else round(specificity, 6),
        "f1": None if f1 is None else round(f1, 6),
        "brier": round(brier_score(y, p), 6),
        "ece": round(expected_calibration_error(y, p, n_bins=n_bins), 6),
        "calibration_bins": calibration_bins(y, p, n_bins=n_bins),
    }


def fit_f1_threshold(y_true: Iterable[int], probabilities: Iterable[float]) -> float:
    """Choose a threshold on calibration data only; ties prefer the higher threshold."""
    y, p = _validated(y_true, probabilities)
    candidates = sorted({0.0, 1.0, *p})
    best = (float("-inf"), float("-inf"))
    for threshold in candidates:
        metrics = operational_metrics(y, p, threshold=threshold, n_bins=10)
        score = metrics["f1"] if metrics["f1"] is not None else 0.0
        rank = (float(score), float(threshold))
        if rank > best:
            best = rank
    return best[1]


def fit_temperature(
    calibration_labels: Iterable[int],
    calibration_logits: Iterable[float],
    evaluation_labels: Iterable[int],
    evaluation_logits: Iterable[float],
) -> dict[str, Any]:
    """Fit temperature on calibration rows and evaluate once on a holdout."""
    cal_y = list(calibration_labels)
    eval_y = list(evaluation_labels)
    cal_logits = [float(value) for value in calibration_logits]
    eval_logits = [float(value) for value in evaluation_logits]
    # Reuse strict label validation without treating logits as probabilities.
    _validated(cal_y, [0.5] * len(cal_y))
    _validated(eval_y, [0.5] * len(eval_y))
    if len(cal_y) < 2 or len(eval_y) < 2 or set(cal_y) != {0, 1} or set(eval_y) != {0, 1}:
        raise ValueError("temperature calibration and evaluation halves must each contain both classes")
    if len(cal_y) != len(cal_logits) or len(eval_y) != len(eval_logits):
        raise ValueError("temperature labels and logits must have equal lengths")
    if any(not math.isfinite(value) for value in cal_logits + eval_logits):
        raise ValueError("temperature logits must be finite")

    def probabilities(logits: list[float], temperature: float) -> list[float]:
        return [1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, logit / temperature)))) for logit in logits]

    def score(labels: list[int], logits: list[float], temperature: float) -> float:
        return brier_score(labels, probabilities(logits, temperature))

    best_temperature = 1.0
    best_calibration_brier = score(cal_y, cal_logits, best_temperature)
    for candidate in [1.0 + 0.25 * index for index in range(21)]:
        candidate_brier = score(cal_y, cal_logits, candidate)
        if candidate_brier < best_calibration_brier - 1e-6:
            best_temperature = candidate
            best_calibration_brier = candidate_brier
    cal_probabilities = probabilities(cal_logits, best_temperature)
    eval_probabilities = probabilities(eval_logits, best_temperature)
    return {
        "temperature": round(best_temperature, 6),
        "brier_calibration_T1": round(score(cal_y, cal_logits, 1.0), 6),
        "brier_calibration_best": round(best_calibration_brier, 6),
        "brier_evaluation_T1": round(score(eval_y, eval_logits, 1.0), 6),
        "brier_evaluation_best": round(brier_score(eval_y, eval_probabilities), 6),
        "calibration_probabilities": cal_probabilities,
        "evaluation_probabilities": eval_probabilities,
        "evaluation_metrics": operational_metrics(eval_y, eval_probabilities),
    }
