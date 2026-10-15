"""Evaluation metrics for decision model benchmarks.

Metrics computed:
    accuracy          — top-1 correct rate (excluding abstentions)
    covered_accuracy  — top-1 correct rate including abstentions (abstain = wrong)
    abstention_rate   — fraction of inputs where model declined to decide
    ece               — Expected Calibration Error (10 equal-width bins)
    p50_latency_ms    — median inference latency
    p95_latency_ms    — 95th percentile inference latency

Guardrail-specific:
    precision         — TP / (TP + FP)
    recall            — TP / (TP + FN)  (true positive = correctly flagged unsafe)
    f1
    fpr               — false positive rate (safe flagged as unsafe)
    fnr               — false negative rate (unsafe missed)
    expected_risk     — fpr * fp_cost + fnr * fn_cost
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
def top1_accuracy(
    predictions: List[str],
    ground_truth: List[str],
    abstentions: Optional[List[bool]] = None,
    include_abstentions: bool = False,
) -> float:
    """Fraction of non-abstained predictions that match ground truth.

    If include_abstentions=True, abstained cases count as wrong.
    """
    if not predictions:
        return 0.0
    mask = abstentions or [False] * len(predictions)
    correct = total = 0
    for pred, gt, abst in zip(predictions, ground_truth, mask):
        if abst and not include_abstentions:
            continue
        total += 1
        if not abst and pred == gt:
            correct += 1
    return correct / total if total else 0.0


def abstention_rate(abstentions: List[bool]) -> float:
    if not abstentions:
        return 0.0
    return sum(abstentions) / len(abstentions)


def ece(
    confidences: List[float],
    correct: List[bool],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error.  Lower is better (0 = perfect calibration)."""
    if not confidences:
        return 0.0
    bins: List[Tuple[List[float], List[bool]]] = [
        ([], []) for _ in range(n_bins)
    ]
    for conf, corr in zip(confidences, correct):
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx][0].append(conf)
        bins[idx][1].append(corr)

    total = len(confidences)
    error = 0.0
    for confs_in_bin, corrects_in_bin in bins:
        if not confs_in_bin:
            continue
        avg_conf = sum(confs_in_bin) / len(confs_in_bin)
        avg_acc = sum(corrects_in_bin) / len(corrects_in_bin)
        error += (len(confs_in_bin) / total) * abs(avg_conf - avg_acc)
    return error


def latency_percentiles(latencies_ms: List[float]) -> Dict[str, float]:
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    s = sorted(latencies_ms)
    n = len(s)
    def pct(p):
        idx = max(0, min(n - 1, int(math.ceil(p * n)) - 1))
        return round(s[idx], 1)
    return {
        "p50": pct(0.50),
        "p95": pct(0.95),
        "mean": round(sum(s) / n, 1),
    }


# ---------------------------------------------------------------------------
def guardrail_metrics(
    predictions: List[bool],    # True = flagged as unsafe
    ground_truth: List[bool],   # True = actually unsafe
    confidences: List[float],
    fp_cost: float = 1.0,
    fn_cost: float = 10.0,
) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for pred, gt in zip(predictions, ground_truth):
        if gt and pred:
            tp += 1
        elif not gt and pred:
            fp += 1
        elif not gt and not pred:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    expected_risk = fpr * fp_cost + fnr * fn_cost
    corr = [pred == gt for pred, gt in zip(predictions, ground_truth)]

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "expected_risk": round(expected_risk, 4),
        "ece": round(ece(confidences, corr), 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


# ---------------------------------------------------------------------------
def evaluate_choice_suite(
    results: List[Dict[str, Any]],
    ground_truth: List[str],
    model_name: str,
    suite_name: str,
) -> Dict[str, Any]:
    """Aggregate metrics for a choice benchmark run."""
    predictions = [r["choice"] for r in results]
    abstentions = [r["abstained"] for r in results]
    confidences = [r["probability"] for r in results if not r["abstained"]]
    correct_flags = [
        p == g for p, g, a in zip(predictions, ground_truth, abstentions) if not a
    ]
    latencies = [r["latency_ms"] for r in results]

    acc = top1_accuracy(predictions, ground_truth, abstentions, include_abstentions=False)
    covered_acc = top1_accuracy(predictions, ground_truth, abstentions, include_abstentions=True)
    abst = abstention_rate(abstentions)
    ece_val = ece(confidences, correct_flags) if confidences else 0.0
    lat = latency_percentiles(latencies)

    return {
        "model": model_name,
        "suite": suite_name,
        "n": len(results),
        "accuracy": round(acc, 4),
        "covered_accuracy": round(covered_acc, 4),
        "abstention_rate": round(abst, 4),
        "ece": round(ece_val, 4),
        **{f"latency_{k}": v for k, v in lat.items()},
    }


def evaluate_guardrail_suite(
    results: List[Dict[str, Any]],
    ground_truth: List[bool],
    model_name: str,
    fp_cost: float = 1.0,
    fn_cost: float = 10.0,
) -> Dict[str, Any]:
    """Aggregate metrics for a guardrail (noul) benchmark run."""
    predictions = [r["value"] for r in results]
    confidences = [max(r["probability"], 1 - r["probability"]) for r in results]
    abstentions = [r["abstained"] for r in results]
    latencies = [r["latency_ms"] for r in results]

    gm = guardrail_metrics(predictions, ground_truth, confidences, fp_cost, fn_cost)
    lat = latency_percentiles(latencies)

    return {
        "model": model_name,
        "suite": "guardrails",
        "n": len(results),
        "abstention_rate": round(abstention_rate(abstentions), 4),
        **gm,
        **{f"latency_{k}": v for k, v in lat.items()},
    }
