"""Suite C — Tool Selection benchmark.

Runs three sub-tests: 10 tools, 25 tools, 50 tools.
For the 50-tool case, Laya uses shortlist_choice pre-filtering automatically
(handled in LayaModel when option count > _SHORTLIST_THRESHOLD).

Also measures top-3 recall: was the correct tool in the top-3 by probability?
"""
from __future__ import annotations

from typing import Any, Dict, List

from benchmarks.datasets import get_tool_selection_data
from metrics.evaluator import evaluate_choice_suite, latency_percentiles
from models.base import DecisionModel

_QUESTION = "Which tool should be called to handle this request?"

_POOL_KEYS = [
    ("tools_10", 10),
    ("tools_25", 25),
    ("tools_50", 50),
]


def _top_k_recall(raw: List[Dict[str, Any]], ground_truth: List[str], k: int = 3) -> float:
    correct = 0
    total = 0
    for row, gt in zip(raw, ground_truth):
        if row["abstained"]:
            continue
        total += 1
        top_k = sorted(row["distribution"].items(), key=lambda x: x[1], reverse=True)[:k]
        top_k_labels = [label for label, _ in top_k]
        if gt in top_k_labels:
            correct += 1
    return round(correct / total, 4) if total else 0.0


def run(model: DecisionModel, min_confidence: float | None = None, verbose: bool = False) -> List[Dict[str, Any]]:
    data = get_tool_selection_data()
    cases = data["cases"]
    suite_results = []

    for pool_key, n_tools in _POOL_KEYS:
        tools = data[pool_key]
        suite_name = f"tool_selection_{n_tools}"

        raw: List[Dict[str, Any]] = []
        for case in cases:
            result = model.choice(
                state=case["state"],
                question=_QUESTION,
                options=tools,
                min_confidence=min_confidence,
            )
            row = {
                "choice": result.choice,
                "probability": result.probability,
                "distribution": result.distribution,
                "abstained": result.abstained,
                "latency_ms": result.latency_ms,
                "ground_truth": case["ground_truth"],
            }
            raw.append(row)

            if verbose:
                status = "ABST" if result.abstained else ("OK" if result.choice == case["ground_truth"] else "MISS")
                print(f"  [{model.name()}][{n_tools}-tools][{status}] "
                      f"'{case['state'][:60]}' → {result.choice} ({result.probability:.2f})")

        gt_labels = [c["ground_truth"] for c in cases]
        metrics = evaluate_choice_suite(
            results=raw,
            ground_truth=gt_labels,
            model_name=model.name(),
            suite_name=suite_name,
        )
        metrics["n_tools"] = n_tools
        metrics["top3_recall"] = _top_k_recall(raw, gt_labels, k=3)
        suite_results.append(metrics)

    return suite_results
