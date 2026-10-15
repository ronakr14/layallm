"""Suite D — Guardrail benchmark.

Uses the `noul` primitive to classify each LLM output as safe or unsafe.
Applies asymmetric cost weighting: FN_cost >> FP_cost because missing an
unsafe output is much more harmful than over-blocking a safe one.
"""
from __future__ import annotations

from typing import Any, Dict, List

from benchmarks.datasets import get_guardrail_data
from metrics.evaluator import evaluate_guardrail_suite
from models.base import DecisionModel

_FP_COST = 1.0    # safe output wrongly blocked
_FN_COST = 10.0   # unsafe output wrongly passed


def run(
    model: DecisionModel,
    min_confidence: float | None = None,
    fp_cost: float = _FP_COST,
    fn_cost: float = _FN_COST,
    verbose: bool = False,
) -> Dict[str, Any]:
    data = get_guardrail_data()
    cases = data["cases"]
    question = data["question"]

    raw: List[Dict[str, Any]] = []
    for case in cases:
        result = model.noul(
            state=case["state"],
            question=question,
            min_confidence=min_confidence,
        )
        row = {
            "value": result.value,
            "probability": result.probability,
            "abstained": result.abstained,
            "latency_ms": result.latency_ms,
            "ground_truth_unsafe": case["unsafe"],
        }
        raw.append(row)

        if verbose:
            pred_str = "UNSAFE" if result.value else "SAFE"
            gt_str = "unsafe" if case["unsafe"] else "safe"
            correct = (result.value == case["unsafe"])
            status = "ABST" if result.abstained else ("OK" if correct else "MISS")
            print(f"  [{model.name()}][guardrails][{status}] "
                  f"'{case['state'][:60]}' → {pred_str} p={result.probability:.2f} (gt={gt_str})")

    predictions = [r["value"] for r in raw]
    ground_truth = [r["ground_truth_unsafe"] for r in raw]

    metrics = evaluate_guardrail_suite(
        results=raw,
        ground_truth=ground_truth,
        model_name=model.name(),
        fp_cost=fp_cost,
        fn_cost=fn_cost,
    )
    return metrics
