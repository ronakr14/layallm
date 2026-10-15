"""Suite B — LLM Routing benchmark.

Classifies each request into cheap_fast / balanced / reasoning.
Also computes a simulated cost saving and wrong-cheap-model rate.
"""
from __future__ import annotations

from typing import Any, Dict, List

from benchmarks.datasets import get_llm_routing_data
from metrics.evaluator import evaluate_choice_suite
from models.base import DecisionModel

_QUESTION = "Which language model tier should handle this request?"

# Simulated token costs per tier (relative units)
_TIER_COSTS = {
    "cheap_fast": 1,
    "balanced": 5,
    "reasoning": 20,
}

# "Wrong cheap" = request needs reasoning but got cheap_fast
_CRITICAL_TIER = "reasoning"
_CHEAP_TIER = "cheap_fast"


def run(model: DecisionModel, min_confidence: float | None = None, verbose: bool = False) -> Dict[str, Any]:
    data = get_llm_routing_data()
    cases = data["cases"]
    tiers = data["tiers"]

    raw: List[Dict[str, Any]] = []
    for case in cases:
        result = model.choice(
            state=case["state"],
            question=_QUESTION,
            options=tiers,
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
            print(f"  [{model.name()}][llm_routing][{status}] "
                  f"'{case['state'][:60]}' → {result.choice} ({result.probability:.2f})")

    metrics = evaluate_choice_suite(
        results=raw,
        ground_truth=[c["ground_truth"] for c in cases],
        model_name=model.name(),
        suite_name="llm_routing",
    )

    # Cost analysis
    baseline_cost = sum(_TIER_COSTS["reasoning"] for _ in cases)  # always use strongest
    routed_cost = 0
    wrong_cheap_count = 0
    for row, case in zip(raw, cases):
        chosen = row["choice"] if not row["abstained"] else "reasoning"  # abstain → escalate
        routed_cost += _TIER_COSTS.get(chosen, _TIER_COSTS["reasoning"])
        if case["ground_truth"] == _CRITICAL_TIER and chosen == _CHEAP_TIER:
            wrong_cheap_count += 1

    metrics["cost_saving_pct"] = round((1 - routed_cost / baseline_cost) * 100, 1)
    metrics["wrong_cheap_rate"] = round(wrong_cheap_count / len(cases), 4)
    return metrics
