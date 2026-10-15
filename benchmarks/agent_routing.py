"""Suite A — Agent Routing benchmark.

Runs three sub-tests with increasing agent pool sizes (5, 10, 20) to measure
how both models degrade as the decision space expands.
"""
from __future__ import annotations

from typing import Any, Dict, List

from benchmarks.datasets import get_agent_routing_data
from metrics.evaluator import evaluate_choice_suite
from models.base import DecisionModel

_QUESTION = "Which agent should handle this request?"

_POOL_SIZES = [5, 10, 20]
_POOL_KEYS = ["agents_5", "agents_10", "agents_20"]


def run(model: DecisionModel, min_confidence: float | None = None, verbose: bool = False) -> List[Dict[str, Any]]:
    data = get_agent_routing_data()
    cases = data["cases"]
    suite_results = []

    for pool_key in _POOL_KEYS:
        agents = data[pool_key]
        n_agents = len(agents)
        suite_name = f"agent_routing_{n_agents}"

        raw: List[Dict[str, Any]] = []
        for case in cases:
            result = model.choice(
                state=case["state"],
                question=_QUESTION,
                options=agents,
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
                print(f"  [{model.name()}][{n_agents}-agent][{status}] "
                      f"'{case['state'][:60]}' → {result.choice} ({result.probability:.2f})")

        metrics = evaluate_choice_suite(
            results=raw,
            ground_truth=[c["ground_truth"] for c in cases],
            model_name=model.name(),
            suite_name=suite_name,
        )
        metrics["n_agents"] = n_agents
        suite_results.append(metrics)

    return suite_results
