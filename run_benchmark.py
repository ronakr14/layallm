"""Jev vs Laya benchmark runner.

Usage:
    python run_benchmark.py                         # all suites, both models
    python run_benchmark.py --suite A               # agent routing only
    python run_benchmark.py --suite D --model laya  # guardrails, Laya only
    python run_benchmark.py --suite all --min-confidence 0.85
    python run_benchmark.py --verbose               # show per-case output

Suites: A=agent_routing  B=llm_routing  C=tool_selection  D=guardrails
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

# Force UTF-8 stdout on Windows so tabulate box-drawing chars render correctly
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Path setup so sub-packages resolve from repo root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import benchmarks.agent_routing as suite_a
import benchmarks.llm_routing as suite_b
import benchmarks.tool_selection as suite_c
import benchmarks.guardrails as suite_d
from models.jev_model import JevModel
from models.laya_model import LayaModel

try:
    from tabulate import tabulate
    _HAS_TABULATE = True
except ImportError:
    _HAS_TABULATE = False


# ---------------------------------------------------------------------------
def _fmt(val: Any) -> str:
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _print_table(rows: List[Dict[str, Any]], title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    if not rows:
        print("  (no results)")
        return
    if _HAS_TABULATE:
        # Format floats
        formatted = [{k: _fmt(v) for k, v in row.items()} for row in rows]
        print(tabulate(formatted, headers="keys", tablefmt="rounded_outline"))
    else:
        for row in rows:
            print({k: _fmt(v) for k, v in row.items()})


def _save_results(all_results: Dict[str, Any]) -> str:
    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("results", f"benchmark_{ts}.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    return path


# ---------------------------------------------------------------------------
def build_models(model_arg: str, fast: bool = False):
    models = []
    if model_arg in ("jev", "both"):
        models.append(JevModel())
    if model_arg in ("laya", "both"):
        print("[Laya] Loading model (first run downloads ~400 MB)...")
        models.append(LayaModel(fast=fast))
    return models


# ---------------------------------------------------------------------------
def _comparison_table(all_results: Dict[str, Any]) -> None:
    """Print a compact Jev vs Laya side-by-side summary across all suites."""
    KEY_METRICS = {
        "agent_routing": ["accuracy", "ece", "abstention_rate"],
        "llm_routing": ["accuracy", "cost_saving_pct", "wrong_cheap_rate", "ece"],
        "tool_selection": ["accuracy", "top3_recall", "ece"],
        "guardrails": ["f1", "fnr", "fpr", "expected_risk"],
    }

    rows = []
    for suite_key, suite_data in all_results.get("suites", {}).items():
        if isinstance(suite_data, list):
            items = suite_data
        else:
            items = [suite_data]

        for item in items:
            suite_name = item.get("suite", "")
            base = suite_name.rsplit("_", 1)[0] if suite_name else suite_name
            metrics_keys = KEY_METRICS.get(base, [])
            row: Dict[str, Any] = {
                "suite": item.get("suite", suite_key),
                "model": item.get("model", "?"),
            }
            for k in metrics_keys:
                if k in item:
                    row[k] = item[k]
            rows.append(row)

    if rows:
        _print_table(rows, "COMPARISON SUMMARY — all suites")


# ---------------------------------------------------------------------------
def run_all(models, suites: List[str], min_confidence, verbose: bool) -> Dict[str, Any]:
    all_results: Dict[str, Any] = {"timestamp": datetime.now().isoformat(), "suites": {}}

    # ------ Suite A: Agent Routing ------
    if "A" in suites:
        print("\n[Suite A] Agent Routing — 5 / 10 / 20 agents")
        a_results = []
        for model in models:
            t0 = time.time()
            rows = suite_a.run(model, min_confidence=min_confidence, verbose=verbose)
            elapsed = time.time() - t0
            for row in rows:
                row["total_time_s"] = round(elapsed / len(rows), 1)
            a_results.extend(rows)

        # Group by n_agents for a degradation view
        for n in [5, 10, 20]:
            subset = [r for r in a_results if r.get("n_agents") == n]
            _print_table(
                [{k: v for k, v in r.items() if k not in ("suite",)} for r in subset],
                f"Suite A — Agent Routing ({n} agents)",
            )
        all_results["suites"]["A"] = a_results

    # ------ Suite B: LLM Routing ------
    if "B" in suites:
        print("\n[Suite B] LLM Routing — cheap / balanced / reasoning")
        b_results = []
        for model in models:
            row = suite_b.run(model, min_confidence=min_confidence, verbose=verbose)
            b_results.append(row)
        _print_table(b_results, "Suite B — LLM Routing")
        all_results["suites"]["B"] = b_results

    # ------ Suite C: Tool Selection ------
    if "C" in suites:
        print("\n[Suite C] Tool Selection — 10 / 25 / 50 tools")
        c_results = []
        for model in models:
            rows = suite_c.run(model, min_confidence=min_confidence, verbose=verbose)
            c_results.extend(rows)
        for n in [10, 25, 50]:
            subset = [r for r in c_results if r.get("n_tools") == n]
            _print_table(
                [{k: v for k, v in r.items() if k not in ("suite",)} for r in subset],
                f"Suite C — Tool Selection ({n} tools)",
            )
        all_results["suites"]["C"] = c_results

    # ------ Suite D: Guardrails ------
    if "D" in suites:
        print("\n[Suite D] Guardrails — safe / borderline / unsafe / adversarial")
        d_results = []
        for model in models:
            row = suite_d.run(model, min_confidence=min_confidence, verbose=verbose)
            d_results.append(row)
        _print_table(d_results, "Suite D — Guardrails (FP_cost=1, FN_cost=10)")
        all_results["suites"]["D"] = d_results

    # Print comparison summary when both models ran
    if len(models) > 1:
        _comparison_table(all_results)

    return all_results


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Jev vs Laya decision model benchmark")
    parser.add_argument("--suite", default="all",
                        help="Suites to run: A, B, C, D, or 'all' (default: all)")
    parser.add_argument("--model", default="both",
                        choices=["jev", "laya", "both"],
                        help="Which model(s) to benchmark (default: both)")
    parser.add_argument("--min-confidence", type=float, default=None,
                        help="Abstention threshold (e.g. 0.80).  None = no abstention.")
    parser.add_argument("--fast", action="store_true",
                        help="Use Laya fast mode (lower latency, slightly less accurate)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-case results")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving results to results/ directory")
    args = parser.parse_args()

    suites_arg = args.suite.upper()
    if suites_arg == "ALL":
        suites = ["A", "B", "C", "D"]
    else:
        suites = [s.strip() for s in suites_arg.replace(",", " ").split()]

    invalid = [s for s in suites if s not in ("A", "B", "C", "D")]
    if invalid:
        print(f"Unknown suite(s): {invalid}.  Valid: A, B, C, D, all")
        sys.exit(1)

    models = build_models(args.model, fast=args.fast)
    if not models:
        print("No models selected.")
        sys.exit(1)

    print(f"\nRunning suites: {', '.join(suites)}")
    print(f"Models: {', '.join(m.name() for m in models)}")
    if args.min_confidence:
        print(f"Min confidence threshold: {args.min_confidence}")
    if args.fast:
        print("Laya fast mode: ON")

    all_results = run_all(models, suites, args.min_confidence, args.verbose)

    if not args.no_save:
        path = _save_results(all_results)
        print(f"\nResults saved → {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
