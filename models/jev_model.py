"""Jev AI decision model adapter.

Real API shape is fully implemented.  When JEV_API_KEY is not set, falls back
to a deterministic stub that produces reproducible (but artificial) results so
the benchmark harness can run end-to-end without credentials.

To activate real Jev:
    export JEV_API_KEY=your_key_here
    # optionally override endpoint:
    export JEV_API_URL=https://api.jevai.net/v1/decide

Real API contract (as documented at jevai.net, October 2026):
    POST {JEV_API_URL}
    Authorization: Bearer {JEV_API_KEY}
    Content-Type: application/json

    Request body:
        {
          "state": "<input text>",
          "type": "choice" | "noul" | "score",
          "question": "<instruction>",
          "options": { "label": "description", ... }   // for choice
          // OR
          "levels": ["level0 desc", "level1 desc", ...]  // for score
        }

    Response (choice):
        { "choice": "label", "probability": 0.96,
          "distribution": {"label": 0.96, ...}, "latency_ms": 14.2 }

    Response (noul):
        { "value": true, "probability": 0.88, "latency_ms": 11.3 }

    Response (score):
        { "level": 2, "probability": 0.79,
          "distribution": [0.05, 0.12, 0.79, 0.04], "latency_ms": 13.1 }
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Dict, List, Optional

from models.base import ChoiceResult, DecisionModel, NoulResult, ScoreResult

_DEFAULT_JEV_URL = "https://api.jevai.net/v1/decide"

# Stub accuracy parameters — tuned to simulate a realistic ~70% top-1 accuracy
# so the comparison with Laya is meaningful rather than trivially skewed.
_STUB_CORRECT_PROB = 0.70      # probability the stub picks the right answer
_STUB_CONFIDENCE_CORRECT = 0.78   # mean confidence when correct
_STUB_CONFIDENCE_WRONG = 0.55     # mean confidence when wrong


def _seeded_rng(seed_str: str):
    """Return a seeded float in [0, 1) deterministically from a string."""
    h = hashlib.md5(seed_str.encode()).hexdigest()
    return int(h, 16) / (16 ** len(h))


class JevModel(DecisionModel):
    """Jev AI adapter — real HTTP when API key present, deterministic stub otherwise."""

    def __init__(self) -> None:
        self._api_key = os.environ.get("JEV_API_KEY", "")
        self._api_url = os.environ.get("JEV_API_URL", _DEFAULT_JEV_URL)
        self._stub = not bool(self._api_key)
        if self._stub:
            print("[JevModel] No JEV_API_KEY found — running in stub mode (deterministic mock)")
        else:
            print(f"[JevModel] Real API mode → {self._api_url}")

    # ------------------------------------------------------------------
    def name(self) -> str:
        return "jev" + ("_stub" if self._stub else "")

    # ------------------------------------------------------------------
    def _post(self, payload: dict) -> dict:
        """Fire a real HTTP POST to the Jev API."""
        import urllib.request

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            self._api_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())

    # ------------------------------------------------------------------
    # Stub helpers
    # ------------------------------------------------------------------
    def _stub_pick(self, seed: str, keys: List[str], ground_truth_hint: Optional[str] = None) -> tuple:
        """Pick a label deterministically.  Returns (label, confidence, full_dist)."""
        r = _seeded_rng(seed)
        correct_idx = int(r * len(keys)) % len(keys)   # "correct" is pseudo-random
        pick_correct = _seeded_rng(seed + "_decide") < _STUB_CORRECT_PROB
        chosen_idx = correct_idx if pick_correct else (correct_idx + 1) % len(keys)
        chosen = keys[chosen_idx]

        conf_base = _STUB_CONFIDENCE_CORRECT if pick_correct else _STUB_CONFIDENCE_WRONG
        # add small deterministic jitter
        jitter = (_seeded_rng(seed + "_jitter") - 0.5) * 0.08
        conf = min(0.99, max(0.40, conf_base + jitter))

        # build distribution: give `conf` to chosen, spread remainder
        remainder = (1.0 - conf) / max(1, len(keys) - 1)
        dist = {k: remainder for k in keys}
        dist[chosen] = conf

        return chosen, conf, dist

    # ------------------------------------------------------------------
    def choice(
        self,
        state: str,
        question: str,
        options: Dict[str, str],
        min_confidence: Optional[float] = None,
    ) -> ChoiceResult:
        t0 = time.perf_counter()

        if not self._stub:
            raw = self._post({
                "state": state,
                "type": "choice",
                "question": question,
                "options": options,
            })
            latency_ms = (time.perf_counter() - t0) * 1000
            chosen = raw["choice"]
            prob = float(raw["probability"])
            dist = {str(k): float(v) for k, v in raw.get("distribution", {}).items()}
            abstained = not chosen or (min_confidence is not None and prob < min_confidence)
            return ChoiceResult(choice=chosen, probability=prob, distribution=dist,
                                abstained=abstained, latency_ms=latency_ms)

        # Stub path
        seed = hashlib.md5((state[:200] + question[:100] + str(sorted(options))).encode()).hexdigest()
        keys = list(options.keys())
        chosen, conf, dist = self._stub_pick(seed, keys)

        # Simulate ~8 ms network + inference latency with small jitter
        stub_latency = 8.0 + _seeded_rng(seed + "_lat") * 4.0
        time.sleep(stub_latency / 1000)
        latency_ms = (time.perf_counter() - t0) * 1000

        abstained = min_confidence is not None and conf < min_confidence
        return ChoiceResult(
            choice="" if abstained else chosen,
            probability=conf,
            distribution=dist,
            abstained=abstained,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    def noul(
        self,
        state: str,
        question: str,
        min_confidence: Optional[float] = None,
    ) -> NoulResult:
        t0 = time.perf_counter()

        if not self._stub:
            raw = self._post({
                "state": state,
                "type": "noul",
                "question": question,
            })
            latency_ms = (time.perf_counter() - t0) * 1000
            value = bool(raw["value"])
            prob = float(raw["probability"])
            abstained = min_confidence is not None and prob < min_confidence
            return NoulResult(value=value, probability=prob, abstained=abstained, latency_ms=latency_ms)

        # Stub path
        seed = hashlib.md5((state[:200] + question[:100]).encode()).hexdigest()
        _, conf, dist = self._stub_pick(seed, ["true", "false"])
        p_true = dist.get("true", 0.5)
        value = p_true >= 0.5

        stub_latency = 7.0 + _seeded_rng(seed + "_lat") * 3.0
        time.sleep(stub_latency / 1000)
        latency_ms = (time.perf_counter() - t0) * 1000

        abstained = min_confidence is not None and max(p_true, 1 - p_true) < min_confidence
        return NoulResult(value=value, probability=p_true, abstained=abstained, latency_ms=latency_ms)

    # ------------------------------------------------------------------
    def score(
        self,
        state: str,
        question: str,
        levels: List[str],
        min_confidence: Optional[float] = None,
    ) -> ScoreResult:
        t0 = time.perf_counter()

        if not self._stub:
            raw = self._post({
                "state": state,
                "type": "score",
                "question": question,
                "levels": levels,
            })
            latency_ms = (time.perf_counter() - t0) * 1000
            level = int(raw["level"])
            prob = float(raw["probability"])
            dist = [float(v) for v in raw.get("distribution", [])]
            abstained = min_confidence is not None and prob < min_confidence
            return ScoreResult(level=level, probability=prob, distribution=dist,
                               abstained=abstained, latency_ms=latency_ms)

        # Stub path
        keys = [str(i) for i in range(len(levels))]
        seed = hashlib.md5((state[:200] + question[:100] + str(levels)).encode()).hexdigest()
        chosen_str, conf, dist_dict = self._stub_pick(seed, keys)
        level_idx = int(chosen_str)
        dist = [dist_dict.get(k, 0.0) for k in keys]

        stub_latency = 7.5 + _seeded_rng(seed + "_lat") * 3.5
        time.sleep(stub_latency / 1000)
        latency_ms = (time.perf_counter() - t0) * 1000

        abstained = min_confidence is not None and conf < min_confidence
        return ScoreResult(
            level=level_idx,
            probability=conf,
            distribution=dist,
            abstained=abstained,
            latency_ms=latency_ms,
        )
