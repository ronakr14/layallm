"""Laya decision model — wraps the locally installed `laya` package (v0.3.28+).

Model downloads on first call (~400 MB for the english checkpoint).
CPU inference: ~100-300 ms per decision.  GPU: ~10-30 ms.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import config
from models.base import ChoiceResult, DecisionModel, NoulResult, ScoreResult

_SHORTLIST_THRESHOLD = 10   # use shortlist pre-filter when option count exceeds this
_SHORTLIST_K = 8            # candidates kept after shortlist


class LayaModel(DecisionModel):
    """Wraps `laya.Agent` for the benchmark harness."""

    def __init__(
        self,
        model_id: str = config.LAYA_MODEL,
        device: Optional[str] = config.LAYA_DEVICE,
        fast: bool = False,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._fast = fast
        self._agent = None          # lazy-loaded on first call
        self._embed_fn = None

    # ------------------------------------------------------------------
    def name(self) -> str:
        return "laya"

    # ------------------------------------------------------------------
    def _agent_instance(self):
        if self._agent is None:
            import laya
            self._agent = laya.Agent(
                model_id_or_path=self._model_id,
                device=self._device,
                fast=self._fast,
            )
        return self._agent

    def _embed(self):
        if self._embed_fn is None:
            import laya
            self._embed_fn = laya.cached_embed_fn(laya.embed_fn_from_agent(self._agent_instance()))
        return self._embed_fn

    # ------------------------------------------------------------------
    def choice(
        self,
        state: str,
        question: str,
        options: Dict[str, str],
        min_confidence: Optional[float] = None,
    ) -> ChoiceResult:
        import laya

        agent = self._agent_instance()

        # For large option sets, run a shortlist pass first
        if len(options) > _SHORTLIST_THRESHOLD:
            shortlisted_labels = laya.shortlist_choice(
                state,
                options,
                embed_fn=self._embed(),
                k=_SHORTLIST_K,
                instructions=question,
            )
            options = {k: options[k] for k in shortlisted_labels if k in options}

        questions = {
            "decision": {
                "type": "choice",
                "instructions": question,
                "criteria": options,
            }
        }

        t0 = time.perf_counter()
        try:
            result = agent.decide(
                state,
                questions=questions,
                return_details=True,
                **({"min_confidence": min_confidence} if min_confidence is not None else {}),
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            dec = result.values.get("decision")
            # When using questions= format, values["decision"] is a dict with
            # {"type": "choice", "choice": <label>, "probabilities": {...}, ...}
            if isinstance(dec, dict):
                chosen = dec.get("choice")
                probs = dec.get("probabilities") or result.probabilities.get("decision", {})
                conf = dec.get("answer_confidence") or result.answer_confidence.get("decision") or 0.0
            else:
                chosen = dec
                probs = result.probabilities.get("decision", {})
                conf = result.answer_confidence.get("decision") or 0.0
            abstained = chosen is None

            return ChoiceResult(
                choice=chosen or "",
                probability=conf,
                distribution={str(k): float(v) for k, v in probs.items()},
                abstained=abstained,
                latency_ms=latency_ms,
            )
        except Exception:
            latency_ms = (time.perf_counter() - t0) * 1000
            return ChoiceResult(
                choice="",
                probability=0.0,
                distribution={},
                abstained=True,
                latency_ms=latency_ms,
            )

    # ------------------------------------------------------------------
    def noul(
        self,
        state: str,
        question: str,
        min_confidence: Optional[float] = None,
    ) -> NoulResult:
        agent = self._agent_instance()

        questions = {
            "decision": {
                "type": "noul",
                "instructions": question,
            }
        }

        t0 = time.perf_counter()
        try:
            result = agent.decide(
                state,
                questions=questions,
                return_details=True,
                **({"min_confidence": min_confidence} if min_confidence is not None else {}),
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            dec = result.values.get("decision")
            # When using questions= format, values["decision"] is a dict with
            # {"type": "noul", "noul": <P(true) float>, "answer_confidence": ..., ...}
            if isinstance(dec, dict):
                p_true = float(dec.get("noul", 0.5))
                conf = dec.get("answer_confidence") or result.answer_confidence.get("decision") or 0.0
                abstained = dec is None
            else:
                abstained = dec is None
                probs = result.probabilities.get("decision", {})
                p_true = float(probs.get(True, probs.get("true", 0.5)))
                conf = result.answer_confidence.get("decision") or 0.0

            return NoulResult(
                value=p_true >= 0.5,
                probability=p_true,
                abstained=abstained,
                latency_ms=latency_ms,
            )
        except Exception:
            latency_ms = (time.perf_counter() - t0) * 1000
            return NoulResult(value=False, probability=0.0, abstained=True, latency_ms=latency_ms)

    # ------------------------------------------------------------------
    def score(
        self,
        state: str,
        question: str,
        levels: List[str],
        min_confidence: Optional[float] = None,
    ) -> ScoreResult:
        agent = self._agent_instance()

        questions = {
            "decision": {
                "type": "score",
                "instructions": question,
                "criteria": levels,
            }
        }

        t0 = time.perf_counter()
        try:
            result = agent.decide(
                state,
                questions=questions,
                return_details=True,
                **({"min_confidence": min_confidence} if min_confidence is not None else {}),
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            dec = result.values.get("decision")
            # When using questions= format, values["decision"] is a dict with
            # {"type": "score", "score": <float>, "probabilities": {"0": p, "1": p, ...},
            #  "answer_confidence": ..., ...}
            if isinstance(dec, dict):
                probs_dict = dec.get("probabilities") or {}
                conf = dec.get("answer_confidence") or result.answer_confidence.get("decision") or 0.0
                abstained = False
                # pick argmax from probabilities (string keys "0", "1", ...)
                if probs_dict:
                    level_idx = int(max(probs_dict, key=lambda k: probs_dict[k]))
                else:
                    level_idx = min(len(levels) - 1, round(float(dec.get("score", 0))))
            else:
                abstained = dec is None
                probs_dict = {str(i): 0.0 for i in range(len(levels))}
                conf = result.answer_confidence.get("decision") or 0.0
                level_idx = 0

            dist = [float(probs_dict.get(str(i), 0.0)) for i in range(len(levels))]

            return ScoreResult(
                level=level_idx,
                probability=conf,
                distribution=dist,
                abstained=abstained,
                latency_ms=latency_ms,
            )
        except Exception:
            latency_ms = (time.perf_counter() - t0) * 1000
            return ScoreResult(level=0, probability=0.0, distribution=[], abstained=True, latency_ms=latency_ms)
