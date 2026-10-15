"""Abstract interface for System 1 decision models (Jev, Laya, etc.)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ChoiceResult:
    choice: str                          # winning label key
    probability: float                   # P(choice)
    distribution: Dict[str, float]       # full label → probability map
    abstained: bool = False              # True when below min_confidence threshold
    latency_ms: float = 0.0


@dataclass
class NoulResult:
    value: bool                          # True = unsafe/yes/positive
    probability: float                   # P(true)
    abstained: bool = False
    latency_ms: float = 0.0


@dataclass
class ScoreResult:
    level: int                           # 0-based ordinal index
    probability: float                   # P(level)
    distribution: List[float] = field(default_factory=list)
    abstained: bool = False
    latency_ms: float = 0.0


class DecisionModel(ABC):
    """Minimal interface every decision model must implement."""

    @abstractmethod
    def choice(
        self,
        state: str,
        question: str,
        options: Dict[str, str],          # label → description
        min_confidence: Optional[float] = None,
    ) -> ChoiceResult: ...

    @abstractmethod
    def noul(
        self,
        state: str,
        question: str,
        min_confidence: Optional[float] = None,
    ) -> NoulResult: ...

    @abstractmethod
    def score(
        self,
        state: str,
        question: str,
        levels: List[str],                # ordered list of level descriptions
        min_confidence: Optional[float] = None,
    ) -> ScoreResult: ...

    @abstractmethod
    def name(self) -> str: ...
