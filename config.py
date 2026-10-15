"""Configuration — reads from environment variables.

Set before running:
    export JEV_API_KEY=your_key_here          # activates real Jev API
    export JEV_API_URL=https://...            # optional URL override
    export LAYA_MODEL=convaiinnovations/laya  # optional model override
    export LAYA_DEVICE=cuda                   # optional: cpu | cuda | mps
"""
import os

JEV_API_KEY: str = os.environ.get("JEV_API_KEY", "")
JEV_API_URL: str = os.environ.get("JEV_API_URL", "https://api.jevai.net/v1/decide")

LAYA_MODEL: str = os.environ.get("LAYA_MODEL", "convaiinnovations/laya")
LAYA_DEVICE: str | None = os.environ.get("LAYA_DEVICE") or None  # None = auto

# Benchmark defaults
DEFAULT_MIN_CONFIDENCE: float | None = None   # None = no abstention gate
