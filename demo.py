#!/usr/bin/env python3
"""NiceGUI demo — Jev vs Laya Decision Model Playground.

Run:
    python demo.py
Then open http://localhost:8080
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nicegui import run, ui

from benchmarks.datasets import (
    AGENT_ROUTING_CASES, AGENTS_5,
    GUARDRAIL_CASES, GUARDRAIL_QUESTION,
    LLM_ROUTING_CASES, LLM_TIERS,
    TOOL_SELECTION_CASES, TOOLS_10,
)


# ── model registry (lazy) ─────────────────────────────────────────────────────
_models: Dict[str, Any] = {}

def _get_model(name: str) -> Any:
    if name not in _models:
        if name == "jev":
            from models.jev_model import JevModel
            _models["jev"] = JevModel()
        else:
            from models.laya_model import LayaModel
            _models["laya"] = LayaModel()
    return _models[name]


# ── inference (blocking — runs in thread pool via run.io_bound) ───────────────
def _infer(
    model_name: str,
    dec_type: str,
    state_text: str,
    question: str,
    options: Optional[Dict[str, str]],
    levels: Optional[List[str]],
    min_confidence: Optional[float],
) -> Dict[str, Any]:
    try:
        m = _get_model(model_name)
    except ImportError as exc:
        raise RuntimeError(f"Model '{model_name}' not available: {exc}") from exc

    if dec_type == "choice":
        r = m.choice(state_text, question, options or {}, min_confidence)
        return dict(type="choice", model=model_name, choice=r.choice,
                    probability=r.probability, distribution=r.distribution,
                    abstained=r.abstained, latency_ms=r.latency_ms)
    elif dec_type == "noul":
        r = m.noul(state_text, question, min_confidence)
        return dict(type="noul", model=model_name, value=r.value,
                    probability=r.probability, abstained=r.abstained,
                    latency_ms=r.latency_ms)
    else:
        r = m.score(state_text, question, levels or [], min_confidence)
        return dict(type="score", model=model_name, level=r.level,
                    probability=r.probability, distribution=r.distribution,
                    abstained=r.abstained, latency_ms=r.latency_ms,
                    levels=levels or [])


# ── text helpers ──────────────────────────────────────────────────────────────
def _parse_opts(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if ":" in s:
            k, v = s.split(":", 1)
            out[k.strip()] = v.strip()
        else:
            out[s] = s
    return out

def _fmt_opts(d: Dict[str, str]) -> str:
    return "\n".join(f"{k}: {v}" for k, v in d.items())

def _parse_lvls(text: str) -> List[str]:
    return [l.strip() for l in text.splitlines() if l.strip()]


# ── presets ───────────────────────────────────────────────────────────────────
PRESETS = [
    dict(group="Agent Routing", type="choice",
         question="Which agent should handle this request?",
         opts=_fmt_opts(AGENTS_5), cases=AGENT_ROUTING_CASES[:6]),
    dict(group="LLM Routing", type="choice",
         question="Which language model tier should handle this request?",
         opts=_fmt_opts(LLM_TIERS), cases=LLM_ROUTING_CASES[:6]),
    dict(group="Tool Selection", type="choice",
         question="Which tool should be called to handle this request?",
         opts=_fmt_opts(TOOLS_10), cases=TOOL_SELECTION_CASES[:6]),
    dict(group="Guardrails", type="noul",
         question=GUARDRAIL_QUESTION,
         opts="", cases=GUARDRAIL_CASES[:6]),
]

SCORE_LEVELS_DEFAULT = (
    "Poor: incorrect, harmful, or completely off-topic\n"
    "Below average: partially correct but missing key details\n"
    "Average: correct but generic, lacks depth\n"
    "Good: accurate and relevant, minor gaps\n"
    "Excellent: comprehensive, accurate, well-structured"
)


# ── distribution chart ────────────────────────────────────────────────────────
def _dist_chart(result: Dict[str, Any]) -> None:
    dec_type = result["type"]

    if dec_type == "choice":
        dist = result.get("distribution") or {}
        if not dist:
            return
        sorted_items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        labels = [k for k, _ in sorted_items]
        values = [v for _, v in sorted_items]
        winner = result.get("choice", "")
        colors = ["#4ade80" if lbl == winner else "#60a5fa" for lbl in labels]

    elif dec_type == "score":
        dist = result.get("distribution") or []
        if not dist:
            return
        lvls = result.get("levels") or []
        win_idx = result.get("level", -1)
        labels = [f"L{i}: {lvls[i][:28]}" if i < len(lvls) else f"L{i}" for i in range(len(dist))]
        values = list(dist)
        colors = ["#4ade80" if i == win_idx else "#60a5fa" for i in range(len(dist))]

    else:
        return

    # Reverse so first item appears at top of horizontal bar chart
    rev_labels = labels[::-1]
    rev_data = [
        {"value": v, "itemStyle": {"color": c}}
        for v, c in zip(values[::-1], colors[::-1])
    ]

    chart_opts = {
        "animation": False,
        "grid": {"left": "38%", "right": "14%", "top": "4%", "bottom": "4%"},
        "tooltip": {"trigger": "axis"},
        "xAxis": {
            "type": "value", "max": 1.0,
            "axisLabel": {"formatter": "{value:.0%}", "fontSize": 10},
        },
        "yAxis": {
            "type": "category", "data": rev_labels,
            "axisLabel": {"fontSize": 10, "width": 145, "overflow": "truncate"},
        },
        "series": [{
            "type": "bar",
            "data": rev_data,
            "label": {"show": True, "position": "right", "formatter": "{c:.0%}", "fontSize": 10},
            "barMaxWidth": 22,
        }],
    }
    h = max(100, len(labels) * 34 + 30)
    ui.echart(chart_opts).classes("w-full").style(f"height: {h}px")


# ── result card ───────────────────────────────────────────────────────────────
def _result_card(result: Dict[str, Any]) -> None:
    dec_type = result["type"]
    model_name = result["model"]
    abstained = result.get("abstained", False)
    prob = result.get("probability", 0.0)
    latency = result.get("latency_ms", 0.0)

    badge_color = "purple" if model_name == "laya" else "blue"
    display_name = model_name.upper().replace("_STUB", " (stub)")

    with ui.card().classes("w-full").style("background:#334155"):
        with ui.row().classes("items-center gap-2 px-4 pt-3 pb-1"):
            ui.badge(display_name, color=badge_color)
            if abstained:
                ui.badge("ABSTAINED", color="orange")
            ui.space()
            ui.label(f"{latency:.1f} ms").classes("text-xs").style("color:#94a3b8")

        with ui.column().classes("px-4 pb-4 gap-2 w-full"):

            if dec_type == "choice":
                choice = result.get("choice") or "—"
                col = "#4ade80" if not abstained else "#f59e0b"
                ui.label(choice).classes("text-xl font-bold").style(f"color:{col}")
                ui.label(f"Confidence  {prob:.1%}").classes("text-sm").style("color:#94a3b8")
                dist = result.get("distribution") or {}
                if dist:
                    with ui.expansion("Probability distribution").classes("w-full").style("color:#cbd5e1"):
                        _dist_chart(result)

            elif dec_type == "noul":
                value = result.get("value", False)
                lbl = "UNSAFE / TRUE" if value else "SAFE / FALSE"
                col = "#f87171" if value else "#4ade80"
                ui.label(lbl).classes("text-xl font-bold").style(f"color:{col}")
                with ui.grid(columns=2).classes("w-full gap-x-4 gap-y-1 mt-1"):
                    ui.label(f"P(true)  {prob:.1%}").classes("text-sm").style("color:#94a3b8")
                    ui.label(f"P(false)  {1-prob:.1%}").classes("text-sm").style("color:#94a3b8")
                    ui.linear_progress(prob, color="red" if value else "green", size="6px")
                    ui.linear_progress(1 - prob, color="green" if not value else "grey", size="6px")

            else:  # score
                level = result.get("level", 0)
                lvls = result.get("levels") or []
                desc = lvls[level] if lvls and level < len(lvls) else f"Level {level}"
                ui.label(f"Level {level}").classes("text-2xl font-bold").style("color:#818cf8")
                ui.label(desc).classes("text-sm").style("color:#e2e8f0;word-break:break-word")
                ui.label(f"Confidence  {prob:.1%}").classes("text-sm").style("color:#94a3b8")
                dist = result.get("distribution") or []
                if dist:
                    with ui.expansion("Level distribution").classes("w-full").style("color:#cbd5e1"):
                        _dist_chart(result)


# ── shared header ────────────────────────────────────────────────────────────
def _render_header(current: str = "playground") -> None:
    _active = "color:#a5b4fc;font-weight:600;text-decoration:none"
    _dim    = "color:#94a3b8;text-decoration:none"
    with ui.header(elevated=True).style("background:#1e1b4b"):
        with ui.row().classes("items-center gap-3 px-6 w-full"):
            ui.icon("psychology", size="sm").classes("text-indigo-300")
            ui.label("Decision Model Playground").classes("text-base font-bold text-white")
            ui.badge("Jev  ·  Laya", color="indigo")
            ui.space()
            ui.link("Playground", "/").style(
                _active if current == "playground" else _dim
            ).classes("text-sm")
            ui.link("Architecture", "/architecture").style(
                _active if current == "architecture" else _dim
            ).classes("text-sm")


# ── architecture SVG ──────────────────────────────────────────────────────────
_ARCH_SVG = """
<svg viewBox="0 0 960 740" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;height:auto;display:block;font-family:ui-monospace,'Cascadia Code',monospace">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#475569"/>
    </marker>
  </defs>

  <rect width="960" height="740" fill="#0f172a"/>

  <!-- USER INPUT -->
  <rect x="40" y="15" width="880" height="60" fill="#1e293b" rx="8" stroke="#334155" stroke-width="1.5"/>
  <text x="480" y="33" fill="#64748b" font-size="9" text-anchor="middle" font-weight="700" letter-spacing="2">USER  INPUT</text>
  <rect x="58"  y="39" width="160" height="22" fill="#0f172a" rx="4" stroke="#475569" stroke-width="1"/>
  <text x="138" y="54" fill="#cbd5e1" font-size="9.5" text-anchor="middle">state  (context text)</text>
  <rect x="228" y="39" width="120" height="22" fill="#0f172a" rx="4" stroke="#475569" stroke-width="1"/>
  <text x="288" y="54" fill="#cbd5e1" font-size="9.5" text-anchor="middle">question</text>
  <rect x="358" y="39" width="175" height="22" fill="#0f172a" rx="4" stroke="#475569" stroke-width="1"/>
  <text x="445" y="54" fill="#cbd5e1" font-size="9.5" text-anchor="middle">options / levels</text>
  <rect x="543" y="39" width="155" height="22" fill="#0f172a" rx="4" stroke="#475569" stroke-width="1"/>
  <text x="620" y="54" fill="#cbd5e1" font-size="9.5" text-anchor="middle">decision type</text>
  <rect x="708" y="39" width="170" height="22" fill="#0f172a" rx="4" stroke="#475569" stroke-width="1"/>
  <text x="793" y="54" fill="#cbd5e1" font-size="9.5" text-anchor="middle">min_confidence  (opt)</text>

  <line x1="480" y1="75" x2="480" y2="108" stroke="#475569" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- DEMO APP -->
  <rect x="130" y="110" width="700" height="72" fill="#1e293b" rx="8" stroke="#334155" stroke-width="1.5"/>
  <text x="480" y="130" fill="#94a3b8" font-size="9" text-anchor="middle" font-weight="700" letter-spacing="2">DEMO APP  &#xB7;  demo.py</text>
  <text x="480" y="149" fill="#e2e8f0" font-size="11" text-anchor="middle" font-weight="500">_infer()  &#x2192;  run.io_bound()  (thread pool, non-blocking)</text>
  <text x="480" y="168" fill="#64748b" font-size="9.5" text-anchor="middle">lazy model registry  &#xB7;  async dispatch  &#xB7;  comparison strip when both models run</text>

  <line x1="480" y1="182" x2="480" y2="218" stroke="#475569" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- DecisionModel ABC -->
  <rect x="270" y="220" width="420" height="80" fill="#1e1b4b" rx="8" stroke="#6366f1" stroke-width="1.5"/>
  <text x="480" y="240" fill="#a5b4fc" font-size="9" text-anchor="middle" font-weight="700" letter-spacing="2">DecisionModel  (ABC)  &#xB7;  models/base.py</text>
  <text x="480" y="259" fill="#c7d2fe" font-size="10.5" text-anchor="middle">.choice( state, question, options )  &#x2192;  ChoiceResult</text>
  <text x="480" y="275" fill="#c7d2fe" font-size="10.5" text-anchor="middle">.noul( state, question )  &#x2192;  NoulResult</text>
  <text x="480" y="291" fill="#c7d2fe" font-size="10.5" text-anchor="middle">.score( state, question, levels )  &#x2192;  ScoreResult</text>

  <path d="M 360 300 L 215 380" stroke="#475569" stroke-width="1.5" fill="none" marker-end="url(#arr)"/>
  <path d="M 600 300 L 745 380" stroke="#475569" stroke-width="1.5" fill="none" marker-end="url(#arr)"/>

  <!-- LAYA -->
  <rect x="15" y="382" width="400" height="210" fill="#1e1b4b" rx="8" stroke="#a855f7" stroke-width="2"/>
  <text x="215" y="404" fill="#e9d5ff" font-size="11" text-anchor="middle" font-weight="700" letter-spacing="1">LAYA  &#xB7;  LOCAL  MODEL</text>
  <text x="215" y="421" fill="#7c3aed" font-size="9.5" text-anchor="middle">LayaModel  &#xB7;  models/laya_model.py</text>
  <line x1="35" y1="428" x2="395" y2="428" stroke="#2e1065" stroke-width="1"/>
  <text x="35" y="445" fill="#d8b4fe" font-size="9.5">&#x25B8;  Package:  laya  (pip install laya)</text>
  <text x="35" y="462" fill="#d8b4fe" font-size="9.5">&#x25B8;  Checkpoint:  convaiinnovations/laya  (~400 MB)</text>
  <text x="35" y="479" fill="#d8b4fe" font-size="9.5">&#x25B8;  Device:  auto  |  CPU  |  CUDA  |  MPS</text>
  <text x="35" y="496" fill="#d8b4fe" font-size="9.5">&#x25B8;  Latency:  100&#x2013;300 ms (CPU)   10&#x2013;30 ms (GPU)</text>
  <text x="35" y="513" fill="#d8b4fe" font-size="9.5">&#x25B8;  Shortlist pre-filter when options &gt; 10  (keeps k=8)</text>
  <text x="35" y="530" fill="#d8b4fe" font-size="9.5">&#x25B8;  Lazy init &#x2014; model downloads on first call</text>
  <text x="35" y="547" fill="#d8b4fe" font-size="9.5">&#x25B8;  Fully offline &#x2014; no network calls during inference</text>
  <text x="35" y="564" fill="#d8b4fe" font-size="9.5">&#x25B8;  Config:  LAYA_MODEL  /  LAYA_DEVICE  env vars</text>

  <!-- JEV -->
  <rect x="545" y="382" width="400" height="210" fill="#0c1a2e" rx="8" stroke="#3b82f6" stroke-width="2"/>
  <text x="745" y="404" fill="#bfdbfe" font-size="11" text-anchor="middle" font-weight="700" letter-spacing="1">JEV  &#xB7;  REMOTE  API</text>
  <text x="745" y="421" fill="#1d4ed8" font-size="9.5" text-anchor="middle">JevModel  &#xB7;  models/jev_model.py</text>
  <line x1="565" y1="428" x2="925" y2="428" stroke="#0c2044" stroke-width="1"/>
  <text x="565" y="445" fill="#93c5fd" font-size="9.5">&#x25B8;  Endpoint:  POST  api.jevai.net/v1/decide</text>
  <text x="565" y="462" fill="#93c5fd" font-size="9.5">&#x25B8;  Auth:  Authorization: Bearer {JEV_API_KEY}</text>
  <text x="565" y="479" fill="#93c5fd" font-size="9.5">&#x25B8;  Latency:  ~8&#x2013;12 ms  (real API)</text>
  <text x="565" y="496" fill="#93c5fd" font-size="9.5">&#x25B8;  Stub mode:  auto when JEV_API_KEY not set</text>
  <text x="565" y="513" fill="#93c5fd" font-size="9.5">&#x25B8;  Stub:  hash-seeded, ~70 % simulated accuracy</text>
  <text x="565" y="530" fill="#93c5fd" font-size="9.5">&#x25B8;  Stateless &#x2014; no local weights or cache</text>
  <text x="565" y="547" fill="#93c5fd" font-size="9.5">&#x25B8;  Config:  JEV_API_KEY  /  JEV_API_URL  env vars</text>
  <text x="565" y="564" fill="#93c5fd" font-size="9.5">&#x25B8;  Body:  { state, type, question, options/levels }</text>

  <path d="M 215 592 L 355 641" stroke="#475569" stroke-width="1.5" fill="none" marker-end="url(#arr)"/>
  <path d="M 745 592 L 605 641" stroke="#475569" stroke-width="1.5" fill="none" marker-end="url(#arr)"/>

  <!-- RESULT OBJECTS -->
  <rect x="185" y="643" width="590" height="80" fill="#1e293b" rx="8" stroke="#10b981" stroke-width="1.5"/>
  <text x="480" y="663" fill="#6ee7b7" font-size="9" text-anchor="middle" font-weight="700" letter-spacing="2">RESULT  OBJECTS  &#xB7;  models/base.py</text>
  <text x="480" y="681" fill="#e2e8f0" font-size="11" text-anchor="middle">ChoiceResult   &#xB7;   NoulResult   &#xB7;   ScoreResult</text>
  <text x="480" y="698" fill="#94a3b8" font-size="9.5" text-anchor="middle">choice/level/value  &#xB7;  probability  &#xB7;  distribution  &#xB7;  abstained  &#xB7;  latency_ms</text>
  <text x="480" y="714" fill="#475569" font-size="9" text-anchor="middle">abstained = True  when  probability &lt; min_confidence  (optional threshold)</text>
</svg>
"""


# ── architecture page ─────────────────────────────────────────────────────────
@ui.page("/architecture")
def architecture_page() -> None:
    ui.add_head_html(
        "<style>"
        "body{background:#0f172a!important}"
        ".nicegui-content{padding:0!important;max-width:100%!important}"
        "</style>"
    )
    _render_header("architecture")

    with ui.column().classes("w-full max-w-6xl mx-auto gap-6 p-4"):

        ui.label("System Architecture").classes("text-2xl font-bold text-indigo-300 mt-2")
        ui.label(
            "How Laya (local model) and Jev (remote API) implement the same DecisionModel "
            "interface to serve three decision types: choice, noul, and score."
        ).classes("text-sm text-slate-400 -mt-4")

        with ui.card().classes("w-full").style("background:#1e293b;padding:0;overflow:hidden"):
            ui.html(_ARCH_SVG)

        ui.label("Decision Types").classes(
            "text-base font-bold text-indigo-300 uppercase tracking-wider"
        )
        with ui.row().classes("w-full gap-4"):
            for badge_col, badge_lbl, title, body, use_for in [
                (
                    "indigo", "choice",
                    "Pick one from N labeled options",
                    "Returns the winning label key plus the full probability distribution over all "
                    "options. Laya applies a shortlist pre-filter (k=8) when option count exceeds 10.",
                    "Agent routing  ·  LLM tier selection  ·  tool dispatch  ·  classification",
                ),
                (
                    "green", "noul",
                    "Binary yes / no decision",
                    "Returns P(true) and a boolean value. Named from neural boolean. "
                    "Abstention available via min_confidence threshold.",
                    "Guardrails  ·  safety checks  ·  feature flags  ·  yes/no routing",
                ),
                (
                    "purple", "score",
                    "Ordinal level from a rubric",
                    "Returns the winning level index (0-based) plus a distribution over all levels. "
                    "Levels are user-defined strings ordered from lowest to highest.",
                    "Quality scoring  ·  severity triage  ·  priority ranking  ·  rubric eval",
                ),
            ]:
                with ui.card().classes("flex-1").style("background:#1e293b;min-width:200px"):
                    with ui.row().classes("items-center gap-2 mb-1"):
                        ui.badge(badge_lbl, color=badge_col)
                    ui.label(title).classes("text-sm font-semibold text-white")
                    ui.label(body).classes("text-xs text-slate-400 mt-1")
                    ui.separator().style("opacity:.2")
                    ui.label("Use for:").classes(
                        "text-xs text-slate-500 uppercase tracking-wider mt-1"
                    )
                    ui.label(use_for).classes("text-xs text-slate-300")

        ui.label("Model Comparison").classes(
            "text-base font-bold text-indigo-300 uppercase tracking-wider"
        )
        with ui.row().classes("w-full gap-4"):
            with ui.card().classes("flex-1").style(
                "background:#1e1b4b;border:1px solid #7c3aed;min-width:200px"
            ):
                ui.badge("LAYA", color="purple").classes("mb-2")
                for lbl, val in [
                    ("Type",      "Local model — runs on your hardware"),
                    ("Size",      "~400 MB  (first-run HuggingFace download)"),
                    ("Latency",   "~100–300 ms CPU  /  ~10–30 ms GPU"),
                    ("Network",   "None — fully offline inference"),
                    ("Config",    "LAYA_MODEL, LAYA_DEVICE env vars"),
                    ("Large opt", "Shortlist pre-filter when > 10 options"),
                ]:
                    with ui.row().classes("w-full gap-2 items-start"):
                        ui.label(lbl).classes("text-xs text-slate-500 w-20 flex-shrink-0")
                        ui.label(val).classes("text-xs text-slate-300")

            with ui.card().classes("flex-1").style(
                "background:#0c1a2e;border:1px solid #1d4ed8;min-width:200px"
            ):
                ui.badge("JEV", color="blue").classes("mb-2")
                for lbl, val in [
                    ("Type",    "Remote REST API — cloud inference"),
                    ("Size",    "No local weights"),
                    ("Latency", "~8–12 ms (real API)  /  ~8 ms stub"),
                    ("Network", "POST api.jevai.net/v1/decide"),
                    ("Config",  "JEV_API_KEY, JEV_API_URL env vars"),
                    ("No key",  "Auto-fallback to deterministic stub"),
                ]:
                    with ui.row().classes("w-full gap-2 items-start"):
                        ui.label(lbl).classes("text-xs text-slate-500 w-20 flex-shrink-0")
                        ui.label(val).classes("text-xs text-slate-300")


@ui.page("/")
def page() -> None:
    ui.add_head_html(
        "<style>"
        "body{background:#0f172a!important}"
        ".nicegui-content{padding:0!important;max-width:100%!important}"
        "</style>"
    )

    _render_header("playground")

    # UI element refs — filled during layout construction
    r: Dict[str, Any] = {}

    with ui.column().classes("w-full max-w-6xl mx-auto gap-4 p-4"):

        # ── presets panel ─────────────────────────────────────────────────────
        with ui.expansion("Quick Presets", icon="bolt").classes("w-full").style(
            "background:#1e293b;border-radius:8px;color:white"
        ):
            with ui.tabs().classes("text-white") as tabs_el:
                for p in PRESETS:
                    ui.tab(p["group"])

            with ui.tab_panels(tabs_el, value=PRESETS[0]["group"]).classes("bg-transparent"):
                for preset in PRESETS:
                    with ui.tab_panel(preset["group"]):
                        with ui.row().classes("flex-wrap gap-2"):
                            for case in preset["cases"]:
                                gt_raw = case.get("ground_truth",
                                                   "unsafe" if case.get("unsafe") else "safe")
                                s = case["state"]
                                short = s[:52] + ("…" if len(s) > 52 else "")

                                def _load(p=preset, txt=s):
                                    r["dec_type"].set_value(p["type"])
                                    r["state_ta"].set_value(txt)
                                    r["question"].set_value(p["question"])
                                    dt = p["type"]
                                    if dt == "choice":
                                        r["opts_ta"].set_value(p["opts"])
                                    r["opts_row"].set_visibility(dt == "choice")
                                    r["lvls_row"].set_visibility(dt == "score")

                                ui.button(
                                    f"[{gt_raw}] {short}",
                                    on_click=_load,
                                ).props("flat no-caps dense").classes(
                                    "text-xs text-left bg-slate-700 text-slate-200 rounded px-2"
                                ).style("max-width:440px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap")

        # ── two-column layout ─────────────────────────────────────────────────
        with ui.row().classes("w-full gap-4 items-start"):

            # INPUT CARD ───────────────────────────────────────────────────────
            with ui.card().classes("flex-1").style("background:#1e293b;min-width:320px"):
                ui.label("Input").classes("text-sm font-bold text-indigo-300 uppercase tracking-wider")
                ui.separator().style("opacity:.2")

                r["dec_type"] = ui.select(
                    {"choice": "choice — pick one from labeled options",
                     "noul":   "noul — yes / no binary",
                     "score":  "score — ordinal level"},
                    label="Decision Type", value="choice",
                ).classes("w-full")

                r["model"] = ui.select(
                    {"jev":  "Jev  (API / stub mode when no key set)",
                     "laya": "Laya  (local model, ~400 MB first run)",
                     "both": "Both  (side-by-side comparison)"},
                    label="Model", value="jev",
                ).classes("w-full")

                r["state_ta"] = ui.textarea(
                    label="State  (context / input text)",
                    placeholder="Enter the text the model should reason about…",
                ).classes("w-full").props("rows=4 outlined")

                r["question"] = ui.input(
                    label="Question",
                    placeholder="What decision should the model make?",
                ).classes("w-full").props("outlined")

                # options section (choice)
                r["opts_row"] = ui.column().classes("w-full gap-1")
                with r["opts_row"]:
                    ui.label("Options  (label: description, one per line)").classes("text-xs text-slate-400")
                    r["opts_ta"] = ui.textarea(
                        value=_fmt_opts(AGENTS_5),
                        placeholder="agent_a: handles SQL and data pipelines\nagent_b: writes code",
                    ).classes("w-full font-mono text-xs").props("rows=6 outlined")

                # levels section (score)
                r["lvls_row"] = ui.column().classes("w-full gap-1")
                with r["lvls_row"]:
                    ui.label("Levels  (ordered: lowest → highest, one per line)").classes("text-xs text-slate-400")
                    r["lvls_ta"] = ui.textarea(
                        value=SCORE_LEVELS_DEFAULT,
                        placeholder="Level 0: lowest\nLevel 1: medium\nLevel 2: highest",
                    ).classes("w-full font-mono text-xs").props("rows=5 outlined")
                r["lvls_row"].set_visibility(False)

                # noul hint
                r["noul_hint"] = ui.label(
                    "No extra inputs needed — the model returns P(true) and P(false)."
                ).classes("text-xs text-slate-400 italic mt-1")
                r["noul_hint"].set_visibility(False)

                def _on_type_change(e) -> None:
                    dt = e.args
                    r["opts_row"].set_visibility(dt == "choice")
                    r["lvls_row"].set_visibility(dt == "score")
                    r["noul_hint"].set_visibility(dt == "noul")

                r["dec_type"].on("update:model-value", _on_type_change)

                # min confidence
                with ui.row().classes("items-center gap-3 w-full mt-1"):
                    use_conf = ui.checkbox("Abstention threshold").classes("text-sm text-slate-300")
                    conf_num = ui.number(
                        min=0.50, max=0.99, step=0.01, value=0.80, format="%.2f",
                        label="min_conf",
                    ).classes("w-28").props("outlined dense")
                    conf_num.bind_visibility_from(use_conf, "value")

                with ui.row().classes("justify-end items-center gap-2 mt-3"):
                    r["spinner"] = ui.spinner("dots", size="sm", color="indigo").style("visibility:hidden")
                    r["run_btn"] = ui.button("Run", icon="play_arrow").props("unelevated color=indigo")

            # RESULTS CARD ─────────────────────────────────────────────────────
            with ui.card().classes("flex-1").style("background:#1e293b;min-width:320px"):
                ui.label("Results").classes("text-sm font-bold text-indigo-300 uppercase tracking-wider")
                ui.separator().style("opacity:.2")
                r["results"] = ui.column().classes("w-full gap-3")
                with r["results"]:
                    ui.label("Run a query to see results.").classes(
                        "text-slate-500 italic py-8 text-center w-full"
                    )

    # ── run handler ───────────────────────────────────────────────────────────
    async def on_run() -> None:
        dec_type: str = r["dec_type"].value   # "choice" | "noul" | "score"
        model: str = r["model"].value          # "jev" | "laya" | "both"
        state_text: str = (r["state_ta"].value or "").strip()
        question: str = (r["question"].value or "").strip()

        if not state_text:
            ui.notify("State text is required.", type="warning")
            return
        if not question:
            ui.notify("Question is required.", type="warning")
            return

        options: Optional[Dict[str, str]] = None
        levels: Optional[List[str]] = None

        if dec_type == "choice":
            options = _parse_opts(r["opts_ta"].value or "")
            if len(options) < 2:
                ui.notify("Need at least 2 options for choice.", type="warning")
                return

        elif dec_type == "score":
            levels = _parse_lvls(r["lvls_ta"].value or "")
            if len(levels) < 2:
                ui.notify("Need at least 2 levels for score.", type="warning")
                return

        min_conf: Optional[float] = float(conf_num.value) if use_conf.value else None
        models_to_run = ["jev", "laya"] if model == "both" else [model]

        r["run_btn"].disable()
        r["spinner"].style("visibility:visible")
        r["results"].clear()

        results: List[Dict[str, Any]] = []
        error: Optional[str] = None

        try:
            for m in models_to_run:
                res = await run.io_bound(
                    _infer, m, dec_type, state_text, question, options, levels, min_conf
                )
                results.append(res)
        except Exception as exc:
            error = str(exc)

        if error:
            ui.notify(f"Error: {error}", type="negative", timeout=8000)
            with r["results"]:
                ui.label(f"Error: {error}").classes("text-red-400 text-sm")
        else:
            with r["results"]:
                for res in results:
                    _result_card(res)

                # comparison strip when both models ran
                if len(results) == 2:
                    r0, r1 = results[0], results[1]
                    with ui.card().classes("w-full mt-1").style("background:#475569"):
                        ui.label("Comparison").classes(
                            "text-xs font-bold text-indigo-300 uppercase px-4 pt-3"
                        )
                        with ui.row().classes("px-4 pb-3 gap-8 flex-wrap"):
                            for res in [r0, r1]:
                                nm = res["model"].upper().replace("_STUB", " ★")
                                with ui.column().classes("gap-0"):
                                    ui.label(nm).classes("text-sm font-bold text-white")
                                    ui.label(f"Confidence: {res['probability']:.1%}").classes("text-xs text-slate-300")
                                    ui.label(f"Latency: {res['latency_ms']:.1f} ms").classes("text-xs text-slate-300")
                                    if res["type"] == "choice":
                                        ui.label(f"→ {res.get('choice') or '—'}").classes("text-xs font-medium text-white")
                                    elif res["type"] == "score":
                                        ui.label(f"→ Level {res.get('level', '?')}").classes("text-xs font-medium text-white")
                                    elif res["type"] == "noul":
                                        val_str = "TRUE" if res.get("value") else "FALSE"
                                        ui.label(f"→ {val_str}").classes("text-xs font-medium text-white")

                        if r0["type"] == "choice":
                            agree = r0.get("choice") == r1.get("choice")
                            col = "#4ade80" if agree else "#fbbf24"
                            with ui.row().classes("px-4 pb-2"):
                                ui.label(
                                    "✓ Models agree" if agree else "✗ Models disagree"
                                ).style(f"font-size:13px;color:{col}")
                        elif r0["type"] == "noul":
                            agree = r0.get("value") == r1.get("value")
                            col = "#4ade80" if agree else "#fbbf24"
                            with ui.row().classes("px-4 pb-2"):
                                ui.label(
                                    "✓ Models agree" if agree else "✗ Models disagree"
                                ).style(f"font-size:13px;color:{col}")

        r["run_btn"].enable()
        r["spinner"].style("visibility:hidden")

    r["run_btn"].on("click", on_run)


ui.run(title="Decision Model Playground", port=8080, dark=True, favicon="🧠")
