"""MARITIME SENTINEL // MISSION CONTROL.

High-density Streamlit telemetry surface for the maritime supply-chain
stress-testing engine. Everything on screen is derived from a live discrete
event simulation or from the Monte Carlo ensemble - there are no hard-coded
numbers.

    streamlit run app/mission_control.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from maritime_sentinel.config import DEFAULT_SCENARIO
from maritime_sentinel.engine import simulate
from maritime_sentinel.graph.build import graph_mermaid, run_workflow
from maritime_sentinel.graph.state import RunContext
from maritime_sentinel.metrics import compute_metrics, risk_band, throughput_timeline
from maritime_sentinel.montecarlo import run_monte_carlo, scenario_key
from maritime_sentinel.schemas import (
    MonteCarloSpec,
    PerturbationSpec,
    ScenarioSpec,
    ShockSpec,
    StressSpec,
)
from maritime_sentinel.topology import build_graph, criticality

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

INK = "#0b0f14"
PANEL = "#111823"
GRID = "#1e2b3a"
TEXT = "#c8d6e5"
MUTED = "#6b8199"
CYAN = "#2ee6c5"
AMBER = "#ffb648"
RED = "#ff5c72"
BLUE = "#59a6ff"
VIOLET = "#a97bff"

RISK_COLOR = {
    "NOMINAL": CYAN,
    "GUARDED": BLUE,
    "ELEVATED": AMBER,
    "CRITICAL": RED,
}

st.set_page_config(
    page_title="MARITIME SENTINEL // Mission Control",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
<style>
  .stApp {{
      background:
        radial-gradient(1200px 600px at 15% -10%, #12202e 0%, transparent 60%),
        radial-gradient(900px 500px at 95% 0%, #101d2b 0%, transparent 55%),
        {INK};
      color: {TEXT};
  }}
  html, body, [class*="css"] {{
      font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
  }}
  section[data-testid="stSidebar"] {{
      background: linear-gradient(180deg, #0d141d 0%, #0a0f16 100%);
      border-right: 1px solid {GRID};
  }}
  h1, h2, h3, h4 {{ letter-spacing: 0.04em; }}
  .mc-header {{
      display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
      gap: 12px; padding: 14px 18px; margin-bottom: 10px;
      border: 1px solid {GRID}; border-radius: 10px;
      background: linear-gradient(90deg, rgba(46,230,197,0.06), rgba(89,166,255,0.03) 45%, transparent);
  }}
  .mc-title {{ font-size: 1.30rem; font-weight: 700; letter-spacing: 0.16em; color: #eaf6ff; }}
  .mc-sub {{ font-size: 0.72rem; color: {MUTED}; letter-spacing: 0.08em; margin-top: 2px; }}
  .pill {{
      display: inline-flex; align-items: center; gap: 6px;
      padding: 4px 11px; border-radius: 999px; font-size: 0.68rem;
      letter-spacing: 0.10em; text-transform: uppercase;
      border: 1px solid {GRID}; background: #0e161f; color: {TEXT};
      white-space: nowrap;
  }}
  .pill b {{ color: #ffffff; font-weight: 600; }}
  .dot {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; }}
  .kpi {{
      border: 1px solid {GRID}; border-radius: 10px; padding: 11px 13px;
      background: linear-gradient(160deg, #101a25, #0c131b);
      min-height: 86px;
  }}
  .kpi .k-label {{ font-size: 0.63rem; letter-spacing: 0.13em; color: {MUTED}; text-transform: uppercase; }}
  .kpi .k-value {{ font-size: 1.42rem; font-weight: 700; color: #eef7ff; line-height: 1.25; margin-top: 4px; }}
  .kpi .k-note {{ font-size: 0.63rem; color: {MUTED}; margin-top: 3px; }}
  .kpi.accent {{ border-left: 2px solid {CYAN}; }}
  .kpi.warn {{ border-left: 2px solid {AMBER}; }}
  .kpi.bad {{ border-left: 2px solid {RED}; }}
  div[data-testid="stMetricValue"] {{ font-size: 1.3rem; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {GRID}; }}
  .stTabs [data-baseweb="tab"] {{
      background: transparent; border: 1px solid transparent; border-radius: 8px 8px 0 0;
      padding: 7px 15px; font-size: 0.74rem; letter-spacing: 0.10em; text-transform: uppercase;
      color: {MUTED};
  }}
  .stTabs [aria-selected="true"] {{
      background: #101a25; border-color: {GRID}; border-bottom-color: #101a25; color: {CYAN};
  }}
  div[data-testid="stExpander"] {{ border: 1px solid {GRID}; border-radius: 8px; background: #0d141c; }}
  .stDataFrame {{ border: 1px solid {GRID}; border-radius: 8px; }}
  .trace-node {{ font-size: 0.72rem; color: {MUTED}; }}
  .mono-note {{ font-size: 0.68rem; color: {MUTED}; letter-spacing: 0.05em; }}
  code {{ color: {CYAN}; }}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def style_fig(fig: go.Figure, height: int = 300, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin={"l": 44, "r": 18, "t": 34 if title else 14, "b": 34},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "ui-monospace, monospace", "size": 10, "color": TEXT},
        hoverlabel={"bgcolor": "#0d1720", "bordercolor": GRID, "font_size": 10},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "font_size": 9,
            "bgcolor": "rgba(0,0,0,0)",
        },
    )
    if title:
        fig.update_layout(
            title={"text": title, "font": {"size": 11, "color": MUTED}, "x": 0, "xanchor": "left"}
        )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def pchart(fig: go.Figure, key: str) -> None:
    """Render a Plotly figure with an explicit key (Streamlit needs unique ids)."""

    st.plotly_chart(fig, width="stretch", key=key)


def kpi(label: str, value: str, note: str = "", tone: str = "") -> str:
    cls = f"kpi {tone}".strip()
    return (
        f'<div class="{cls}"><div class="k-label">{label}</div>'
        f'<div class="k-value">{value}</div><div class="k-note">{note}</div></div>'
    )


def pill(label: str, value: str, color: str = CYAN) -> str:
    return (
        f'<span class="pill"><span class="dot" style="background:{color};'
        f'box-shadow:0 0 7px {color}"></span>{label} <b>{value}</b></span>'
    )


# ---------------------------------------------------------------------------
# Cached execution
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def run_reference(spec_json: str) -> dict:
    spec = ScenarioSpec.model_validate_json(spec_json)
    result = simulate(spec)
    return {
        "telemetry": result.telemetry,
        "vessels": result.vessels,
        "ports": pd.DataFrame(result.ports),
        "events": result.events,
        "weather": result.weather_timeline,
        "metrics": compute_metrics(result),
        "timeline": throughput_timeline(result),
        "diverged": result.diverged,
        "divergence_reason": result.divergence_reason,
        "runtime": result.runtime_seconds,
    }


@st.cache_data(show_spinner=False)
def run_ensemble(spec_json: str, reps: int, seed: int) -> dict:
    spec = ScenarioSpec.model_validate_json(spec_json)
    mc = run_monte_carlo(spec, n_replications=reps, seed=seed)
    return {
        "outcomes": mc.outcomes,
        "samples": mc.samples,
        "fan": mc.fan_chart,
        "ports": mc.port_table,
        "tornado": mc.tornado,
        "aggregate": mc.aggregate,
        "baseline": mc.baseline_throughput,
        "baseline_effective": mc.baseline.effective_throughput_teu_day,
        "runtime": mc.runtime_seconds,
    }


# ---------------------------------------------------------------------------
# Scenario controls
# ---------------------------------------------------------------------------


def sidebar_controls() -> tuple[ScenarioSpec, dict]:
    base = DEFAULT_SCENARIO
    sb = st.sidebar
    sb.markdown("<div class='mc-title' style='font-size:0.9rem'>CONTROL DECK</div>", unsafe_allow_html=True)
    sb.markdown("---")

    with sb.expander("STRESS ENVELOPE", expanded=True):
        horizon = st.slider("Horizon (days)", 60.0, 365.0, float(base.stress.horizon_days), 10.0)
        warmup = st.slider("Warm-up (days)", 5.0, 60.0, float(base.stress.warmup_days), 5.0)
        demand = st.slider("Demand multiplier", 0.70, 1.50, float(base.stress.demand_multiplier), 0.01)
        weather = st.slider("Weather intensity", 0.50, 2.50, float(base.stress.weather_intensity), 0.05)
        geo = st.slider("Geopolitical friction", 0.00, 2.50, float(base.stress.geo_intensity), 0.05)
        svc_cv = st.slider("Handling variability (CV)", 0.10, 0.90, float(base.stress.service_cv), 0.02)
        season = st.selectbox("Monsoon season", ["NW_MONSOON", "SE_MONSOON", "TRANSITION"], index=0)
        policy = st.selectbox("Routing policy", ["adaptive", "static"], index=0)

    with sb.expander("CAPACITY & SHOCK", expanded=False):
        capacity_scale = st.slider(
            "Berth capacity scale", 0.80, 1.80, float(base.stress.capacity_scale), 0.01
        )
        st.caption("as-built 1.00 · the AM-SES build graph prescribes ~1.21 for this envelope")
        shock_on = st.checkbox("Scheduled Malacca blockade", value=True)
        shock_start = st.slider("Shock start (day)", 10.0, float(max(20.0, horizon - 30.0)), 55.0, 1.0)
        shock_days = st.slider("Shock duration (days)", 3.0, 60.0, 18.0, 1.0)
        shock_sev = st.slider("Shock severity", 0.2, 0.95, 0.72, 0.01)

    with sb.expander("MONTE CARLO", expanded=True):
        reps = st.slider("Replications", 4, 96, 32, 2)
        seed = st.number_input("Master seed", 0, 999_999, int(base.seed), 1)
        ranges = {}
        for p in base.montecarlo.perturbations:
            lo, hi = st.slider(
                p.label or p.key,
                round(p.low * 0.5, 2),
                round(p.high * 1.5, 2),
                (float(p.low), float(p.high)),
                0.01,
            )
            ranges[p.key] = (lo, hi)

    with sb.expander("BUILD PIPELINE", expanded=False):
        max_retries = st.slider("Max self-healing retries", 0, 3, 3, 1)
        fault_mode = st.selectbox(
            "Fault injection", ["none", "schema", "runtime", "divergence", "timeout"], index=0
        )
        run_build = st.button("RUN AM-SES BUILD GRAPH", width="stretch")

    shock = (
        ShockSpec(
            id="SHOCK-MALACCA",
            target="MALACCA",
            kind="blockade",
            start_day=float(shock_start),
            duration_days=float(shock_days),
            severity=float(shock_sev),
            description="Scheduled Malacca blockade used to measure resilience and recovery.",
        )
        if shock_on
        else None
    )

    stress = StressSpec(
        horizon_days=float(horizon),
        warmup_days=float(warmup),
        demand_multiplier=float(demand),
        weather_intensity=float(weather),
        geo_intensity=float(geo),
        service_cv=float(svc_cv),
        season=season,
        routing_policy=policy,
        capacity_scale=float(capacity_scale),
        shock=shock,
    )
    perturbations = [
        PerturbationSpec(key=p.key, low=float(ranges[p.key][0]), high=float(ranges[p.key][1]), label=p.label)
        for p in base.montecarlo.perturbations
    ]
    montecarlo = MonteCarloSpec(
        n_replications=int(reps),
        baseline_replications=base.montecarlo.baseline_replications,
        perturbations=perturbations,
        acceptance=base.montecarlo.acceptance,
        percentiles=base.montecarlo.percentiles,
    )
    spec = ScenarioSpec(
        spec_id=base.spec_id,
        title=base.title,
        seed=int(seed),
        network=base.network,
        stress=stress,
        montecarlo=montecarlo,
    )
    sb.markdown(
        f"<div class='mono-note'>network {len(spec.network.ports)} ports · "
        f"{len(spec.network.chokepoints)} chokepoints<br>"
        f"offered demand {spec.nominal_daily_teu() * demand:,.0f} TEU/day</div>",
        unsafe_allow_html=True,
    )
    return spec, {
        "reps": int(reps),
        "seed": int(seed),
        "max_retries": int(max_retries),
        "fault_mode": fault_mode,
        "run_build": run_build,
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def chart_fan(fan: pd.DataFrame, baseline: float) -> go.Figure:
    fig = go.Figure()
    if not fan.empty:
        fig.add_trace(
            go.Scatter(x=fan["t_day"], y=fan["p95"], line={"width": 0}, showlegend=False, hoverinfo="skip")
        )
        fig.add_trace(
            go.Scatter(
                x=fan["t_day"],
                y=fan["p05"],
                fill="tonexty",
                line={"width": 0},
                fillcolor="rgba(46,230,197,0.14)",
                name="p05-p95",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(x=fan["t_day"], y=fan["p75"], line={"width": 0}, showlegend=False, hoverinfo="skip")
        )
        fig.add_trace(
            go.Scatter(
                x=fan["t_day"],
                y=fan["p25"],
                fill="tonexty",
                line={"width": 0},
                fillcolor="rgba(46,230,197,0.28)",
                name="p25-p75",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(x=fan["t_day"], y=fan["p50"], line={"color": CYAN, "width": 2}, name="median")
        )
    fig.add_hline(
        y=baseline,
        line={"color": BLUE, "width": 1.4, "dash": "dot"},
        annotation_text=f"baseline {baseline:,.0f}",
        annotation_font_size=9,
        annotation_font_color=BLUE,
    )
    fig.update_yaxes(title_text="TEU/day")
    fig.update_xaxes(title_text="day")
    return style_fig(fig, 320, "THROUGHPUT ENVELOPE // MONTE CARLO ENSEMBLE")


def chart_backlog(timeline: pd.DataFrame, gate: float) -> go.Figure:
    fig = go.Figure()
    if not timeline.empty:
        fig.add_trace(
            go.Scatter(
                x=timeline["t_day"],
                y=timeline["queue_total"],
                line={"color": AMBER, "width": 1.6},
                name="queued vessels",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=timeline["t_day"],
                y=timeline["inflight"],
                line={"color": BLUE, "width": 1.2, "dash": "dot"},
                name="in flight",
            )
        )
    fig.update_yaxes(title_text="vessels")
    fig.update_xaxes(title_text="day")
    return style_fig(fig, 240, f"SYSTEM BACKLOG // SLOPE GATE {gate:.1f} VESSELS/DAY")


def chart_utilization(ports: pd.DataFrame, gate: float) -> go.Figure:
    frame = ports.sort_values("utilization", ascending=True)
    colors = [RED if u > gate else (AMBER if u > gate - 0.08 else CYAN) for u in frame["utilization"]]
    fig = go.Figure(
        go.Bar(
            x=frame["utilization"],
            y=frame["name"],
            orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            text=[f"{u:.0%}" for u in frame["utilization"]],
            textposition="outside",
            textfont={"size": 9},
            hovertemplate="%{y}<br>rho=%{x:.3f}<extra></extra>",
        )
    )
    fig.add_vline(
        x=gate,
        line={"color": RED, "width": 1.2, "dash": "dash"},
        annotation_text=f"gate {gate:.2f}",
        annotation_font_size=9,
        annotation_font_color=RED,
    )
    fig.update_xaxes(
        range=[0, max(1.0, float(frame["utilization"].max()) * 1.25)], title_text="berth utilisation"
    )
    return style_fig(fig, 260, "BERTH UTILISATION // STABILITY GATE")


def chart_network(spec: ScenarioSpec, ports: pd.DataFrame) -> go.Figure:
    coords = {p.id: (p.lon, p.lat) for p in spec.network.ports}
    for c in spec.network.chokepoints:
        coords[c.id] = (c.lon, c.lat)
    util = dict(zip(ports["port_id"], ports["utilization"], strict=False))
    teu = dict(zip(ports["port_id"], ports["teu"], strict=False))
    fig = go.Figure()
    for lane in spec.network.lanes:
        if lane.a not in coords or lane.b not in coords:
            continue
        x0, y0 = coords[lane.a]
        x1, y1 = coords[lane.b]
        color = AMBER if lane.contingency else GRID
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line={
                    "color": color,
                    "width": 2.0 if lane.contingency else 1.1,
                    "dash": "dot" if lane.contingency else "solid",
                },
                hoverinfo="text",
                text=[f"{lane.a}-{lane.b} · {lane.distance_nm:.0f}nm · {lane.zone}"],
                showlegend=False,
            )
        )
    px = [p.lon for p in spec.network.ports]
    py = [p.lat for p in spec.network.ports]
    labels = [p.name for p in spec.network.ports]
    sizes = [
        12 + 26 * min(1.0, teu.get(p.id, 0) / max(1.0, max(teu.values() or [1]))) for p in spec.network.ports
    ]
    colors = [
        RISK_COLOR.get("CRITICAL", RED)
        if util.get(p.id, 0) > 0.92
        else (AMBER if util.get(p.id, 0) > 0.84 else CYAN)
        for p in spec.network.ports
    ]
    fig.add_trace(
        go.Scatter(
            x=px,
            y=py,
            mode="markers+text",
            text=labels,
            textposition="top center",
            textfont={"size": 9, "color": TEXT},
            marker={
                "size": sizes,
                "color": colors,
                "line": {"width": 1, "color": "#04121a"},
                "opacity": 0.92,
            },
            customdata=[[util.get(p.id, 0), teu.get(p.id, 0), p.berths] for p in spec.network.ports],
            hovertemplate="%{text}<br>rho=%{customdata[0]:.2f}<br>TEU=%{customdata[1]:,.0f}"
            "<br>berths=%{customdata[2]}<extra></extra>",
            name="terminals",
        )
    )
    cx = [c.lon for c in spec.network.chokepoints]
    cy = [c.lat for c in spec.network.chokepoints]
    fig.add_trace(
        go.Scatter(
            x=cx,
            y=cy,
            mode="markers+text",
            text=[c.name for c in spec.network.chokepoints],
            textposition="bottom center",
            textfont={"size": 8, "color": MUTED},
            marker={
                "size": 11,
                "color": VIOLET,
                "symbol": "diamond",
                "line": {"width": 1, "color": "#0b0f14"},
            },
            hovertemplate="%{text}<extra>chokepoint</extra>",
            name="chokepoints",
        )
    )
    fig.update_xaxes(range=[94, 122], title_text="longitude", showgrid=True)
    fig.update_yaxes(range=[-11.5, 7.5], title_text="latitude", scaleanchor="x", scaleratio=1)
    return style_fig(fig, 470, "MARITIME LANE GRAPH // INDONESIA")


def chart_queue_heatmap(telemetry: pd.DataFrame, port_ids: list[str]) -> go.Figure:
    if telemetry.empty:
        return style_fig(go.Figure(), 280)
    cols = [f"q_{p}" for p in port_ids if f"q_{p}" in telemetry.columns]
    if not cols:
        return style_fig(go.Figure(), 280)
    z = telemetry[cols].to_numpy(dtype=float).T
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=telemetry["t_day"],
            y=[c.replace("q_", "") for c in cols],
            colorscale=[[0, "#0d1a24"], [0.35, "#134e4a"], [0.7, AMBER], [1, RED]],
            colorbar={"thickness": 9, "len": 0.75, "tickfont": {"size": 8}},
            hovertemplate="day %{x:.0f}<br>%{y}: %{z:.0f} waiting<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="day")
    return style_fig(fig, 280, "BERTH QUEUE HEATMAP // VESSELS WAITING")


def chart_weather(weather: pd.DataFrame) -> go.Figure:
    zones = sorted(weather["zone"].unique()) if not weather.empty else []
    if not zones:
        return style_fig(go.Figure(), 240)
    horizon = float(weather["day"].max())
    grid = np.linspace(0, horizon, 220)
    z = []
    for zone in zones:
        rows = weather[weather["zone"] == zone].sort_values("day")
        days = rows["day"].to_numpy(dtype=float)
        states = rows["state"].to_numpy(dtype=float)
        idx = np.searchsorted(days, grid, side="right") - 1
        idx = np.clip(idx, 0, len(states) - 1)
        z.append(states[idx])
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=grid,
            y=zones,
            colorscale=[[0, "#0d2430"], [0.33, "#1b6d63"], [0.66, AMBER], [1, RED]],
            zmin=0,
            zmax=3,
            colorbar={
                "thickness": 9,
                "len": 0.75,
                "tickvals": [0, 1, 2, 3],
                "ticktext": ["CALM", "ADVERSE", "SEVERE", "EXTREME"],
                "tickfont": {"size": 7},
            },
            hovertemplate="%{y}<br>day %{x:.0f}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="day")
    return style_fig(fig, 240, "SEA-STATE FIELD // MARKOV CHAIN OCCUPANCY")


def chart_events(events: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if events.empty:
        return style_fig(fig, 210, "DISRUPTION REGISTER // EMPTY")
    frame = events.copy()
    frame["duration"] = (frame["end_day"] - frame["start_day"]).clip(lower=0.2)
    palette = {"blockade": RED, "port_blackout": AMBER, "congestion_surge": VIOLET}
    for kind, group in frame.groupby("kind"):
        fig.add_trace(
            go.Bar(
                x=group["duration"],
                y=group["target"],
                base=group["start_day"],
                orientation="h",
                marker={"color": palette.get(kind, BLUE), "line": {"width": 0}},
                name=str(kind),
                opacity=0.85,
                customdata=group[["start_day", "severity", "label"]].to_numpy(),
                hovertemplate="%{y}<br>day %{customdata[0]:.0f} · sev %{customdata[1]:.2f}"
                "<br>%{customdata[2]}<extra></extra>",
            )
        )
    fig.update_xaxes(title_text="day")
    return style_fig(fig, 210, "DISRUPTION REGISTER // INCIDENT WINDOWS")


def chart_hist(values: np.ndarray, baseline: float, title: str, color: str) -> go.Figure:
    fig = go.Figure(
        go.Histogram(
            x=values,
            nbinsx=18,
            marker={"color": color, "line": {"width": 0}},
            opacity=0.85,
            hovertemplate="%{x:.3g}<br>count %{y}<extra></extra>",
        )
    )
    if baseline and np.isfinite(baseline):
        fig.add_vline(
            x=baseline,
            line={"color": BLUE, "width": 1.2, "dash": "dot"},
            annotation_text="baseline",
            annotation_font_size=9,
            annotation_font_color=BLUE,
        )
    return style_fig(fig, 230, title)


def chart_tornado(tornado: pd.DataFrame) -> go.Figure:
    if tornado.empty:
        return style_fig(go.Figure(), 250)
    column = "rho_resilience" if "rho_resilience" in tornado else "rho_throughput"
    frame = tornado.sort_values(column)
    colors = [RED if v < 0 else CYAN for v in frame[column]]
    fig = go.Figure(
        go.Bar(
            x=frame[column],
            y=frame["label"],
            orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            hovertemplate="%{y}<br>rho=%{x:.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line={"color": GRID, "width": 1})
    fig.update_xaxes(title_text="Spearman rho vs resilience index", range=[-1, 1])
    return style_fig(fig, 260, "SENSITIVITY // RANK CORRELATION")


def chart_scatter(outcomes: pd.DataFrame, samples: pd.DataFrame) -> go.Figure:
    frame = outcomes.merge(samples, on="replication", how="left", suffixes=("", "_in"))
    demand_col = "demand_multiplier" if "demand_multiplier" in frame else None
    x = frame[demand_col] if demand_col else frame["replication"]
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=frame["throughput_teu_day"],
            mode="markers",
            marker={
                "size": 9,
                "color": frame["resilience_index"],
                "colorscale": [[0, RED], [0.5, AMBER], [1, CYAN]],
                "colorbar": {
                    "thickness": 9,
                    "len": 0.75,
                    "tickfont": {"size": 8},
                    "title": {"text": "R", "font": {"size": 9}},
                },
                "line": {"width": 0.5, "color": "#0b0f14"},
                "showscale": True,
            },
            customdata=frame[["replication", "service_level", "max_utilization"]].to_numpy(),
            hovertemplate="rep %{customdata[0]}<br>%{y:,.0f} TEU/day<br>"
            "SL %{customdata[1]:.2f} · rho %{customdata[2]:.2f}<extra></extra>",
        )
    )
    if demand_col:
        fig.update_xaxes(title_text="demand multiplier")
    fig.update_yaxes(title_text="TEU/day")
    return style_fig(fig, 280, "REPLICATE CLOUD // RESILIENCE COLOURED")


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def render_header(spec: ScenarioSpec, metrics: dict, agg: dict) -> None:
    band = risk_band(metrics)
    color = RISK_COLOR.get(band, CYAN)
    divergence = metrics.get("diverged", 0.0) > 0
    status = "DIVERGED" if divergence else band
    st.markdown(
        f"""
        <div class="mc-header">
          <div>
            <div class="mc-title">MARITIME SENTINEL <span style="color:{MUTED};font-size:0.7rem">// MISSION CONTROL</span></div>
            <div class="mc-sub">{spec.spec_id} · {spec.title}</div>
          </div>
          <div style="display:flex;gap:7px;flex-wrap:wrap">
            {pill("status", status, RED if divergence else color)}
            {pill("reps", str(agg.get("replications", 0)))}
            {pill("horizon", f"{spec.stress.horizon_days:.0f}d")}
            {pill("season", spec.stress.season.replace("_", " "))}
            {pill("seed", str(spec.seed))}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(spec: ScenarioSpec, metrics: dict, agg: dict) -> None:
    tp = agg.get("throughput_percentiles", {})
    rs = agg.get("resilience_percentiles", {})
    baseline = agg.get("baseline_throughput", 0.0)
    cols = st.columns(6, gap="small")
    tiles = [
        (
            "THROUGHPUT P50",
            f"{tp.get('p50', 0):,.0f}",
            f"TEU/day · baseline {baseline:,.0f}",
            "accent" if tp.get("p50", 0) >= baseline * 0.9 else "warn",
        ),
        (
            "RESILIENCE P05",
            f"{rs.get('p5', 0):.2f}",
            f"CVaR95 {agg.get('resilience_cvar95', 0):.2f}",
            "bad" if rs.get("p5", 1) < 0.6 else ("warn" if rs.get("p5", 1) < 0.8 else "accent"),
        ),
        (
            "SERVICE LEVEL",
            f"{agg.get('mean_service_level', 0):.0%}",
            f"mean over {agg.get('replications', 0)} reps",
            "bad" if agg.get("mean_service_level", 1) < 0.55 else "accent",
        ),
        (
            "PEAK UTILISATION",
            f"{metrics.get('max_utilization', 0):.2f}",
            "gate 0.92",
            "bad" if metrics.get("max_utilization", 0) > 0.92 else "accent",
        ),
        (
            "MEAN BERTH WAIT",
            f"{metrics.get('mean_wait_hours', 0):.1f}h",
            f"p95 sojourn {metrics.get('p95_sojourn_hours', 0):.0f}h",
            "warn" if metrics.get("mean_wait_hours", 0) > 24 else "",
        ),
        (
            "P(INSTABILITY)",
            f"{agg.get('p_instability', 0):.0%}",
            f"demurrage ${agg.get('mean_total_cost_usd', 0) / 1e6:.1f}M",
            "bad" if agg.get("p_instability", 0) > 0.15 else "accent",
        ),
    ]
    for col, (label, value, note, tone) in zip(cols, tiles, strict=False):
        col.markdown(kpi(label, value, note, tone), unsafe_allow_html=True)


def render_overview(spec: ScenarioSpec, ref: dict, mc: dict) -> None:
    gate = spec.montecarlo.acceptance.max_berth_utilization
    left, right = st.columns([2.05, 1.0], gap="small")
    with left:
        pchart(chart_fan(mc["fan"], mc["baseline"]), key="chart-01")
        pchart(
            chart_backlog(ref["timeline"], spec.montecarlo.acceptance.max_backlog_slope_per_day),
            key="chart-02",
        )
    with right:
        pchart(chart_utilization(ref["ports"], gate), key="chart-03")
        pchart(
            chart_hist(
                mc["outcomes"]["throughput_teu_day"].to_numpy(dtype=float),
                mc["baseline"],
                "ENSEMBLE THROUGHPUT DISTRIBUTION",
                CYAN,
            ),
            key="chart-14",
        )
        pchart(chart_events(ref["events"]), key="chart-04")

    st.markdown("#### PORT QUEUE DYNAMICS")
    pchart(chart_queue_heatmap(ref["telemetry"], [p.id for p in spec.network.ports]), key="chart-05")


def render_network(spec: ScenarioSpec, ref: dict) -> None:
    graph = build_graph(spec.network)
    cent = criticality(graph)
    left, right = st.columns([2.0, 1.0], gap="small")
    with left:
        pchart(chart_network(spec, ref["ports"]), key="chart-06")
    with right:
        frame = pd.DataFrame(
            sorted(cent.items(), key=lambda kv: kv[1], reverse=True)[:8], columns=["node", "betweenness"]
        )
        fig = go.Figure(
            go.Bar(
                x=frame["betweenness"],
                y=frame["node"],
                orientation="h",
                marker={"color": VIOLET, "line": {"width": 0}},
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            )
        )
        fig.update_xaxes(title_text="betweenness centrality")
        pchart(style_fig(fig, 300, "CRITICALITY // SHORTEST-PATH EXPOSURE"), key="chart-07")
        pchart(chart_weather(ref["weather"]), key="chart-08")

    st.markdown("#### TERMINAL LEDGER")
    table = ref["ports"].copy()
    table["utilization"] = table["utilization"].map(lambda v: f"{v:.3f}")
    table["mean_wait_h"] = table["mean_wait_h"].map(lambda v: f"{v:.1f}")
    table["p95_wait_h"] = table["p95_wait_h"].map(lambda v: f"{v:.1f}")
    table["teu"] = table["teu"].map(lambda v: f"{v:,.0f}")
    st.dataframe(table, width="stretch", hide_index=True)


def render_port_ops(spec: ScenarioSpec, ref: dict) -> None:
    telemetry = ref["telemetry"]
    port_ids = [p.id for p in spec.network.ports]
    selected = st.multiselect("Terminals", port_ids, default=port_ids[:4])
    if telemetry.empty or not selected:
        st.info("No telemetry for the current selection.")
        return
    fig = go.Figure()
    for pid, color in zip(selected, [CYAN, BLUE, AMBER, VIOLET, RED, "#7bd88f", "#ff9f9f"], strict=False):
        col = f"q_{pid}"
        if col in telemetry:
            fig.add_trace(
                go.Scatter(
                    x=telemetry["t_day"], y=telemetry[col], name=pid, line={"color": color, "width": 1.5}
                )
            )
    fig.update_yaxes(title_text="vessels waiting")
    fig.update_xaxes(title_text="day")
    pchart(style_fig(fig, 300, "BERTH QUEUE // PER TERMINAL"), key="chart-09")

    left, right = st.columns(2, gap="small")
    with left:
        pchart(chart_queue_heatmap(telemetry, port_ids), key="chart-10")
    with right:
        vessels = ref["vessels"]
        if not vessels.empty:
            fig2 = go.Figure(
                go.Histogram(x=vessels["sojourn_h"], nbinsx=40, marker={"color": BLUE, "line": {"width": 0}})
            )
            fig2.add_vline(
                x=float(vessels["sla_h"].median()),
                line={"color": RED, "width": 1.2, "dash": "dash"},
                annotation_text="median SLA",
                annotation_font_size=9,
                annotation_font_color=RED,
            )
            fig2.update_xaxes(title_text="hours door-to-door")
            pchart(style_fig(fig2, 280, "SOJOURN TIME DISTRIBUTION"), key="chart-11")

    st.markdown("#### ROUTING & COST")
    vessels = ref["vessels"]
    if not vessels.empty:
        c1, c2, c3 = st.columns(3, gap="small")
        c1.markdown(
            kpi("VESSELS SERVED", f"{len(vessels):,}", "post warm-up completions"), unsafe_allow_html=True
        )
        c2.markdown(
            kpi("REROUTED", f"{vessels['rerouted'].mean():.1%}", "share using the contingency lane"),
            unsafe_allow_html=True,
        )
        c3.markdown(
            kpi(
                "DEMURRAGE",
                f"${vessels['demurrage_usd'].sum() / 1e6:.1f}M",
                f"freight ${vessels['freight_usd'].sum() / 1e6:.1f}M",
            ),
            unsafe_allow_html=True,
        )
        top = (
            vessels.groupby("channel")
            .agg(
                vessels=("vessel_id", "count"),
                sojourn_h=("sojourn_h", "mean"),
                on_time=("on_time", "mean"),
                demurrage=("demurrage_usd", "sum"),
            )
            .reset_index()
            .sort_values("demurrage", ascending=False)
        )
        top["sojourn_h"] = top["sojourn_h"].round(1)
        top["on_time"] = top["on_time"].map(lambda v: f"{v:.0%}")
        top["demurrage"] = top["demurrage"].map(lambda v: f"${v:,.0f}")
        st.dataframe(top, width="stretch", hide_index=True)


def render_montecarlo(spec: ScenarioSpec, mc: dict) -> None:
    agg = mc["aggregate"]
    outcomes = mc["outcomes"]
    left, right = st.columns(2, gap="small")
    with left:
        pchart(
            chart_hist(
                outcomes["resilience_index"].to_numpy(dtype=float),
                1.0,
                "RESILIENCE INDEX DISTRIBUTION",
                VIOLET,
            ),
            key="chart-15",
        )
        pchart(chart_tornado(mc["tornado"]), key="chart-12")
    with right:
        pchart(chart_scatter(outcomes, mc["samples"]), key="chart-13")
        pchart(
            chart_hist(
                outcomes["max_utilization"].to_numpy(dtype=float),
                spec.montecarlo.acceptance.max_berth_utilization,
                "PEAK UTILISATION DISTRIBUTION",
                AMBER,
            ),
            key="chart-16",
        )

    st.markdown("#### ENSEMBLE STATISTICS")
    cols = st.columns(4, gap="small")
    stats = [
        ("RESILIENCE CVaR95", f"{agg['resilience_cvar95']:.3f}", "expected shortfall"),
        ("THROUGHPUT CVaR95", f"{agg['throughput_cvar95']:,.0f}", "TEU/day lower tail"),
        ("P(SL < 0.60)", f"{agg['p_service_breach']:.1%}", "service breach probability"),
        ("ENSEMBLE RUNTIME", f"{mc['runtime']:.1f}s", f"{agg['replications']} replications"),
    ]
    for col, (label, value, note) in zip(cols, stats, strict=False):
        col.markdown(kpi(label, value, note), unsafe_allow_html=True)

    table = outcomes.copy()
    for col_name in ("throughput_teu_day", "delivered_teu"):
        table[col_name] = table[col_name].map(lambda v: f"{v:,.0f}")
    for col_name in ("service_level", "resilience_index", "max_utilization", "convergence_rse"):
        table[col_name] = table[col_name].map(lambda v: f"{v:.3f}")
    table["mean_wait_hours"] = table["mean_wait_hours"].map(lambda v: f"{v:.1f}")
    st.dataframe(table, width="stretch", hide_index=True, height=280)

    st.download_button(
        "DOWNLOAD OUTCOMES CSV",
        outcomes.to_csv(index=False).encode(),
        file_name="mc_outcomes.csv",
        mime="text/csv",
    )


@st.cache_data(show_spinner=False)
def latest_snapshot() -> dict | None:
    """Most recent finished build, so the tab paints instantly on first load."""

    runs = sorted((PROJECT_ROOT / "artifacts").glob("run-*/state_snapshot.json"))
    if not runs:
        return None
    try:
        return json.loads(runs[-1].read_text())
    except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupt run dir
        return None


def render_pipeline(spec: ScenarioSpec, opts: dict) -> None:
    st.markdown("#### AM-SES v2.0 // SELF-HEALING BUILD GRAPH")
    st.markdown(
        "<div class='mono-note'>planner → synthesizer → sandbox_tester → (reflector ↺) → finalize · "
        f"retry budget {opts['max_retries']} · fault injection <b>{opts['fault_mode']}</b></div>",
        unsafe_allow_html=True,
    )
    state_key = f"build::{scenario_key(spec)}::{opts['max_retries']}::{opts['fault_mode']}"
    if opts["run_build"] or state_key not in st.session_state:
        snapshot = latest_snapshot()
        if snapshot is not None and not opts["run_build"]:
            st.session_state[state_key] = (snapshot, str(snapshot.get("run_id", "")))
        else:
            with st.spinner("executing state graph · compiling artifacts · running sandbox..."):
                fresh, context = run_workflow(
                    spec,
                    project_root=PROJECT_ROOT,
                    max_retries=opts["max_retries"],
                    fault_mode=opts["fault_mode"],
                )
                st.session_state[state_key] = (fresh, context.run_id)

    state, run_id = st.session_state.get(state_key, ({}, ""))
    if not state:
        st.info("No build executed yet.")
        return

    verdict = state.get("verdict", "UNKNOWN")
    color = {"PASS": CYAN, "PASS_WITH_RETRIES": BLUE}.get(verdict, RED)
    report = state.get("test_report") or {}

    c1, c2, c3, c4 = st.columns(4, gap="small")
    c1.markdown(
        kpi("VERDICT", verdict, f"run {run_id[-12:]}", "accent" if color != RED else "bad"),
        unsafe_allow_html=True,
    )
    c2.markdown(
        kpi("ATTEMPTS", str(state.get("retry_count", 0) + 1), f"cap {opts['max_retries'] + 1} executions"),
        unsafe_allow_html=True,
    )
    c3.markdown(
        kpi("SANDBOX TIME", f"{report.get('duration_seconds', 0):.2f}s", f"exit {report.get('exit_code')}"),
        unsafe_allow_html=True,
    )
    c4.markdown(
        kpi(
            "SIZING",
            str((state.get("architectural_blueprint") or {}).get("sizing_strategy", "-")),
            f"margin {(state.get('architectural_blueprint') or {}).get('resilience_margin', 1):.2f}",
        ),
        unsafe_allow_html=True,
    )

    st.markdown("##### EXECUTION TRACE")
    rows = []
    for entry in state.get("execution_logs", []):
        rows.append(
            {
                "node": entry["node"].replace("node_", ""),
                "level": entry["level"],
                "message": entry["message"],
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown("##### ACCEPTANCE GATES")
    crit = pd.DataFrame(report.get("criteria", []))
    if not crit.empty:
        crit["passed"] = crit["passed"].map(lambda v: "PASS" if v else "FAIL")
        crit["observed"] = crit["observed"].map(lambda v: f"{v:.4f}")
        st.dataframe(crit, width="stretch", hide_index=True)

    trail = state.get("reflection_trail", [])
    if trail:
        st.markdown("##### REFLECTION TRAIL")
        for item in trail:
            with st.expander(
                f"attempt {item['attempt'] + 1} · {item['failure_class']} · confidence {item['confidence']:.2f}"
            ):
                st.markdown(f"**Diagnosis** {item['diagnosis']}")
                for patch in item["patches"]:
                    st.markdown(
                        f"- `{patch['pointer']}`: `{patch['previous']}` → `{patch['updated']}`  \n"
                        f"  <span class='mono-note'>{patch['rationale']}</span>",
                        unsafe_allow_html=True,
                    )
    else:
        st.success("First build passed every gate - no reflection required.")

    blueprint = state.get("architectural_blueprint") or {}
    artifacts = state.get("code_artifacts") or {}
    c1, c2 = st.columns(2, gap="small")
    with c1, st.expander("GENERATED ARTIFACTS"):
        if artifacts.get("artifacts"):
            st.dataframe(pd.DataFrame(artifacts["artifacts"]), width="stretch", hide_index=True)
        st.caption(f"build dir: `{artifacts.get('build_dir', '-')}`")
    with c2, st.expander("GRAPH DEFINITION (MERMAID)"):
        ctx = RunContext(
            run_id="preview", project_root=PROJECT_ROOT, artifacts_root=PROJECT_ROOT / "artifacts"
        )
        st.code(graph_mermaid(ctx), language="mermaid")

    with st.expander("ARCHITECTURAL BLUEPRINT"):
        st.markdown(
            f"**{blueprint.get('blueprint_id', '-')}** · objective: {blueprint.get('objective', '-')}"
        )
        sizing = pd.DataFrame(blueprint.get("sizing", []))
        if not sizing.empty:
            st.dataframe(sizing, width="stretch", hide_index=True)
        equations = pd.DataFrame(blueprint.get("equations", []))
        if not equations.empty:
            st.dataframe(equations[["name", "domain", "latex"]], width="stretch", hide_index=True)
        if blueprint.get("risk_notes"):
            for note in blueprint["risk_notes"]:
                st.markdown(f"- {note}")

    build_dir = Path(artifacts.get("build_dir", ""))
    memo = build_dir / "blueprint.md"
    if memo.exists():
        with st.expander("GENERATED DESIGN MEMO"):
            st.markdown(memo.read_text())
    source = build_dir / "generated_simulation.py"
    if source.exists():
        with st.expander("GENERATED SIMULATION SOURCE"):
            st.code(source.read_text(), language="python")


def render_data(spec: ScenarioSpec, ref: dict, mc: dict) -> None:
    st.markdown("#### RAW TELEMETRY")
    tabs = st.tabs(["REFERENCE TELEMETRY", "VESSEL RECORDS", "INCIDENTS", "ENSEMBLE SAMPLES"])
    with tabs[0]:
        st.dataframe(ref["telemetry"], width="stretch", hide_index=True, height=320)
        st.download_button(
            "DOWNLOAD TELEMETRY CSV",
            ref["telemetry"].to_csv(index=False).encode(),
            file_name="telemetry.csv",
            mime="text/csv",
        )
    with tabs[1]:
        st.dataframe(ref["vessels"], width="stretch", hide_index=True, height=320)
        st.download_button(
            "DOWNLOAD VESSELS CSV",
            ref["vessels"].to_csv(index=False).encode(),
            file_name="vessels.csv",
            mime="text/csv",
        )
    with tabs[2]:
        events = ref["events"]
        st.dataframe(events, width="stretch", hide_index=True, height=320)
    with tabs[3]:
        st.dataframe(mc["samples"], width="stretch", hide_index=True, height=320)
    st.markdown("#### SCENARIO MANIFEST")
    st.json(json.loads(spec.model_dump_json()), expanded=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    spec, opts = sidebar_controls()
    spec_json = spec.model_dump_json()

    ref = run_reference(spec_json)

    cache_key = f"mc::{scenario_key(spec)}::{opts['reps']}::{opts['seed']}"
    if cache_key not in st.session_state:
        holder = st.empty()
        with holder.container():
            st.markdown(
                f"<div class='mono-note'>executing {opts['reps']} Monte Carlo replications · "
                "latin hypercube perturbation ...</div>",
                unsafe_allow_html=True,
            )
            bar = st.progress(0.0)
            mc = run_ensemble(spec_json, opts["reps"], opts["seed"])
            bar.progress(1.0, text="ensemble complete")
        holder.empty()
        st.session_state[cache_key] = mc
    mc = st.session_state[cache_key]

    metrics = ref["metrics"]
    agg = mc["aggregate"]
    render_header(spec, metrics, agg)
    render_kpis(spec, metrics, agg)

    tabs = st.tabs(["OVERVIEW", "NETWORK", "PORT OPS", "MONTE CARLO", "BUILD PIPELINE", "RAW DATA"])
    with tabs[0]:
        render_overview(spec, ref, mc)
    with tabs[1]:
        render_network(spec, ref)
    with tabs[2]:
        render_port_ops(spec, ref)
    with tabs[3]:
        render_montecarlo(spec, mc)
    with tabs[4]:
        render_pipeline(spec, opts)
    with tabs[5]:
        render_data(spec, ref, mc)

    st.markdown(
        f"<div class='mono-note' style='margin-top:18px'>reference replication "
        f"{ref['runtime']:.2f}s · ensemble {mc['runtime']:.1f}s · "
        f"generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} · "
        "SIMULATION ONLY - NO LIVE VESSEL CONTROL</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
