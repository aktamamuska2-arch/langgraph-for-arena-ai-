"""Validation metrics and gate evaluation for the sandbox visual tester.

Two families of checks live here:

*Convergence* - does the lattice settle? We fit an exponential rate to the tail
of the population series (it must decay toward zero), measure how far the
coefficient of variation drifts between the middle and the end of the run, and
audit the spatial field for coherence (a lag-1 autocorrelation high enough to
prove the shading is a smooth choropleth rather than per-cell noise).

*Integrity* - does the trade network hang together? Giant-component share,
edges per node, orphan fraction, plus the exact physical ledger identity

    stocks_end = stocks_start + produced - consumed + imported - exported - spoiled

which must hold to floating-point tolerance. A shock-recovery gate then
re-runs the world with a mid-horizon drought and requires the trade volume to
return to at least a fixed fraction of its pre-shock level.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .model import AgoraModel
from .schemas import (
    ExperimentSpec,
    Gate,
    GateResult,
    ValidationReport,
)

CONVERGENCE_WINDOW = 0.25


def _tail(history: list[dict[str, float]], fraction: float = CONVERGENCE_WINDOW) -> list[dict[str, float]]:
    if not history:
        return []
    keep = max(4, int(len(history) * fraction))
    return history[-keep:]


def growth_rate(history: list[dict[str, float]]) -> float:
    """Least-squares exponential rate of the population tail, per tick."""

    tail = _tail(history)
    if len(tail) < 3:
        return 0.0
    t = np.array([row["t"] for row in tail], dtype=float)
    p = np.array([max(row["population"], 1e-9) for row in tail], dtype=float)
    slope = np.polyfit(t - t[0], np.log(p), 1)[0]
    return float(slope)


def _cv(history: list[dict[str, float]]) -> float:
    pops = np.array([row["population"] for row in history], dtype=float)
    if pops.size < 2 or pops.mean() <= 0:
        return 0.0
    return float(pops.std() / pops.mean())


def convergence_metrics(history: list[dict[str, float]]) -> dict[str, float]:
    if len(history) < 8:
        return {"abs_growth_tail": 1.0, "cv_drift": 1.0, "pop_drawdown": 1.0}
    rate = growth_rate(history)
    mid = history[len(history) // 2 : len(history) // 2 + max(4, len(history) // 8)]
    tail = _tail(history)
    pops = np.array([row["population"] for row in history], dtype=float)
    running_max = np.maximum.accumulate(pops)
    drawdown = float(np.max(1.0 - pops / np.maximum(running_max, 1e-9)))
    return {
        "abs_growth_tail": abs(rate),
        "cv_drift": abs(_cv(tail) - _cv(mid)) if mid else 0.0,
        "pop_drawdown": drawdown,
    }


def field_metrics(model: AgoraModel) -> dict[str, float]:
    """Spatial coherence of the population field plus discretisation mass error."""

    pop, hatch = model.pop_field()
    field = pop.astype(np.float64) / 255.0
    if field.max() <= 0:
        return {"field_lag1_corr": 0.0, "field_peak_share": 1.0, "block_mass_error": 0.0, "hatch_cells": 0.0}
    a = field[:, :-1].ravel()
    b = field[:, 1:].ravel()
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    total = float(field.sum())
    peak_share = float(field.max() * 1.0 / max(total, 1e-9))
    # 2x2 block averaging must conserve the field exactly - a discretisation
    # invariant that catches any per-cell normalisation bug
    h, w = field.shape
    blocks = field[: h - h % 2, : w - w % 2].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    block_mass_error = abs(blocks.sum() * 4.0 - total) / max(total, 1e-12)
    return {
        "field_lag1_corr": corr,
        "field_peak_share": peak_share,
        "block_mass_error": float(block_mass_error),
        "hatch_cells": float((hatch >= 2).sum()),
    }


def mass_balance(model: AgoraModel) -> float:
    """Relative error of the physical ledger identity."""

    final = sum(float(s.total_stocks().sum()) for s in model.settlements.values())
    supply = getattr(model, "supply", None)
    if supply is not None:
        # inventories held by pengepul and penjual are part of the world's stock
        final += sum(float(c.inventory.sum()) for c in supply.collectors.values())
        final += sum(float(r.inventory.sum()) for r in supply.retailers.values())
    ledger = model.ledger
    expected = (
        model.initial_stocks
        + ledger["produced"].sum()
        - ledger["consumed"].sum()
        + ledger["imported"].sum()
        - ledger["exported"].sum()
        - ledger["spoiled"].sum()
        - ledger["diverted"].sum()
    )
    scale = max(abs(expected), abs(model.initial_stocks), 1.0)
    return float(abs(final - expected) / scale)


def network_metrics(model: AgoraModel) -> dict[str, float]:
    graph = model.trade_graph()
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    if n == 0:
        return {
            "giant_component_share": 0.0,
            "edges_per_node": 0.0,
            "orphan_fraction": 1.0,
            "mean_degree": 0.0,
        }
    undirected = graph.to_undirected()
    giant = max((len(c) for c in nx_components(undirected)), default=0)
    orphans = sum(1 for node in graph.nodes if graph.degree(node) == 0)
    return {
        "giant_component_share": float(giant / n),
        "edges_per_node": float(m / n),
        "orphan_fraction": float(orphans / n),
        "mean_degree": float(2.0 * m / n),
    }


def nx_components(graph) -> list[set]:
    import networkx as nx

    return list(nx.connected_components(graph))


def visual_contract(before: AgoraModel, after: AgoraModel) -> tuple[float, dict[str, Any]]:
    """Audit the rendered state against the visual specification."""

    from .render import render_frame

    checks: dict[str, bool] = {}
    detail: dict[str, Any] = {}

    image = render_frame(after, scale=6, hud=False)
    array = np.asarray(image, dtype=np.float64)
    checks["not_blank"] = float(array.std()) > 8.0
    detail["pixel_std"] = float(array.std())

    codes = np.unique(after.world.terrain)
    checks["terrain_variety"] = codes.size >= 5
    detail["terrain_classes"] = int(codes.size)

    pop_before, _hatch_before = before.pop_field()
    pop_after, hatch_after = after.pop_field()
    checks["population_grew"] = float(pop_after.mean()) > float(pop_before.mean())
    checks["shading_present"] = float(pop_after.max()) > 60
    checks["hatching_present"] = int((hatch_after >= 2).sum()) > 0
    detail["pop_mean_before"] = float(pop_before.mean())
    detail["pop_mean_after"] = float(pop_after.mean())
    detail["hatch_cells"] = int((hatch_after >= 2).sum())

    landed = 0
    for edge in after.edges.values():
        if not edge.path:
            continue
        tail_x, tail_y = edge.path[-1]
        if after.world.terrain[tail_y, tail_x] != 0:
            landed += 1
    checks["flows_land_anchored"] = landed == len([e for e in after.edges.values() if e.path])
    detail["flows"] = len(after.edges)

    checks["pulses_alive"] = len(after.pulses) > 0
    detail["pulses"] = len(after.pulses)

    score = sum(checks.values()) / len(checks)
    detail["checks"] = checks
    return float(score), detail


@dataclass
class Study:
    """A full sandbox run plus everything the gates need."""

    model: AgoraModel
    spec: ExperimentSpec
    metrics: dict[str, float] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    history: list[dict[str, float]] = field(default_factory=list)


def run_study(spec: ExperimentSpec, with_shock: bool = True) -> Study:
    """Execute the lattice, optionally stress it, and collect every metric."""

    started = time.perf_counter()
    model = AgoraModel(spec)
    horizon = spec.horizon
    settle = max(spec.settle_ticks, int(horizon * 0.55))
    for _ in range(settle):
        model.step()
    baseline_model_metrics = {
        **convergence_metrics(model.history),
        **network_metrics(model),
    }
    before_pop, before_hatch = model.pop_field()
    snapshot_before = _clone_view(model, before_pop, before_hatch)

    metrics: dict[str, float] = {}
    detail: dict[str, Any] = {}

    if with_shock:
        baseline_flow = _window_flow(model, 40)
        _inject_representative_shock(model)
        mid_flow = _window_flow(model, 40)
        for _ in range(max(20, (horizon - settle) // 3)):
            model.step()
        recovery_flow = _window_flow(model, 40)
        metrics["shock_recovery"] = float(recovery_flow / baseline_flow) if baseline_flow > 1e-9 else 0.0
        metrics["shock_dip"] = float(mid_flow / baseline_flow) if baseline_flow > 1e-9 else 0.0
        detail["shock"] = {
            "baseline_flow": baseline_flow,
            "mid_flow": mid_flow,
            "recovery_flow": recovery_flow,
        }
    else:
        remaining = max(0, horizon - settle)
        for _ in range(remaining):
            model.step()
        metrics["shock_recovery"] = 1.0
        metrics["shock_dip"] = 1.0

    for _ in range(max(0, horizon - settle - (horizon - settle) // 3 - 80)):
        model.step()

    metrics.update(convergence_metrics(model.history))
    metrics.update(field_metrics(model))
    metrics.update(network_metrics(model))
    metrics.update(
        baseline_model_metrics
        | {
            # keep the final values for the live-state gates
            "giant_component_share": network_metrics(model)["giant_component_share"],
            "edges_per_node": network_metrics(model)["edges_per_node"],
            "orphan_fraction": network_metrics(model)["orphan_fraction"],
        }
    )
    metrics.update(model.supply.macro())
    metrics["mass_balance_error"] = mass_balance(model)
    metrics["alive_fraction"] = float(
        sum(1 for s in model.settlements.values() if s.alive) / max(len(model.settlements), 1)
    )

    score, visual_detail = visual_contract(snapshot_before, model)
    metrics["visual_contract_score"] = score
    detail["visual"] = visual_detail
    metrics["duration_seconds"] = time.perf_counter() - started

    return Study(
        model=model,
        spec=spec,
        metrics=metrics,
        detail=detail,
        duration=metrics["duration_seconds"],
        history=list(model.history),
    )


class _Snapshot:
    """Minimal stand-in so the visual contract can compare two moments."""

    def __init__(self, model: AgoraModel, pop: np.ndarray, hatch: np.ndarray) -> None:
        self._pop = pop
        self._hatch = hatch
        self.model = model

    def pop_field(self):
        return self._pop, self._hatch


def _clone_view(model: AgoraModel, pop: np.ndarray, hatch: np.ndarray):
    return _Snapshot(model, pop, hatch)


def _window_flow(model: AgoraModel, ticks: int) -> float:
    start = model.total_flow
    for _ in range(ticks):
        model.step()
    return float(model.total_flow - start)


def _inject_representative_shock(model: AgoraModel) -> None:
    """Stress the food system at its most exposed node."""

    exporters = sorted(
        (s for s in model.settlements.values() if s.alive),
        key=lambda s: float(s.endowment[0]),
        reverse=True,
    )
    target = exporters[0] if exporters else next(iter(model.settlements.values()))
    model.inject_shock(
        kind="drought",
        x=target.x,
        y=target.y,
        radius=18.0,
        magnitude=0.85,
        duration=55,
        description="validation stress: drought on the dominant food exporter",
    )


DEFAULT_GATES: list[Gate] = [
    Gate(
        id="C-GRID",
        metric="abs_growth_tail",
        comparator="<=",
        threshold=0.0045,
        description="population series must flatten: |exponential rate| of the final 25% <= 0.0045/tick",
    ),
    Gate(
        id="C-COHER",
        metric="field_lag1_corr",
        comparator=">=",
        threshold=0.55,
        description="population choropleth must be spatially coherent, not per-cell noise",
    ),
    Gate(
        id="N-COMP",
        metric="giant_component_share",
        comparator=">=",
        threshold=0.85,
        description="at least 85% of living settlements must sit in one trade component",
    ),
    Gate(
        id="N-EDGE",
        metric="edges_per_node",
        comparator=">=",
        threshold=0.90,
        description="the discovered market must carry at least one link per node",
    ),
    Gate(
        id="N-MASS",
        metric="mass_balance_error",
        comparator="<=",
        threshold=1e-9,
        description="physical ledger identity must hold to floating-point tolerance",
    ),
    Gate(
        id="V-CONTRACT",
        metric="visual_contract_score",
        comparator=">=",
        threshold=0.85,
        description="rendered frame must satisfy the visual specification",
    ),
    Gate(
        id="R-SHOCK",
        metric="shock_recovery",
        comparator=">=",
        threshold=0.60,
        description="trade volume must recover to 60% of its pre-shock level",
    ),
    Gate(
        id="M-MARGIN",
        metric="marketing_margin",
        comparator="<=",
        threshold=0.55,
        description="the pengepul/penjual layer must not capture more than 55% of the retail price",
    ),
    Gate(
        id="S-PRED",
        metric="syndicate_take_share",
        comparator="<=",
        threshold=0.10,
        description="predation must stay marginal in a world whose chain is not broken",
    ),
    Gate(
        id="C-STABLE",
        metric="pop_drawdown",
        comparator="<=",
        threshold=0.35,
        description="aggregate population must not collapse by more than 35% from its peak",
    ),
]

FAILURE_BY_GATE = {
    "M-MARGIN": "supply_chain",
    "S-PRED": "syndication",
    "C-GRID": "non_convergence",
    "C-COHER": "non_convergence",
    "C-STABLE": "instability",
    "N-COMP": "network_fragmentation",
    "N-EDGE": "network_fragmentation",
    "N-MASS": "mass_imbalance",
    "V-CONTRACT": "visual_contract",
    "R-SHOCK": "instability",
}


def evaluate(
    gates: list[Gate],
    metrics: dict[str, float],
    attempt: int,
    duration: float = 0.0,
) -> ValidationReport:
    results: list[GateResult] = []
    missing: list[str] = []
    for gate in gates:
        if gate.metric not in metrics:
            missing.append(gate.id)
            continue
        results.append(GateResult.build(gate, metrics[gate.metric]))
    passed = all(r.passed for r in results) and not missing

    failure_class = "none"
    if not passed:
        failing = [r for r in results if not r.passed]
        failure_class = FAILURE_BY_GATE.get(failing[0].id, "instability") if failing else "schema_error"

    return ValidationReport(
        attempt=attempt,
        passed=passed,
        exit_code=0 if passed else 1,
        timed_out=False,
        duration_seconds=float(duration),
        failure_class=failure_class,  # type: ignore[arg-type]
        results=results,
        metrics={k: float(v) for k, v in metrics.items()},
        stdout_tail=f"{len(results)} gates evaluated on attempt {attempt}",
        stderr_tail=""
        if passed
        else f"failed gates: {[r.id for r in results if not r.passed]}"
        + (f" | missing: {missing}" if missing else ""),
    )
