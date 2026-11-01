"""Monte Carlo perturbation driver with Latin Hypercube sampling.

Uncertainty model
-----------------
Let ``theta = (demand_multiplier, weather_intensity, geo_intensity,
service_cv, capacity_scale)`` live in a box ``[low_i, high_i]``. We draw ``N``
stratified samples per dimension (Latin Hypercube), so every replicate explores
a distinct stratum of every marginal while the joint sample stays uniform over
the box. The mapping to KPIs is the full discrete-event engine - there is no
surrogate model, hence no surrogate error.

Resilience index
----------------
``R = Theta_stress / Theta_baseline`` where the baseline is the same network
run with benign forcing (weather intensity 0.55, no geopolitical incidents, no
shock). ``R`` is reported with a lower-tail CVaR so the dashboard can show
"how bad is a bad day", not just the mean.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .engine import RunResult, simulate
from .metrics import (
    compute_metrics,
    cvar,
    percentile_table,
    spearman,
    throughput_timeline,
)
from .schemas import (
    DeploymentPlan,
    PerturbationSpec,
    ScenarioSpec,
    StressSpec,
)

BASELINE_WEATHER_INTENSITY = 0.55
BASELINE_GEO_INTENSITY = 0.0


@dataclass(frozen=True)
class BaselineReference:
    """Benign-forcing reference used as the resilience denominator."""

    throughput_teu_day: float
    effective_throughput_teu_day: float
    service_level: float
    rows: tuple[dict[str, float], ...] = ()


@dataclass
class MCResult:
    spec: ScenarioSpec
    outcomes: pd.DataFrame
    samples: pd.DataFrame
    aggregate: dict[str, Any]
    fan_chart: pd.DataFrame
    port_table: pd.DataFrame
    tornado: pd.DataFrame
    baseline: BaselineReference
    runtime_seconds: float
    reference_result: RunResult | None = field(default=None, repr=False)

    @property
    def baseline_throughput(self) -> float:
        return self.baseline.throughput_teu_day

    def summary(self) -> dict[str, Any]:
        return dict(self.aggregate)


def latin_hypercube(n: int, dimensions: int, rng: np.random.Generator) -> np.ndarray:
    """Return an ``(n, dimensions)`` matrix of stratified samples in [0, 1)."""

    cut = np.linspace(0.0, 1.0, n + 1)
    u = rng.random((n, dimensions))
    samples = np.empty((n, dimensions), dtype=float)
    for j in range(dimensions):
        strata = cut[:-1] + u[:, j] * (1.0 / n)
        samples[:, j] = rng.permutation(strata)
    return samples


def apply_perturbation(spec: ScenarioSpec, key: str, value: float) -> ScenarioSpec:
    stress = spec.stress
    if key in {"demand_multiplier", "weather_intensity", "geo_intensity", "service_cv", "capacity_scale"}:
        updated: StressSpec = stress.model_copy(update={key: float(value)})
        return spec.model_copy(update={"stress": updated})
    raise KeyError(f"unknown perturbation key: {key}")


def baseline_spec(spec: ScenarioSpec) -> ScenarioSpec:
    stress = spec.stress.model_copy(
        update={
            "weather_intensity": BASELINE_WEATHER_INTENSITY,
            "geo_intensity": BASELINE_GEO_INTENSITY,
            "demand_multiplier": 1.0,
            "capacity_scale": 1.0,
            "shock": None,
        }
    )
    return spec.model_copy(update={"stress": stress})


def run_baseline(
    spec: ScenarioSpec,
    deployment: DeploymentPlan | None = None,
    replications: int | None = None,
) -> BaselineReference:
    """Simulate the same network under benign forcing.

    The reference is the denominator of the resilience index. ``effective``
    throughput counts only the TEU that arrived inside their contractual SLA,
    which is the quantity that actually degrades when the network congests -
    raw throughput saturates at offered demand.
    """

    reps = replications or max(1, spec.montecarlo.baseline_replications)
    base = baseline_spec(spec)
    rows: list[dict[str, float]] = []
    for i in range(reps):
        result = simulate(base, deployment=deployment, seed=spec.seed + 7000 + i)
        m = compute_metrics(result)
        m["replication"] = float(i)
        rows.append(m)
    throughput = float(np.median([r["throughput_teu_day"] for r in rows]))
    effective = float(np.median([r["effective_throughput_teu_day"] for r in rows]))
    service = float(np.median([r["service_level"] for r in rows]))
    return BaselineReference(
        throughput_teu_day=throughput,
        effective_throughput_teu_day=effective,
        service_level=service,
        rows=tuple(rows),
    )


def run_monte_carlo(
    spec: ScenarioSpec,
    deployment: DeploymentPlan | None = None,
    n_replications: int | None = None,
    seed: int | None = None,
    progress: Callable[[int, int, float], None] | None = None,
    include_timelines: bool = True,
    baseline: BaselineReference | None = None,
) -> MCResult:
    started = time.perf_counter()
    n = int(n_replications or spec.montecarlo.n_replications)
    master_seed = int(seed if seed is not None else spec.seed)
    rng = np.random.default_rng(master_seed + 91)

    perturbations: list[PerturbationSpec] = list(spec.montecarlo.perturbations)
    if not perturbations:
        perturbations = [PerturbationSpec(key="demand_multiplier", low=1.0, high=1.0 + 1e-6, label="Demand")]

    unit = latin_hypercube(n, len(perturbations), rng)
    values = np.empty_like(unit)
    for j, p in enumerate(perturbations):
        values[:, j] = p.low + unit[:, j] * (p.high - p.low)

    samples = pd.DataFrame({p.key: values[:, j] for j, p in enumerate(perturbations)})
    samples.insert(0, "replication", np.arange(n))

    if baseline is None:
        baseline = run_baseline(spec, deployment=deployment)
    effective_reference = max(baseline.effective_throughput_teu_day, 1e-9)

    rows: list[dict[str, Any]] = []
    timelines: list[pd.DataFrame] = []
    port_accumulator: dict[str, list[dict[str, float]]] = {}
    reference: RunResult | None = None

    for i in range(n):
        rep_spec = spec
        for j, p in enumerate(perturbations):
            rep_spec = apply_perturbation(rep_spec, p.key, float(values[i, j]))
        result = simulate(rep_spec, deployment=deployment, seed=master_seed + 1000 * (i + 1))
        if reference is None:
            reference = result
        m = compute_metrics(result)
        m["replication"] = i
        m["resilience_index"] = float(min(1.5, m["effective_throughput_teu_day"] / effective_reference))
        m["seed"] = result.seed
        rows.append(m)
        for p_row in result.ports:
            port_accumulator.setdefault(p_row["port_id"], []).append(p_row)
        if include_timelines:
            tl = throughput_timeline(result)
            tl = tl.assign(replication=i)
            timelines.append(tl)
        if progress is not None:
            progress(i + 1, n, time.perf_counter() - started)

    outcomes = pd.DataFrame(rows)
    aggregate = aggregate_outcomes(outcomes, spec, baseline)
    fan = build_fan_chart(timelines) if timelines else pd.DataFrame()
    ports = summarise_ports(port_accumulator)
    tornado = build_tornado(samples, outcomes, perturbations)

    return MCResult(
        spec=spec,
        outcomes=outcomes,
        samples=samples,
        aggregate=aggregate,
        fan_chart=fan,
        port_table=ports,
        tornado=tornado,
        baseline=baseline,
        runtime_seconds=time.perf_counter() - started,
        reference_result=reference,
    )


def aggregate_outcomes(
    outcomes: pd.DataFrame, spec: ScenarioSpec, baseline: BaselineReference
) -> dict[str, Any]:
    percentiles = spec.montecarlo.percentiles
    throughput = outcomes["throughput_teu_day"].to_numpy(dtype=float)
    resilience = outcomes["resilience_index"].to_numpy(dtype=float)
    service = outcomes["service_level"].to_numpy(dtype=float)
    wait = outcomes["mean_wait_hours"].to_numpy(dtype=float)
    util = outcomes["max_utilization"].to_numpy(dtype=float)
    cost = (outcomes["freight_cost_usd"] + outcomes["demurrage_cost_usd"]).to_numpy(dtype=float)
    diverged = outcomes["diverged"].to_numpy(dtype=float)

    return {
        "replications": int(len(outcomes)),
        "baseline_throughput": float(baseline.throughput_teu_day),
        "baseline_effective_throughput": float(baseline.effective_throughput_teu_day),
        "baseline_service_level": float(baseline.service_level),
        "effective_throughput_percentiles": percentile_table(
            outcomes["effective_throughput_teu_day"].to_numpy(dtype=float), percentiles
        ),
        "effective_throughput_mean": float(outcomes["effective_throughput_teu_day"].mean()),
        "throughput_percentiles": percentile_table(throughput, percentiles),
        "resilience_percentiles": percentile_table(resilience, percentiles),
        "service_level_percentiles": percentile_table(service, percentiles),
        "wait_hours_percentiles": percentile_table(wait, percentiles),
        "utilization_percentiles": percentile_table(util, percentiles),
        "throughput_mean": float(np.mean(throughput)),
        "throughput_std": float(np.std(throughput, ddof=1)) if len(throughput) > 1 else 0.0,
        "resilience_mean": float(np.mean(resilience)),
        "resilience_cvar95": cvar(resilience, 95.0),
        "throughput_cvar95": cvar(throughput, 95.0),
        "mean_wait_hours": float(np.mean(wait)),
        "mean_service_level": float(np.mean(service)),
        "mean_max_utilization": float(np.mean(util)),
        "p_instability": float(np.mean(diverged)),
        "p_service_breach": float(np.mean(service < 0.6)),
        "mean_total_cost_usd": float(np.mean(cost)),
        "total_cost_usd": float(np.sum(cost)),
        "runtime_seconds": float(outcomes.get("runtime_seconds", pd.Series([0.0])).mean()),
        "worst_replication": int(outcomes["throughput_teu_day"].idxmin()) if len(outcomes) else 0,
        "best_replication": int(outcomes["throughput_teu_day"].idxmax()) if len(outcomes) else 0,
    }


def build_fan_chart(timelines: list[pd.DataFrame]) -> pd.DataFrame:
    if not timelines:
        return pd.DataFrame()
    stacked = pd.concat(timelines, ignore_index=True)
    grouped = stacked.groupby("t_day")["throughput_teu_day"]
    fan = pd.DataFrame(
        {
            "p05": grouped.quantile(0.05),
            "p25": grouped.quantile(0.25),
            "p50": grouped.quantile(0.50),
            "p75": grouped.quantile(0.75),
            "p95": grouped.quantile(0.95),
            "mean": grouped.mean(),
        }
    ).reset_index()
    queue = stacked.groupby("t_day")["queue_total"].quantile(0.95).reset_index(name="queue_p95")
    fan = fan.merge(queue, on="t_day", how="left")
    return fan


def summarise_ports(port_accumulator: dict[str, list[dict[str, float]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for port_id, entries in sorted(port_accumulator.items()):
        frame = pd.DataFrame(entries)
        rows.append(
            {
                "port_id": port_id,
                "name": frame["name"].iloc[0],
                "berths": int(frame["berths"].iloc[0]),
                "calls_mean": float(frame["calls"].mean()),
                "teu_mean": float(frame["teu"].mean()),
                "utilization_mean": float(frame["utilization"].mean()),
                "utilization_p95": float(frame["utilization"].quantile(0.95)),
                "mean_wait_h": float(frame["mean_wait_h"].mean()),
                "p95_wait_h": float(frame["p95_wait_h"].mean()),
                "max_queue": int(frame["max_queue"].max()),
                "closed_hours": float(frame["closed_hours"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_tornado(
    samples: pd.DataFrame, outcomes: pd.DataFrame, perturbations: list[PerturbationSpec]
) -> pd.DataFrame:
    """Rank-correlation sensitivity of each perturbation.

    Raw throughput is a poor response variable - it saturates at offered demand
    because late cargo is still delivered before the horizon ends. The ranking
    therefore uses the service-constrained KPIs (resilience, tail sojourn),
    which is where congestion actually shows up.
    """

    rows = []
    throughput = outcomes["throughput_teu_day"].to_numpy(dtype=float)
    resilience = outcomes["resilience_index"].to_numpy(dtype=float)
    wait = outcomes["mean_wait_hours"].to_numpy(dtype=float)
    p95 = outcomes["p95_sojourn_hours"].to_numpy(dtype=float)
    for p in perturbations:
        if p.key not in samples:
            continue
        x = samples[p.key].to_numpy(dtype=float)
        rows.append(
            {
                "key": p.key,
                "label": p.label or p.key,
                "low": p.low,
                "high": p.high,
                "rho_throughput": spearman(x, throughput),
                "rho_resilience": spearman(x, resilience),
                "rho_mean_wait": spearman(x, wait),
                "rho_p95_sojourn": spearman(x, p95),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("rho_resilience", key=lambda s: s.abs(), ascending=False)
    return frame


def scenario_key(spec: ScenarioSpec, extra: str = "") -> str:
    import hashlib
    import json

    payload = json.dumps({"spec": spec.model_dump(mode="json"), "extra": extra}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
