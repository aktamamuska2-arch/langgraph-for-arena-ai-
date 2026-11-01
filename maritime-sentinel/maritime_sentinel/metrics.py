"""KPI extraction, convergence diagnostics and resilience scoring."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .engine import SOJOURN_GUARD_DAYS, RunResult
from .schemas import ScenarioOutcome

BATCH_COUNT = 10
DIVERGENCE_RSE = 1.0
DIVERGENCE_SLOPE = 100.0


def service_weight(vessels: pd.DataFrame) -> np.ndarray:
    """Exponential lateness penalty applied to each delivered TEU.

    ``w = exp(-max(0, sojourn - SLA) / SLA)``

    A vessel that arrives inside its contractual window counts fully; late
    cargo decays smoothly instead of dropping to zero at the first minute of
    delay. The weighted sum is the *service-constrained throughput* used as the
    numerator of the resilience index, which keeps the metric sensitive to
    congestion without collapsing to a binary on-time share.
    """

    if vessels.empty:
        return np.array([])
    lateness = np.maximum(
        0.0, vessels["sojourn_h"].to_numpy(dtype=float) - vessels["sla_h"].to_numpy(dtype=float)
    )
    sla = np.maximum(vessels["sla_h"].to_numpy(dtype=float), 1e-6)
    return np.exp(-lateness / sla)


def batch_means(values: np.ndarray, batches: int = BATCH_COUNT) -> tuple[float, float, float]:
    """Method-of-batch-means: returns ``(grand_mean, standard_error, relative_se)``.

    The replication is only accepted when the relative standard error of the
    steady-state mean falls below the blueprint threshold, i.e. when the
    estimator has converged rather than merely produced a number.
    """

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < batches * 2:
        return (float(arr.mean()) if arr.size else 0.0, float("nan"), 1.0)
    usable = arr[: (arr.size // batches) * batches]
    chunk = usable.reshape(batches, -1)
    means = chunk.mean(axis=1)
    grand = float(means.mean())
    se = float(means.std(ddof=1) / math.sqrt(batches))
    rse = float(abs(se / grand)) if grand > 0 else 1.0
    return grand, se, rse


def backlog_slope(telemetry: pd.DataFrame, warmup: float) -> float:
    if telemetry.empty or "t_day" not in telemetry:
        return 0.0
    window = telemetry[telemetry["t_day"] >= warmup]
    if len(window) < 3:
        return 0.0
    x = window["t_day"].to_numpy(dtype=float)
    y = window["queue_total"].to_numpy(dtype=float)
    try:
        slope, _ = np.polyfit(x, y, 1)
    except (np.linalg.LinAlgError, ValueError):  # pragma: no cover
        return 0.0
    return float(slope)


def interval_throughput(telemetry: pd.DataFrame) -> np.ndarray:
    if telemetry.empty or "delivered_teu" not in telemetry:
        return np.array([])
    cum = telemetry["delivered_teu"].to_numpy(dtype=float)
    t = telemetry["t_day"].to_numpy(dtype=float)
    if len(cum) < 3:
        return np.array([])
    dt = np.diff(t)
    dteu = np.diff(cum)
    mask = dt > 1e-9
    return dteu[mask] / dt[mask]


def compute_metrics(result: RunResult) -> dict[str, float]:
    spec = result.spec
    vessels = result.vessels
    measured = max(result.measured_days, 1e-6)
    nominal_rate = (
        spec.network.channels
        and sum(c.base_daily_calls * c.teu_mean for c in spec.network.channels)
        * spec.stress.demand_multiplier
    )

    effective_throughput = 0.0
    if vessels.empty:
        throughput = 0.0
        mean_wait = 0.0
        p95_sojourn = float(SOJOURN_GUARD_DAYS * 24.0)
        service_level = 0.0
        freight = 0.0
        demurrage = 0.0
        served = 0
    else:
        throughput = float(vessels["teu"].sum() / measured)
        mean_wait = float(vessels["total_wait_h"].mean())
        p95_sojourn = float(np.percentile(vessels["sojourn_h"].to_numpy(dtype=float), 95))
        service_level = float(vessels["on_time"].mean())
        freight = float(vessels["freight_usd"].sum())
        demurrage = float(vessels["demurrage_usd"].sum())
        served = int(len(vessels))
        weights = service_weight(vessels)
        effective_throughput = float((vessels["teu"] * weights).sum() / measured)

    telemetry = (
        result.telemetry[result.telemetry["t_day"] >= spec.stress.warmup_days]
        if not result.telemetry.empty
        else result.telemetry
    )
    series = interval_throughput(telemetry)
    _, _, rse = batch_means(series) if series.size else (0.0, float("nan"), 1.0)

    utilization = [float(p["utilization"]) for p in result.ports]
    max_util = max(utilization) if utilization else 0.0
    slope = backlog_slope(result.telemetry, spec.stress.warmup_days)
    max_queue = int(max((p["max_queue"] for p in result.ports), default=0))

    if result.diverged:
        rse = max(rse, DIVERGENCE_RSE)
        slope = max(slope, DIVERGENCE_SLOPE)

    throughput_ratio = throughput / nominal_rate if nominal_rate > 0 else 0.0
    metrics = {
        "throughput_teu_day": throughput,
        "effective_throughput_teu_day": effective_throughput,
        "throughput_ratio": throughput_ratio,
        "delivered_teu": float(vessels["teu"].sum()) if not vessels.empty else 0.0,
        "served_vessels": float(served),
        "mean_wait_hours": mean_wait,
        "p95_sojourn_hours": p95_sojourn,
        "service_level": service_level,
        "max_utilization": max_util,
        "max_queue": float(max_queue),
        "backlog_slope_per_day": slope,
        "convergence_rse": rse,
        "freight_cost_usd": freight,
        "demurrage_cost_usd": demurrage,
        "runtime_seconds": result.runtime_seconds,
        "reroute_share": float(vessels["rerouted"].mean()) if not vessels.empty else 0.0,
        "diverged": 1.0 if result.diverged else 0.0,
    }
    return metrics


def summarize(
    result: RunResult,
    replication: int = 0,
    baseline_effective_throughput: float | None = None,
) -> ScenarioOutcome:
    """Wrap a replication in the typed outcome model.

    Resilience is measured on *service-constrained* throughput (TEU delivered
    inside SLA), because raw throughput saturates at offered demand and would
    report a congested network as healthy.
    """

    m = compute_metrics(result)
    resilience = 1.0
    if baseline_effective_throughput and baseline_effective_throughput > 0:
        resilience = float(
            min(1.5, max(0.0, m["effective_throughput_teu_day"] / baseline_effective_throughput))
        )
    return ScenarioOutcome(
        replication=replication,
        seed=result.seed,
        throughput_teu_day=m["throughput_teu_day"],
        delivered_teu=m["delivered_teu"],
        serve_ratio=m["throughput_ratio"],
        mean_wait_hours=m["mean_wait_hours"],
        p95_sojourn_hours=m["p95_sojourn_hours"],
        service_level=m["service_level"],
        max_utilization=min(1.5, m["max_utilization"]),
        max_backlog=int(m["max_queue"]),
        backlog_slope_per_day=m["backlog_slope_per_day"],
        freight_cost_usd=m["freight_cost_usd"],
        demurrage_cost_usd=m["demurrage_cost_usd"],
        resilience_index=resilience,
        convergence_rse=m["convergence_rse"],
        diverged=bool(result.diverged),
        runtime_seconds=result.runtime_seconds,
    )


def risk_band(metrics: dict[str, float]) -> str:
    """Coarse operational risk classification for the mission-control header."""

    if metrics.get("diverged", 0.0) > 0 or metrics.get("max_utilization", 0.0) > 0.98:
        return "CRITICAL"
    if metrics.get("max_utilization", 0.0) > 0.90 or metrics.get("service_level", 1.0) < 0.6:
        return "ELEVATED"
    if metrics.get("max_utilization", 0.0) > 0.82:
        return "GUARDED"
    return "NOMINAL"


def throughput_timeline(result: RunResult) -> pd.DataFrame:
    """Warm-up-excluded throughput and backlog timeline for charts."""

    if result.telemetry.empty:
        return pd.DataFrame(columns=["t_day", "throughput_teu_day", "queue_total", "inflight"])
    frame = result.telemetry.copy()
    frame["d_teu"] = frame["delivered_teu"].diff().fillna(0.0)
    frame["d_t"] = frame["t_day"].diff().replace(0, np.nan)
    frame["throughput_teu_day"] = (frame["d_teu"] / frame["d_t"]).fillna(0.0)
    window = frame[frame["t_day"] >= result.spec.stress.warmup_days]
    if window.empty:
        window = frame
    return window[["t_day", "throughput_teu_day", "queue_total", "inflight", "generated"]]


def port_frame(result: RunResult) -> pd.DataFrame:
    return pd.DataFrame(result.ports)


def percentile_table(values: np.ndarray, percentiles: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"p{p:g}": float("nan") for p in percentiles}
    return {f"p{p:g}": float(np.percentile(arr, p)) for p in percentiles}


def cvar(values: np.ndarray, level: float = 95.0) -> float:
    """Conditional value at risk (expected shortfall) of the lower tail."""

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    threshold = np.percentile(arr, 100.0 - level)
    tail = arr[arr <= threshold]
    return float(tail.mean()) if tail.size else float(threshold)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation without SciPy: Pearson on average ranks."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or y.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0
    # pandas>=3 returns read-only views, so copy before centring
    rx = np.array(pd.Series(x).rank(), dtype=float)
    ry = np.array(pd.Series(y).rank(), dtype=float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt(float((rx**2).sum()) * float((ry**2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0
