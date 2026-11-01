"""Behavioural tests for the SimPy discrete-event core."""

from __future__ import annotations

import math

import pytest

from maritime_sentinel.engine import (
    CapacityPool,
    default_deployment,
    simulate,
)
from maritime_sentinel.metrics import compute_metrics, risk_band


def test_capacity_pool_shrinks_and_absorbs_deficit() -> None:
    import simpy

    env = simpy.Environment()
    pool = CapacityPool(env, capacity=3, name="test")
    assert pool.available_slots() == 3

    events = [pool.request() for _ in range(3)]
    assert all(ev.triggered for ev in events)
    assert pool.available_slots() == 0

    queued = pool.request()
    assert not queued.triggered
    assert pool.queue_length() == 1

    pool.set_online(1)  # two servers go offline while three are occupied
    assert pool.available_slots() == 0

    pool.release()  # busy 3 -> 2, still above the 1 online server
    assert not queued.triggered
    pool.release()  # busy 2 -> 1
    assert not queued.triggered
    pool.release()  # busy 1 -> 0, the deficit is absorbed and the queue drains
    assert queued.triggered, "waiting vessel must be admitted once capacity frees up"
    assert pool.busy == 1 and pool.queue_length() == 0

    pool.set_online(3)
    assert pool.available_slots() == 2


def test_simulation_is_deterministic_given_the_seed(short_scenario) -> None:  # noqa: ANN001
    a = compute_metrics(simulate(short_scenario, seed=99))
    b = compute_metrics(simulate(short_scenario, seed=99))
    c = compute_metrics(simulate(short_scenario, seed=100))
    for key in ("throughput_teu_day", "mean_wait_hours", "service_level", "max_utilization"):
        assert a[key] == pytest.approx(b[key], rel=1e-12)
    assert a["throughput_teu_day"] != pytest.approx(c["throughput_teu_day"], rel=1e-9)


def test_replication_is_stable_across_processes(project_root) -> None:  # noqa: ANN001
    """Guards against process-salted state (e.g. builtin ``hash``) leaking into
    the RNG streams: two interpreters with different PYTHONHASHSEED must agree."""

    import json
    import os
    import subprocess
    import sys

    script = (
        "import json;"
        "from maritime_sentinel.config import DEFAULT_SCENARIO;"
        "from maritime_sentinel.engine import simulate;"
        "from maritime_sentinel.metrics import compute_metrics;"
        "m = compute_metrics(simulate(DEFAULT_SCENARIO, seed=DEFAULT_SCENARIO.seed));"
        "print(json.dumps({k: m[k] for k in "
        "('throughput_teu_day','mean_wait_hours','service_level','max_utilization')}))"
    )
    outputs = []
    for hash_seed in ("0", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(project_root)}
        proc = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=env, check=True
        )
        outputs.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    assert outputs[0] == outputs[1], "replication must not depend on the process hash seed"


def test_more_capacity_means_less_waiting(scenario) -> None:  # noqa: ANN001
    lean = scenario.model_copy(update={"stress": scenario.stress.model_copy(update={"capacity_scale": 1.0})})
    rich = scenario.model_copy(update={"stress": scenario.stress.model_copy(update={"capacity_scale": 1.45})})
    lean_m = compute_metrics(simulate(lean, seed=7))
    rich_m = compute_metrics(simulate(rich, seed=7))
    assert rich_m["max_utilization"] < lean_m["max_utilization"]
    assert rich_m["mean_wait_hours"] < lean_m["mean_wait_hours"]
    assert rich_m["service_level"] >= lean_m["service_level"]


def test_starved_network_diverges_and_is_caught(short_scenario) -> None:  # noqa: ANN001
    plan = default_deployment(short_scenario).model_copy(
        update={"berths": {p.id: 1 for p in short_scenario.network.ports}}
    )
    result = simulate(short_scenario, deployment=plan, seed=3)
    assert result.diverged is True
    assert "queue" in result.divergence_reason or "sojourn" in result.divergence_reason
    metrics = compute_metrics(result)
    assert metrics["diverged"] == 1.0
    assert metrics["backlog_slope_per_day"] >= 1.0
    assert risk_band(metrics) == "CRITICAL"


def test_outputs_are_finite_and_consistent(short_scenario) -> None:  # noqa: ANN001
    result = simulate(short_scenario, seed=11)
    metrics = compute_metrics(result)
    for key, value in metrics.items():
        assert math.isfinite(value), f"{key} is not finite"
    assert metrics["throughput_teu_day"] > 0
    assert 0.0 <= metrics["service_level"] <= 1.0
    assert not result.telemetry.empty
    assert not result.vessels.empty
    assert set(result.telemetry.columns) >= {"t_day", "delivered_teu", "queue_total"}
    assert len(result.ports) == len(short_scenario.network.ports)


def test_vessel_records_are_internally_consistent(short_scenario) -> None:  # noqa: ANN001
    result = simulate(short_scenario, seed=5)
    vessels = result.vessels
    assert (vessels["finish_day"] >= vessels["start_day"]).all()
    assert (vessels["sojourn_h"] > 0).all()
    assert (vessels["teu"] > 0).all()
    assert (vessels["freight_usd"] >= 0).all()
    assert vessels["sojourn_h"].max() <= 45 * 24 + 1e-6


def test_shock_reduces_resilience(scenario) -> None:  # noqa: ANN001
    benign_stress = scenario.stress.model_copy(
        update={"shock": None, "weather_intensity": 0.55, "geo_intensity": 0.0}
    )
    benign = scenario.model_copy(update={"stress": benign_stress})
    base_m = compute_metrics(simulate(benign, seed=21))
    stressed_m = compute_metrics(simulate(scenario, seed=21))
    assert stressed_m["p95_sojourn_hours"] > base_m["p95_sojourn_hours"]
    assert stressed_m["service_level"] < base_m["service_level"]
