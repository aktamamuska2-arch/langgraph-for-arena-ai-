"""Discrete-event simulation core.

Every replication builds one :class:`World` and advances it with SimPy. The
engine is deliberately deterministic given ``(scenario, deployment, seed)``:
all stochastic draws come from ``numpy.random.Generator`` streams spawned from
a single ``SeedSequence``, and SimPy's event calendar is deterministic.

Entities
--------
``CapacityPool``  finite server with dynamic availability (berths / strait slots)
``PortOps``       process that re-evaluates berth availability every 6 h
``ChokepointOps`` process that re-evaluates strait slot availability
``vessel``        load -> sail -> discharge voyage with live re-routing
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import simpy

from .geopolitics import GeopoliticalField
from .schemas import (
    ChokepointSpec,
    DeploymentPlan,
    PortSpec,
    ScenarioSpec,
)
from .topology import build_graph, lane_pairs
from .weather import WeatherField, zone_statistics

BASE_SPEED_KNOTS = 15.5
DEMURRAGE_USD_PER_HOUR = 900.0
QUEUE_GUARD_VESSELS = 320
SOJOURN_GUARD_DAYS = 45.0
OPS_INTERVAL_DAYS = 0.25


class DivergenceError(RuntimeError):
    """Raised when the replicated queue is non-stationary (unbounded backlog)."""


# ---------------------------------------------------------------------------
# Dynamic capacity pool
# ---------------------------------------------------------------------------


class CapacityPool:
    """FIFO multi-server queue whose number of online servers can change at runtime.

    SimPy's ``Resource`` cannot shrink or grow its capacity, so berth and strait
    availability is modelled with an explicit server count. ``online`` may drop
    below ``busy`` — the deficit is absorbed as vessels finish service.
    """

    __slots__ = ("env", "capacity", "name", "online", "busy", "_queue", "_granted")

    def __init__(self, env: simpy.Environment, capacity: int, name: str) -> None:
        self.env = env
        self.capacity = max(1, capacity)
        self.name = name
        self.online = self.capacity
        self.busy = 0
        self._queue: list[simpy.Event] = []
        self._granted = 0

    # -- queries ---------------------------------------------------------
    def available_slots(self) -> int:
        return max(0, self.online - self.busy)

    def queue_length(self) -> int:
        return len(self._queue)

    # -- mutation --------------------------------------------------------
    def request(self) -> simpy.Event:
        ev = self.env.event()
        self._queue.append(ev)
        self._dispatch()
        return ev

    def release(self) -> None:
        self.busy = max(0, self.busy - 1)
        self._dispatch()

    def set_online(self, value: int) -> None:
        self.online = int(max(0, min(self.capacity, value)))
        self._dispatch()

    def _dispatch(self) -> None:
        while self._queue and self.available_slots() > 0:
            ev = self._queue.pop(0)
            self.busy += 1
            self._granted += 1
            if not ev.triggered:
                ev.succeed()


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------


@dataclass
class PortRuntime:
    port: PortSpec
    pool: CapacityPool
    berths: int
    busy_berth_days: float = 0.0
    calls: int = 0
    teu: float = 0.0
    waits: list[float] = field(default_factory=list)
    closed_until: float = 0.0
    closed_hours: float = 0.0
    max_queue: int = 0


@dataclass
class ChokepointRuntime:
    spec: ChokepointSpec
    pool: CapacityPool
    transits: int = 0
    max_queue: int = 0


@dataclass
class RunResult:
    spec: ScenarioSpec
    deployment: DeploymentPlan
    seed: int
    telemetry: pd.DataFrame
    vessels: pd.DataFrame
    ports: list[dict[str, Any]]
    events: pd.DataFrame
    weather_timeline: pd.DataFrame
    runtime_seconds: float
    diverged: bool
    divergence_reason: str
    measured_days: float

    def to_summary(self) -> dict[str, Any]:
        return {
            "diverged": self.diverged,
            "divergence_reason": self.divergence_reason,
            "runtime_seconds": self.runtime_seconds,
            "vessels_completed": int(len(self.vessels)),
            "measured_days": self.measured_days,
        }


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------


class World:
    def __init__(self, spec: ScenarioSpec, deployment: DeploymentPlan, seed: int) -> None:
        self.spec = spec
        self.deployment = deployment
        self.seed = seed
        self.env = simpy.Environment()
        streams = np.random.SeedSequence(seed).spawn(6)
        (
            self.rng_weather,
            self.rng_geo,
            self.rng_demand,
            self.rng_service,
            self.rng_ops,
            self.rng_aux,
        ) = (np.random.default_rng(s) for s in streams)

        self.graph = build_graph(spec.network)
        self.lanes = lane_pairs(spec)
        self.ports_spec = {p.id: p for p in spec.network.ports}
        self.chokepoint_specs = {c.id: c for c in spec.network.chokepoints}
        self.horizon = float(spec.stress.horizon_days)
        self.warmup = float(spec.stress.warmup_days)

        self.weather = WeatherField(
            self.env,
            spec.network.zones,
            self.rng_weather,
            spec.stress.weather_intensity,
            spec.stress.season,
        )
        self.geo = GeopoliticalField(self.env, spec.network, spec.stress, self.rng_geo)

        self.ports: dict[str, PortRuntime] = {}
        for port in spec.network.ports:
            berths = int(deployment.berths.get(port.id, port.berths))
            self.ports[port.id] = PortRuntime(
                port=port, pool=CapacityPool(self.env, berths, port.id), berths=berths
            )

        self.chokepoints: dict[str, ChokepointRuntime] = {}
        for cp in spec.network.chokepoints:
            online = max(1, int(round(cp.concurrent_slots * deployment.capacity_scale)))
            self.chokepoints[cp.id] = ChokepointRuntime(spec=cp, pool=CapacityPool(self.env, online, cp.id))

        self.vessel_records: list[dict[str, Any]] = []
        self.telemetry: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.inflight = 0
        self.generated = 0
        self.delivered_teu = 0.0
        self._static_paths: dict[tuple[str, str], list[str]] = {}
        self._precompute_static_paths()
        self._terminal_nodes = set(self.ports_spec) | set(self.chokepoint_specs)

    # -- routing ---------------------------------------------------------
    def _precompute_static_paths(self) -> None:
        g = self.graph.copy()
        for _u, _v, d in g.edges(data=True):
            if d["contingency"]:
                d["_w"] = math.inf
            else:
                d["_w"] = d["distance_nm"] / BASE_SPEED_KNOTS
        for origin in self.ports_spec:
            for destination in self.ports_spec:
                if origin == destination:
                    continue
                try:
                    self._static_paths[(origin, destination)] = nx.shortest_path(
                        g, origin, destination, weight="_w"
                    )
                except (nx.NetworkXNoPath, nx.NodeNotFound):  # pragma: no cover
                    self._static_paths[(origin, destination)] = [origin, destination]

    def _contingency_open(self) -> bool:
        if self.spec.stress.routing_policy == "static":
            return False
        return self.geo.capacity_factor("MALACCA") < 0.55

    def _edge_weight(self, u: str, v: str) -> float:
        data = self.lanes.get((u, v))
        if data is None:  # pragma: no cover - defensive
            return math.inf
        if data["contingency"] and not self._contingency_open():
            return math.inf
        if self.spec.stress.routing_policy == "static":
            hours = data["distance_nm"] / BASE_SPEED_KNOTS
        else:
            hours = data["distance_nm"] / (BASE_SPEED_KNOTS * self.weather.speed_factor(data["zone"]))
        for node in (u, v):
            rt = self.chokepoints.get(node)
            if rt is not None:
                contention = rt.pool.queue_length() / max(1, rt.pool.online)
                hours += rt.spec.transit_hours * (1.0 + contention)
        return hours + 1e-6

    def _route(self, origin: str, destination: str) -> list[str]:
        if self.spec.stress.routing_policy == "static":
            return self._static_paths.get((origin, destination), [origin, destination])
        try:
            return nx.shortest_path(
                self.graph, origin, destination, weight=lambda u, v, _d: self._edge_weight(u, v)
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return self._static_paths.get((origin, destination), [origin, destination])

    # -- service ---------------------------------------------------------
    def _sample_service_days(self, port: PortSpec, teu: float) -> float:
        productivity = self.weather.productivity_factor(port.zone)
        mu = port.service_teu_per_berth_day * productivity
        median = teu / max(mu, 1e-6)
        sigma = float(self.spec.stress.service_cv)
        sampled = float(self.rng_service.lognormal(mean=math.log(max(median, 1e-6)), sigma=sigma))
        return max(0.02, min(sampled, 20.0))

    @staticmethod
    def _season_factor(channel, day: float) -> float:
        """Monthly demand index of a channel (1.0 = flat year-round)."""

        month = int((day % 365.0) / 365.0 * 12.0) % 12
        return float(channel.seasonality[month])

    # -- processes -------------------------------------------------------
    def ops_process(self, rt: PortRuntime):
        zone = self.weather.zones[rt.port.zone]
        while True:
            yield self.env.timeout(OPS_INTERVAL_DAYS)
            now = self.env.now
            rt.max_queue = max(rt.max_queue, rt.pool.queue_length())
            if now < rt.closed_until:
                rt.pool.set_online(0)
                continue
            geo = self.geo.capacity_factor(rt.port.id)
            online = int(round(rt.berths * geo))
            if self.weather.is_extreme(rt.port.zone):
                hazard = zone.closure_prob_extreme * OPS_INTERVAL_DAYS
                if float(self.rng_ops.random()) < hazard:
                    duration = float(np.clip(self.rng_ops.lognormal(math.log(0.4), 0.6), 0.1, 1.5))
                    rt.closed_until = now + duration
                    rt.closed_hours += duration * 24.0
                    self.events.append(
                        {
                            "kind": "port_closure",
                            "target": rt.port.id,
                            "start_day": round(now, 3),
                            "end_day": round(now + duration, 3),
                            "severity": 1.0,
                            "label": f"{rt.port.name} closed: extreme weather",
                        }
                    )
                    online = 0
            rt.pool.set_online(online)

    def chokepoint_ops_process(self, rt: ChokepointRuntime):
        while True:
            yield self.env.timeout(OPS_INTERVAL_DAYS)
            rt.max_queue = max(rt.max_queue, rt.pool.queue_length())
            geo = self.geo.capacity_factor(rt.spec.id)
            online = int(round(rt.pool.capacity * geo))
            if geo > 0:
                online = max(1, online)
            rt.pool.set_online(online)

    def guard_process(self):
        while True:
            yield self.env.timeout(1.0)
            total_queue = sum(rt.pool.queue_length() for rt in self.ports.values())
            worst_port = max((rt.pool.queue_length(), pid) for pid, rt in self.ports.items())
            if total_queue > QUEUE_GUARD_VESSELS:
                raise DivergenceError(
                    f"unbounded berth queue at t={self.env.now:.1f}d "
                    f"({total_queue} vessels waiting, worst port {worst_port[1]}={worst_port[0]})"
                )

    def telemetry_process(self, step: float = 0.5):
        while True:
            self._sample()
            yield self.env.timeout(step)

    def _sample(self) -> None:
        row: dict[str, Any] = {
            "t_day": round(self.env.now, 3),
            "delivered_teu": self.delivered_teu,
            "inflight": self.inflight,
            "queue_total": sum(rt.pool.queue_length() for rt in self.ports.values()),
            "generated": self.generated,
        }
        for pid, rt in self.ports.items():
            row[f"q_{pid}"] = rt.pool.queue_length()
            row[f"busy_{pid}"] = rt.pool.busy
        self.telemetry.append(row)

    def demand_process(self, channel):
        rate = channel.base_daily_calls * self.spec.stress.demand_multiplier
        while True:
            intensity = rate * self._season_factor(channel, self.env.now)
            dt = float(self.rng_demand.exponential(1.0 / max(intensity, 1e-9)))
            yield self.env.timeout(max(dt, 1e-4))
            if self.env.now > self.horizon:
                return
            self.generated += 1
            vid = self.generated
            teu = float(self.rng_demand.lognormal(math.log(channel.teu_mean), channel.teu_cv))
            teu = float(np.clip(teu, 80.0, channel.teu_mean * 4.0))
            self.env.process(self.vessel_process(vid, channel, teu))

    def _berth_service(self, port_id: str, teu: float):
        rt = self.ports[port_id]
        wait_start = self.env.now
        yield rt.pool.request()
        wait_days = self.env.now - wait_start
        service_start = self.env.now
        service = self._sample_service_days(rt.port, teu)
        yield self.env.timeout(service)
        service_end = self.env.now
        # Only the portion of the service interval that falls inside the
        # measured (post-warm-up) window counts towards berth occupancy,
        # otherwise utilisation is inflated by the warm-up transient.
        overlap = max(0.0, min(service_end, self.horizon) - max(service_start, self.warmup))
        rt.busy_berth_days += overlap
        if service_end >= self.warmup:
            rt.waits.append(wait_days * 24.0)
            rt.calls += 1
            rt.teu += teu
        rt.pool.release()
        return wait_days, service

    def vessel_process(self, vessel_id: int, channel, teu: float):
        t_start = self.env.now
        origin, destination = channel.origin, channel.destination
        load_wait, load_service = yield from self._berth_service(origin, teu)
        depart = self.env.now
        self.inflight += 1
        path = self._route(origin, destination)
        rerouted = bool(
            any(
                self.lanes.get((a, b), {}).get("contingency", False)
                for a, b in zip(path, path[1:], strict=False)
            )
        )
        transit_hours = 0.0
        for a, b in zip(path, path[1:], strict=False):
            lane = self.lanes.get((a, b))
            if lane is None:  # pragma: no cover - defensive
                continue
            speed = BASE_SPEED_KNOTS * (1.0 if False else self.weather.speed_factor(lane["zone"]))
            hours = lane["distance_nm"] / max(speed, 1e-6)
            transit_hours += hours
            yield self.env.timeout(hours / 24.0)
            cp = self.chokepoints.get(b)
            if cp is not None:
                yield cp.pool.request()
                dwell = cp.spec.transit_hours * (0.8 + 0.4 * float(self.rng_aux.random()))
                transit_hours += dwell
                cp.transits += 1
                yield self.env.timeout(dwell / 24.0)
                cp.pool.release()
            if self.env.now > self.horizon + 5.0:
                break
        self.inflight -= 1
        arrive = self.env.now
        discharge_wait, discharge_service = yield from self._berth_service(destination, teu)
        finish = self.env.now
        sojourn_hours = (finish - t_start) * 24.0
        if sojourn_hours > SOJOURN_GUARD_DAYS * 24.0:
            raise DivergenceError(
                f"vessel {vessel_id} sojourn {sojourn_hours:.0f}h exceeds guard at t={self.env.now:.1f}d"
            )
        if finish < self.warmup:
            return
        freight = 0.0
        for a, b in zip(path, path[1:], strict=False):
            lane = self.lanes.get((a, b), {})
            freight += float(lane.get("cost_usd_per_teu", 0.0)) * teu
        dwell_hours = max(0.0, sojourn_hours - float(channel.sla_hours))
        record = {
            "vessel_id": vessel_id,
            "channel": channel.id,
            "origin": origin,
            "destination": destination,
            "teu": round(teu, 2),
            "start_day": round(t_start, 3),
            "depart_day": round(depart, 3),
            "arrive_day": round(arrive, 3),
            "finish_day": round(finish, 3),
            "load_wait_h": round(load_wait * 24.0, 3),
            "discharge_wait_h": round(discharge_wait * 24.0, 3),
            "total_wait_h": round((load_wait + discharge_wait) * 24.0, 3),
            "transit_h": round(transit_hours, 3),
            "service_h": round((load_service + discharge_service) * 24.0, 3),
            "sojourn_h": round(sojourn_hours, 3),
            "sla_h": float(channel.sla_hours),
            "on_time": bool(sojourn_hours <= float(channel.sla_hours)),
            "path": "-".join(path),
            "hops": len(path) - 1,
            "rerouted": rerouted,
            "freight_usd": round(freight, 2),
            "demurrage_usd": round(dwell_hours * DEMURRAGE_USD_PER_HOUR, 2),
        }
        self.vessel_records.append(record)
        self.delivered_teu += teu

    # -- driver ----------------------------------------------------------
    def run(self) -> RunResult:
        started = time.perf_counter()
        divergence_reason = ""
        self.weather.start()
        self.geo.start()
        self.env.process(self.guard_process())
        self.env.process(self.telemetry_process())
        for rt in self.ports.values():
            self.env.process(self.ops_process(rt))
        for rt in self.chokepoints.values():
            self.env.process(self.chokepoint_ops_process(rt))
        for channel in self.spec.network.channels:
            self.env.process(self.demand_process(channel))
        try:
            self.env.run(until=self.horizon)
        except DivergenceError as exc:
            divergence_reason = str(exc)
        runtime = time.perf_counter() - started
        self._sample()

        telemetry = pd.DataFrame(self.telemetry)
        vessels = pd.DataFrame(self.vessel_records)
        events = pd.DataFrame(self.events)
        weather_rows: list[dict[str, Any]] = []
        for zone_id, st in self.weather.state.items():
            for row in st.transitions:
                weather_rows.append(
                    {
                        "zone": zone_id,
                        "day": float(row["day"]),
                        "state": int(row["state"]),
                        "label": str(row["label"]),
                    }
                )
        weather_timeline = pd.DataFrame(weather_rows)
        geo_rows = self.geo.log()
        if geo_rows:
            geo_frame = pd.DataFrame(geo_rows)
            geo_frame["kind"] = "geo_incident"
            events = pd.concat([events, geo_frame], ignore_index=True) if not events.empty else geo_frame

        measured_days = max(1e-6, self.env.now - self.warmup)
        port_rows = []
        for pid, rt in self.ports.items():
            waits = np.array(rt.waits) if rt.waits else np.array([0.0])
            port_rows.append(
                {
                    "port_id": pid,
                    "name": rt.port.name,
                    "berths": rt.berths,
                    "calls": rt.calls,
                    "teu": round(rt.teu, 2),
                    "mean_wait_h": float(np.mean(waits)),
                    "p95_wait_h": float(np.percentile(waits, 95)),
                    "utilization": float(min(1.5, rt.busy_berth_days / max(rt.berths * measured_days, 1e-9))),
                    "max_queue": int(rt.max_queue),
                    "closed_hours": float(rt.closed_hours),
                }
            )
        return RunResult(
            spec=self.spec,
            deployment=self.deployment,
            seed=self.seed,
            telemetry=telemetry,
            vessels=vessels,
            ports=port_rows,
            events=events,
            weather_timeline=weather_timeline,
            runtime_seconds=runtime,
            diverged=bool(divergence_reason),
            divergence_reason=divergence_reason,
            measured_days=measured_days,
        )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def default_deployment(spec: ScenarioSpec, seed: int | None = None) -> DeploymentPlan:
    scale = float(spec.stress.capacity_scale)
    return DeploymentPlan(
        plan_id=f"{spec.spec_id}-PROPORTIONAL",
        blueprint_id="UNPLANNED",
        strategy="proportional",
        berths={p.id: max(1, int(round(p.berths * scale))) for p in spec.network.ports},
        capacity_scale=scale,
        routing_policy=spec.stress.routing_policy,
        seed=int(spec.seed if seed is None else seed),
        horizon_days=float(spec.stress.horizon_days),
        warmup_days=float(spec.stress.warmup_days),
        notes=["Proportional allocation: physical berths scaled by the scenario capacity factor."],
    )


def simulate(
    spec: ScenarioSpec,
    deployment: DeploymentPlan | None = None,
    seed: int | None = None,
) -> RunResult:
    plan = deployment or default_deployment(spec, seed)
    effective_seed = int(seed if seed is not None else spec.seed)
    world = World(spec, plan, effective_seed)
    return world.run()


def expected_weather_productivity(spec: ScenarioSpec, zone_id: str) -> float:
    zone = next(z for z in spec.network.zones if z.id == zone_id)
    return zone_statistics(zone, spec.stress.weather_intensity, spec.stress.season)["expected_productivity"]
