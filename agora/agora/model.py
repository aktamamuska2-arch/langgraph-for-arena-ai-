"""The Mesa model: a 2D cellular world of autonomous settlements.

Tick pipeline
-------------
1. activate/deactivate scheduled and injected shocks
2. every settlement produces, consumes, adapts its labour mix and demography
3. every ``discovery_interval`` ticks, settlements scan for trade partners
4. the market executes the arbitrage round and infrastructure upgrades accrue
5. edge weights decay; stale links are pruned
6. telemetry is recorded and the visual pulse buffer ages

There is no global optimiser. Prices, specialisation, roads and the trade graph
are all consequences of local decisions.
"""

from __future__ import annotations

import base64
import itertools
import math
from collections import deque
from typing import Any

import mesa
import networkx as nx
import numpy as np
from mesa.space import MultiGrid

from .market import MarketEngine, TradeEdge
from .schemas import (
    RESOURCES,
    ExperimentSpec,
    SettlementSnapshot,
    ShockEvent,
    ShockSpec,
)
from .settlement import Settlement, ShockEffect, settlement_name
from .world import WorldData, astar_path, generate_world

PULSE_TTL = 13
MAX_PULSES = 480


def _b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


class AgoraModel(mesa.Model):
    """Open-world spatial economy with emergent trade networks."""

    def __init__(self, spec: ExperimentSpec) -> None:
        super().__init__(seed=spec.world.seed)
        self.spec = spec
        self.world_spec = spec.world
        self.econ = spec.economy
        self.world: WorldData = generate_world(spec.world)

        h, w = self.world.terrain.shape
        self.height, self.width = h, w
        self.grid = MultiGrid(w, h, torus=False)

        self.base_cost = self.world.cost.copy()
        self.road_level = np.zeros((h, w), dtype=np.float32)
        self.cost = self.base_cost.copy()
        self.road_epoch = 0
        self.roads_upgraded = 0

        self.settlements: dict[int, Settlement] = {}
        self.edges: dict[tuple[int, int], TradeEdge] = {}
        self.market = MarketEngine(self.econ)

        name_rng = np.random.default_rng(spec.world.seed + 7717)
        for sid, (y, x) in enumerate(self.world.sites):
            agent = Settlement(self, sid, (int(x), int(y)), settlement_name(name_rng))
            self.grid.place_agent(agent, (int(x), int(y)))
            self.settlements[sid] = agent

        self.tick_count = 0
        self.pulses: deque[dict[str, Any]] = deque(maxlen=MAX_PULSES)
        self.history: list[dict[str, float]] = []
        self.total_flow = 0.0
        self.trade_events = 0

        # physical accounting ledger: every unit of matter that enters or
        # leaves a stock is booked here so the mass-balance invariant can be
        # audited externally (see metrics.mass_balance)
        self.ledger = {
            name: np.zeros(len(RESOURCES), dtype=np.float64)
            for name in ("produced", "consumed", "imported", "exported", "spoiled")
        }
        self.initial_stocks = sum(float(s.stocks.sum()) for s in self.settlements.values())

        self._path_cache: dict[tuple[tuple[int, int], tuple[int, int]], tuple[int, list, float]] = {}
        self.scheduled: list[dict[str, Any]] = [self._shock_record(s) for s in spec.shocks]
        self.active: list[dict[str, Any]] = []
        self.event_log: deque[dict[str, Any]] = deque(maxlen=200)

        self.running = True
        self._apply_one_shot(sorted(self.scheduled, key=lambda s: s["start_tick"]))

    # ------------------------------------------------------------------
    # Shocks
    # ------------------------------------------------------------------
    @staticmethod
    def _shock_record(spec: ShockSpec) -> dict[str, Any]:
        centre_x = spec.x if spec.x is not None else -1
        centre_y = spec.y if spec.y is not None else -1
        return {
            "id": spec.id,
            "kind": spec.kind,
            "x": centre_x,
            "y": centre_y,
            "radius": float(spec.radius),
            "magnitude": float(spec.magnitude),
            "start_tick": int(spec.start_tick),
            "end_tick": int(spec.start_tick + spec.duration),
            "description": spec.description,
        }

    def inject_shock(
        self,
        kind: str,
        x: int | None = None,
        y: int | None = None,
        radius: float = 14.0,
        magnitude: float = 0.6,
        duration: int = 45,
        description: str = "",
    ) -> dict[str, Any]:
        """Add a shock at the current tick (used by the interactive UI)."""

        record = {
            "id": f"ui-{kind}-{self.tick_count}",
            "kind": kind,
            "x": int(x) if x is not None else -1,
            "y": int(y) if y is not None else -1,
            "radius": float(radius),
            "magnitude": float(magnitude),
            "start_tick": self.tick_count,
            "end_tick": self.tick_count + int(duration),
            "description": description or f"{kind} injected at tick {self.tick_count}",
        }
        self.scheduled.append(record)
        self.active.append(record)
        self._apply_one_shot([record])
        self._pulse(record["x"], record["y"], radius=3.5, kind="shock", strength=1.0)
        self.event_log.append(
            {"tick": self.tick_count, "kind": "shock", "label": f"{kind} @ {record['x']},{record['y']}"}
        )
        return record

    def _apply_one_shot(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            if record["start_tick"] > self.tick_count:
                continue
            kind = record["kind"]
            if kind == "ore_discovery":
                self._boost_deposit(record)
            elif kind == "road_collapse":
                clear = self._region_mask(record)
                self.road_level[clear] = 0.0
                self._rebuild_cost()
                self.road_epoch += 1

    def _region_mask(self, record: dict[str, Any]) -> np.ndarray:
        if record["x"] < 0 or record["y"] < 0:
            return np.ones((self.height, self.width), dtype=bool)
        yy, xx = np.mgrid[0 : self.height, 0 : self.width]
        return np.hypot(xx - record["x"], yy - record["y"]) <= record["radius"]

    def _boost_deposit(self, record: dict[str, Any]) -> None:
        mask = self._region_mask(record)
        gain = 0.6 + record["magnitude"]
        for settlement in self.settlements.values():
            if mask[settlement.y, settlement.x]:
                settlement.endowment[2] *= 1.0 + gain * record["magnitude"]
                settlement.capacity *= 1.0 + 0.25 * record["magnitude"]
                discovered = 2.0 * record["magnitude"]
                settlement.stocks[2] += discovered
                self.ledger["produced"][2] += discovered
                self._pulse(settlement.x, settlement.y, radius=2.5, kind="discovery", strength=1.0)

    def _advance_shocks(self) -> None:
        t = self.tick_count
        for record in self.scheduled:
            if record["start_tick"] == t and record not in self.active:
                self.active.append(record)
                self.event_log.append({"tick": t, "kind": "shock", "label": f"{record['kind']} begins"})
                self._pulse(record["x"], record["y"], radius=3.5, kind="shock", strength=1.0)
        still: list[dict[str, Any]] = []
        for record in self.active:
            if record["end_tick"] > t:
                still.append(record)
            else:
                self.event_log.append({"tick": t, "kind": "shock_end", "label": f"{record['kind']} ends"})
        self.active = still

    def shock_effect_at(self, x: int, y: int) -> ShockEffect:
        """Aggregate all active shocks into multiplicative modifiers."""

        effect = ShockEffect()
        for record in self.active:
            if record["x"] >= 0 and record["y"] >= 0:
                distance = math.hypot(x - record["x"], y - record["y"])
                if distance > record["radius"]:
                    continue
                falloff = 1.0 - 0.6 * (distance / max(record["radius"], 1e-6))
            else:
                falloff = 1.0
            mag = record["magnitude"] * falloff
            kind = record["kind"]
            if kind == "drought":
                effect.production[0] *= 1.0 - mag
            elif kind == "blight":
                effect.production[1] *= 1.0 - mag
            elif kind == "demand_boom":
                effect.consumption *= 1.0 + 1.1 * mag
            elif kind == "plague":
                effect.mortality += 0.02 * mag
            elif kind == "embargo":
                effect.trade_friction = max(effect.trade_friction, 2.2 * mag)
        return effect

    def freight_multiplier(self, edge: TradeEdge) -> float:
        """Embargoes raise the effective freight rate on affected routes."""

        multiplier = 1.0
        for record in self.active:
            if record["kind"] != "embargo":
                continue
            if record["x"] < 0:
                multiplier *= 1.0 + 3.0 * record["magnitude"]
                continue
            settlement = self.settlements.get(edge.dst)
            if settlement is None:
                continue
            if math.hypot(settlement.x - record["x"], settlement.y - record["y"]) <= record["radius"]:
                multiplier *= 1.0 + 3.0 * record["magnitude"]
        return multiplier

    # ------------------------------------------------------------------
    # Geography helpers
    # ------------------------------------------------------------------
    def road_factor_at(self, pos: tuple[int, int]) -> float:
        x, y = pos
        return 1.0 + 0.18 * float(self.road_level[y, x])

    def cheapest_path(self, a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
        """Cached cheapest route between two cells, in ``(x, y)`` coordinates."""

        key = (a, b)
        cached = self._path_cache.get(key)
        if cached is not None and cached[0] == self.road_epoch:
            return cached[1]
        path_yx = astar_path(self.cost, (a[1], a[0]), (b[1], b[0]))
        path = [(int(x), int(y)) for y, x in path_yx]
        cost = self.route_cost_of(path)
        self._path_cache[key] = (self.road_epoch, path, cost)
        return path

    def route_cost(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        key = (a, b)
        cached = self._path_cache.get(key)
        if cached is not None and cached[0] == self.road_epoch:
            return cached[2]
        self.cheapest_path(a, b)
        return self._path_cache[key][2]

    def path_cost(self, path: list[tuple[int, int]]) -> float:
        if len(path) < 2:
            return 0.0
        return self.route_cost_of(path)

    def route_cost_of(self, path: list[tuple[int, int]]) -> float:
        total = 0.0
        for (x0, y0), (x1, y1) in itertools.pairwise(path):
            step = 1.4142 if (x0 != x1 and y0 != y1) else 1.0
            total += float(self.cost[y1, x1]) * step
        return total

    def _rebuild_cost(self) -> None:
        gain = 1.0 + self.econ.road_gain * self.road_level
        updated = np.where(np.isfinite(self.base_cost), self.base_cost / gain, np.inf)
        self.cost = updated.astype(np.float32)

    def on_trade(self, edge: TradeEdge, volume: float) -> None:
        """Infrastructure feedback: sustained flow paves the route it uses."""

        self.total_flow += volume
        self.trade_events += 1
        if volume > 0.30 and self.tick_count % 2 == 0:
            buyer = self.settlements[edge.dst]
            self._pulse(buyer.x, buyer.y, radius=1.0, kind="trade", strength=min(1.0, volume / 2.0))
        if edge.cum_flow >= self.econ.road_threshold and edge.path:
            upgrade = float(self.econ.road_gain)
            changed = False
            for x, y in edge.path:
                if self.road_level[y, x] < self.econ.road_max:
                    self.road_level[y, x] = float(min(self.econ.road_max, self.road_level[y, x] + upgrade))
                    changed = True
            edge.cum_flow -= self.econ.road_threshold
            if changed:
                self.roads_upgraded += 1
                self._rebuild_cost()
                self.road_epoch += 1
                self.event_log.append({"tick": self.tick_count, "kind": "road", "label": "trade route paved"})
                mid = edge.path[len(edge.path) // 2]
                self._pulse(mid[0], mid[1], radius=2.0, kind="road", strength=1.0)

    # ------------------------------------------------------------------
    # Pulses
    # ------------------------------------------------------------------
    def _pulse(self, x: int, y: int, radius: float, kind: str, strength: float) -> None:
        if x < 0 or y < 0:
            return
        self.pulses.append(
            {
                "x": float(x),
                "y": float(y),
                "r0": float(radius),
                "kind": kind,
                "strength": float(strength),
                "t0": int(self.tick_count),
            }
        )

    # ------------------------------------------------------------------
    # Tick pipeline
    # ------------------------------------------------------------------
    def step(self) -> None:
        self.tick_count += 1
        t = self.tick_count
        self._advance_shocks()
        self._apply_one_shot(self.scheduled)

        for settlement in self.settlements.values():
            if not settlement.alive:
                continue
            effect = self.shock_effect_at(settlement.x, settlement.y)
            produced = settlement.produce(self.econ, effect)
            settlement.consume(self.econ, effect)
            settlement.spoil(self.econ)
            settlement.update_population(self.econ, effect)
            if t % 3 == 0:
                settlement.adapt_labour(self.econ)
            # harvest pulses are staggered per settlement so the map breathes
            # instead of flashing in unison
            if float(produced.sum()) > 1.2 and (t + settlement.sid * 3) % 11 == 0:
                self._pulse(
                    settlement.x,
                    settlement.y,
                    radius=0.8 + 1.6 * settlement.population / max(settlement.capacity, 1.0),
                    kind="production",
                    strength=min(1.0, float(produced.sum()) / 6.0),
                )

        if t % self.econ.discovery_interval == 0:
            self.market.discover(self)

        for edge in self.edges.values():
            if self.freight_multiplier(edge) > 1.5:
                edge.weights *= 0.92
        self.market.execute(self)
        self.market.decay(self)

        if t % self.spec.record_every == 0:
            self._record()

    def _record(self) -> None:
        population = sum(s.population for s in self.settlements.values() if s.alive)
        wealth = sum(s.wealth for s in self.settlements.values())
        self.history.append(
            {
                "t": float(self.tick_count),
                "population": float(population),
                "wealth": float(wealth),
                "edges": float(len(self.edges)),
                "flow": float(self.total_flow),
                "alive": float(sum(1 for s in self.settlements.values() if s.alive)),
            }
        )

    # ------------------------------------------------------------------
    # Network analytics
    # ------------------------------------------------------------------
    def trade_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        for sid, settlement in self.settlements.items():
            if settlement.alive:
                graph.add_node(sid, population=settlement.population)
        for (src, dst), edge in self.edges.items():
            if src in graph and dst in graph and edge.weight > 0:
                graph.add_edge(src, dst, weight=edge.weight, flow=edge.total_flow)
        return graph

    def network_summary(self) -> dict[str, float]:
        graph = self.trade_graph()
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        if n == 0:
            return {
                "nodes": 0.0,
                "edges": 0.0,
                "density": 0.0,
                "mean_degree": 0.0,
                "giant_component_share": 0.0,
                "total_flow": float(self.total_flow),
                "roads_upgraded": float(self.roads_upgraded),
            }
        undirected = graph.to_undirected()
        giant = max((len(c) for c in nx.connected_components(undirected)), default=0)
        return {
            "nodes": float(n),
            "edges": float(m),
            "density": float(nx.density(graph)),
            "mean_degree": float(2.0 * m / n),
            "giant_component_share": float(giant / n),
            "total_flow": float(self.total_flow),
            "roads_upgraded": float(self.roads_upgraded),
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def pop_field(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-cell population shading (0-255) plus a density hatch tier (0-3).

        The hatch tier is derived from the *intensity* of the same field, so the
        pattern forms concentric density bands around each settlement: plain
        shading at the fringe, a single diagonal through the built-up ring and a
        cross hatch over the dense core. Growth therefore thickens the texture
        rather than just brightening it.
        """

        alive = [s for s in self.settlements.values() if s.alive]
        field = np.zeros((self.height, self.width), dtype=np.float32)
        if not alive:
            return field.astype(np.uint8), np.zeros_like(field, dtype=np.uint8)
        # shading is relative to the *typical carrying capacity* rather than to the
        # current peak: growth then visibly brightens the map over time instead of
        # being renormalised away by whichever settlement happens to lead
        reference = max(float(np.mean([s.capacity for s in alive])), 1.0)
        yy, xx = np.mgrid[0 : self.height, 0 : self.width]
        for settlement in alive:
            level = float(np.clip(settlement.population / reference, 0.02, 1.0))
            sigma = 1.5 + 2.6 * level
            influence = np.exp(-(((xx - settlement.x) ** 2 + (yy - settlement.y) ** 2) / (2 * sigma**2)))
            field = np.maximum(field, influence.astype(np.float32) * (0.35 + 0.65 * level))
        field = np.clip(field, 0.0, 1.0)
        hatch = np.zeros_like(field, dtype=np.uint8)
        hatch[field > 0.24] = 1
        hatch[field > 0.44] = 2
        hatch[field > 0.64] = 3
        return (field * 255).astype(np.uint8), hatch

    def meta(self) -> dict[str, Any]:
        """Static payload: sent once, cached by the client."""

        return {
            "width": self.width,
            "height": self.height,
            "resources": list(RESOURCES),
            "terrain": self.world.terrain.flatten().tolist(),
            "depth": np.round(self.world.depth, 2).flatten().tolist(),
            "experiment": {
                "id": self.spec.experiment_id,
                "title": self.spec.title,
                "horizon": self.spec.horizon,
                "seed": self.world_spec.seed,
                "settlements": len(self.settlements),
            },
            "settlements": [{"id": s.sid, "name": s.name, "x": s.x, "y": s.y} for s in self.settlements.values()],
        }

    def frame(self) -> dict[str, Any]:
        """Dynamic payload consumed by the canvas every animation frame."""

        pop, hatch = self.pop_field()
        flows = []
        for (src, dst), edge in sorted(self.edges.items(), key=lambda kv: kv[1].weight, reverse=True)[:90]:
            if not edge.path:
                continue
            step = max(1, len(edge.path) // 24)
            points = [[x, y] for x, y in edge.path[::step]]
            if points[-1] != [edge.path[-1][0], edge.path[-1][1]]:
                points.append([edge.path[-1][0], edge.path[-1][1]])
            flows.append(
                {
                    "s": src,
                    "d": dst,
                    "w": round(float(edge.weight), 3),
                    "res": int(edge.dominant),
                    "p": points,
                    "v": round(float(edge.total_flow), 2),
                }
            )
        pulses = [
            {
                "x": p["x"],
                "y": p["y"],
                "age": self.tick_count - p["t0"],
                "r": p["r0"],
                "k": p["kind"],
                "s": p["strength"],
            }
            for p in self.pulses
            if self.tick_count - p["t0"] <= PULSE_TTL
        ]
        degree: dict[int, int] = dict.fromkeys(self.settlements, 0)
        for src, dst in self.edges:
            degree[src] = degree.get(src, 0) + 1
            degree[dst] = degree.get(dst, 0) + 1

        return {
            "tick": self.tick_count,
            "pop": _b64(pop),
            "hatch": _b64(hatch),
            "roads": _b64(np.clip(self.road_level / max(self.econ.road_max, 1) * 255, 0, 255).astype(np.uint8)),
            "settlements": [
                SettlementSnapshot.model_validate(s.snapshot(self.econ, degree.get(s.sid, 0))).model_dump()
                for s in self.settlements.values()
            ],
            "flows": flows,
            "pulses": pulses,
            "network": self.network_summary(),
            "history": self.history[-160:],
            "shocks": [ShockEvent(**{k: r[k] for k in ShockEvent.model_fields}).model_dump() for r in self.active],
            "events": list(self.event_log)[-12:],
        }
