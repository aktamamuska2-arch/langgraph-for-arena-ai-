"""Network topology: graph construction, criticality analysis and demand load."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

from .schemas import NetworkSpec, ScenarioSpec


@dataclass(frozen=True)
class PortLoad:
    """Static berth-day demand implied by the O-D channels of one port."""

    port_id: str
    calls_per_day: float
    teu_per_day: float
    berth_days_per_day: float
    base_berths: int
    service_teu_per_berth_day: float

    @property
    def nominal_utilization(self) -> float:
        capacity = self.base_berths * self.service_teu_per_berth_day
        return (
            self.berth_days_per_day / max(self.base_berths, 1e-9)
            if capacity <= 0
            else ((self.teu_per_day / self.service_teu_per_berth_day) / self.base_berths)
        )


def build_graph(network: NetworkSpec) -> nx.Graph:
    """Undirected lane graph with node kind/zone attributes."""

    g = nx.Graph()
    for z in network.zones:
        g.graph.setdefault("zones", {})[z.id] = z.model_dump()
    for idx, p in enumerate(network.ports):
        g.add_node(
            p.id,
            kind="port",
            label=p.name,
            lat=p.lat,
            lon=p.lon,
            zone=p.zone,
            berths=p.berths,
            service_rate=p.service_teu_per_berth_day,
            role=p.role,
            order=idx,
        )
    for c in network.chokepoints:
        g.add_node(
            c.id,
            kind="chokepoint",
            label=c.name,
            lat=c.lat,
            lon=c.lon,
            zone=c.zone,
            slots=c.concurrent_slots,
            transit_hours=c.transit_hours,
            geo_rate=c.geo_rate_per_year,
            max_severity=c.max_severity,
            reroute_hours=c.reroute_hours,
        )
    for lane in network.lanes:
        g.add_edge(
            lane.a,
            lane.b,
            distance_nm=lane.distance_nm,
            zone=lane.zone,
            cost_usd_per_teu=lane.cost_usd_per_teu,
            capacity_teu_day=lane.capacity_teu_day,
            contingency=lane.contingency,
            note=lane.note,
        )
    return g


def port_loads(scenario: ScenarioSpec, capacity_scale: float = 1.0) -> dict[str, PortLoad]:
    """Allocate daily calls and TEU to each terminal from the O-D channels.

    A vessel call consumes one berth slot at both ends, so a port's berth-day
    demand is the sum of service demands of all channels terminating there.
    """

    network = scenario.network
    calls: dict[str, float] = {p.id: 0.0 for p in network.ports}
    teu: dict[str, float] = {p.id: 0.0 for p in network.ports}
    for ch in network.channels:
        rate = ch.base_daily_calls * scenario.stress.demand_multiplier
        teu_rate = rate * ch.teu_mean
        for endpoint in (ch.origin, ch.destination):
            calls[endpoint] += rate
            teu[endpoint] += teu_rate
    loads: dict[str, PortLoad] = {}
    for p in network.ports:
        max(1.0, p.berths * capacity_scale)
        berth_days = teu[p.id] / p.service_teu_per_berth_day if p.service_teu_per_berth_day else 0.0
        loads[p.id] = PortLoad(
            port_id=p.id,
            calls_per_day=calls[p.id],
            teu_per_day=teu[p.id],
            berth_days_per_day=berth_days,
            base_berths=p.berths,
            service_teu_per_berth_day=p.service_teu_per_berth_day,
        )
    return loads


def nominal_utilization(load: PortLoad, capacity_scale: float = 1.0) -> float:
    return load.berth_days_per_day / max(load.base_berths * capacity_scale, 1e-9)


def criticality(graph: nx.Graph) -> dict[str, float]:
    """Betweenness centrality restricted to port-to-port traffic.

    Chokepoints score highly when they sit on many shortest sea routes, which
    is exactly the quantity a geopolitical blockade attacks.
    """

    port_nodes = [n for n, d in graph.nodes(data=True) if d["kind"] == "port"]
    for _u, _v, d in graph.edges(data=True):
        d["_w"] = d["distance_nm"]
    centrality = nx.betweenness_centrality_subset(
        graph, sources=port_nodes, targets=port_nodes, weight="_w", normalized=True
    )
    return {k: round(v, 6) for k, v in centrality.items()}


def articulation_points(graph: nx.Graph) -> list[str]:
    return sorted(nx.articulation_points(graph))


def connectivity_report(graph: nx.Graph) -> dict[str, Any]:
    is_connected = nx.is_connected(graph)
    components = [sorted(c) for c in nx.connected_components(graph)]
    ports = [n for n, d in graph.nodes(data=True) if d["kind"] == "port"]
    unreachable: list[tuple[str, str]] = []
    if is_connected:
        for i, a in enumerate(ports):
            for b in ports[i + 1 :]:
                if not nx.has_path(graph, a, b):  # pragma: no cover - defensive
                    unreachable.append((a, b))
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": round(nx.density(graph), 5),
        "connected": is_connected,
        "components": len(components),
        "articulation_points": articulation_points(graph),
        "unreachable_pairs": unreachable,
    }


def lane_pairs(scenario: ScenarioSpec) -> dict[tuple[str, str], dict[str, Any]]:
    """Mirrored lane lookup keyed by ordered node pair."""

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for lane in scenario.network.lanes:
        payload = {
            "distance_nm": lane.distance_nm,
            "zone": lane.zone,
            "cost_usd_per_teu": lane.cost_usd_per_teu,
            "capacity_teu_day": lane.capacity_teu_day,
            "contingency": lane.contingency,
        }
        out[(lane.a, lane.b)] = payload
        out[(lane.b, lane.a)] = payload
    return out


def node_coordinates(scenario: ScenarioSpec) -> dict[str, tuple[float, float, str]]:
    coords: dict[str, tuple[float, float, str]] = {}
    for p in scenario.network.ports:
        coords[p.id] = (p.lon, p.lat, "port")
    for c in scenario.network.chokepoints:
        coords[c.id] = (c.lon, c.lat, "chokepoint")
    return coords
