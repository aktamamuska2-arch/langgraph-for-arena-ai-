"""Decentralised market discovery and the spatial gravity model.

Discovery
---------
Every ``discovery_interval`` ticks each settlement scans the settlements inside
its ``discovery_radius`` and scores a directed, resource-specific link by

    A_ij^(r) = G * S_i^alpha * D_j^beta / (d_ij + d0)^gamma
               * exp(-tau * c_ij) * (1 + mu * m_ij)

where ``S`` is the seller's surplus, ``D`` the buyer's deficit, ``d`` the
Euclidean separation, ``c`` the terrain-weighted transport cost along the
cheapest route and ``m`` the accumulated repeat-partner trust. An edge is
created only when ``A`` crosses ``edge_threshold``; nothing is assigned
centrally and no settlement can see prices beyond its own horizon.

Execution
---------
A link trades when the spatial arbitrage condition holds

    p_j - p_i > tau * c_ij        (buyer price - seller price > freight)

with volume ``q = min(surplus_i, deficit_j) * trade_fraction * min(1, w)``,
``transfer_loss`` burned in transit and ``tau * c_ij * q`` burned as freight.

Edges decay geometrically every tick and are pruned when their weight falls
below ``edge_min_weight`` or they stay idle for ``edge_max_idle`` ticks, so the
network has to be *maintained* by real exchange.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .schemas import RESOURCES, EconomySpec, GravitySpec

if TYPE_CHECKING:  # pragma: no cover
    from .model import AgoraModel

EPS = 1e-6


@dataclass
class TradeEdge:
    """One directed link between two settlements, with per-resource weights."""

    src: int
    dst: int
    weights: np.ndarray = field(default_factory=lambda: np.zeros(len(RESOURCES)))
    path: list[tuple[int, int]] = field(default_factory=list)
    cost: float = 0.0
    cum_flow: float = 0.0
    total_flow: float = 0.0
    created_tick: int = 0
    last_active: int = -10_000
    dominant: int = 0
    epoch: int = -1

    @property
    def key(self) -> tuple[int, int]:
        return (self.src, self.dst)

    @property
    def weight(self) -> float:
        return float(self.weights.sum())

    def refresh_path(self, model: AgoraModel, force: bool = False) -> None:
        """Recompute the cheapest route when the road network has changed."""

        if not force and self.epoch == model.road_epoch and self.path:
            return
        src = model.settlements[self.src].pos
        dst = model.settlements[self.dst].pos
        path = model.cheapest_path(src, dst)
        self.path = path
        self.cost = model.path_cost(path)
        self.epoch = model.road_epoch


class MarketEngine:
    """Stateless-ish policy object holding the gravity parameters."""

    def __init__(self, econ: EconomySpec) -> None:
        self.econ = econ
        self.gravity: GravitySpec = econ.gravity

    # -- gravity ---------------------------------------------------------
    def attractiveness(
        self,
        surplus: float,
        deficit: float,
        distance: float,
        transport_cost: float,
        trust: float,
    ) -> float:
        g = self.gravity
        if surplus <= EPS or deficit <= EPS or distance < 0:
            return 0.0
        mass = (surplus**g.alpha) * (deficit**g.beta)
        separation = (distance + g.d0) ** g.gamma
        friction = float(np.exp(-g.tau * transport_cost))
        memory = 1.0 + g.memory_bonus * min(trust, 6.0)
        return float(g.scale * mass / separation * friction * memory)

    # -- discovery -------------------------------------------------------
    def discover(self, model: AgoraModel) -> int:
        """One decentralised scan pass; returns the number of edges touched."""

        econ = self.econ
        alive = [s for s in model.settlements.values() if s.alive]
        if len(alive) < 2:
            return 0
        touched = 0
        radius = econ.discovery_radius

        for seller in alive:
            surplus = seller.marketable(econ)
            if not np.any(surplus > EPS):
                continue
            scored: list[tuple[float, int, int]] = []
            for buyer in alive:
                if buyer.sid == seller.sid:
                    continue
                dx = buyer.x - seller.x
                dy = buyer.y - seller.y
                distance = float(np.hypot(dx, dy))
                if distance > radius:
                    continue
                deficit = buyer.deficit(econ)
                if not np.any(deficit > EPS):
                    continue
                trust = seller.partners.get(buyer.sid, 0.0)
                edge = model.edges.get((seller.sid, buyer.sid))
                transport_cost = edge.cost if edge is not None else model.route_cost(seller.pos, buyer.pos)
                for r in range(len(RESOURCES)):
                    score = self.attractiveness(float(surplus[r]), float(deficit[r]), distance, transport_cost, trust)
                    if score >= econ.edge_threshold:
                        scored.append((score, buyer.sid, r))

            # relative selection: a partner is kept only if its gravity score is
            # competitive for that resource - this is what makes the network
            # sparse and structured instead of a complete graph
            best_for_resource: dict[int, float] = {}
            for score, _, r in scored:
                best_for_resource[r] = max(best_for_resource.get(r, 0.0), score)
            per_buyer: dict[int, list[tuple[float, int]]] = {}
            for score, buyer_id, r in scored:
                cut = max(econ.edge_threshold, econ.candidate_cut * best_for_resource[r])
                if score >= cut:
                    per_buyer.setdefault(buyer_id, []).append((score, r))
            ranked = sorted(
                ((max(sc for sc, _ in items), bid, items) for bid, items in per_buyer.items()),
                reverse=True,
            )[: econ.max_partners]
            for top_score, buyer_id, items in ranked:
                touched += self._upsert(model, seller.sid, buyer_id, items, top_score)
        return touched

    def _upsert(
        self,
        model: AgoraModel,
        src: int,
        dst: int,
        items: list[tuple[float, int]],
        top_score: float = 0.0,
    ) -> int:
        econ = self.econ
        edge = model.edges.get((src, dst))
        if edge is None:
            edge = TradeEdge(src=src, dst=dst, created_tick=model.tick_count)
            model.edges[(src, dst)] = edge
            edge.refresh_path(model, force=True)
        else:
            edge.refresh_path(model)
        reference = max(top_score, econ.edge_threshold)
        updated = edge.weights.copy()
        for score, r in items:
            value = min(4.0, 2.0 * score / reference)
            updated[r] = max(float(updated[r]) * econ.edge_decay, float(value))
        edge.weights = updated
        edge.dominant = int(np.argmax(edge.weights))
        return 1

    # -- execution -------------------------------------------------------
    def execute(self, model: AgoraModel) -> float:
        """Run the trade round; returns total delivered volume."""

        econ = self.econ
        delivered_total = 0.0
        if not model.edges:
            return 0.0
        order = sorted(model.edges.values(), key=lambda e: e.weight, reverse=True)

        for edge in order:
            seller = model.settlements.get(edge.src)
            buyer = model.settlements.get(edge.dst)
            if seller is None or buyer is None or not seller.alive or not buyer.alive:
                continue
            seller_prices = seller.prices(econ)
            buyer_prices = buyer.prices(econ)
            freight_unit = self.gravity.tau * max(edge.cost, 0.0) * model.freight_multiplier(edge)
            traded_here = 0.0

            for r in range(len(RESOURCES)):
                weight = float(edge.weights[r])
                if weight < econ.edge_min_weight:
                    continue
                if buyer_prices[r] - seller_prices[r] <= freight_unit:
                    continue
                available = max(0.0, float(seller.stocks[r] - 0.35 * seller.target_stock(econ)[r]))
                needed = max(0.0, float(buyer.target_stock(econ)[r] - buyer.stocks[r]))
                quantity = min(available, needed) * econ.trade_fraction * min(1.0, weight)
                if quantity <= EPS:
                    continue

                delivered = quantity * (1.0 - econ.transfer_loss)
                seller.stocks[r] -= quantity
                buyer.stocks[r] += delivered
                seller.wealth += quantity * float(seller_prices[r])
                buyer.wealth -= quantity * (float(seller_prices[r]) + freight_unit)
                model.ledger["exported"][r] += quantity
                model.ledger["imported"][r] += delivered
                seller.traded[r] += quantity
                buyer.traded[r] += delivered
                edge.cum_flow += quantity
                edge.total_flow += quantity
                edge.last_active = model.tick_count
                traded_here += delivered
                delivered_total += delivered

            if traded_here > 0:
                seller.partners[buyer.sid] = seller.partners.get(buyer.sid, 0.0) + 0.05
                buyer.partners[seller.sid] = buyer.partners.get(seller.sid, 0.0) + 0.05
                seller.advance_tech(econ, traded_here)
                buyer.advance_tech(econ, traded_here)
                model.on_trade(edge, traded_here)

        return delivered_total

    # -- decay -----------------------------------------------------------
    def decay(self, model: AgoraModel) -> int:
        econ = self.econ
        removed = 0
        for key in list(model.edges):
            edge = model.edges[key]
            edge.weights = edge.weights * econ.edge_decay
            idle = model.tick_count - edge.last_active
            if float(edge.weights.max()) < econ.edge_min_weight or idle > econ.edge_max_idle:
                del model.edges[key]
                removed += 1
        return removed
