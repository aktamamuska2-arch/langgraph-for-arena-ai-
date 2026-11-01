"""Decentralised market discovery and the structural gravity kernel.

Discovery
---------
Every ``discovery_interval`` ticks each settlement scans the settlements inside
its ``discovery_radius``. A directed, resource-specific link is scored by the
structural gravity kernel

    A_ij = G * (S_i)^a * (D_j)^b / [ (d_ij+d0)^g * Pi_i^w * P_j^w ]
           * exp(-tau c_ij) * (1 + mu m_ij)

with two *multilateral resistance* terms built from the settlement's own
horizon: ``Pi_i`` is seller i's access to rival buyers and ``P_j`` is buyer j's
menu of rival suppliers. Dividing by them turns a raw bilateral mass into the
share of commerce an agent can expect, which is what Anderson & van Wincoop
(2003) show is required for the distance coefficient to be identified at all.
Without it the kernel is size-biased: a big settlement scores highly against
everyone, and the *relative* cut that keeps the graph sparse then starves small
peripheral nodes.

Discovery runs two passes:

1. **seller-initiated** - a settlement with marketable surplus solicits buyers;
2. **buyer-initiated** - a settlement with an unserved deficit solicits
   suppliers. This is the asymmetry fix: a pure importer, which owns nothing
   anyone wants, used to be unreachable and remained an orphan forever.

Execution
---------
A link trades when the spatial arbitrage condition holds

    p_j - p_i > tau * c_ij * embargo

with volume ``q = min(available_i, needed_j) * trade_fraction * min(1, w)``,
``transfer_loss`` burned in transit and ``tau c q`` burned as freight.

Relationship hysteresis
-----------------------
Weights decay every tick and links are pruned when idle, but *relationship
capital* buys time: the decay exponent is divided by ``1 + kappa m`` and the idle
grace period is multiplied by ``1 + grace m``. Trade relationships are known to
be hysteretic - they survive adverse shocks and re-form along existing channels
rather than being rebuilt from scratch (Besedes & Prusa, 2006). Without this,
a shock that suspends arbitrage for a few ticks deletes the whole network just
when it is most needed.
"""

from __future__ import annotations

import math
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
    initiator: str = "seller"

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
    """Gravity scoring, decentralised discovery, arbitrage and decay."""

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
        resistance_out: float = 1.0,
        resistance_in: float = 1.0,
    ) -> float:
        """Bilateral gravity score, normalised by multilateral resistance."""

        g = self.gravity
        if surplus <= EPS or deficit <= EPS or distance < 0:
            return 0.0
        mass = (surplus**g.alpha) * (deficit**g.beta)
        separation = (distance + g.d0) ** g.gamma
        resistance = (max(resistance_out, g.mr_floor) * max(resistance_in, g.mr_floor)) ** g.mr_weight
        friction = math.exp(-g.tau * transport_cost)
        memory = 1.0 + g.memory_bonus * min(trust, 6.0)
        return float(g.scale * mass / (separation * resistance) * friction * memory)

    def access_indices(self, model: AgoraModel, alive: list) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Outward (Pi) and inward (P) resistance, computed locally.

        Each settlement only sums over neighbours inside its own discovery
        radius, so this stays a decentralised computation - there is no global
        market-clearing solve anywhere in the loop.
        """

        g = self.gravity
        radius = self.econ.discovery_radius
        surplus = {s.sid: s.marketable(self.econ) for s in alive}
        deficit = {s.sid: s.deficit(self.econ) for s in alive}
        out_res: dict[int, np.ndarray] = {}
        in_res: dict[int, np.ndarray] = {}
        for agent in alive:
            pi = np.zeros(len(RESOURCES))
            pj = np.zeros(len(RESOURCES))
            for other in alive:
                if other.sid == agent.sid:
                    continue
                distance = math.hypot(other.x - agent.x, other.y - agent.y)
                if distance > radius:
                    continue
                kernel = (distance + g.d0) ** (-g.gamma)
                pi += deficit[other.sid] * kernel
                pj += surplus[other.sid] * kernel
            out_res[agent.sid] = pi
            in_res[agent.sid] = pj
        return out_res, in_res

    # -- discovery -------------------------------------------------------
    def discover(self, model: AgoraModel) -> int:
        """One decentralised scan pass; returns the number of edges touched."""

        alive = [s for s in model.settlements.values() if s.alive]
        if len(alive) < 2:
            return 0
        resistance_out, resistance_in = self.access_indices(model, alive)
        touched = self._seller_pass(model, alive, resistance_out, resistance_in)
        touched += self._buyer_pass(model, alive, resistance_out, resistance_in)
        return touched

    def _candidate_score(
        self,
        model: AgoraModel,
        seller,
        buyer,
        resource: int,
        resistance_out: dict[int, np.ndarray],
        resistance_in: dict[int, np.ndarray],
    ) -> tuple[float, float]:
        distance = math.hypot(buyer.x - seller.x, buyer.y - seller.y)
        if distance > self.econ.discovery_radius:
            return 0.0, 0.0
        edge = model.edges.get((seller.sid, buyer.sid))
        cost = edge.cost if edge is not None else model.route_cost(seller.pos, buyer.pos)
        trust = seller.partners.get(buyer.sid, 0.0)
        score = self.attractiveness(
            float(seller.marketable(self.econ)[resource]),
            float(buyer.deficit(self.econ)[resource]),
            distance,
            cost,
            trust,
            float(resistance_out[seller.sid][resource]),
            float(resistance_in[buyer.sid][resource]),
        )
        return score, cost

    def _seller_pass(
        self,
        model: AgoraModel,
        alive: list,
        resistance_out: dict[int, np.ndarray],
        resistance_in: dict[int, np.ndarray],
    ) -> int:
        econ = self.econ
        touched = 0
        for seller in alive:
            surplus = seller.marketable(econ)
            if not np.any(surplus > EPS):
                continue
            scored: list[tuple[float, int, int]] = []
            for buyer in alive:
                if buyer.sid == seller.sid:
                    continue
                if not np.any(buyer.deficit(econ) > EPS):
                    continue
                for r in range(len(RESOURCES)):
                    if surplus[r] <= EPS:
                        continue
                    score, _ = self._candidate_score(model, seller, buyer, r, resistance_out, resistance_in)
                    if score > 0:
                        scored.append((score, buyer.sid, r))
            if not scored:
                continue
            touched += self._commit(model, seller.sid, scored, "seller")
        return touched

    def _buyer_pass(
        self,
        model: AgoraModel,
        alive: list,
        resistance_out: dict[int, np.ndarray],
        resistance_in: dict[int, np.ndarray],
    ) -> int:
        """A deficit with no supplier inside its horizon solicits one.

        Without this pass the graph is only as connected as the *sellers* choose
        it to be, so any settlement that owns nothing in demand becomes a
        permanent orphan - artificial fragmentation caused purely by the
        direction of search.
        """

        econ = self.econ
        touched = 0
        for buyer in alive:
            deficit = buyer.deficit(econ)
            if not np.any(deficit > EPS):
                continue
            served = np.zeros(len(RESOURCES), dtype=bool)
            for (_src, dst), edge in model.edges.items():
                if dst == buyer.sid and float(edge.weights.max()) >= econ.edge_min_weight:
                    served |= edge.weights >= econ.edge_min_weight
            if served.all():
                continue
            initiated = 0
            # staple food is the survival-critical channel; wood and ore are
            # sourced opportunistically by the seller pass. Restricting the
            # initiative keeps the safety net without letting it saturate the
            # graph and drown out the gravity kernel's distance decay.
            for resource in (0,):
                if served[resource] or initiated >= econ.buyer_initiative:
                    continue
                options: list[tuple[float, int, int]] = []
                for seller in alive:
                    if seller.sid == buyer.sid:
                        continue
                    if seller.marketable(econ)[resource] <= EPS:
                        continue
                    score, _ = self._candidate_score(
                        model, seller, buyer, int(resource), resistance_out, resistance_in
                    )
                    if score > 0:
                        options.append((score, seller.sid, int(resource)))
                if not options:
                    continue
                options.sort(reverse=True)
                best = options[0][0]
                kept = [o for o in options if o[0] >= max(econ.edge_threshold, econ.candidate_cut * best)][:2]
                for score, seller_id, r in kept:
                    touched += self._upsert(model, seller_id, buyer.sid, [(score, r)], score, "buyer")
                    initiated += 1
        return touched

    def _commit(self, model: AgoraModel, seller_id: int, scored: list[tuple[float, int, int]], who: str) -> int:
        """Apply relative selection and cap the partner list."""

        econ = self.econ
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
        return sum(
            self._upsert(model, seller_id, buyer_id, items, top_score, who) for top_score, buyer_id, items in ranked
        )

    def _upsert(
        self,
        model: AgoraModel,
        src: int,
        dst: int,
        items: list[tuple[float, int]],
        top_score: float = 0.0,
        initiator: str = "seller",
    ) -> int:
        econ = self.econ
        edge = model.edges.get((src, dst))
        if edge is None:
            edge = TradeEdge(src=src, dst=dst, created_tick=model.tick_count, initiator=initiator)
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
                seller.add_stocks(-np.eye(len(RESOURCES))[r] * quantity)
                buyer.add_stocks(np.eye(len(RESOURCES))[r] * delivered)
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
                seller.trade_volume += traded_here
                buyer.trade_volume += traded_here
                model.on_trade(edge, traded_here)

        return delivered_total

    # -- decay -----------------------------------------------------------
    def decay(self, model: AgoraModel) -> int:
        """Weight decay with relationship hysteresis."""

        econ = self.econ
        base_rate = -math.log(max(econ.edge_decay, 1e-6))
        removed = 0
        for key in list(model.edges):
            edge = model.edges[key]
            seller = model.settlements.get(edge.src)
            trust = seller.partners.get(edge.dst, 0.0) if seller is not None else 0.0
            rate = base_rate / (1.0 + econ.trust_halflife_decay * min(trust, 6.0))
            edge.weights = edge.weights * math.exp(-rate)
            idle = model.tick_count - edge.last_active
            grace = econ.edge_max_idle * (1.0 + econ.trust_grace * min(trust, 4.0))
            if float(edge.weights.max()) < econ.edge_min_weight or idle > grace:
                del model.edges[key]
                removed += 1
        return removed
