"""The vertical supply chain: producers, pengepul, penjual — and the syndicates.

Why a second layer
------------------
A settlement that both produces and consumes is a *node*, not an economy. This
module disaggregates it into the actors a macroeconomist would actually ask
about:

```
      produsen ──► pengepul ──► penjual ──► rumah tangga
     (producers)  (collectors)  (retailers)   (the granary)
         ▲
         └──────── sindikat (predation, appears when the chain is unstable)
```

Every margin is a share of the *retail* price, which is the scarcity price the
settlement already computes. Consumer prices are therefore untouched by the
existence of the chain; what the chain changes is **who captures the value**,
how much of it never arrives, and how quickly the whole thing fails when it is
stressed.

The three things this layer adds that a flat model cannot express
-----------------------------------------------------------------
1. **Marketing margin.** Households pay the retail price; the producer receives
   only ``(1-m_retail)(1-m_wholesale)(1-m_gate)`` of it. The gap is the chain's
   cut, and it is a *transfer*, not a loss — which is exactly why a flat,
   single-actor model cannot show a farm-gate crisis.
2. **Concentration.** Ownership within a resource node is drawn from a power law,
   so one producer usually holds a majority. When the Herfindahl index crosses
   ``vertical_threshold`` that producer is *vertically integrated*: it withholds
   supply to defend its price, and the collectors that depend on it become
   fragile. Concentration is measured, not assumed.
3. **Predation.** Syndicates are not scripted. Their power obeys an entry/exit
   logistic driven by a stress index (marketing spread, stockouts, relative
   scarcity). In a calm world they stay at zero; a supply-chain shock is what
   makes them multiply, and their growth is bounded by ``syndicate_cap``.

Conservation
------------
Nothing here creates or destroys matter except through an explicit, booked loss:
``loss_gate``, ``loss_handling``, ``loss_retail`` and syndicate diversion. Every
kilogram either reaches the granary, stays in an inventory, or is written to the
``diverted`` ledger — which is why ``mass_balance`` still holds to 1e-13.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .schemas import RESOURCES, EconomySpec, SupplySpec

__all__ = ["Collector", "Firm", "Retailer", "SupplyChain", "Syndicate"]

STRESS_SPREAD = 1.6  # weight on the marketing spread blowing out
STRESS_STOCKOUT = 1.4  # weight on households going unserved
STRESS_SCARCITY = 0.6  # weight on relative local scarcity


@dataclass
class Firm:
    """A producer: one owner of a slice of a settlement's resource node."""

    fid: int
    sid: int
    resource: int
    share: float
    capacity: float
    vertical: bool = False
    collector: int = -1
    cash: float = 0.0
    revenue: float = 0.0
    sold: float = 0.0

    @property
    def name(self) -> str:
        return f"F{self.fid}"


@dataclass
class Collector:
    """A pengepul: buys from producers in its catchment, sells downstream.

    Inventory is what makes a pengepul more than a pipe: it is the buffer that
    decides whether a bad week reaches the household, and it is also the stock a
    vertically integrated upstream can starve.
    """

    cid: int
    sid: int
    weight: float
    throughput: np.ndarray
    inventory: np.ndarray = field(default_factory=lambda: np.zeros(len(RESOURCES)))
    cash: float = 100.0
    dependency: float = 0.0
    firms: list[int] = field(default_factory=list)
    bought: np.ndarray = field(default_factory=lambda: np.zeros(len(RESOURCES)))
    shipped: np.ndarray = field(default_factory=lambda: np.zeros(len(RESOURCES)))
    stranded: float = 0.0

    @property
    def name(self) -> str:
        return f"P{self.cid}"


@dataclass
class Retailer:
    """A penjual: the last hop, facing the households directly."""

    rid: int
    sid: int
    weight: float
    outlet: np.ndarray
    inventory: np.ndarray = field(default_factory=lambda: np.zeros(len(RESOURCES)))
    cash: float = 50.0
    markup: float = 0.10
    sold: np.ndarray = field(default_factory=lambda: np.zeros(len(RESOURCES)))
    unmet: float = 0.0

    @property
    def name(self) -> str:
        return f"R{self.rid}"


@dataclass
class Syndicate:
    """A sindikat: rents extracted from a chain that is already under stress.

    ``power`` is the only state. Itself it neither produces nor consumes: it
    diverts a share of the physical flow (a booked loss) and levies a fee on the
    intermediaries (a transfer).
    """

    zid: int
    sid: int
    power: float = 0.0
    stress: float = 0.0
    diverted: float = 0.0
    fees: float = 0.0
    active_ticks: int = 0

    @property
    def active(self) -> bool:
        return self.power > 0.05

    @property
    def name(self) -> str:
        return f"Z{self.zid}"


class SupplyChain:
    """Owns every firm, pengepul, penjual and sindikat in the world."""

    def __init__(self, model) -> None:
        self.model = model
        self.spec: SupplySpec = model.spec.supply
        self.econ: EconomySpec = model.econ
        self.firms: dict[int, Firm] = {}
        self.collectors: dict[int, Collector] = {}
        self.retailers: dict[int, Retailer] = {}
        self.syndicates: dict[int, Syndicate] = {}
        # (sid, resource) -> owning firm id and its node concentration
        self.ownership: dict[tuple[int, int], tuple[int, float]] = {}
        self.by_node: dict[tuple[int, int], list[int]] = {}
        self.by_settlement: dict[int, list[int]] = {}
        self.chain_loss: dict[str, float] = {
            "gate": 0.0,
            "handling": 0.0,
            "retail": 0.0,
            "diverted": 0.0,
        }
        self.fees_paid = 0.0
        self.gross_value = 0.0
        self.delivered_value = 0.0
        self.cum_produced = 0.0
        # exponential moving averages: the console and the gates care about the
        # *current* state of the chain, not about a cumulative total dominated
        # by the calm opening centuries
        self.ema = {"gross": 0.0, "lost": 0.0, "take": 0.0}
        self.ema_alpha = 0.02
        self._build()

    # -- construction ----------------------------------------------------
    def _build(self) -> None:
        spec = self.spec
        rng = np.random.default_rng(self.model.spec.world.seed + 9059)
        fid = cid = rid = 0
        for settlement in self.model.settlements.values():
            sid = settlement.sid
            labour = np.maximum(settlement.labor, 1e-6)
            counts = np.maximum(1, np.round(spec.producers_per_settlement * labour / labour.sum())).astype(int)
            ceiling = 1.0 + self.econ.tech_ceiling
            firm_ids: list[int] = []
            for r in range(len(RESOURCES)):
                node: list[int] = []
                tail = np.arange(1, counts[r] + 1, dtype=float) ** (-spec.ownership_tail)
                tail = tail[rng.permutation(len(tail))]
                shares = tail / tail.sum()
                for share in shares:
                    self.firms[fid] = Firm(
                        fid=fid,
                        sid=sid,
                        resource=r,
                        share=float(share),
                        capacity=float(share * settlement.endowment[r] * ceiling),
                        cash=np.float64(rng.uniform(10.0, 40.0)).item(),
                    )
                    node.append(fid)
                    firm_ids.append(fid)
                    fid += 1
                self.by_node[(sid, r)] = node
                hhi = float(np.sum(shares**2))
                dominant = int(node[int(np.argmax(shares))])
                self.ownership[(sid, r)] = (dominant, hhi)
                if hhi >= spec.vertical_threshold:
                    self.firms[dominant].vertical = True
            self.by_settlement[sid] = firm_ids

            n_col = max(1, spec.collectors_per_settlement)
            for _k in range(n_col):
                base = settlement.endowment * ceiling * spec.collector_throughput_mult / n_col
                self.collectors[cid] = Collector(
                    cid=cid, sid=sid, weight=1.0 / n_col, throughput=base.astype(float)
                )
                cid += 1
            n_ret = max(1, spec.retailers_per_settlement)
            for _k in range(n_ret):
                base = settlement.endowment * ceiling * spec.retailer_outlet_mult / n_ret
                self.retailers[rid] = Retailer(
                    rid=rid, sid=sid, weight=1.0 / n_ret, outlet=base.astype(float), markup=spec.margin_retail
                )
                rid += 1
            self.syndicates[sid] = Syndicate(zid=sid, sid=sid)

        self._assign_firms()

    def _assign_firms(self) -> None:
        """Round-robin the largest producers across collectors, so each pengepul
        has a catchment — and its dependency is the biggest name in it."""

        for sid in self.by_settlement:
            collectors = [c for c in self.collectors.values() if c.sid == sid]
            order = sorted(self.by_settlement[sid], key=lambda f: -self.firms[f].share)
            for i, f in enumerate(order):
                target = collectors[i % len(collectors)]
                self.firms[f].collector = target.cid
                target.firms.append(f)
            for c in collectors:
                total = sum(self.firms[f].share for f in c.firms)
                c.dependency = (
                    max((self.firms[f].share for f in c.firms), default=0.0) / total if total > 0 else 0.0
                )

    def seed_inventories(self, share: float = 0.22) -> None:
        """Prime the chain from the opening granaries.

        A chain that starts empty starves the first households for two ticks.
        Moving stock out of the granary into pengepul and penjual shelves keeps
        the world's total stock exactly unchanged - a transfer, not a gift.
        """

        for settlement in self.model.settlements.values():
            moved = settlement.stocks * share
            settlement.add_stocks(-moved)
            collectors = self.collectors_of(settlement.sid)
            retailers = self.retailers_of(settlement.sid)
            if collectors:
                for c in collectors:
                    c.inventory += moved * 0.6 * c.weight / max(sum(x.weight for x in collectors), 1e-9)
            if retailers:
                for r in retailers:
                    r.inventory += moved * 0.4 * r.weight / max(sum(x.weight for x in retailers), 1e-9)

    # -- helpers ---------------------------------------------------------
    def collectors_of(self, sid: int) -> list[Collector]:
        return [c for c in self.collectors.values() if c.sid == sid]

    def retailers_of(self, sid: int) -> list[Retailer]:
        return [r for r in self.retailers.values() if r.sid == sid]

    def node_hhi(self, sid: int, resource: int) -> float:
        return self.ownership.get((sid, resource), (-1, 0.0))[1]

    def dominant_firm(self, sid: int, resource: int) -> Firm | None:
        fid = self.ownership.get((sid, resource), (-1, 0.0))[0]
        return self.firms.get(fid) if fid >= 0 else None

    def vertical_share(self) -> float:
        if not self.ownership:
            return 0.0
        hits = sum(1 for _key, (_fid, hhi) in self.ownership.items() if hhi >= self.spec.vertical_threshold)
        return hits / len(self.ownership)

    def max_dependency(self) -> float:
        return max((c.dependency for c in self.collectors.values()), default=0.0)

    # -- the tick --------------------------------------------------------
    def route(self, settlement, output: np.ndarray) -> np.ndarray:
        """Move one tick of a settlement's output down the chain.

        Returns the volume that actually lands in the granary. Everything else is
        either still in an inventory (conserved, arriving later) or is written to
        the ``diverted`` ledger as an explicit loss.
        """

        spec = self.spec
        sid = settlement.sid
        collectors = self.collectors_of(sid)
        retailers = self.retailers_of(sid)
        syndicate = self.syndicates.get(sid)
        power = syndicate.power if syndicate else 0.0
        landed = np.zeros(len(RESOURCES))
        lost = np.zeros(len(RESOURCES))
        taken = np.zeros(len(RESOURCES))
        demand = settlement.consumption_rate(self.econ)
        for collector in collectors:
            collector.bought[:] = 0.0
            collector.shipped[:] = 0.0
        for retailer in retailers:
            retailer.sold[:] = 0.0
            retailer.unmet = 0.0

        for r in range(len(RESOURCES)):
            gross = float(output[r])
            if gross <= 1e-12 and float(settlement.stocks[r]) <= 1e-9:
                continue
            self.gross_value += gross * float(settlement.prices(self.econ)[r])
            self.cum_produced += gross

            # 1. farm gate: the first handling loss happens before anyone is paid
            gate_loss = gross * spec.loss_gate
            self.chain_loss["gate"] += gate_loss
            lost[r] += gate_loss
            flow = gross - gate_loss

            # 2. predation happens in transit, before the pengepul sees the cargo
            diverted = flow * spec.syndicate_diversion * power
            if diverted > 0.0:
                self.chain_loss["diverted"] += diverted
                lost[r] += diverted
                taken[r] += diverted
                if syndicate:
                    syndicate.diverted += diverted
            flow -= diverted

            # 3. producers sell into their pengepul, capped by its throughput
            if flow > 0.0 and collectors:
                for c in collectors:
                    share = c.weight if gross > 0 else 0.0
                    intake = flow * share
                    headroom = float(c.throughput[r] - c.inventory[r])
                    if headroom <= 0.0:
                        # saturated pengepul: the cargo bypasses it and is sold
                        # at the farm gate directly into the granary
                        landed[r] += intake
                        continue
                    take = min(intake, headroom)
                    c.inventory[r] += take
                    c.bought[r] += take
                    landed[r] += intake - take
            else:
                landed[r] += flow

            # 4. short haul from pengepul to penjual
            pool = 0.0
            for c in collectors:
                stock = float(c.inventory[r])
                if stock <= 0.0:
                    continue
                # a dominant upstream withholds supply through its catchment:
                # the goods stay in the pengepul, so this is market power, not loss
                hold = 0.0
                owner = self.dominant_firm(sid, r)
                if owner is not None and owner.vertical and owner.collector == c.cid:
                    hold = stock * spec.vertical_withhold
                shipping = min(stock - hold, float(c.throughput[r]))
                if shipping <= 0.0:
                    continue
                c.inventory[r] = stock - shipping
                c.shipped[r] += shipping
                pool += shipping * (1.0 - spec.loss_handling)
                self.chain_loss["handling"] += shipping * spec.loss_handling
                lost[r] += shipping * spec.loss_handling
                c.stranded = float(c.inventory[r])

            # 5. penjual to household: demand is split across outlets by weight,
            #    and whatever nobody buys stays on the shelf (conserved)
            remaining = float(demand[r])
            if retailers:
                for retailer in sorted(retailers, key=lambda x: -x.weight):
                    if remaining <= 0.0 or pool <= 0.0:
                        break
                    room = min(remaining, float(demand[r]) * retailer.weight * 1.25)
                    take = min(pool, room, float(retailer.outlet[r]))
                    if take <= 0.0:
                        continue
                    pool -= take
                    delivered = take * (1.0 - spec.loss_retail)
                    self.chain_loss["retail"] += take * spec.loss_retail
                    lost[r] += take * spec.loss_retail
                    landed[r] += delivered
                    retailer.sold[r] += delivered
                    remaining -= take
                retailer.unmet = max(0.0, remaining)
                if pool > 0.0:
                    shelf = max(retailers, key=lambda x: float(x.outlet[r]))
                    room = float(shelf.outlet[r]) * 3.0 - float(shelf.inventory[r])
                    stored = max(0.0, min(pool, room))
                    shelf.inventory[r] += stored
                    if pool - stored > 0.0:
                        # the shelves are full and nobody is buying: it goes back
                        # to the granary rather than evaporating
                        landed[r] += pool - stored
            else:
                landed[r] += pool

            if power > 0.0:
                fee = flow * spec.syndicate_fee * power * float(settlement.prices(self.econ)[r]) * 0.01
                for actor in (*collectors, *retailers):
                    charge = min(actor.cash, fee / max(len(collectors) + len(retailers), 1))
                    actor.cash -= charge
                    self.fees_paid += charge
                if syndicate:
                    syndicate.fees += fee * 0.01

        # the ledger is per-resource: a scalar here would be broadcast into all
        # three resource slots and triple-count the loss
        self.model.ledger["diverted"] += lost
        gross_tick = float(np.sum(output))
        take_tick = float(np.sum(taken))
        a = self.ema_alpha
        self.ema["gross"] = (1.0 - a) * self.ema["gross"] + a * gross_tick
        self.ema["lost"] = (1.0 - a) * self.ema["lost"] + a * float(np.sum(lost))
        self.ema["take"] = (1.0 - a) * self.ema["take"] + a * take_tick
        delivered = float(landed.sum())
        self.delivered_value += delivered * float(settlement.prices(self.econ).mean())
        return landed

    def update_predation(self, settlement, produced_total: float, landed_total: float) -> None:
        """Entry and exit of syndicates, driven by measured chain stress."""

        spec = self.spec
        syndicate = self.syndicates.get(settlement.sid)
        if syndicate is None:
            return
        demand = float(settlement.consumption_rate(self.econ).sum())
        stockout = 0.0 if demand <= 1e-9 else max(0.0, 1.0 - landed_total / demand)
        nominal = max(spec.marketing_margin, 1e-6)
        realised = self.realised_margin(settlement)
        spread = max(0.0, realised / nominal - 1.0) if nominal > 0 else 0.0
        prices = settlement.prices(self.econ)
        reference = self.reference_price()
        # a settlement sitting at the price ceiling is scarce, but one famine
        # node must not be able to finance an organisation on its own: every
        # term is capped before it enters the stress index
        scarcity = float(np.max(prices / np.maximum(reference, 1e-9)) - 1.0)
        stress = (
            STRESS_SPREAD * min(spread, 1.0)
            + STRESS_STOCKOUT * min(stockout, 1.0)
            + STRESS_SCARCITY * min(max(0.0, scarcity), 1.0)
        )
        # model-level stress is a path property: a single bad tick must not
        # conjure an organisation, but a sustained breakdown must
        syndicate.stress = 0.82 * syndicate.stress + 0.18 * stress

        entry = spec.syndicate_entry * max(0.0, syndicate.stress - spec.syndicate_threshold)
        headroom = max(0.0, 1.0 - syndicate.power / max(spec.syndicate_cap, 1e-6))
        syndicate.power += entry * headroom - spec.syndicate_exit * syndicate.power
        syndicate.power = float(np.clip(syndicate.power, 0.0, spec.syndicate_cap))
        if syndicate.active:
            syndicate.active_ticks += 1

    # -- prices ----------------------------------------------------------
    def realised_margin(self, settlement) -> float:
        """``1 - farm_gate/retail`` after the vertical and predatory wedges.

        A dominant producer's withholding narrows what the pengepul can move, so
        the marketing margin *widens* — the classic signature of market power
        being exercised somewhere upstream of the household.
        """

        spec = self.spec
        syndicate = self.syndicates.get(settlement.sid)
        power = syndicate.power if syndicate else 0.0
        vertical = float(np.mean([self.node_hhi(settlement.sid, r) for r in range(len(RESOURCES))]))
        wedge = spec.vertical_wedge * max(0.0, vertical - spec.vertical_threshold) / max(
            1.0 - spec.vertical_threshold, 1e-6
        )
        return min(
            0.85,
            spec.marketing_margin * (1.0 + wedge + spec.syndicate_fee * power * 2.0),
        )

    def price_ladder(self, settlement) -> tuple[float, float, float]:
        """Farm-gate, wholesale and retail price for a settlement, value weighted."""

        prices = settlement.prices(self.econ)
        produced = np.maximum(settlement.produced, 1e-9)
        weights = produced / produced.sum()
        retail = float(np.dot(prices, weights))
        margin = self.realised_margin(settlement)
        gate = retail * (1.0 - margin)
        wholesale = retail * (1.0 - margin * 0.55)
        return gate, wholesale, retail

    def reference_price(self) -> np.ndarray:
        """A world reference price per resource: production-weighted mean."""

        total = np.zeros(len(RESOURCES))
        weight = 0.0
        for settlement in self.model.settlements.values():
            if not settlement.alive:
                continue
            q = max(float(settlement.produced.sum()), 0.0)
            if q <= 0.0:
                continue
            total += settlement.prices(self.econ) * q
            weight += q
        return total / weight if weight > 0 else np.ones(len(RESOURCES))

    # -- reporting -------------------------------------------------------
    def macro(self) -> dict[str, float]:
        """The macroscopic observables of the layered economy."""

        alive = [s for s in self.model.settlements.values() if s.alive]
        if not alive:
            return dict.fromkeys(
                ("farm_gate_price", "wholesale_price", "retail_price", "marketing_margin"), 0.0
            )

        gates, wholes, retails, weights = 0.0, 0.0, 0.0, 0.0
        margins = []
        for settlement in alive:
            gate, wholesale, retail = self.price_ladder(settlement)
            w = max(float(settlement.produced.sum()), 1e-9)
            gates += gate * w
            wholes += wholesale * w
            retails += retail * w
            weights += w
            margins.append(self.realised_margin(settlement))
        weights = max(weights, 1e-9)

        produced = float(sum(float(s.produced.sum()) for s in alive))
        take_value = 0.0
        for settlement in alive:
            syndicate = self.syndicates.get(settlement.sid)
            if syndicate is not None:
                take_value += syndicate.diverted * float(settlement.prices(self.econ).mean())
        take_value += self.fees_paid

        food_prices = np.array([float(s.prices(self.econ)[0]) for s in alive])
        stockout = float(np.mean([r.unmet for r in self.retailers.values()])) if self.retailers else 0.0
        active = sum(1 for z in self.syndicates.values() if z.active)

        return {
            "producers": float(len(self.firms)),
            "collectors": float(len(self.collectors)),
            "retailers": float(len(self.retailers)),
            "syndicates_active": float(active),
            "farm_gate_price": gates / weights,
            "wholesale_price": wholes / weights,
            "retail_price": retails / weights,
            "marketing_margin": float(np.mean(margins)) if margins else 0.0,
            "chain_loss_share": self.ema["lost"] / max(self.ema["gross"], 1e-9),
            "syndicate_take_share": self.ema["take"] / max(self.ema["gross"], 1e-9),
            "hhi_mean": float(np.mean([h for _k, (_f, h) in self.ownership.items()])) if self.ownership else 0.0,
            "vertical_share": self.vertical_share(),
            "max_dependency": self.max_dependency(),
            "stockout_rate": stockout,
            "cpi_dispersion": float(food_prices.std() / max(food_prices.mean(), 1e-9)),
            "gross_output": produced,
            "traded_value": self.delivered_value,
        }

    def snapshot(self, limit: int = 220) -> list[dict[str, object]]:
        """A compact agent list for the console renderer."""

        out: list[dict[str, object]] = []
        for firm in list(self.firms.values())[:limit]:
            settlement = self.model.settlements.get(firm.sid)
            if settlement is not None:
                out.append(
                    {
                        "k": "firm",
                        "x": settlement.x,
                        "y": settlement.y,
                        "r": firm.resource,
                        "w": round(firm.share, 3),
                        "v": 1 if firm.vertical else 0,
                    }
                )
        for collector in self.collectors.values():
            settlement = self.model.settlements.get(collector.sid)
            if settlement is not None:
                out.append(
                    {
                        "k": "collector",
                        "x": settlement.x,
                        "y": settlement.y,
                        "w": round(float(collector.dependency), 3),
                        "v": 0,
                    }
                )
        for retailer in self.retailers.values():
            settlement = self.model.settlements.get(retailer.sid)
            if settlement is not None:
                out.append({"k": "retail", "x": settlement.x, "y": settlement.y, "w": 0.4, "v": 0})
        for syndicate in self.syndicates.values():
            if not syndicate.active:
                continue
            settlement = self.model.settlements.get(syndicate.sid)
            if settlement is not None:
                out.append(
                    {
                        "k": "syndicate",
                        "x": settlement.x,
                        "y": settlement.y,
                        "w": round(float(syndicate.power), 3),
                        "v": 1,
                    }
                )
        return out
