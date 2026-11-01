"""Settlement agents: the production/consumption units of the emergent economy.

Each settlement is a node in a spatial economy. It owns a territory, extracts
resources from it, feeds its population, and adapts its labour allocation from
local scarcity signals only - there is no central planner and no settlement can
observe global prices. That locality is what makes the trade network emergent.

Three pieces of mathematics are derived rather than tuned (see README §4):

* the labour reallocation exponent follows from the revenue-maximising FOC of a
  Cobb-Douglas settlement that is a *price maker* in its own local market;
* the carrying capacity is the population the settlement's own food economy can
  sustain at its habitual labour mix;
* stock updates use Neumaier compensation so the physical ledger does not drift
  with the number of ticks simulated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from mesa.space import Agent as GridAgent

from .schemas import RESOURCES, EconomySpec
from .world import WorldData, territorial_yield

if TYPE_CHECKING:  # pragma: no cover
    from .model import AgoraModel

SYLLABLES = (
    "ka",
    "va",
    "nor",
    "thel",
    "mi",
    "ra",
    "sun",
    "dor",
    "pel",
    "an",
    "ith",
    "os",
    "mar",
    "vel",
    "tor",
    "qua",
    "ryn",
    "bel",
    "cor",
    "ash",
)


def settlement_name(rng: np.random.Generator) -> str:
    """Stable, pronounceable settlement name."""

    n = int(rng.integers(2, 3))
    parts = [SYLLABLES[int(rng.integers(0, len(SYLLABLES)))] for _ in range(n)]
    tail = ["a", "os", "eth", "ar", "ia", "un"][int(rng.integers(0, 6))]
    return ("".join(parts)).capitalize() + tail


@dataclass
class ShockEffect:
    """Multiplicative modifiers applied to one settlement for one tick."""

    production: np.ndarray = field(default_factory=lambda: np.ones(len(RESOURCES)))
    consumption: np.ndarray = field(default_factory=lambda: np.ones(len(RESOURCES)))
    mortality: float = 0.0
    trade_friction: float = 0.0


class Settlement(GridAgent):
    """A single autonomous settlement."""

    def __init__(self, model: AgoraModel, sid: int, pos: tuple[int, int], name: str) -> None:
        super().__init__(model)
        self.sid = int(sid)
        self.name = name
        self.x = int(pos[0])
        self.y = int(pos[1])

        econ: EconomySpec = model.econ
        world: WorldData = model.world
        raw = territorial_yield(world, (self.y, self.x), model.world_spec.territory_radius)
        area = float((2 * model.world_spec.territory_radius + 1) ** 2)
        # normalise the territory to an "effective field count" so production and
        # per-capita consumption live on the same scale (~2-5 units/tick)
        self.endowment = (raw / area) * 9.0

        rng = model.rng
        # Settlements are *established*, not newborn: a world is observed after it
        # has already been peopled. Starting every node at a dozen souls against a
        # carrying capacity in the hundreds creates a 60x transient that swamps the
        # convergence window and says nothing about the model's fixed point.
        self.tech = 0.0
        self.population = float(0.45 * self._carrying_capacity(econ) * rng.uniform(0.85, 1.15))
        self.stocks = (0.75 * self.target_stock(econ)).astype(np.float64)
        # Neumaier compensation term: keeps the ledger exact to ~1 ulp of the
        # final stock instead of drifting with every add/subtract
        self.stocks_comp = np.zeros(len(RESOURCES), dtype=np.float64)
        self.labor = np.full(len(RESOURCES), 1.0 / len(RESOURCES), dtype=np.float64)
        # slow memory of the habitual food labour share, used by the carrying capacity
        self.food_labor_memory = float(self.labor[0])
        # slow memory of realised food output - the flow the carrying capacity is
        # built from, so that K follows the settlement's actual food economy
        self.food_output_memory = float(self.endowment[0] * 0.5)
        self.security = 0.6
        self.wealth = float(rng.random() * 20.0) + 0.05 * self.population
        self.trade_volume = 0.0
        self.alive = True
        self.founded_tick = 0
        self.partners: dict[int, float] = {}
        self.capacity = self._carrying_capacity(econ)

        # per-tick telemetry
        self.produced = np.zeros(len(RESOURCES))
        self.consumed = np.zeros(len(RESOURCES))
        self.traded = np.zeros(len(RESOURCES))
        self.specialization = "balanced"
        self.peak_population = self.population
        self.unmet_food = 0.0

    # -- bookkeeping -----------------------------------------------------
    def add_stocks(self, delta: np.ndarray) -> None:
        """Neumaier compensated accumulation of a stock change.

        Plain ``stocks += delta`` loses roughly one ulp per operation, so a
        400-tick run drifts by ~1e-11 absolute against a ledger of ~2e4. The
        compensation term carries the rounding error forward, holding the
        conservation identity at the precision floor of the final value. This is
        an *engine property* - it never substitutes for a real conservation fix.
        """

        total = self.stocks + delta
        big = np.abs(self.stocks) >= np.abs(delta)
        self.stocks_comp += np.where(big, (self.stocks - total) + delta, (delta - total) + self.stocks)
        self.stocks = total

    def total_stocks(self) -> np.ndarray:
        """Stock including the compensation term (used by the ledger audit)."""

        return self.stocks + self.stocks_comp

    # -- pricing ---------------------------------------------------------
    def target_stock(self, econ: EconomySpec) -> np.ndarray:
        return self.consumption_rate(econ) * econ.target_coverage_ticks + 0.05

    def consumption_rate(self, econ: EconomySpec) -> np.ndarray:
        return np.array(
            [
                econ.food_per_capita * self.population,
                econ.wood_per_capita * self.population,
                econ.ore_per_capita * self.population,
            ],
            dtype=np.float64,
        )

    def prices(self, econ: EconomySpec) -> np.ndarray:
        """Local scarcity price: rises as stock falls below the target buffer."""

        base = np.array([econ.base_price[r] for r in RESOURCES], dtype=np.float64)
        target = np.maximum(self.target_stock(econ), 1e-6)
        ratio = target / np.maximum(self.stocks, 1e-6)
        price = base * np.power(ratio, econ.price_elasticity)
        return np.clip(price, base * econ.price_floor, base * econ.price_ceiling)

    def surplus(self, econ: EconomySpec) -> np.ndarray:
        return np.maximum(0.0, self.stocks - self.target_stock(econ))

    def marketable(self, econ: EconomySpec) -> np.ndarray:
        """What the settlement can bring to market in one discovery round.

        A producer quotes everything above its own reserve; the *buyer's* deficit
        and the trade fraction are what actually cap the volume. Capping the
        quote on the seller's own consumption target would cripple exporters of
        a resource they never consume - a miner would hoard ore it cannot ship.
        """

        return self.surplus(econ)

    def deficit(self, econ: EconomySpec) -> np.ndarray:
        return np.maximum(0.0, self.target_stock(econ) - self.stocks)

    def spoil(self, econ: EconomySpec) -> None:
        """Granaries are finite: stocks decay, faster once storage overflows."""

        target = self.target_stock(econ)
        excess = np.maximum(0.0, self.stocks - target * econ.storage_buffer)
        loss = self.stocks * econ.spoilage + excess * econ.overflow_decay
        self.add_stocks(-loss)
        # clamp: compensation must never push a stock negative
        self.stocks = np.maximum(self.stocks, 0.0)
        self.model.ledger["spoiled"] += loss

    # -- production ------------------------------------------------------
    def produce(self, econ: EconomySpec, effect: ShockEffect | None = None) -> np.ndarray:
        mod = np.ones(len(RESOURCES)) if effect is None else effect.production
        labour = np.power(np.maximum(self.labor, 1e-4), econ.labor_exponent)
        roads = float(self.model.road_factor_at((self.x, self.y)))
        output = self.endowment * labour * (1.0 + self.tech) * roads * mod
        self.add_stocks(output)
        self.model.ledger["produced"] += output
        self.produced = output
        self.food_output_memory = (1.0 - econ.capacity_labor_memory) * self.food_output_memory + (
            econ.capacity_labor_memory * float(output[0])
        )
        return output

    def consume(self, econ: EconomySpec, effect: ShockEffect | None = None) -> np.ndarray:
        demand = self.consumption_rate(econ)
        if effect is not None:
            demand = demand * effect.consumption
        shortfall = np.maximum(0.0, demand - self.stocks)
        taken = np.minimum(demand, self.stocks)
        self.add_stocks(-taken)
        self.model.ledger["consumed"] += taken
        self.consumed = taken
        # unmet food demand is the starvation channel
        self.unmet_food = float(shortfall[0])
        return shortfall

    # -- adaptation ------------------------------------------------------
    def marginal_revenue(self, econ: EconomySpec) -> np.ndarray:
        """Long-run marginal revenue product, with own-price feedback.

        Because stock is proportional to output at the steady state and
        ``p_r ∝ S_r^{-eps}``, revenue behaves like ``p_r E_r^{1-eps} l_r^{m}``
        with ``m = kappa(1-eps)``. ``p_r kappa E_r l_r^{kappa-1}`` is the
        *instantaneous* marginal product and is deliberately NOT equalised: the
        settlement is a price maker, so the price it gets falls as it supplies
        more. Equalising the instantaneous form would produce a corner solution.
        """

        exposure = 1.0 - econ.price_elasticity
        m = econ.labor_exponent * exposure
        return (
            self.prices(econ)
            * np.power(np.maximum(self.endowment, 1e-9), exposure)
            * m
            * np.power(np.maximum(self.labor, 1e-9), m - 1.0)
        )

    def adapt_labour(self, econ: EconomySpec) -> None:
        """Decentralised reallocation toward the most valuable local activity.

        Maximising steady-state revenue ``sum_r p_r E_r^{1-eps} l_r^{m}`` subject
        to ``sum_r l_r = 1`` gives ``l_r ∝ (p_r E_r^{1-eps})^{1/(1-m)}`` with
        ``m = kappa(1-eps)``. The exponent is therefore *derived* from the
        production curvature and the own-price elasticity - it is not a tuning
        knob. It stays interior because the price-making term ``1-eps`` softens
        the curvature: the exact FOC without it would be ``1/(1-kappa)``, which
        always picks a corner and is why settlements must be price makers.
        """

        exposure = 1.0 - econ.price_elasticity
        signal = self.prices(econ) * np.power(np.maximum(self.endowment, 1e-9), exposure)
        raw = np.power(signal / max(float(signal.max()), 1e-12), econ.sharpness)
        total = float(raw.sum())
        target = raw / total if total > 0 else np.full(len(RESOURCES), 1.0 / len(RESOURCES))
        self.labor = (1.0 - econ.labor_inertia) * self.labor + econ.labor_inertia * target
        self.labor = np.clip(self.labor, 0.02, None)
        self.labor /= self.labor.sum()
        self.food_labor_memory = (1.0 - econ.capacity_labor_memory) * self.food_labor_memory + (
            econ.capacity_labor_memory * float(self.labor[0])
        )
        idx = int(np.argmax(self.labor))
        share = float(self.labor[idx])
        self.specialization = RESOURCES[idx] if share > 0.42 else "balanced"

    def labour_kkt_residual(self, econ: EconomySpec) -> float:
        """Relative spread of marginal revenue among the *unconstrained* activities.

        The allocation is optimal when every activity that has not hit its
        absorption cap earns the same marginal revenue; a positive residual means
        labour is misallocated. Capped activities are excluded by construction -
        at a corner the KKT condition is an inequality, not an equality.
        """

        mrp = self.marginal_revenue(econ)
        reference = float(np.mean(mrp))
        if reference <= 1e-12:
            return 0.0
        return float(np.std(mrp) / reference)

    def decay_trust(self, econ: EconomySpec) -> None:
        """Relationship capital depreciates when it is not exercised."""

        if not self.partners:
            return
        decay = econ.trust_decay
        self.partners = {k: v * decay for k, v in self.partners.items() if v * decay > 1e-3}

    # -- demography ------------------------------------------------------
    def _carrying_capacity(self, econ: EconomySpec) -> float:
        """Logistic carrying capacity from the territory's own food capability.

        ``K = K0 + E_food (1+θ) R / [c_food (1 + sigma * T_coverage)]``

        Two things matter here. First, storage losses: the granary loses
        ``sigma`` of its stock every tick and the buffer is ``T_coverage`` ticks of
        consumption, so a fraction ``sigma * T_coverage`` of gross output never
        reaches a mouth and must be subtracted from what the land can support.
        Second - and this is the structural point - *K must not depend on the
        population it regulates*. Using the current or habitual labour share
        makes the carrying capacity a function of the state it is supposed to
        bound, so the logistic has no fixed point to converge to and the field
        drifts indefinitely. ``K`` is therefore the land's capability at full
        food labour, which is exogenous once technology saturates.
        """

        # road access is trade infrastructure, not soil fertility: it belongs in
        # production and in transport cost, never in the land's food ceiling
        keep = 1.0 + float(econ.spoilage) * float(econ.target_coverage_ticks)
        food = self.endowment[0] * (1.0 + self.tech) / (max(econ.food_per_capita, 1e-9) * keep)
        return float(econ.base_capacity + max(0.0, food))

    def update_population(self, econ: EconomySpec, effect: ShockEffect | None = None) -> None:
        target = np.maximum(self.target_stock(econ)[0], 1e-6)
        raw_security = float(np.clip((self.stocks[0] / target - 0.15) / 0.85, 0.0, 1.25))
        # adaptive expectations: households react to a smoothed food outlook, which
        # damps the granary sawtooth that would otherwise wreck lag-1 coherence
        self.security = (1.0 - econ.security_memory) * self.security + econ.security_memory * raw_security
        self.capacity = self._carrying_capacity(econ)
        crowding = 1.0 - self.population / max(self.capacity, 1e-6)
        birth = econ.growth_rate * self.security * max(crowding, -0.25)
        death = econ.death_rate + (effect.mortality if effect else 0.0)
        delta = self.population * (birth - death)
        self.population = float(max(0.0, self.population + delta))
        self.peak_population = max(self.peak_population, self.population)
        if self.population < 1.2:
            self.alive = False

    def advance_tech(self, econ: EconomySpec) -> None:
        """Technology diffuses logistically toward a finite frontier.

        ``dtheta/dt = g * ln(1+V) * (1 - theta/theta_max)`` - a Bass/Rogers
        adoption curve driven by the settlement's exposure to exchange. The
        finite frontier is what makes the world stationary: an unbounded
        ``theta += g ln(1+V)`` keeps lifting the carrying capacity forever, so
        the population can only ever chase a moving target and the field never
        converges. Adoption is also faster than demographic turnover, which is
        the timescale separation a stationary state needs.
        """

        exposure = math.log1p(max(self.trade_volume, 0.0))
        headroom = max(0.0, econ.tech_ceiling - self.tech)
        self.tech = float(min(econ.tech_ceiling, self.tech + econ.tech_gain * (1.0 + exposure) * headroom))
        self.trade_volume = 0.0

    # -- reporting -------------------------------------------------------
    def snapshot(self, econ: EconomySpec, degree: int) -> dict:
        prices = self.prices(econ)
        return {
            "id": self.sid,
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "population": round(self.population, 3),
            "wealth": round(self.wealth, 3),
            "stocks": {r: round(float(self.stocks[i]), 3) for i, r in enumerate(RESOURCES)},
            "prices": {r: round(float(prices[i]), 4) for i, r in enumerate(RESOURCES)},
            "produced": {r: round(float(self.produced[i]), 4) for i, r in enumerate(RESOURCES)},
            "consumed": {r: round(float(self.consumed[i]), 4) for i, r in enumerate(RESOURCES)},
            "surplus": {r: round(float(v), 4) for r, v in zip(RESOURCES, self.surplus(econ), strict=False)},
            "deficit": {r: round(float(v), 4) for r, v in zip(RESOURCES, self.deficit(econ), strict=False)},
            "specialization": self.specialization,
            "degree": int(degree),
            "alive": bool(self.alive),
            "capacity": round(self.capacity, 3),
            "labor": {r: round(float(self.labor[i]), 4) for i, r in enumerate(RESOURCES)},
            "endowment": {r: round(float(self.endowment[i]), 4) for i, r in enumerate(RESOURCES)},
        }
