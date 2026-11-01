"""Settlement agents: the production/consumption units of the emergent economy.

Each settlement is a node in a spatial economy. It owns a territory, extracts
resources from it, feeds its population, and adapts its labour allocation from
local scarcity signals only - there is no central planner and no settlement can
observe global prices. That locality is what makes the trade network emergent.
"""

from __future__ import annotations

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
        self.capacity = float(econ.base_capacity + 9.0 * self.endowment[0])

        rng = model.rng
        self.population = float(8.0 + rng.random() * 6.0)
        self.stocks = (self.endowment * 0.7).astype(np.float64)
        self.labor = np.full(len(RESOURCES), 1.0 / len(RESOURCES), dtype=np.float64)
        self.wealth = float(rng.random() * 20.0)
        self.tech = 0.0
        self.alive = True
        self.founded_tick = 0
        self.partners: dict[int, float] = {}

        # per-tick telemetry
        self.produced = np.zeros(len(RESOURCES))
        self.consumed = np.zeros(len(RESOURCES))
        self.traded = np.zeros(len(RESOURCES))
        self.specialization = "balanced"
        self.peak_population = self.population
        self.unmet_food = 0.0

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

    def spoil(self, econ: EconomySpec) -> None:
        """Granaries are finite: stocks decay, faster once storage overflows."""

        target = self.target_stock(econ)
        excess = np.maximum(0.0, self.stocks - target * econ.storage_buffer)
        loss = self.stocks * econ.spoilage + excess * econ.overflow_decay
        self.stocks = np.maximum(0.0, self.stocks - loss)
        self.model.ledger["spoiled"] += loss

    def deficit(self, econ: EconomySpec) -> np.ndarray:
        return np.maximum(0.0, self.target_stock(econ) - self.stocks)

    # -- production ------------------------------------------------------
    def produce(self, econ: EconomySpec, effect: ShockEffect | None = None) -> np.ndarray:
        mod = np.ones(len(RESOURCES)) if effect is None else effect.production
        labour = np.power(np.maximum(self.labor, 1e-4), econ.labor_exponent)
        roads = float(self.model.road_factor_at(self.pos)) if hasattr(self.model, "road_factor_at") else 1.0
        output = self.endowment * labour * (1.0 + self.tech) * roads * mod
        self.stocks += output
        self.model.ledger["produced"] += output
        self.produced = output
        return output

    def consume(self, econ: EconomySpec, effect: ShockEffect | None = None) -> np.ndarray:
        demand = self.consumption_rate(econ)
        if effect is not None:
            demand = demand * effect.consumption
        shortfall = np.maximum(0.0, demand - self.stocks)
        taken = np.minimum(demand, self.stocks)
        self.stocks -= taken
        self.model.ledger["consumed"] += taken
        self.consumed = taken
        # unmet food demand is the starvation channel
        self.unmet_food = float(shortfall[0])
        return shortfall

    # -- adaptation ------------------------------------------------------
    def adapt_labour(self, econ: EconomySpec) -> None:
        """Decentralised reallocation toward the most valuable local activity."""

        prices = self.prices(econ)
        signal = prices * np.maximum(self.endowment, 1e-6)
        sharp = np.power(signal / max(float(signal.max()), 1e-9), econ.labor_sharpness)
        total = float(sharp.sum())
        target = sharp / total if total > 0 else np.full(len(RESOURCES), 1.0 / len(RESOURCES))
        self.labor = (1.0 - econ.labor_inertia) * self.labor + econ.labor_inertia * target
        self.labor = np.clip(self.labor, 0.02, None)
        self.labor /= self.labor.sum()
        idx = int(np.argmax(self.labor))
        share = float(self.labor[idx])
        self.specialization = RESOURCES[idx] if share > 0.42 else "balanced"

    # -- demography ------------------------------------------------------
    def update_population(self, econ: EconomySpec, effect: ShockEffect | None = None) -> None:
        target = np.maximum(self.target_stock(econ)[0], 1e-6)
        security = float(np.clip((self.stocks[0] / target - 0.15) / 0.85, 0.0, 1.25))
        self.capacity = float(
            econ.base_capacity
            + 9.0 * self.endowment[0] * (1.0 + self.tech)
            + 14.0 * self.model.road_factor_at(self.pos)
        )
        crowding = 1.0 - self.population / max(self.capacity, 1e-6)
        birth = econ.growth_rate * security * max(crowding, -0.25)
        death = econ.death_rate + (effect.mortality if effect else 0.0)
        delta = self.population * (birth - death)
        self.population = float(max(0.0, self.population + delta))
        self.peak_population = max(self.peak_population, self.population)
        if self.population < 1.2:
            self.alive = False

    def advance_tech(self, econ: EconomySpec, traded_volume: float) -> None:
        """Knowledge diffuses with exchange volume, with diminishing returns."""

        self.tech = float(min(1.6, self.tech + econ.tech_gain * np.log1p(traded_volume)))

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
