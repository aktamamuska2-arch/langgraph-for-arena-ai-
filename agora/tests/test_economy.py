"""Economy invariants: pricing, labour, conservation, demography."""

from __future__ import annotations

import numpy as np
import pytest

from agora.market import MarketEngine
from agora.metrics import mass_balance
from agora.model import AgoraModel
from agora.schemas import RESOURCES, EconomySpec


def test_initial_state_is_sane(default_spec):
    model = AgoraModel(default_spec)
    assert len(model.settlements) == default_spec.world.settlement_count
    for settlement in model.settlements.values():
        assert settlement.population > 0
        assert (settlement.stocks >= 0).all()
        assert settlement.endowment.shape == (3,)
        assert np.isclose(settlement.labor.sum(), 1.0)


def test_prices_respect_the_clamp(warm_model):
    econ = warm_model.econ
    for settlement in warm_model.settlements.values():
        prices = settlement.prices(econ)
        base = np.array([econ.base_price[r] for r in RESOURCES])
        assert (prices >= base * econ.price_floor - 1e-9).all()
        assert (prices <= base * econ.price_ceiling + 1e-9).all()


def test_prices_fall_as_stocks_rise(default_spec):
    model = AgoraModel(default_spec)
    settlement = next(iter(model.settlements.values()))
    econ = model.econ
    poor = settlement.prices(econ).copy()
    settlement.stocks = settlement.stocks * 4 + 50
    rich = settlement.prices(econ)
    assert (rich <= poor + 1e-9).all()


def test_labour_reallocates_toward_comparative_advantage(warm_model):
    """Specialisation must follow endowment, not a global plan."""

    for settlement in warm_model.settlements.values():
        assert np.isclose(settlement.labor.sum(), 1.0, atol=1e-9)
        assert (settlement.labor > 0).all()
    specialists = sum(1 for s in warm_model.settlements.values() if s.specialization != "balanced")
    assert specialists >= 8, "most settlements should have specialised by tick 240"


def test_mass_balance_holds_exactly(warm_model):
    error = mass_balance(warm_model)
    assert error < 1e-9, f"ledger identity violated: relative error {error}"


def test_mass_balance_holds_with_shocks(default_spec):
    model = AgoraModel(default_spec)
    for _ in range(120):
        model.step()
    model.inject_shock("drought", x=48, y=32, radius=20, magnitude=0.9, duration=40)
    for _ in range(80):
        model.step()
    assert mass_balance(model) < 1e-9


def test_population_converges_and_stays_finite(default_spec):
    model = AgoraModel(default_spec)
    for _ in range(400):
        model.step()
    populations = np.array([s.population for s in model.settlements.values()])
    assert np.isfinite(populations).all()
    assert (populations >= 0).all()
    history = np.array([row["population"] for row in model.history])
    tail = history[-len(history) // 4 :]
    growth = (tail[-1] / tail[0]) ** (1.0 / max(len(tail), 1)) - 1.0
    assert abs(growth) < 0.006, f"population still growing at {growth:.4f}/tick"


def test_spoilage_keeps_stocks_bounded(default_spec):
    """Granaries reach a steady level instead of hoarding without limit."""

    model = AgoraModel(default_spec)
    for _ in range(300):
        model.step()
    mid = np.array([s.stocks.copy() for s in model.settlements.values()])
    for _ in range(120):
        model.step()
    end = np.array([s.stocks.copy() for s in model.settlements.values()])
    assert np.isfinite(end).all()
    drift = np.abs(end - mid).mean() / max(float(np.abs(mid).mean()), 1e-9)
    assert drift < 0.35, f"stocks still drifting by {drift:.0%} over 120 ticks"
    for settlement, row in zip(model.settlements.values(), end, strict=False):
        cap = settlement.target_stock(model.econ) * model.econ.storage_buffer
        assert (row <= cap * 8.0 + 5.0).all(), "storage ran away"


def test_gravity_kernel_is_monotone():
    engine = MarketEngine(EconomySpec())
    near = engine.attractiveness(5.0, 5.0, 5.0, 6.0, 0.0)
    far = engine.attractiveness(5.0, 5.0, 25.0, 40.0, 0.0)
    assert near > far
    heavy = engine.attractiveness(10.0, 10.0, 10.0, 12.0, 0.0)
    light = engine.attractiveness(1.0, 1.0, 10.0, 12.0, 0.0)
    assert heavy > light
    trusted = engine.attractiveness(5.0, 5.0, 10.0, 12.0, 2.0)
    straight = engine.attractiveness(5.0, 5.0, 10.0, 12.0, 0.0)
    assert trusted > straight


def test_shock_reduces_output(default_spec):
    model = AgoraModel(default_spec)
    for _ in range(80):
        model.step()
    target = next(iter(model.settlements.values()))
    baseline = target.produce(model.econ, None)
    effect = model.shock_effect_at(target.x, target.y)
    assert np.allclose(effect.production, 1.0)
    model.inject_shock("drought", x=target.x, y=target.y, radius=12, magnitude=0.9, duration=20)
    stressed = model.shock_effect_at(target.x, target.y)
    assert stressed.production[0] < 0.5
    stressed_output = target.produce(model.econ, stressed)
    assert stressed_output[0] < baseline[0]


@pytest.mark.parametrize(
    "kind", ["drought", "blight", "ore_discovery", "demand_boom", "plague", "embargo", "road_collapse"]
)
def test_every_shock_kind_is_accepted(default_spec, kind):
    model = AgoraModel(default_spec)
    for _ in range(30):
        model.step()
    record = model.inject_shock(kind, x=50, y=30, radius=12, magnitude=0.7, duration=25)
    assert record["kind"] == kind
    for _ in range(25):
        model.step()
    assert mass_balance(model) < 1e-6
