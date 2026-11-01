"""The layered economy: producers, pengepul, penjual and the syndicates.

These tests are about *structure and conservation*, not about a particular
number: a chain that creates matter, or a sindikat that exists without a broken
chain, is a bug regardless of how pretty the aggregate looks.
"""

from __future__ import annotations

import numpy as np
import pytest

from agora.metrics import mass_balance
from agora.model import AgoraModel
from agora.schemas import RESOURCES, ExperimentSpec


@pytest.fixture()
def warm(default_spec):
    model = AgoraModel(default_spec)
    for _ in range(180):
        model.step()
    return model


# ---------------------------------------------------------------------------
# Census and structure
# ---------------------------------------------------------------------------


def test_the_world_has_a_layered_agent_population(warm):
    supply = warm.supply
    assert len(supply.firms) >= 14 * warm.spec.supply.producers_per_settlement // 2
    assert len(supply.collectors) == len(warm.settlements) * warm.spec.supply.collectors_per_settlement
    assert len(supply.retailers) == len(warm.settlements) * warm.spec.supply.retailers_per_settlement
    assert len(supply.syndicates) == len(warm.settlements)

    macro = supply.macro()
    assert macro["producers"] > 100
    assert macro["collectors"] > macro["syndicates_active"]


def test_every_resource_node_is_fully_owned(warm):
    """Shares inside a node are a partition, not a wish list."""

    for (sid, resource), node in warm.supply.by_node.items():
        total = sum(warm.supply.firms[fid].share for fid in node)
        assert total == pytest.approx(1.0, abs=1e-9), (sid, resource)
        assert len(node) >= 1


def test_ownership_is_concentrated_and_flags_a_vertical_producer(warm):
    supply = warm.supply
    hhis = [hhi for _key, (_firm, hhi) in supply.ownership.items()]
    assert np.mean(hhis) > supply.spec.vertical_threshold
    vertical = [f for f in supply.firms.values() if f.vertical]
    assert vertical, "a power-law draw must produce at least one dominant producer"
    for firm in vertical:
        assert supply.node_hhi(firm.sid, firm.resource) >= supply.spec.vertical_threshold
        assert firm is supply.dominant_firm(firm.sid, firm.resource)


def test_dependency_is_measured_per_collector(warm):
    for collector in warm.supply.collectors.values():
        if not collector.firms:
            continue
        total = sum(warm.supply.firms[f].share for f in collector.firms)
        biggest = max(warm.supply.firms[f].share for f in collector.firms)
        assert collector.dependency == pytest.approx(biggest / total, abs=1e-9)
        assert collector.dependency <= 1.0


# ---------------------------------------------------------------------------
# The price ladder
# ---------------------------------------------------------------------------


def test_price_ladder_descends_from_household_to_farm_gate(warm):
    for settlement in warm.settlements.values():
        gate, wholesale, retail = warm.supply.price_ladder(settlement)
        assert gate < wholesale < retail
        assert retail > 0


def test_marketing_margin_matches_the_specification_in_a_calm_world(warm):
    macro = warm.supply.macro()
    nominal = warm.spec.supply.marketing_margin
    # the vertical wedge widens it, but it must stay in a sane band
    assert nominal <= macro["marketing_margin"] <= nominal + 0.10
    assert macro["farm_gate_price"] < macro["retail_price"]


def test_vertical_power_widens_the_margin(default_spec):
    """Market power exercised upstream shows up as a wider marketing margin."""

    baseline = AgoraModel(default_spec)
    for _ in range(120):
        baseline.step()
    calm = baseline.supply.macro()["marketing_margin"]

    weak = ExperimentSpec(
        world=default_spec.world,
        economy=default_spec.economy,
        supply=default_spec.supply.model_copy(
            update={"vertical_wedge": 0.0, "vertical_threshold": 0.99}
        ),
    )
    model = AgoraModel(weak)
    for _ in range(120):
        model.step()
    assert model.supply.macro()["marketing_margin"] < calm


# ---------------------------------------------------------------------------
# Conservation
# ---------------------------------------------------------------------------


def test_the_chain_conserves_mass(warm):
    assert mass_balance(warm) <= 1e-12


def test_priming_the_chain_does_not_create_stock(default_spec):
    model = AgoraModel(default_spec)
    supply = model.supply
    on_shelves = sum(float(c.inventory.sum()) for c in supply.collectors.values())
    on_shelves += sum(float(r.inventory.sum()) for r in supply.retailers.values())
    assert on_shelves > 0.0
    total = sum(float(s.total_stocks().sum()) for s in model.settlements.values()) + on_shelves
    assert total == pytest.approx(model.initial_stocks, rel=1e-12)


def test_chain_losses_stay_bounded_in_a_calm_world(warm):
    macro = warm.supply.macro()
    assert 0.0 < macro["chain_loss_share"] < 0.10


def test_losses_are_the_sum_of_explicit_buckets(warm):
    supply = warm.supply
    booked = float(warm.ledger["diverted"].sum())
    buckets = sum(supply.chain_loss.values())
    assert booked == pytest.approx(buckets, rel=1e-9)


# ---------------------------------------------------------------------------
# Predation
# ---------------------------------------------------------------------------


def test_syndicates_are_dormant_in_a_calm_world(warm):
    active = sum(1 for z in warm.supply.syndicates.values() if z.active)
    assert active <= 1
    assert warm.supply.macro()["syndicate_take_share"] < 0.02


def test_a_broken_chain_breeds_syndicates(warm):
    """Entry is endogenous: no script, only a stress index."""

    before = sum(1 for z in warm.supply.syndicates.values() if z.active)
    exporter = max(
        (s for s in warm.settlements.values() if s.alive), key=lambda s: float(s.endowment[0])
    )
    warm.inject_shock(kind="drought", x=exporter.x, y=exporter.y, radius=26.0, magnitude=0.95, duration=90)
    warm.inject_shock(kind="embargo", x=exporter.x, y=exporter.y, radius=30.0, magnitude=0.9, duration=70)
    peak = before
    for _ in range(90):
        warm.step()
        peak = max(peak, sum(1 for z in warm.supply.syndicates.values() if z.active))
    assert peak >= before + 3, "a severe supply-chain shock must multiply rackets"
    assert warm.supply.macro()["syndicate_take_share"] > 0.01

    # and they must retreat once the chain recovers, or they are not predation
    for _ in range(260):
        warm.step()
    after = sum(1 for z in warm.supply.syndicates.values() if z.active)
    assert after <= peak


def test_syndicate_power_is_bounded(default_spec):
    hot = ExperimentSpec(
        world=default_spec.world,
        economy=default_spec.economy,
        supply=default_spec.supply.model_copy(
            update={"syndicate_entry": 0.4, "syndicate_diversion": 0.6, "syndicate_fee": 0.5}
        ),
    )
    model = AgoraModel(hot)
    for _ in range(160):
        model.step()
        for syndicate in model.supply.syndicates.values():
            assert 0.0 <= syndicate.power <= hot.supply.syndicate_cap + 1e-12


def test_predation_raises_the_cost_of_trade(warm):
    edge = next(iter(warm.edges.values()), None)
    if edge is None:
        pytest.skip("no corridor to tax")
    clean = warm.freight_multiplier(edge)
    warm.supply.syndicates[edge.src].power = 0.5
    warm.supply.syndicates[edge.dst].power = 0.5
    assert warm.freight_multiplier(edge) > clean


def test_diversion_is_booked_as_loss_not_as_a_transfer(warm):
    syndicate = warm.supply.syndicates[0]
    syndicate.power = 0.5
    before = float(warm.ledger["diverted"].sum())
    for _ in range(10):
        warm.step()
    assert float(warm.ledger["diverted"].sum()) > before
    assert syndicate.diverted > 0.0
    assert mass_balance(warm) <= 1e-12


# ---------------------------------------------------------------------------
# The macro view
# ---------------------------------------------------------------------------


def test_macro_snapshot_is_complete_and_finite(warm):
    macro = warm.supply.macro()
    for key in (
        "producers",
        "collectors",
        "retailers",
        "syndicates_active",
        "farm_gate_price",
        "retail_price",
        "marketing_margin",
        "chain_loss_share",
        "syndicate_take_share",
        "hhi_mean",
        "vertical_share",
        "max_dependency",
        "stockout_rate",
        "cpi_dispersion",
    ):
        assert key in macro
        assert np.isfinite(macro[key]), key
    assert 0.0 <= macro["vertical_share"] <= 1.0
    assert 0.0 <= macro["stockout_rate"] <= 1.0


def test_chain_efficiency_reaches_the_carrying_capacity(default_spec):
    """A lossier chain must support fewer people, or the demography is lying."""

    lean = ExperimentSpec(
        world=default_spec.world,
        economy=default_spec.economy,
        supply=default_spec.supply.model_copy(
            update={"loss_gate": 0.02, "loss_handling": 0.03, "loss_retail": 0.02}
        ),
    )
    fat = AgoraModel(lean)
    nominal = AgoraModel(default_spec)
    settlement_a = fat.settlements[0]
    settlement_b = nominal.settlements[0]
    assert settlement_a._carrying_capacity(fat.econ) < settlement_b._carrying_capacity(nominal.econ)


def test_frame_payload_carries_the_chain(warm):
    frame = warm.frame()
    assert "macro" in frame and "agents" in frame
    kinds = {agent["k"] for agent in frame["agents"]}
    assert {"firm", "retail"} <= kinds
    assert frame["macro"]["producers"] == len(warm.supply.firms)
    assert all(0 <= agent["r"] < len(RESOURCES) for agent in frame["agents"] if agent["k"] == "firm")
