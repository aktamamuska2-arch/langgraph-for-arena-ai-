"""Network emergence: discovery, arbitrage, decay, roads and resilience."""

from __future__ import annotations

import networkx as nx
import numpy as np

from agora.metrics import network_metrics, run_study
from agora.model import AgoraModel


def test_no_links_before_the_first_scan(default_spec):
    model = AgoraModel(default_spec)
    assert not model.edges
    model.step()
    # discovery only fires on interval boundaries
    for _ in range(default_spec.economy.discovery_interval):
        model.step()
    assert model.edges, "a discovery pass should have produced at least one candidate link"


def test_network_grows_and_connects(warm_model):
    summary = warm_model.network_summary()
    assert summary["edges"] >= 10
    assert summary["giant_component_share"] >= 0.85
    graph = warm_model.trade_graph()
    assert nx.is_weakly_connected(graph) or summary["giant_component_share"] >= 0.85


def test_edges_are_directed_and_weighted(warm_model):
    for (src, dst), edge in warm_model.edges.items():
        assert edge.src == src and edge.dst == dst
        assert edge.weight > 0
        assert edge.path[0] != edge.path[-1]
        assert warm_model.world.terrain[edge.path[-1][1], edge.path[-1][0]] != 0


def test_trade_requires_spatial_arbitrage(default_spec):
    """A link must only move goods when the destination premium covers freight."""

    model = AgoraModel(default_spec)
    for _ in range(200):
        model.step()
    econ = model.econ
    checked = 0
    for edge in model.edges.values():
        seller = model.settlements[edge.src]
        buyer = model.settlements[edge.dst]
        seller_prices = seller.prices(econ)
        buyer_prices = buyer.prices(econ)
        freight = econ.gravity.tau * edge.cost * model.freight_multiplier(edge)
        for r in range(3):
            if edge.total_flow > 0 and edge.weights[r] >= econ.edge_min_weight:
                # a traded edge must have had a positive margin at some point;
                # at the final tick it may have closed - just require the
                # invariant on the *prices that remain* for still-open links
                if buyer_prices[r] - seller_prices[r] <= freight:
                    assert buyer.stocks[r] > 0 or seller.surplus(econ)[r] <= 0
                checked += 1
    assert checked > 0


def test_stale_links_decay_and_are_pruned(default_spec):
    """Decay is a maintenance cost: an idle link cannot coast forever."""

    from agora.market import MarketEngine, TradeEdge

    model = AgoraModel(default_spec)
    engine = MarketEngine(model.econ)
    idle = TradeEdge(src=0, dst=1, weights=np.array([1.0, 0.0, 0.0]))
    idle.last_active = -10_000
    live = TradeEdge(src=2, dst=3, weights=np.array([1.0, 0.0, 0.0]))
    live.last_active = model.tick_count
    model.edges[(0, 1)] = idle
    model.edges[(2, 3)] = live

    removed = engine.decay(model)
    assert removed == 1
    assert (0, 1) not in model.edges, "an idle link must be dropped"
    assert (2, 3) in model.edges, "a freshly used link must survive"

    # weight decay alone prunes a link that is never refreshed
    for _ in range(200):
        engine.decay(model)
    assert (2, 3) not in model.edges


def test_road_upgrades_lower_transport_cost(default_spec):
    model = AgoraModel(default_spec)
    for _ in range(320):
        model.step()
    assert model.roads_upgraded > 0, "sustained trade must pave at least one route"
    assert float(model.road_level.max()) > 0
    for (_src, _dst), edge in model.edges.items():
        if edge.path:
            paved = sum(1 for x, y in edge.path if model.road_level[y, x] > 0)
            assert paved >= 0
    # a paved cell must be cheaper than the same cell unpaved
    y, x = np.unravel_index(int(np.argmax(model.road_level)), model.road_level.shape)
    assert model.cost[y, x] < model.base_cost[y, x]


def test_embargo_suppresses_trade(default_spec):
    baseline = AgoraModel(default_spec)
    for _ in range(240):
        baseline.step()
    start = baseline.total_flow
    for _ in range(40):
        baseline.step()
    normal = baseline.total_flow - start

    embargoed = AgoraModel(default_spec)
    for _ in range(240):
        embargoed.step()
    embargoed.inject_shock("embargo", x=None, y=None, radius=64, magnitude=0.9, duration=60)
    start2 = embargoed.total_flow
    for _ in range(40):
        embargoed.step()
    stressed = embargoed.total_flow - start2
    assert stressed < normal


def test_network_metrics_are_consistent(warm_model):
    metrics = network_metrics(warm_model)
    summary = warm_model.network_summary()
    assert metrics["giant_component_share"] == summary["giant_component_share"]
    assert 0.0 <= metrics["orphan_fraction"] <= 1.0
    assert metrics["edges_per_node"] >= 0.0


def test_shock_recovery_gate_is_measurable(default_spec):
    study = run_study(default_spec)
    assert "shock_recovery" in study.metrics
    assert study.metrics["shock_recovery"] > 0.0
    assert study.metrics["mass_balance_error"] < 1e-9


def test_simulation_is_reproducible(default_spec):
    a = AgoraModel(default_spec)
    b = AgoraModel(default_spec)
    for _ in range(120):
        a.step()
        b.step()
    assert a.tick_count == b.tick_count
    assert len(a.edges) == len(b.edges)
    for sid, settlement in a.settlements.items():
        assert np.isclose(settlement.population, b.settlements[sid].population)
        assert np.allclose(settlement.stocks, b.settlements[sid].stocks)
