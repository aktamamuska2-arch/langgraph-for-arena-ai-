"""Unit tests for the analytic layer: queueing math, topology and weather."""

from __future__ import annotations

import math

import numpy as np
import pytest

from maritime_sentinel.codegen import (
    derive_deployment,
    erlang_c,
    expected_wait_hours,
    required_servers,
    sha256_bytes,
    static_scan,
)
from maritime_sentinel.schemas import ArchitecturalBlueprint
from maritime_sentinel.topology import (
    articulation_points,
    build_graph,
    connectivity_report,
    criticality,
    lane_pairs,
    port_loads,
)
from maritime_sentinel.weather import (
    effective_transition,
    stationary_distribution,
    zone_statistics,
)

# -- Erlang C ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("servers", "load", "expected"),
    [
        (1, 0.5, 0.5),  # M/M/1: C = rho
        (2, 1.0, 1.0 / 3.0),  # textbook value
        (3, 2.0, 4.0 / 9.0),  # textbook value
    ],
)
def test_erlang_c_matches_closed_forms(servers: float, load: float, expected: float) -> None:
    assert erlang_c(int(servers), load) == pytest.approx(expected, rel=1e-6)


def test_erlang_c_is_bounded_and_monotone() -> None:
    values = [erlang_c(c, 4.0) for c in range(5, 20)]
    assert all(0.0 <= v <= 1.0 for v in values)
    assert all(a >= b for a, b in zip(values, values[1:], strict=False))
    assert erlang_c(4, 4.0) == 1.0  # overloaded


def test_required_servers_tracks_the_wait_target() -> None:
    loose = required_servers(8000.0, 1600.0, 24.0)
    tight = required_servers(8000.0, 1600.0, 1.0)
    assert tight >= loose
    assert expected_wait_hours(tight, 8000.0, 1600.0) <= 1.0 + 1e-9
    assert math.isinf(expected_wait_hours(4, 8000.0, 1600.0))


def test_expected_wait_is_infinite_without_capacity() -> None:
    assert math.isinf(expected_wait_hours(0, 100.0, 100.0))


# -- deployment derivation --------------------------------------------------


def test_deployment_strategies_differ(scenario, project_root) -> None:  # noqa: ANN001
    graph = build_graph(scenario.network)
    loads = port_loads(scenario)
    assert loads

    base = ArchitecturalBlueprint(
        blueprint_id="BP-TEST",
        spec_id=scenario.spec_id,
        objective="test",
        sizing_strategy="proportional",
        resilience_margin=1.1,
        network_signature={"nodes": graph.number_of_nodes()},
        equations=[],
        sizing=[],
        modules=[],
        test_criteria=[],
        risk_notes=[],
        total_berths=0,
    )
    proportional = derive_deployment(base, scenario, seed=1, capacity_scale=1.0)
    assert proportional.berths == {p.id: p.berths for p in scenario.network.ports}
    assert proportional.strategy == "proportional"


# -- static analysis of generated code --------------------------------------


def test_static_scan_flags_forbidden_tokens() -> None:
    assert static_scan("import subprocess\n") == ["subprocess"]
    assert static_scan("value = eval('1+1')\n") == ["eval("]
    assert static_scan("x = 1\n") == []


def test_static_scan_flags_syntax_errors() -> None:
    violations = static_scan("def broken(:\n")
    assert any(v.startswith("syntax_error") for v in violations)


def test_sha256_is_stable() -> None:
    assert sha256_bytes(b"abc") == sha256_bytes(b"abc")
    assert len(sha256_bytes(b"abc")) == 64


# -- topology ---------------------------------------------------------------


def test_graph_is_connected_and_dense_enough(scenario) -> None:  # noqa: ANN001
    graph = build_graph(scenario.network)
    report = connectivity_report(graph)
    assert report["nodes"] == len(scenario.network.ports) + len(scenario.network.chokepoints)
    assert report["edges"] == len(scenario.network.lanes)
    assert report["connected"] is True
    assert 0.0 < report["density"] <= 1.0


def test_every_port_reaches_every_other_port(scenario) -> None:  # noqa: ANN001
    import networkx as nx

    graph = build_graph(scenario.network)
    ports = [p.id for p in scenario.network.ports]
    for origin in ports:
        lengths = nx.single_source_dijkstra_path_length(graph, origin, weight="distance_nm")
        assert all(dest in lengths for dest in ports)


def test_lane_pairs_are_mirrored(scenario) -> None:  # noqa: ANN001
    pairs = lane_pairs(scenario)
    for lane in scenario.network.lanes:
        assert pairs[(lane.a, lane.b)] == pairs[(lane.b, lane.a)]
        assert pairs[(lane.a, lane.b)]["distance_nm"] == lane.distance_nm


def test_criticality_is_a_probability_distribution(scenario) -> None:  # noqa: ANN001
    graph = build_graph(scenario.network)
    cent = criticality(graph)
    assert all(0.0 <= v <= 1.0 for v in cent.values())
    # subset normalisation over port-to-port traffic keeps the total mass <= 1
    assert 0.5 < sum(cent.values()) <= 1.0 + 1e-9
    assert max(cent.values()) > 0.0


def test_articulation_point_actually_disconnects(scenario) -> None:  # noqa: ANN001
    import networkx as nx

    graph = build_graph(scenario.network)
    points = articulation_points(graph)
    assert points, "expected at least one single point of failure"
    target = points[0]
    pruned = graph.copy()
    pruned.remove_node(target)
    assert not nx.is_connected(pruned)


def test_port_load_matches_nominal_throughput(scenario) -> None:  # noqa: ANN001
    loads = port_loads(scenario)
    total_teu = sum(load.teu_per_day for load in loads.values())
    # every channel is counted twice: once at the load port, once at the discharge port
    assert total_teu == pytest.approx(2 * scenario.nominal_daily_teu(), rel=1e-9)
    assert all(load.teu_per_day > 0 for load in loads.values())
    assert all(load.nominal_utilization > 0 for load in loads.values())


# -- weather ----------------------------------------------------------------


def test_stationary_distribution_is_a_valid_law(scenario) -> None:  # noqa: ANN001
    for zone in scenario.network.zones:
        pi = stationary_distribution(zone, 1.0, "NW_MONSOON")
        assert pi.shape == (4,)
        assert np.all(pi >= 0)
        assert pi.sum() == pytest.approx(1.0, abs=1e-9)


def test_stationary_distribution_satisfies_generator_balance(scenario) -> None:  # noqa: ANN001
    from maritime_sentinel.weather import generator

    zone = scenario.network.zones[0]
    pi = stationary_distribution(zone, 1.0, "NW_MONSOON")
    q = generator(zone, 1.0, "NW_MONSOON")
    assert np.allclose(pi @ q, 0.0, atol=1e-9)


def test_weather_intensity_shifts_the_tail(scenario) -> None:  # noqa: ANN001
    zone = next(z for z in scenario.network.zones if z.id == "JAVA_SEA")
    tail = [zone_statistics(zone, k, "NW_MONSOON")["extreme"] for k in (0.5, 1.0, 2.0)]
    assert tail[0] < tail[1] < tail[2]


def test_effective_transition_stays_stochastic(scenario) -> None:  # noqa: ANN001
    for zone in scenario.network.zones:
        for season in ("NW_MONSOON", "SE_MONSOON", "TRANSITION"):
            matrix = effective_transition(zone, 1.7, season)
            assert np.allclose(matrix.sum(axis=1), 1.0, atol=1e-9)
            assert np.allclose(np.diag(matrix), 0.0)


def test_season_changes_expected_productivity(scenario) -> None:  # noqa: ANN001
    zone = next(z for z in scenario.network.zones if z.id == "FLORES_SEA")
    nw = zone_statistics(zone, 1.0, "NW_MONSOON")["expected_productivity"]
    se = zone_statistics(zone, 1.0, "SE_MONSOON")["expected_productivity"]
    assert se < nw, "Flores Sea must be worse during the southeast monsoon"
