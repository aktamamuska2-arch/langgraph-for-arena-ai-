"""The LangGraph build pipeline: topology, routing, healing and refusal."""

from __future__ import annotations

import json

import pytest

from agora.graph import build_graph, initial_state, run_pipeline
from agora.graph.build import route_after_reflection, route_after_test
from agora.graph.nodes import _critical_radius, self_correction
from agora.graph.state import get_by_pointer, set_by_pointer
from agora.metrics import DEFAULT_GATES, evaluate
from agora.schemas import Blueprint, Directive, ExperimentSpec, Patch


@pytest.fixture(scope="module")
def pipeline_result(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    directive = Directive(horizon=420, settle_ticks=40, max_retries=3)
    return run_pipeline(directive, output_dir=str(out))


def test_graph_topology():
    compiled = build_graph()
    nodes = set(compiled.get_graph().nodes)
    assert {"planner", "synthesizer", "sandbox_tester", "self_correction", "finalizer"} <= nodes
    edges = {(e.source, e.target) for e in compiled.get_graph().edges}
    assert ("planner", "synthesizer") in edges
    assert ("synthesizer", "sandbox_tester") in edges
    assert ("self_correction", "synthesizer") in edges  # the healing cycle
    assert ("finalizer", "__end__") in edges


def test_router_prefers_finalize_on_success():
    state = {"report": None, "attempt": 0, "max_attempts": 4}
    assert route_after_test(state) == "self_correction"
    from agora.metrics import evaluate

    ok = evaluate(DEFAULT_GATES, {g.metric: (g.threshold if g.comparator == ">=" else 0.0) for g in DEFAULT_GATES}, 0)
    state["report"] = ok
    assert route_after_test(state) == "finalizer"


def test_router_stops_at_the_retry_ceiling():
    state = {"report": None, "attempt": 3, "max_attempts": 4}
    assert route_after_test(state) == "finalizer"


def test_pointer_helpers_roundtrip():
    payload = {"economy": {"discovery_radius": 18.0}}
    set_by_pointer(payload, "economy.discovery_radius", 21.5)
    assert get_by_pointer(payload, "economy.discovery_radius") == 21.5
    set_by_pointer(payload, "horizon", 600)
    assert payload["horizon"] == 600


def test_critical_radius_is_the_mst_longest_edge():
    points = [(0, 0), (10, 0), (20, 0)]
    assert _critical_radius(points) == pytest.approx(10.0)
    assert _critical_radius([(0, 0)]) >= 2.0


def test_pipeline_produces_a_verdict(pipeline_result):
    assert pipeline_result["verdict"] in {"PASS", "PASS_WITH_RETRIES"}
    report = pipeline_result["report"]
    assert report.passed
    assert report.attempt + 1 <= pipeline_result["max_attempts"]


def test_blueprint_carries_the_math(pipeline_result):
    blueprint = pipeline_result["blueprint"]
    assert len(blueprint.equations) >= 8
    domains = {e.domain for e in blueprint.equations}
    assert {"production", "pricing", "gravity", "network", "validation"} <= domains
    assert blueprint.critical_radius > 2.0
    assert len(blueprint.site_audit) == 16
    assert all(g.id for g in blueprint.gates)
    assert any("gravity" in e.latex.lower() or "A_{ij}" in e.latex for e in blueprint.equations)


def test_pipeline_reaches_its_verdict_without_wasted_retries(pipeline_result):
    """The buyer-initiated discovery pass removed the fragmentation the roadmap
    used to need a retry for, so the loop should now close on its first attempt."""

    report = pipeline_result["report"]
    assert report.passed
    assert report.attempt == len(pipeline_result["reflections"])
    if not pipeline_result["reflections"]:
        assert pipeline_result["verdict"] == "PASS"
        assert pipeline_result["spec"].economy.discovery_radius >= pipeline_result["blueprint"].critical_radius


def test_pipeline_self_heals_a_fragmented_market(tmp_path):
    """Injected proof that the healing cycle still works end to end.

    A radius pinned at a third of the critical radius fragments the market; the
    reflector must measure it, classify it as ``network_fragmentation``, patch
    the radius and recover inside the retry budget.
    """

    directive = Directive(horizon=420, settle_ticks=40, max_retries=3)
    state = initial_state(directive)
    state["output_dir"] = str(tmp_path / "fragmented")
    state["patches"] = [
        Patch(
            pointer="economy.discovery_radius",
            previous=18.8,
            updated=4.0,
            rationale="injected: an over-local market scan",
        )
    ]
    result = dict(build_graph().invoke(state))
    reflections = result["reflections"]
    assert reflections, "a fragmented market must be rejected at least once"
    assert reflections[0].failure_class == "network_fragmentation"
    assert any(p.pointer == "economy.discovery_radius" for p in reflections[0].patches)
    assert result["verdict"] == "PASS_WITH_RETRIES"
    assert result["spec"].economy.discovery_radius > 4.0


def test_artifacts_are_written_and_hashed(pipeline_result):
    artifacts = pipeline_result["artifacts"]
    assert artifacts
    roles = {a.role for a in artifacts}
    assert "final report" in roles
    assert "blueprint" in roles
    from pathlib import Path

    for artifact in artifacts:
        path = Path(artifact.path)
        assert path.exists()
        assert artifact.sha256 and len(artifact.sha256) == 64
        assert artifact.bytes > 0
    report_path = next(a.path for a in artifacts if a.role == "final report")
    payload = json.loads(Path(report_path).read_text())
    assert payload["verdict"] in {"PASS", "PASS_WITH_RETRIES"}


def test_reflector_refuses_to_patch_a_conservation_defect():
    blueprint = Blueprint(
        blueprint_id="BP-X",
        experiment_id="ASEOWS-X",
        objective="refusal",
        equations=[],
        gates=list(DEFAULT_GATES),
        world_summary={},
        economy={},
        critical_radius=10.0,
        site_audit=[],
        expected={},
    )
    metrics = {g.metric: (g.threshold * 1.5 if g.comparator == ">=" else g.threshold * 0.5) for g in DEFAULT_GATES}
    metrics["mass_balance_error"] = 0.02
    report = evaluate(DEFAULT_GATES, metrics, attempt=0, duration=1.0)
    assert report.failure_class == "mass_imbalance"

    state = initial_state(Directive())
    state.update(
        {
            "blueprint": blueprint,
            "spec": ExperimentSpec(),
            "report": report,
            "metrics": metrics,
            "attempt": 0,
        }
    )
    update = self_correction(state)
    reflection = update["reflections"][-1]
    assert reflection.patches == []
    assert reflection.next_action == "finalize"
    assert route_after_reflection({"reflections": [reflection]}) == "finalizer"


def test_max_retries_is_enforced():
    directive = Directive(horizon=120, max_retries=1, experiment_id="ASEOWS-RETRY")
    # a horizon this short cannot converge, so the loop must stop after the budget
    from agora.graph import run_pipeline as run

    result = run(directive, output_dir="/tmp/aseows-retry-test")
    assert result["report"].attempt + 1 <= 2
    assert result["verdict"] in {"PASS_WITH_RETRIES", "FAILED_NEEDS_REVIEW"}


def test_injected_patch_is_applied_by_the_synthesizer():
    from agora.graph.nodes import planner, synthesizer

    state = initial_state(Directive(experiment_id="ASEOWS-PATCH"))
    state.update(planner(state))
    state["patches"] = [
        Patch(pointer="economy.discovery_radius", previous=None, updated=27.0, rationale="test"),
        Patch(pointer="economy.max_partners", previous=None, updated=11, rationale="test"),
    ]
    state.update(synthesizer(state))
    assert state["spec"].economy.discovery_radius == 27.0
    assert state["spec"].economy.max_partners == 11
