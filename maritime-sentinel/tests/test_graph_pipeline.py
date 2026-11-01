"""End-to-end tests for the LangGraph self-healing pipeline."""

from __future__ import annotations

import json

from maritime_sentinel.graph.build import (
    FINALIZE,
    REFLECTOR,
    SYNTHESIZER,
    build_graph,
    make_reflect_router,
    make_test_router,
    run_workflow,
)
from maritime_sentinel.graph.state import RunContext, State, initial_state
from maritime_sentinel.sandbox import execute_build


def test_state_schema_declares_the_mandated_slices() -> None:
    annotations = State.__annotations__
    for field in (
        "scenario_params",
        "architectural_blueprint",
        "code_artifacts",
        "execution_logs",
        "retry_count",
    ):
        assert field in annotations


def test_initial_state_is_immutable_ready(scenario) -> None:  # noqa: ANN001
    state = initial_state(scenario.model_dump(mode="json"), "run-test")
    assert state["retry_count"] == 0
    assert state["execution_logs"] == []
    assert state["scenario_params"]["spec_id"] == scenario.spec_id


def test_compiled_graph_has_the_expected_topology(tmp_path, project_root) -> None:  # noqa: ANN001
    ctx = RunContext(run_id="t", project_root=project_root, artifacts_root=tmp_path)
    app = build_graph(ctx)
    nodes = set(app.get_graph().nodes)
    assert {"planner", "synthesizer", "sandbox_tester", "reflector", "finalize"} <= nodes
    assert "__start__" in nodes and "__end__" in nodes


def test_retry_governor_routing() -> None:
    test_route = make_test_router(3)
    reflect_route = make_reflect_router(3)

    assert test_route({"test_report": {"passed": True}, "retry_count": 0}) == FINALIZE
    assert test_route({"test_report": {"passed": False}, "retry_count": 0}) == REFLECTOR
    assert test_route({"test_report": {"passed": False}, "retry_count": 2}) == REFLECTOR
    assert test_route({"test_report": {"passed": False}, "retry_count": 3}) == FINALIZE

    assert reflect_route({"retry_count": 1}) == SYNTHESIZER
    assert reflect_route({"retry_count": 3}) == SYNTHESIZER
    assert reflect_route({"retry_count": 4}) == FINALIZE


def test_workflow_heals_and_emits_hashed_artifacts(tmp_path, scenario, project_root) -> None:  # noqa: ANN001
    state, context = run_workflow(scenario, project_root=project_root, artifacts_root=tmp_path, max_retries=3)

    assert state["verdict"] in {"PASS", "PASS_WITH_RETRIES"}
    assert state["retry_count"] <= 3

    report = state["test_report"]
    assert report["passed"] is True
    assert all(c["passed"] for c in report["criteria"])

    blueprint = state["architectural_blueprint"]
    assert blueprint["blueprint_id"].startswith("BP-")
    assert blueprint["equations"], "planner must emit the state-space equations"
    assert blueprint["sizing"], "planner must emit one sizing decision per port"

    artifacts = state["code_artifacts"]
    names = {a["role"] for a in artifacts["artifacts"]}
    assert names == {"manifest", "entrypoint", "design_memo", "acceptance"}
    for record in artifacts["artifacts"]:
        assert len(record["sha256"]) == 64

    # the artifact on disk must hash to what the report claims
    from pathlib import Path

    from maritime_sentinel.codegen import sha256_bytes

    for record in artifacts["artifacts"]:
        payload = Path(record["path"]).read_bytes()
        assert sha256_bytes(payload) == record["sha256"]

    run_dir = context.run_dir()
    assert (run_dir / "final_report.json").exists()
    final = json.loads((run_dir / "final_report.json").read_text())
    assert final["verdict"] == state["verdict"]
    assert final["attempts"] == state["retry_count"] + 1


def test_workflow_first_attempt_is_rejected_under_stress(tmp_path, scenario, project_root) -> None:  # noqa: ANN001
    """The naive proportional sizing must fail the stability gate, otherwise the
    self-healing loop would be decorative."""

    state, context = run_workflow(scenario, project_root=project_root, artifacts_root=tmp_path, max_retries=3)
    assert state["retry_count"] >= 1, "expected at least one healing iteration"
    trail = state["reflection_trail"]
    assert trail, "reflector must record its diagnosis"
    first = trail[0]
    assert first["failure_class"] in {"instability", "divergence", "underperformance"}
    assert first["patches"], "reflector must emit machine-applicable patches"
    assert first["next_action"] == "synthesize"

    # the healed build must adopt the tail-risk sizing and more berths than the first draft
    def berths(attempt: int) -> dict[str, int]:
        manifest = json.loads((context.attempt_dir(attempt) / "scenario_manifest.json").read_text())
        return manifest["deployment"]["berths"]

    assert sum(berths(state["retry_count"]).values()) > sum(berths(0).values())
    assert state["architectural_blueprint"]["sizing_strategy"] == "tail_risk"


def test_retry_cap_forces_a_review_verdict(tmp_path, scenario, project_root) -> None:  # noqa: ANN001
    state, _ = run_workflow(
        scenario,
        project_root=project_root,
        artifacts_root=tmp_path,
        max_retries=0,
        fault_mode="runtime",
    )
    assert state["verdict"] == "FAILED_NEEDS_REVIEW"
    assert state["retry_count"] == 0
    assert state["test_report"]["passed"] is False
    assert state["test_report"]["failure_class"] == "runtime_error"


def test_schema_fault_is_repaired(tmp_path, scenario, project_root) -> None:  # noqa: ANN001
    state, _ = run_workflow(
        scenario, project_root=project_root, artifacts_root=tmp_path, max_retries=3, fault_mode="schema"
    )
    assert state["verdict"] in {"PASS", "PASS_WITH_RETRIES"}
    classes = [entry["failure_class"] for entry in state["reflection_trail"]]
    assert "schema_error" in classes
    repair = next(e for e in state["reflection_trail"] if e["failure_class"] == "schema_error")
    assert repair["patches"][0]["pointer"].endswith("transition[0]")


def test_sandbox_reports_missing_entrypoint(tmp_path) -> None:  # noqa: ANN001
    outcome = execute_build(tmp_path)
    assert outcome.exit_code == 127
    assert outcome.error == "missing_entrypoint"
    assert outcome.payload is None


def test_sandbox_scrubs_the_environment(tmp_path, project_root) -> None:  # noqa: ANN001
    script = tmp_path / "leak.py"
    script.write_text(
        "import os, json\n"
        "print('__RESULT__' + json.dumps({'ok': True, 'leaked': [k for k in os.environ if 'TOKEN' in k]}))\n"
    )
    outcome = execute_build(tmp_path, entrypoint="leak.py", timeout=20)
    assert outcome.payload is not None
    assert outcome.payload["leaked"] == []
