"""Graph assembly: nodes, conditional edges and the retry governor.

Topology
--------
::

    START
      |
    planner ---------(plan valid?)-------------------+
      |                                              |
    synthesizer --(build emitted?)--> sandbox_tester  |
                                        |            |
                            (passed?) --+--> finalize |
                                        |            |
                                     reflector ------+
                                        |
                              (retries left?) --> synthesizer
                                        |
                                      finalize --> END

The retry governor is a hard budget: at most ``max_retries`` self-healing
loops (4 executions in total). Once exhausted, routing is forced to
``finalize`` and the artifact is stamped ``FAILED_NEEDS_REVIEW`` rather than
being retried forever.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from ..config import DEFAULT_SCENARIO
from ..schemas import ScenarioSpec
from .nodes import (
    make_finalizer,
    make_planner,
    make_reflector,
    make_synthesizer,
    make_tester,
)
from .state import RunContext, State, initial_state

PLANNER = "planner"
SYNTHESIZER = "synthesizer"
TESTER = "sandbox_tester"
REFLECTOR = "reflector"
FINALIZE = "finalize"


def route_after_planner(_state: State) -> Literal["synthesizer", "finalize"]:
    return "synthesizer"


def route_after_synthesis(state: State) -> Literal["sandbox_tester", "reflector"]:
    artifacts = state.get("code_artifacts") or {}
    return "sandbox_tester" if artifacts.get("entrypoint") else "reflector"


def make_test_router(max_retries: int):
    def route_after_test(state: State) -> Literal["reflector", "finalize"]:
        report = state.get("test_report") or {}
        if report.get("passed"):
            return "finalize"
        if int(state.get("retry_count", 0)) >= max_retries:
            return "finalize"
        return "reflector"

    return route_after_test


def make_reflect_router(max_retries: int):
    def route_after_reflect(state: State) -> Literal["synthesizer", "finalize"]:
        if int(state.get("retry_count", 0)) > max_retries:
            return "finalize"
        return "synthesizer"

    return route_after_reflect


def build_graph(context: RunContext):
    """Compile the AM-SES state graph for one run context."""

    graph: StateGraph = StateGraph(State)
    graph.add_node(PLANNER, make_planner(context))
    graph.add_node(SYNTHESIZER, make_synthesizer(context))
    graph.add_node(TESTER, make_tester(context))
    graph.add_node(REFLECTOR, make_reflector(context))
    graph.add_node(FINALIZE, make_finalizer(context))

    graph.add_edge(START, PLANNER)
    graph.add_conditional_edges(
        PLANNER, route_after_planner, {"synthesizer": SYNTHESIZER, "finalize": FINALIZE}
    )
    graph.add_conditional_edges(
        SYNTHESIZER, route_after_synthesis, {"sandbox_tester": TESTER, "reflector": REFLECTOR}
    )
    graph.add_conditional_edges(
        TESTER,
        make_test_router(context.max_retries),
        {"reflector": REFLECTOR, "finalize": FINALIZE},
    )
    graph.add_conditional_edges(
        REFLECTOR,
        make_reflect_router(context.max_retries),
        {"synthesizer": SYNTHESIZER, "finalize": FINALIZE},
    )
    graph.add_edge(FINALIZE, END)
    return graph.compile()


def new_run_id(prefix: str = "run") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"


def run_workflow(
    scenario: ScenarioSpec | dict[str, Any] | None = None,
    *,
    project_root: Path | None = None,
    artifacts_root: Path | None = None,
    run_id: str | None = None,
    max_retries: int = 3,
    sandbox_timeout: float = 90.0,
    fault_mode: str = "none",
    fault_persist: bool = False,
    recursion_limit: int = 40,
) -> tuple[dict[str, Any], RunContext]:
    """Execute the graph end-to-end and return ``(final_state, context)``."""

    root = project_root or Path(__file__).resolve().parents[2]
    spec = (
        scenario
        if isinstance(scenario, ScenarioSpec)
        else ScenarioSpec.model_validate(scenario)
        if scenario is not None
        else DEFAULT_SCENARIO
    )
    context = RunContext(
        run_id=run_id or new_run_id(),
        project_root=root,
        artifacts_root=artifacts_root or root / "artifacts",
        max_retries=max_retries,
        sandbox_timeout=sandbox_timeout,
        fault_mode=fault_mode,
        fault_persist=fault_persist,
    )
    app = build_graph(context)
    state = initial_state(spec.model_dump(mode="json"), context.run_id)
    final_state = app.invoke(state, {"recursion_limit": recursion_limit})
    return final_state, context


def graph_mermaid(context: RunContext) -> str:
    """Static Mermaid rendering of the compiled graph (for docs and the UI)."""

    app = build_graph(context)
    return app.get_graph().draw_mermaid()
