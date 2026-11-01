"""Graph wiring: a cyclic, stateful build pipeline.

    START -> planner -> synthesizer -> sandbox_tester -> { self_correction | finalizer }
                            ^                                    |
                            +------------------------------------+

The conditional edge out of the tester is what makes the loop self-healing:
it routes back into the synthesizer for at most ``max_retries`` extra passes.
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from ..schemas import Directive
from .nodes import finalizer, planner, sandbox_tester, self_correction, synthesizer
from .state import PipelineState, initial_state


def route_after_test(state: PipelineState) -> Literal["self_correction", "finalizer"]:
    report = state.get("report")
    attempt = int(state.get("attempt", 0))
    max_attempts = int(state.get("max_attempts", 4))
    if report is not None and report.passed:
        return "finalizer"
    if attempt + 1 >= max_attempts:
        return "finalizer"
    return "self_correction"


def route_after_reflection(state: PipelineState) -> Literal["synthesizer", "finalizer"]:
    reflections = state.get("reflections", [])
    if not reflections:
        return "finalizer"
    return "synthesizer" if reflections[-1].next_action == "synthesize" else "finalizer"


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("planner", planner)
    graph.add_node("synthesizer", synthesizer)
    graph.add_node("sandbox_tester", sandbox_tester)
    graph.add_node("self_correction", self_correction)
    graph.add_node("finalizer", finalizer)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "synthesizer")
    graph.add_edge("synthesizer", "sandbox_tester")
    graph.add_conditional_edges(
        "sandbox_tester",
        route_after_test,
        {"self_correction": "self_correction", "finalizer": "finalizer"},
    )
    graph.add_conditional_edges(
        "self_correction",
        route_after_reflection,
        {"synthesizer": "synthesizer", "finalizer": "finalizer"},
    )
    graph.add_edge("finalizer", END)
    return graph.compile()


def run_pipeline(directive: Directive, output_dir: str = "artifacts") -> dict[str, Any]:
    app = build_graph()
    state = initial_state(directive)
    state["output_dir"] = output_dir
    return dict(app.invoke(state))
