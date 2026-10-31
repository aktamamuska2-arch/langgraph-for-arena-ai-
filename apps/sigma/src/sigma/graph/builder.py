"""Assembly of the SIGMA graph.

The topology below is the executable form of PRD section 3.1. Two cycles are
declared explicitly:

* **micro-cycle** `researcher -> researcher` (tool gathering),
* **macro-cycle** `critic -> {planner | researcher | analyst}` (self-correction).

FASE 2 adds the HITL `human_gate` between the accept branch and `finalizer`.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from sigma.agents.nodes import (
    analyst,
    containment,
    critic,
    finalizer,
    human_gate,
    intake,
    planner,
    researcher,
)
from sigma.errors import ToolTimeoutError
from sigma.graph.routing import (
    APPROVAL_NODE,
    HITL_NODE,
    make_route_after_critic,
    make_route_after_human,
    route_after_intake,
    route_after_planner,
    route_after_researcher,
)
from sigma.state import SigmaState

# Transient failures are retried by the runtime, not by application code: a timed
# out tool call is worth one more attempt, a malformed one is not (PRD section 9).
RESEARCH_RETRY = RetryPolicy(
    max_attempts=2,
    initial_interval=0.05,
    backoff_factor=1.0,
    jitter=False,
    retry_on=ToolTimeoutError,
)


def build_graph(
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    approval_node: str = APPROVAL_NODE,
    hitl: bool = False,
) -> CompiledStateGraph[SigmaState, None, SigmaState, SigmaState]:
    """Compile the SIGMA graph.

    Args:
        checkpointer: required for interrupts, time-travel and session recovery.
        approval_node: where an accepted run goes. `finalizer` by default,
            `human_gate` when HITL is enabled.
        hitl: insert the human approval gate before the finalizer (FASE 2).
    """
    builder = StateGraph(SigmaState)

    builder.add_node("intake", intake)
    builder.add_node("planner", planner)
    builder.add_node("researcher", researcher, retry_policy=RESEARCH_RETRY)
    builder.add_node("analyst", analyst)
    builder.add_node("critic", critic)
    if hitl:
        # An interrupt without a checkpointer is a programming error, not a mode.
        if checkpointer is None:
            raise ValueError("hitl requires a checkpointer")
        builder.add_node(HITL_NODE, human_gate)
    builder.add_node("finalizer", finalizer)
    builder.add_node("containment", containment)

    builder.add_edge(START, "intake")
    builder.add_conditional_edges(
        "intake", route_after_intake, ["planner", "containment"])
    builder.add_conditional_edges(
        "planner", route_after_planner, ["researcher", "containment"])
    builder.add_conditional_edges(
        "researcher", route_after_researcher, ["researcher", "analyst", "containment"])
    builder.add_edge("analyst", "critic")
    builder.add_conditional_edges(
        "critic",
        make_route_after_critic(approval_node),
        ["planner", "researcher", "analyst", approval_node, "containment"],
    )
    if hitl:
        builder.add_conditional_edges(
            HITL_NODE, make_route_after_human(), ["finalizer", "planner", "containment"])
    builder.add_edge("finalizer", END)
    builder.add_edge("containment", END)

    return builder.compile(checkpointer=checkpointer, name="sigma-analytics")


__all__ = ["RESEARCH_RETRY", "build_graph"]
