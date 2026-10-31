"""Conditional-edge predicates.

Every transition of the graph is decided here, from state alone. Keeping the
predicates in one module is what makes the cyclic architecture auditable: the
four cycle-targets of the macro-cycle (planner / researcher / analyst / approval)
are four branches of a single function (PRD section 5).
"""

from __future__ import annotations

from typing import Literal

from sigma.errors import SigmaError
from sigma.state import (
    BudgetState,
    ErrorRecord,
    RunStatus,
    SigmaState,
    latest_evaluation,
)

# The target returned when the run is good enough to be approved. FASE 1 sends the
# run straight to `finalizer`; FASE 2 routes it through the HITL `human_gate`.
APPROVAL_NODE = "finalizer"

# Gap class -> the node that can actually fix it (PRD section 5 routing table).
_GAP_TARGET: dict[str, str] = {
    "structural": "planner",
    "missing_evidence": "researcher",
    "numerical": "researcher",
    "interpretation": "analyst",
}


def fatal_error(state: SigmaState) -> ErrorRecord | None:
    """The first fatal error recorded so far, if any."""
    for record in state.get("errors") or []:
        if record.fatal:
            return record
    return None


def has_pending_tools(state: SigmaState) -> bool:
    """True when the last researcher pass queued calls that still need executing."""
    return bool(state.get("queued_call_ids"))


def budget(state: SigmaState) -> BudgetState:
    value = state.get("budget")
    if value is None:
        raise SigmaError("budget missing from state")
    return value


# --------------------------------------------------------------------------- #
# Edge predicates
# --------------------------------------------------------------------------- #


def route_after_intake(state: SigmaState) -> Literal["planner", "containment"]:
    """A fatal intake failure cannot be analysed: contain immediately."""
    if fatal_error(state) or state.get("scenario") is None or state.get("request") is None:
        return "containment"
    return "planner"


def route_after_planner(state: SigmaState) -> Literal["researcher", "containment"]:
    plan = state.get("plan")
    if fatal_error(state) or plan is None or not plan.sub_questions:
        return "containment"
    return "researcher"


def route_after_researcher(
    state: SigmaState,
) -> Literal["researcher", "analyst", "containment"]:
    """The micro-cycle: re-enter `researcher` while calls are still queued.

    Three independent guards prevent an infinite loop:
    1. `queued_call_ids` is only non-empty when genuinely new calls were proposed;
    2. the tool-call budget is monotonic (a replay can never refund it);
    3. `recursion_limit` on invoke is the last-resort backstop.
    """
    if fatal_error(state):
        return "containment"
    if has_pending_tools(state) and budget(state).tool_calls_left > 0:
        return "researcher"
    return "analyst"


def make_route_after_critic(approval_node: str = APPROVAL_NODE):
    """Build the macro-cycle predicate.

    `approval_node` is injected so the same predicate serves FASE 1 (straight to
    `finalizer`) and FASE 2 (through the HITL `human_gate`) without duplication.
    """

    def route_after_critic(
        state: SigmaState,
    ) -> Literal["planner", "researcher", "analyst", "containment", str]:
        if fatal_error(state):
            return "containment"

        evaluation = latest_evaluation(state)
        if evaluation is None:
            return "containment"

        if evaluation.verdict == "accept":
            return approval_node

        if budget(state).revisions_left <= 0:
            # Out of revisions: stop cycling, hand the provisional result over.
            return approval_node

        dominant = evaluation.dominant_gap
        target = _GAP_TARGET.get(dominant or "", "researcher")
        if target == "planner" and budget(state).revisions_left <= 0:
            return approval_node
        return target

    return route_after_critic


def make_route_after_human():
    """Route the human's decision (FASE 2). Fail-safe: unknown -> containment."""

    def route_after_human(
        state: SigmaState,
    ) -> Literal["finalizer", "planner", "containment"]:
        decision = state.get("human_decision")
        if decision is None:
            return "containment"
        if decision.decision == "approve":
            return "finalizer"
        if decision.decision == "revise":
            if budget(state).revisions_left <= 0:
                return "finalizer"  # nothing left to spend: finalise what we have
            return "planner"
        if decision.decision == "reject":
            return "containment"
        return "containment"

    return route_after_human


__all__ = [
    "APPROVAL_NODE",
    "budget",
    "fatal_error",
    "has_pending_tools",
    "make_route_after_critic",
    "make_route_after_human",
    "route_after_intake",
    "route_after_planner",
    "route_after_researcher",
]

# Re-exported for typing convenience in tests.
_STATUS = RunStatus
