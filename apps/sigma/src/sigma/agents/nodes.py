"""Graph nodes: Planner, Researcher (executor), Analyst, Evaluator, Finalizer, Containment.

Contract for every node in this module:

* **pure w.r.t. state** — a node reads `SigmaState` and returns a partial state dict,
* **budgeted** — every node consumes budget before returning,
* **auditable** — every node appends at least one `AuditEvent`,
* **guarded** — failures are classified, never leaked as raw tracebacks (PRD section 9).

Nodes never call `goto` directly: all transitions are declared as conditional
edges in `sigma.graph.builder`, which keeps the control-flow graph inspectable.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable

from langgraph.errors import GraphBubbleUp
from langgraph.types import interrupt
from pydantic import ValidationError

from sigma.agents import reasoning
from sigma.agents.llm_calls import structured
from sigma.config import SETTINGS
from sigma.errors import (BudgetExhausted, InvalidResume, SigmaError,
                          ToolTimeoutError)
from sigma.state import (
    AuditEvent,
    BudgetState,
    DraftBrief,
    ErrorRecord,
    Evaluation,
    HumanDecision,
    PlanState,
    RequestPayload,
    RunStatus,
    ScenarioContext,
    SigmaState,
    ToolCall,
    audit,
    consume_budget,
    active_findings,
    latest_evaluation,
    message,
    new_session_id,
)
from sigma.llm.deterministic import ToolPlan
from sigma.tools.market import SECTORS, build_portfolio
from sigma.tools.registry import ToolContext, ToolRegistry

REGISTRY = ToolRegistry(timeout_ms=SETTINGS.tool_timeout_ms)


# --------------------------------------------------------------------------- #
# Guard: classifies failures and keeps the graph moving
# --------------------------------------------------------------------------- #


def _error_record(node: str, exc: BaseException) -> ErrorRecord:
    if isinstance(exc, SigmaError):
        return ErrorRecord(node=node, kind=type(exc).__name__, message=str(exc),
                           fatal=exc.fatal)
    return ErrorRecord(node=node, kind=type(exc).__name__, message=f"{exc}", fatal=True)


def guarded(name: str) -> Callable[[Callable[[SigmaState], dict[str, Any]]],
                                   Callable[[SigmaState], dict[str, Any]]]:
    """Wrap a node so that only terminal LangGraph control-flow escapes.

    Non-fatal errors are recorded and execution continues (the router decides);
    fatal errors record themselves and the router routes to `containment`.
    """

    def decorator(fn: Callable[[SigmaState], dict[str, Any]]) -> Callable[[SigmaState], dict[str, Any]]:
        @functools.wraps(fn)
        def wrapper(state: SigmaState) -> dict[str, Any]:
            started = time.perf_counter()
            try:
                update = fn(state)
            except GraphBubbleUp:
                # `interrupt()` and friends are control flow, not failures.
                raise
            except InvalidResume:
                # Bad human input must reach the caller, not be contained.
                raise
            except ToolTimeoutError:
                # Transient: let `RESEARCH_RETRY` re-run the node before giving up.
                raise
            except BudgetExhausted as exc:
                update = {
                    "errors": [_error_record(name, exc)],
                    "audit_trail": [audit(state, name, "budget_exhausted", str(exc))],
                    "status": RunStatus.AWAITING_HUMAN,
                }
            except Exception as exc:  # noqa: BLE001 - classified below, never swallowed
                record = _error_record(name, exc)
                update = {
                    "errors": [record],
                    "audit_trail": [audit(state, name, "error", str(exc), fatal=record.fatal)],
                }
            update["audit_trail"] = _renumber(state, update)
            update.setdefault("audit_trail", []).append(
                AuditEvent(seq=(_last_seq(state) + len(update["audit_trail"]) + 1),
                           node=name, kind="completed", message=f"{name} finished",
                           data={"ms": round((time.perf_counter() - started) * 1000, 1)}))
            return update

        wrapper.__name__ = name
        wrapper.node_name = name  # type: ignore[attr-defined]
        return wrapper

    return decorator


def _last_seq(state: SigmaState) -> int:
    trail = state.get("audit_trail") or []
    return max((e.seq for e in trail), default=0)


def _renumber(state: SigmaState, update: dict[str, Any]) -> list[AuditEvent]:
    """Give every event this node produced a fresh, strictly increasing `seq`.

    Nodes build events from the *incoming* state (provider events, memory
    compaction, their own summary), so without renumbering a node that emits
    several events would reuse the same sequence number.
    """
    base = _last_seq(state)
    events: list[AuditEvent] = []
    for offset, event in enumerate(update.get("audit_trail") or [], start=1):
        event.seq = base + offset
        events.append(event)
    return events


def _compact_errors(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                     for e in exc.errors()[:3])


def _require(state: SigmaState, key: str) -> Any:
    value = state.get(key)  # type: ignore[call-overload]
    if value is None:
        raise SigmaError(f"{key} missing from state")
    return value


def _tool_context(state: SigmaState) -> ToolContext:
    scenario: ScenarioContext = _require(state, "scenario")
    return ToolContext(
        portfolio_id=scenario.portfolio_id,
        seed=scenario.seed,
        horizon_quarters=scenario.horizon_quarters,
        monte_carlo_paths=SETTINGS.monte_carlo_paths,
        state_evidence={e.evidence_id: {"summary": e.summary, "tool": e.tool,
                                        "provenance": e.provenance}
                        for e in (state.get("evidence") or [])},
    )


# --------------------------------------------------------------------------- #
# intake
# --------------------------------------------------------------------------- #


@guarded("intake")
def intake(state: SigmaState) -> dict[str, Any]:
    """Normalise the request and materialise the deterministic scenario."""
    request: RequestPayload = _require(state, "request")
    session_id = state.get("session_id") or new_session_id()
    if state.get("scenario") is None:
        portfolio = build_portfolio(request.portfolio_id, request.seed)
        sectors = sorted({v["sector"] for v in portfolio})
        scenario = ScenarioContext(
            portfolio_id=request.portfolio_id,
            seed=request.seed,
            venture_count=len(portfolio),
            sectors=sectors or list(SECTORS[:2]),
            capital_pool=round(sum(v["capital_deployed"] for v in portfolio), 3),
        )
    else:
        scenario = state["scenario"]

    budget = state.get("budget") or BudgetState(
        max_revisions=min(request.max_revisions, SETTINGS.max_revisions),
        max_tool_calls=SETTINGS.max_tool_calls,
        max_tokens=SETTINGS.max_tokens,
    )
    return {
        "session_id": session_id,
        "scenario": scenario,
        "budget": budget,
        "status": RunStatus.RUNNING,
        "errors": state.get("errors") or [],
        "audit_trail": [audit(state, "intake", "session_start",
                              f"session={session_id} portfolio={scenario.portfolio_id} "
                              f"seed={scenario.seed} ventures={scenario.venture_count}")],
        "messages": [message("system",
                             f"Konteks: portofolio {scenario.portfolio_id}, "
                             f"{scenario.venture_count} ventura, seed {scenario.seed}.",
                             kind="scenario")],
    }


# --------------------------------------------------------------------------- #
# planner
# --------------------------------------------------------------------------- #


def _directive(state: SigmaState) -> str | None:
    """Where the next revision should focus: human note wins over critic gaps."""
    decision = state.get("human_decision")
    if decision is not None and decision.decision == "revise" and decision.note:
        return f"human: {decision.note}"
    evaluation = latest_evaluation(state)
    if evaluation is None:
        return None
    kinds = sorted({g.kind for g in evaluation.gaps if g.weight > 0})
    details = "; ".join(g.detail for g in evaluation.gaps if g.weight > 0)[:400]
    return f"kinds={','.join(kinds)} {details}" if kinds else None


@guarded("planner")
def planner(state: SigmaState) -> dict[str, Any]:
    """Decompose the request; on a revision, keep what is answered and add focus."""
    request: RequestPayload = _require(state, "request")
    scenario: ScenarioContext = _require(state, "scenario")
    budget: BudgetState = _require(state, "budget")
    previous: PlanState | None = state.get("plan")

    revision = 0 if previous is None else previous.revision + 1
    if revision > 0 and budget.revisions_left <= 0:
        # Defensive: the router never sends us here without budget, but if it
        # ever does, stop cycling instead of looping (PRD section 5 fail-safe).
        raise BudgetExhausted(f"revision budget exhausted after {budget.revisions_used} revisions")

    directive = _directive(state)
    result = structured(
        state,
        role="planner",
        response_model=PlanState,
        fallback=lambda: reasoning.plan_questions(
            request, scenario, revision=revision, directive=directive,
            existing=previous.sub_questions if previous else []),
        directive=directive,
        revision=revision,
    )
    plan: PlanState = result.value
    # Revisions are charged by the critic, so planning only spends tokens.
    spent = consume_budget(budget, tokens=result.tokens or 250)

    update = dict(result.update)
    update.update({
        "plan": plan,
        "budget": spent,
        "status": RunStatus.RUNNING,
    })
    update.setdefault("audit_trail", []).append(
        audit(state, "planner", "planned",
              f"revision={plan.revision} questions={len(plan.sub_questions)}",
              directive=directive or "", provider=result.provider))
    return update


# --------------------------------------------------------------------------- #
# researcher (executor) — proposes and executes tool calls in a self-loop
# --------------------------------------------------------------------------- #


@guarded("researcher")
def researcher(state: SigmaState) -> dict[str, Any]:
    """Execute queued tool calls, then propose the next ones.

    One pass = execute + propose. If new calls were proposed, `queued_call_ids` is
    non-empty and the conditional edge sends the graph back into this same node:
    this is the **micro-cycle** of the architecture (PRD section 3.1).
    """
    budget: BudgetState = _require(state, "budget")
    ctx = _tool_context(state)

    pending: list[ToolCall] = list(state.get("pending_tools") or [])
    executed_ids = {record.call_id for record in (state.get("tool_calls") or [])}
    runnable = [c for c in pending if c.call_id not in executed_ids][: max(0, budget.tool_calls_left)]

    records: list[Any] = []
    evidences: list[Any] = []
    errors: list[ErrorRecord] = []
    timed_out = False
    for call in runnable:
        result = REGISTRY.execute(call, ctx)
        records.append(result.record)
        timed_out = timed_out or result.timed_out
        if result.ok:
            evidences.append(result.to_evidence(call.sub_question_id))
        else:
            errors.append(ErrorRecord(node="researcher", kind="ToolExecutionError",
                                      message=f"{call.tool}: {result.record.error}", fatal=False))
    if runnable and not evidences:
        # Nothing could be gathered at all: this is an infrastructure failure, not
        # a quality problem, so it must not be answered with another cycle.
        errors.append(ErrorRecord(node="researcher", kind="ToolPassFailed",
                                  message=f"all {len(runnable)} tool calls failed", fatal=True))
    if timed_out:
        # Handed to the runtime retry policy (transient), then to the runner.
        raise ToolTimeoutError("researcher", SETTINGS.tool_timeout_ms)

    # Propose from the *updated* view of the world (new evidence is visible at once).
    merged = dict(state)
    merged["evidence"] = list(state.get("evidence") or []) + evidences
    merged["tool_calls"] = list(state.get("tool_calls") or []) + records
    left = budget.tool_calls_left - len(runnable)
    passes = int(state.get("research_pass") or 0) + 1

    if left > 0:
        result = structured(
            state,
            role="researcher",
            response_model=ToolPlan,
            tools=REGISTRY.schemas(),
            fallback=lambda: ToolPlan(
                calls=reasoning.propose_tool_calls(
                    {**merged, "research_pass": passes}, budget_left=left),
                rationale="deterministic proposal"),
            evidence=merged["evidence"],
            tool_calls=merged["tool_calls"],
            tool_budget_left=left,
            research_pass=passes,
        )
        new_calls = [c for c in result.value.calls][:left]
    else:
        result = None
        new_calls = []

    spent = consume_budget(budget, tools=len(runnable),
                           tokens=(result.tokens if result else 0) + 180 * max(1, len(runnable)))
    update = dict(result.update) if result else {}
    update.update({
        "tool_calls": records,
        "evidence": evidences,
        "errors": errors,
        "pending_tools": new_calls,
        "queued_call_ids": [c.call_id for c in new_calls],
        "research_pass": passes,
        "budget": spent,
        "status": RunStatus.RUNNING,
    })
    update.setdefault("audit_trail", []).append(
        audit(state, "researcher", "tool_pass",
              f"executed={len(runnable)} ok={len(evidences)} "
              f"proposed={len(new_calls)} left={left}",
              tools=[c.tool for c in runnable],
              provider=(result.provider if result else "none")))
    return update


# --------------------------------------------------------------------------- #
# analyst
# --------------------------------------------------------------------------- #


@guarded("analyst")
def analyst(state: SigmaState) -> dict[str, Any]:
    """Convert evidence into quantified findings and a draft brief."""
    budget: BudgetState = _require(state, "budget")
    new_findings = reasoning.build_findings(state)
    # Reason over every live finding, not just the ones derived in this pass:
    # on a revision most evidence is already cited (and therefore not re-derived).
    known = {f.finding_id for f in active_findings(state)}
    findings = active_findings(state) + [f for f in new_findings
                                         if f.finding_id not in known]
    result = structured(
        state,
        role="analyst",
        response_model=DraftBrief,
        fallback=lambda: reasoning.build_draft(state, findings),
        new_findings=findings,
    )
    draft: DraftBrief = result.value
    spent = consume_budget(budget, tokens=result.tokens or 400)
    update = dict(result.update)
    update.update({
        "findings": new_findings,
        "draft": draft,
        "budget": spent,
        "status": RunStatus.RUNNING,
    })
    update.setdefault("audit_trail", []).append(
        audit(state, "analyst", "synthesised",
              f"findings={len(findings)} recommendations={len(draft.recommendations)}",
              provider=result.provider))
    return update


# --------------------------------------------------------------------------- #
# critic (evaluator)
# --------------------------------------------------------------------------- #


@guarded("critic")
def critic(state: SigmaState) -> dict[str, Any]:
    """Score the draft and enumerate the gaps that drive the macro-cycle."""
    request: RequestPayload = _require(state, "request")
    budget: BudgetState = _require(state, "budget")
    result = structured(
        state,
        role="critic",
        response_model=Evaluation,
        fallback=lambda: reasoning.critique(
            state, threshold=request.acceptance_threshold),
        threshold=request.acceptance_threshold,
    )
    evaluation: Evaluation = result.value
    # One revision is charged per self-correction cycle: this is the single
    # counter that guarantees the macro-cycle terminates (PRD NFR-4).
    spent = consume_budget(budget,
                           revisions=1 if evaluation.verdict == "revise" else 0,
                           tokens=result.tokens or 300)
    update = dict(result.update)
    update.update({
        "evaluation": [evaluation],
        "budget": spent,
        "status": RunStatus.RUNNING,
    })
    update.setdefault("audit_trail", []).append(
        audit(state, "critic", "evaluated",
              f"score={evaluation.score:.3f} verdict={evaluation.verdict} "
              f"gaps={len(evaluation.gaps)}",
              dominant=evaluation.dominant_gap or "", provider=result.provider))
    return update


# --------------------------------------------------------------------------- #
# finalizer & containment
# --------------------------------------------------------------------------- #


@guarded("finalizer")
def finalizer(state: SigmaState) -> dict[str, Any]:
    """Emit the deliverable brief."""
    brief = reasoning.build_final_brief(state, RunStatus.COMPLETED)
    return {
        "brief": brief,
        "status": RunStatus.COMPLETED,
        "audit_trail": [audit(state, "finalizer", "brief_emitted",
                              f"score={brief.score:.3f} findings={len(brief.findings)}")],
        "messages": [message("assistant", f"Brief final: {brief.headline}"[:400], kind="final")],
    }


@guarded("human_gate")
def human_gate(state: SigmaState) -> dict[str, Any]:
    """Pause the run and ask a human to approve, revise or reject (PRD section 8).

    `interrupt` raises on the first pass, so the run stops *inside* the node and
    resumes from the start of this node with the human's answer. Because the whole
    transcript lives in the checkpointed `messages` channel, the replay sees the
    same context — the human is answering the same question the agent asked.
    """
    evaluation = latest_evaluation(state)
    draft: DraftBrief | None = state.get("draft")
    request: RequestPayload | None = state.get("request")

    payload = {
        "kind": "approval_request",
        "session_id": state.get("session_id", ""),
        "question": (request.question if request else ""),
        "score": round(evaluation.score, 4) if evaluation else 0.0,
        "threshold": request.acceptance_threshold if request else None,
        "rubric": evaluation.scores.model_dump() if evaluation else {},
        "headline": draft.headline if draft else "",
        "summary": draft.summary if draft else "",
        "recommendations": draft.recommendations if draft else [],
        "open_gaps": [g.model_dump() for g in (evaluation.gaps if evaluation else [])],
        "prompt": "Setujui brief ini (approve), minta revisi (revise), atau tolak (reject)?",
    }
    try:
        decision = interrupt(payload, response_schema=HumanDecision)
    except ValidationError as exc:
        # A malformed human reply is a caller error, not a run failure: bubble it
        # up so the API can answer 400 and the session stays paused at the gate.
        raise InvalidResume(_compact_errors(exc)) from exc

    return {
        "human_decision": decision,
        "status": RunStatus.RUNNING,
        "audit_trail": [audit(state, "human_gate", "human_decision",
                              f"decision={decision.decision} reviewer={decision.reviewer}",
                              note=decision.note[:200])],
        "messages": [message("human",
                             f"[{decision.reviewer}] {decision.decision}: {decision.note}"[:400],
                             kind="hitl")],
    }


@guarded("containment")
def containment(state: SigmaState) -> dict[str, Any]:
    """Fallback: produce an honest degraded brief instead of an opaque failure (US-7)."""
    errors = list(state.get("errors") or [])
    fatal = any(e.fatal for e in errors)
    decision = state.get("human_decision")
    if decision is not None and decision.decision == "reject":
        status = RunStatus.REJECTED
    else:
        status = RunStatus.FAILED if fatal else RunStatus.DEGRADED

    brief = reasoning.build_final_brief(state, status)
    caveats = [f"{e.node}/{e.kind}: {e.message}" for e in errors[:5]]
    brief.caveats = (caveats + brief.caveats)[:6]
    if fatal:
        brief.headline = "Analisis dihentikan: kegagalan fatal pada pipeline agen."
    elif status is RunStatus.REJECTED:
        brief.headline = f"Brief ditolak reviewer: {decision.note[:120] if decision else ''}"
    else:
        brief.headline = f"Brief terdegradasi (sebagian tool gagal): {brief.headline}"

    return {
        "brief": brief,
        "status": status,
        "audit_trail": [audit(state, "containment", "contained",
                              f"status={status.value} errors={len(errors)}")],
    }
