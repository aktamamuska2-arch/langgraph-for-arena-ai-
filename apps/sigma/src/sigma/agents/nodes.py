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

from sigma.agents import reasoning
from sigma.config import SETTINGS
from sigma.errors import BudgetExhausted, SigmaError, ToolTimeoutError
from sigma.state import (
    AuditEvent,
    BudgetState,
    ErrorRecord,
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
            update.setdefault("audit_trail", []).append(
                AuditEvent(seq=_next_seq(state, update), node=name, kind="completed",
                           message=f"{name} finished",
                           data={"ms": round((time.perf_counter() - started) * 1000, 1)}))
            return update

        wrapper.__name__ = name
        wrapper.node_name = name  # type: ignore[attr-defined]
        return wrapper

    return decorator


def _next_seq(state: SigmaState, update: dict[str, Any]) -> int:
    """Next audit sequence number, accounting for events added by this node.

    `audit()` numbers from the incoming state, so a node that appends two events
    (its own plus the guard's) would otherwise emit a duplicate sequence number.
    """
    sequences = [e.seq for e in (state.get("audit_trail") or [])]
    sequences += [e.seq for e in (update.get("audit_trail") or [])]
    return (max(sequences) + 1) if sequences else 1


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
    plan = reasoning.plan_questions(
        request, scenario, revision=revision, directive=directive,
        existing=previous.sub_questions if previous else [],
    )
    spent = consume_budget(budget, tokens=250)  # revisions are charged by the critic

    return {
        "plan": plan,
        "budget": spent,
        "status": RunStatus.RUNNING,
        "audit_trail": [audit(state, "planner", "planned",
                              f"revision={plan.revision} questions={len(plan.sub_questions)}",
                              directive=directive or "")],
        "messages": [message("assistant",
                             f"Rencana rev-{plan.revision}: {len(plan.sub_questions)} sub-pertanyaan. "
                             f"{plan.hypothesis}"[:600], kind="plan")],
    }


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
    new_calls = reasoning.propose_tool_calls(merged, budget_left=max(0, left)) if left > 0 else []

    spent = consume_budget(budget, tools=len(runnable), tokens=180 * max(1, len(runnable)))
    return {
        "tool_calls": records,
        "evidence": evidences,
        "errors": errors,
        "pending_tools": new_calls,
        "queued_call_ids": [c.call_id for c in new_calls],
        "research_pass": int(state.get("research_pass") or 0) + 1,
        "budget": spent,
        "status": RunStatus.RUNNING,
        "audit_trail": [audit(state, "researcher", "tool_pass",
                              f"executed={len(runnable)} ok={len(evidences)} "
                              f"proposed={len(new_calls)} left={left}",
                              tools=[c.tool for c in runnable])],
        "messages": [message("assistant",
                             f"Mengumpulkan bukti: {len(evidences)} baru dari "
                             f"{[c.tool for c in runnable]}."[:400], kind="research")],
    }


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
    draft = reasoning.build_draft(state, findings)
    spent = consume_budget(budget, tokens=400)
    return {
        "findings": findings,
        "draft": draft,
        "budget": spent,
        "status": RunStatus.RUNNING,
        "audit_trail": [audit(state, "analyst", "synthesised",
                              f"findings={len(findings)} recommendations={len(draft.recommendations)}")],
        "messages": [message("assistant", draft.summary[:600], kind="analysis")],
    }


# --------------------------------------------------------------------------- #
# critic (evaluator)
# --------------------------------------------------------------------------- #


@guarded("critic")
def critic(state: SigmaState) -> dict[str, Any]:
    """Score the draft and enumerate the gaps that drive the macro-cycle."""
    request: RequestPayload = _require(state, "request")
    budget: BudgetState = _require(state, "budget")
    evaluation = reasoning.critique(state, threshold=request.acceptance_threshold)
    # One revision is charged per self-correction cycle: this is the single
    # counter that guarantees the macro-cycle terminates (PRD NFR-4).
    spent = consume_budget(budget,
                           revisions=1 if evaluation.verdict == "revise" else 0,
                           tokens=300)
    return {
        "evaluation": [evaluation],
        "budget": spent,
        "status": RunStatus.RUNNING,
        "audit_trail": [audit(state, "critic", "evaluated",
                              f"score={evaluation.score:.3f} verdict={evaluation.verdict} "
                              f"gaps={len(evaluation.gaps)}",
                              dominant=evaluation.dominant_gap or "")],
        "messages": [message("assistant", evaluation.rationale[:600], kind="critique")],
    }


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
