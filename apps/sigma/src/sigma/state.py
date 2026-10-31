"""Global state schema for the SIGMA graph.

Design contract (PRD section 4):

* `SigmaState` is a `TypedDict`: the thin envelope LangGraph understands.
* Every payload inside it is a **Pydantic** model: strongly typed and validated.
* Every growing field has an **explicit bounded reducer** — no list field relies on
  LangGraph's implicit last-value write, because a cycle would silently swallow
  the evidence collected by earlier iterations (this is what AC-CYCLE-3 checks).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Callable, Literal, TypeVar, TypedDict

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- #
# Reducers
# --------------------------------------------------------------------------- #

T = TypeVar("T", bound=BaseModel)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def bounded_add(cap: int) -> Callable[[list[Any], list[Any]], list[Any]]:
    """Append-only reducer that keeps the newest `cap` items.

    Keeps state bounded on long cycles (PRD section 4.3) while preserving order.
    """

    def _reduce(left: list[Any] | None, right: list[Any] | None) -> list[Any]:
        merged = list(left or []) + list(right or [])
        return merged[-cap:] if len(merged) > cap else merged

    _reduce.__name__ = f"bounded_add_{cap}"
    return _reduce


def merge_unique_by(key: str) -> Callable[[list[Any], list[Any]], list[Any]]:
    """Union by `key`; newer items overwrite older ones with the same key.

    This is what makes a node replay after an `interrupt()` idempotent: the same
    tool call re-proposed on replay does not duplicate work (PRD section 14).
    """

    def _reduce(left: list[Any] | None, right: list[Any] | None) -> list[Any]:
        merged: dict[Any, Any] = {}
        for item in list(left or []) + list(right or []):
            merged[_key_of(item, key)] = item
        return list(merged.values())

    _reduce.__name__ = f"merge_unique_by_{key}"
    return _reduce


def _key_of(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def supersede_by(key: str) -> Callable[[list[T], list[T]], list[T]]:
    """Append-only reducer with explicit supersession.

    An item carrying `supersedes=<id>` deactivates the item it replaces instead of
    deleting it: the audit trail stays complete across self-correction cycles.
    """

    def _reduce(left: list[T] | None, right: list[T] | None) -> list[T]:
        merged: list[T] = list(left or [])
        by_id: dict[Any, T] = {_key_of(i, key): i for i in merged}
        for item in list(right or []):
            merged.append(item)
            by_id[_key_of(item, key)] = item
            target = getattr(item, "supersedes", None)
            if target is not None and target in by_id and by_id[target] is not item:
                if hasattr(by_id[target], "superseded"):
                    setattr(by_id[target], "superseded", True)
        return merged

    _reduce.__name__ = f"supersede_by_{key}"
    return _reduce


def merge_budget(left: BudgetState | None, right: BudgetState | None) -> BudgetState:
    """Merge budgets monotonically: usage never decreases, headroom never grows.

    Guarantees a node replay (or a retry) can never "refund" budget — the property
    that makes the termination guarantee in PRD section 5 sound.
    """
    if left is None:
        return right or BudgetState()
    if right is None:
        return left
    return BudgetState(
        max_revisions=min(left.max_revisions, right.max_revisions),
        revisions_used=max(left.revisions_used, right.revisions_used),
        max_tool_calls=min(left.max_tool_calls, right.max_tool_calls),
        tool_calls_used=max(left.tool_calls_used, right.tool_calls_used),
        max_tokens=min(left.max_tokens, right.max_tokens),
        tokens_used=max(left.tokens_used, right.tokens_used),
    )


def merge_scalar_status(left: RunStatus | None, right: RunStatus | None) -> RunStatus:
    """Status is last-write-wins, but a terminal status is never undone."""
    if left in _TERMINAL and right not in _TERMINAL:
        return left  # type: ignore[return-value]
    return right if right is not None else (left or RunStatus.RUNNING)


# --------------------------------------------------------------------------- #
# Enums & small value objects
# --------------------------------------------------------------------------- #


class RunStatus(str, Enum):
    RUNNING = "running"
    AWAITING_HUMAN = "awaiting_human"
    COMPLETED = "completed"
    DEGRADED = "degraded"
    REJECTED = "rejected"
    FAILED = "failed"


_TERMINAL = {RunStatus.COMPLETED, RunStatus.DEGRADED, RunStatus.REJECTED, RunStatus.FAILED}

GapKind = Literal["structural", "missing_evidence", "interpretation", "numerical"]


class _Model(BaseModel):
    """Shared Pydantic base: frozen-ish, cheap, JSON-serialisable for the checkpointer."""

    model_config = ConfigDict(extra="forbid", validate_assignment=False)


# --------------------------------------------------------------------------- #
# Domain models
# --------------------------------------------------------------------------- #


class RequestPayload(_Model):
    """Normalised user request (written once by `intake`)."""

    question: str
    portfolio_id: str = "sigma-core"
    seed: int = 7
    max_revisions: int = 3
    acceptance_threshold: float = 0.78
    created_at: str = Field(default_factory=_now)


class ScenarioContext(_Model):
    """The deterministic world the agents reason about."""

    portfolio_id: str
    seed: int
    horizon_quarters: int = 8
    venture_count: int = 0
    sectors: list[str] = Field(default_factory=list)
    capital_pool: float = 0.0
    created_at: str = Field(default_factory=_now)


class BudgetState(_Model):
    """Hard termination guarantees (PRD NFR-4)."""

    max_revisions: int = 3
    revisions_used: int = 0
    max_tool_calls: int = 24
    tool_calls_used: int = 0
    max_tokens: int = 120_000
    tokens_used: int = 0

    @property
    def revisions_left(self) -> int:
        return max(0, self.max_revisions - self.revisions_used)

    @property
    def tool_calls_left(self) -> int:
        return max(0, self.max_tool_calls - self.tool_calls_used)

    @property
    def exhausted(self) -> bool:
        return self.revisions_left == 0 or self.tool_calls_left == 0


def consume_budget(budget: BudgetState, *, revisions: int = 0, tools: int = 0, tokens: int = 0) -> BudgetState:
    """Return a copy of `budget` with usage increased (budget is monotonic)."""
    return BudgetState(
        max_revisions=budget.max_revisions,
        revisions_used=budget.revisions_used + revisions,
        max_tool_calls=budget.max_tool_calls,
        tool_calls_used=budget.tool_calls_used + tools,
        max_tokens=budget.max_tokens,
        tokens_used=budget.tokens_used + tokens,
    )


class SubQuestion(_Model):
    """One decomposed analytical question produced by the planner."""

    id: str
    question: str
    tool_hint: str
    status: Literal["pending", "answered", "dropped"] = "pending"
    rationale: str = ""
    priority: float = 1.0


class PlanState(_Model):
    """The planner's current plan. `revision` grows on every self-correction cycle."""

    revision: int = 0
    hypothesis: str = ""
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    directive: str | None = None  # injected by HITL "revise"


class ToolCall(_Model):
    """A proposed tool invocation (queued by the researcher)."""

    call_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    sub_question_id: str | None = None


class ToolCallRecord(ToolCall):
    """An executed tool invocation, including its outcome."""

    ok: bool = True
    duration_ms: int = 0
    error: str | None = None
    output_digest: dict[str, Any] = Field(default_factory=dict)


class Evidence(_Model):
    """A tool result that can be cited by a finding (provenance is mandatory)."""

    evidence_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    tool: str
    sub_question_id: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    provenance: str = ""
    retrieved_at: str = Field(default_factory=_now)


class Finding(_Model):
    """A quantified, evidence-backed claim."""

    finding_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    sub_question_id: str | None = None
    claim: str = ""
    metric: float | None = None
    unit: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    supersedes: str | None = None
    superseded: bool = False

    @property
    def active(self) -> bool:
        return not self.superseded

    @property
    def grounded(self) -> bool:
        """PRD NFR-7: a numeric claim without evidence is not a finding."""
        return bool(self.evidence_ids) or self.metric is None


class DraftBrief(_Model):
    """The analyst's draft that the critic evaluates."""

    revision: int = 0
    headline: str = ""
    summary: str = ""
    recommendations: list[str] = Field(default_factory=list)
    finding_ids: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class RubricScores(_Model):
    """The 5 rubric dimensions from PRD FR-4."""

    coverage: float = 0.0
    evidence_density: float = 0.0
    numeric_grounding: float = 0.0
    actionability: float = 0.0
    calibration: float = 0.0

    @property
    def mean(self) -> float:
        values = [self.coverage, self.evidence_density, self.numeric_grounding,
                  self.actionability, self.calibration]
        return round(sum(values) / len(values), 4)


class Gap(_Model):
    """A concrete deficiency. `kind` drives the cyclic routing decision."""

    kind: GapKind
    detail: str = ""
    sub_question_id: str | None = None
    weight: float = 1.0


class Evaluation(_Model):
    """The critic's verdict for one revision."""

    revision: int = 0
    score: float = 0.0
    scores: RubricScores = Field(default_factory=RubricScores)
    gaps: list[Gap] = Field(default_factory=list)
    verdict: Literal["accept", "revise"] = "revise"
    rationale: str = ""

    @property
    def dominant_gap(self) -> GapKind | None:
        """The weighted-dominant gap class -> selects the cycle target."""
        if not self.gaps:
            return None
        totals: dict[str, float] = {}
        for gap in self.gaps:
            totals[gap.kind] = totals.get(gap.kind, 0.0) + gap.weight
        return max(totals.items(), key=lambda kv: (kv[1], kv[0]))[0]


class HumanDecision(_Model):
    """Typed HITL payload used as `interrupt(response_schema=...)` (PRD section 8)."""

    decision: Literal["approve", "revise", "reject"] = "approve"
    note: str = ""
    reviewer: str = "human"


class FinalBrief(_Model):
    """The deliverable."""

    session_id: str
    status: RunStatus = RunStatus.COMPLETED
    headline: str = ""
    summary: str = ""
    recommendations: list[str] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    score: float = 0.0
    revisions: int = 0
    caveats: list[str] = Field(default_factory=list)
    evidence_count: int = 0
    provenance: list[str] = Field(default_factory=list)
    generated_at: str = Field(default_factory=_now)


class ErrorRecord(_Model):
    """A classified failure (PRD section 9)."""

    node: str
    kind: str
    message: str
    fatal: bool = False
    at: str = Field(default_factory=_now)


class AuditEvent(_Model):
    """Append-only audit trail entry (also used as the LLM memory compression sink)."""

    seq: int = 0
    node: str = ""
    kind: str = ""
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    at: str = Field(default_factory=_now)


# --------------------------------------------------------------------------- #
# The graph envelope
# --------------------------------------------------------------------------- #


class SigmaState(TypedDict, total=False):
    """Global state of one analytical session.

    Fields are grouped by lifecycle stage; see PRD section 4.2 for the table.
    """

    # identity & control
    session_id: str
    request: RequestPayload
    scenario: ScenarioContext
    budget: Annotated[BudgetState, merge_budget]

    # planning
    plan: PlanState

    # execution
    pending_tools: Annotated[list[ToolCall], merge_unique_by("call_id")]
    queued_call_ids: list[str]  # ids proposed by the last researcher pass (last-value)
    research_pass: int  # how many gathering passes have run -> query depth
    tool_calls: Annotated[list[ToolCallRecord], bounded_add(200)]
    evidence: Annotated[list[Evidence], bounded_add(200)]

    # reasoning
    messages: Annotated[list[dict[str, Any]], bounded_add(120)]
    findings: Annotated[list[Finding], supersede_by("finding_id")]
    draft: DraftBrief | None

    # evaluation & HITL
    evaluation: Annotated[list[Evaluation], bounded_add(12)]
    human_decision: HumanDecision | None

    # output & failure
    brief: FinalBrief | None
    errors: Annotated[list[ErrorRecord], bounded_add(50)]
    audit_trail: Annotated[list[AuditEvent], bounded_add(400)]
    status: Annotated[RunStatus, merge_scalar_status]


# --------------------------------------------------------------------------- #
# Helpers used by nodes
# --------------------------------------------------------------------------- #


def new_session_id() -> str:
    return uuid.uuid4().hex[:16]


def audit(state: SigmaState, node: str, kind: str, message: str, **data: Any) -> AuditEvent:
    """Build an audit event with a monotonically increasing sequence number."""
    seq = (state.get("audit_trail") or [])[-1].seq + 1 if state.get("audit_trail") else 1
    return AuditEvent(seq=seq, node=node, kind=kind, message=message, data=data or {})


def message(role: str, content: str, **extra: Any) -> dict[str, Any]:
    """Build an LLM transcript entry (plain dicts: cheap to serialise/stream)."""
    return {"role": role, "content": content, **extra}


def latest_evaluation(state: SigmaState) -> Evaluation | None:
    items = [e for e in (state.get("evaluation") or [])]
    return items[-1] if items else None


def active_findings(state: SigmaState) -> list[Finding]:
    return [f for f in (state.get("findings") or []) if f.active]


def evidence_index(state: SigmaState) -> dict[str, Evidence]:
    return {e.evidence_id: e for e in (state.get("evidence") or [])}


__all__ = [
    "AuditEvent",
    "BudgetState",
    "DraftBrief",
    "ErrorRecord",
    "Evaluation",
    "Evidence",
    "FinalBrief",
    "Finding",
    "Gap",
    "GapKind",
    "HumanDecision",
    "PlanState",
    "RequestPayload",
    "RubricScores",
    "RunStatus",
    "ScenarioContext",
    "SigmaState",
    "SubQuestion",
    "ToolCall",
    "ToolCallRecord",
    "active_findings",
    "audit",
    "bounded_add",
    "consume_budget",
    "evidence_index",
    "latest_evaluation",
    "merge_budget",
    "merge_unique_by",
    "message",
    "new_session_id",
    "supersede_by",
]
