"""JSON views over graph state.

The wire format is deliberately explicit rather than `model_dump()` on the whole
state: the UI needs a stable shape, and the checkpoint payloads can be large.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


def to_jsonable(value: Any, *, depth: int = 0) -> Any:
    """Convert Pydantic models / enums / nested containers into JSON primitives."""
    if depth > 12:
        return str(value)[:200]
    if isinstance(value, BaseModel):
        return to_jsonable(value.model_dump(), depth=depth + 1)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v, depth=depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v, depth=depth + 1) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _status(state: dict[str, Any]) -> str:
    status = state.get("status")
    return status.value if isinstance(status, Enum) else (status or "unknown")


def session_view(state: dict[str, Any]) -> dict[str, Any]:
    """The compact view used by the UI and the session list."""
    plan = state.get("plan")
    evaluation = (state.get("evaluation") or [])
    latest = evaluation[-1] if evaluation else None
    brief = state.get("brief")
    request = state.get("request")
    budget = state.get("budget")

    return {
        "session_id": state.get("session_id"),
        "status": _status(state),
        "question": getattr(request, "question", None),
        "seed": getattr(request, "seed", None),
        "portfolio_id": getattr(request, "portfolio_id", None),
        "revision": getattr(plan, "revision", 0),
        "sub_questions": [to_jsonable(q) for q in getattr(plan, "sub_questions", []) or []],
        "evidence_count": len(state.get("evidence") or []),
        "tool_calls": [to_jsonable(c) for c in (state.get("tool_calls") or [])][-25:],
        "findings": [to_jsonable(f) for f in (state.get("findings") or []) if not f.superseded],
        "evaluation": to_jsonable(latest),
        "evaluations": [to_jsonable(e) for e in evaluation],
        "budget": to_jsonable(budget),
        "brief": to_jsonable(brief),
        "errors": [to_jsonable(e) for e in (state.get("errors") or [])],
    }


def audit_view(state: dict[str, Any], limit: int = 200) -> list[dict[str, Any]]:
    """The execution timeline rendered by the UI."""
    events = state.get("audit_trail") or []
    return [to_jsonable(e) for e in events[-limit:]]


def timeline_view(history: list[Any]) -> list[dict[str, Any]]:
    """Checkpoint history (time-travel) view, newest first."""
    items: list[dict[str, Any]] = []
    for snapshot in history:
        values = dict(snapshot.values or {})
        plan = values.get("plan")
        evaluation = values.get("evaluation") or []
        items.append({
            "checkpoint_id": snapshot.config["configurable"].get("checkpoint_id"),
            "next": list(snapshot.next or []),
            "step": snapshot.metadata.get("step") if snapshot.metadata else None,
            "node": (snapshot.metadata.get("langgraph_node")
                     if snapshot.metadata else None),
            "revision": getattr(plan, "revision", 0),
            "score": (evaluation[-1].score if evaluation else None),
            "status": _status(values),
            "evidence": len(values.get("evidence") or []),
            # The very first checkpoint of a thread is written before `intake`
            # runs, so it holds no request and cannot seed a branch.
            "forkable": bool(values.get("request")),
        })
    return items


__all__ = ["audit_view", "session_view", "timeline_view", "to_jsonable"]
