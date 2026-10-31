"""Transactional memory for the LLM transcript.

The conversation lives in the `messages` channel, so it is checkpointed together
with everything else: a replay after an interrupt sees exactly the same
transcript the first attempt saw. That is what "transactional memory" buys us —
there is no second, hidden copy of the context (PRD section 7.2).

Because cycles can be long, the transcript is compressed: older turns are folded
into an `AuditEvent` summary instead of being dropped, so the audit trail gets
*richer* as the context window shrinks.
"""

from __future__ import annotations

from typing import Any

from sigma.config import SETTINGS
from sigma.state import AuditEvent, SigmaState


def window(state: SigmaState, limit: int | None = None) -> list[dict[str, Any]]:
    """The transcript slice handed to the model."""
    size = limit or SETTINGS.context_window
    messages = list(state.get("messages") or [])
    return messages[-size:]


def append(state: SigmaState, *messages: dict[str, Any]) -> dict[str, Any]:
    """Append transcript entries, compressing when the window is exceeded.

    Returns a partial state update (messages + optional audit event).
    """
    existing = list(state.get("messages") or [])
    merged = existing + list(messages)
    update: dict[str, Any] = {"messages": merged}
    overflow = len(merged) - SETTINGS.context_window
    if overflow > 0:
        dropped = merged[:overflow]
        kept = merged[overflow:]
        update["messages"] = kept
        update["audit_trail"] = [_summarise(state, dropped, len(merged))]
    return update


def _summarise(state: SigmaState, dropped: list[dict[str, Any]], total: int) -> AuditEvent:
    seq = (state.get("audit_trail") or [])[-1].seq + 1 if state.get("audit_trail") else 1
    by_kind: dict[str, int] = {}
    for item in dropped:
        kind = item.get("kind", "message")
        by_kind[kind] = by_kind.get(kind, 0) + 1
    head = (dropped[0].get("content", "") or "")[:80]
    return AuditEvent(
        seq=seq,
        node="memory",
        kind="context_compacted",
        message=f"{len(dropped)} entri diringkas dari {total} (window="
                f"{SETTINGS.context_window}); oldest: {head!r}",
        data={"dropped_by_kind": by_kind},
    )


__all__ = ["append", "window"]
