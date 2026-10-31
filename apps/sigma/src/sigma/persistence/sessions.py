"""Session discovery and database maintenance on top of the checkpointer tables.

The checkpointer owns the schema; this module only reads it for operational
concerns (listing sessions, sizing, vacuuming) and never writes checkpoints.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Iterator

from langgraph.checkpoint.sqlite import SqliteSaver

from sigma.config import SETTINGS
from sigma.graph.checkpointer import database_size_bytes, get_connection


def iter_thread_ids(limit: int = 50) -> Iterator[str]:
    """Distinct `thread_id`s in the checkpoint store, most recently touched first."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints "
            "WHERE checkpoint_ns = '' ORDER BY rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()
    except sqlite3.Error:
        return
    for (thread_id,) in rows:
        yield thread_id


def checkpoint_count(thread_id: str | None = None) -> int:
    conn = get_connection()
    try:
        if thread_id:
            row = conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (thread_id,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()
    except sqlite3.Error:
        return 0
    return int(row[0]) if row else 0


def vacuum_if_large(max_bytes: int = 32 * 1024 * 1024, *, saver: SqliteSaver | None = None) -> bool:
    """Compact the database when it grows past `max_bytes` (PRD section 6)."""
    if database_size_bytes() <= max_bytes:
        return False
    conn = get_connection()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("VACUUM")
    return True


def session_summaries(runner: Any, limit: int = 25) -> list[dict[str, Any]]:
    """Operational view of every stored session."""
    from sigma.api.serialization import to_jsonable
    from sigma.state import RunStatus

    summaries: list[dict[str, Any]] = []
    for thread_id in iter_thread_ids(limit * 3):
        snapshot = runner.state(thread_id)
        if snapshot is None or not snapshot.values:
            continue
        values = dict(snapshot.values or {})
        request = values.get("request")
        evaluation = values.get("evaluation") or []
        status = values.get("status", RunStatus.RUNNING)
        status_value = status.value if isinstance(status, RunStatus) else str(status)
        # A paused run reports `running` in its state (an interrupt writes
        # nothing), so the awaited gate has to be read from the snapshot itself.
        pending_gate = bool(snapshot.next) and any(
            getattr(task, "interrupts", None) for task in (snapshot.tasks or ()))
        if pending_gate:
            status_value = RunStatus.AWAITING_HUMAN.value
        summaries.append({
            "session_id": thread_id,
            "status": status_value,
            "question": getattr(request, "question", None),
            "score": (evaluation[-1].score if evaluation else None),
            "revision": getattr(values.get("plan"), "revision", 0),
            "evidence": len(values.get("evidence") or []),
            "created_at": getattr(request, "created_at", None),
            "awaiting_human": pending_gate,
            "checkpoints": checkpoint_count(thread_id),
            "values": to_jsonable(values),
        })
    summaries.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return summaries[:limit]


__all__ = ["checkpoint_count", "iter_thread_ids", "session_summaries", "vacuum_if_large"]

# Re-exported for convenience in scripts.
SETTINGS_REF = SETTINGS
