"""SQLite checkpointer: durable state, time-travel debugging, session recovery.

A single long-lived connection per process (`check_same_thread=False` is safe here:
`SqliteSaver` serialises access with its own lock, PRD section 6).
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from sigma import state as sigma_state
from sigma.config import SETTINGS


def sigma_serde() -> JsonPlusSerializer:
    """A serializer that knows SIGMA's Pydantic types.

    Registering them explicitly keeps checkpoint round-trips warning-free and,
    more importantly, keeps working under `LANGGRAPH_STRICT_MSGPACK=true`
    (unregistered types would be blocked outright).
    """
    types = [
        obj for obj in vars(sigma_state).values()
        if isinstance(obj, type) and issubclass(obj, sigma_state._Model)
    ]
    types.append(sigma_state.RunStatus)
    return JsonPlusSerializer(
        allowed_json_modules=[("sigma", "state", t.__name__) for t in types],
        allowed_msgpack_modules=types,
    )

_lock = threading.Lock()
_connection: sqlite3.Connection | None = None
_saver: SqliteSaver | None = None


def get_connection(path: str | Path | None = None) -> sqlite3.Connection:
    """Return the process-wide SQLite connection (created on first use)."""
    global _connection
    with _lock:
        if _connection is None:
            target = str(path or SETTINGS.db_path)
            if target != ":memory:":
                Path(target).parent.mkdir(parents=True, exist_ok=True)
            _connection = sqlite3.connect(target, check_same_thread=False)
            _connection.execute("PRAGMA journal_mode=WAL")
            _connection.execute("PRAGMA synchronous=NORMAL")
            _connection.execute(f"PRAGMA busy_timeout={SETTINGS.db_busy_timeout_ms}")
        return _connection


def get_saver(path: str | Path | None = None) -> SqliteSaver:
    """Return a set-up `SqliteSaver` backed by the process-wide connection."""
    global _saver
    with _lock:
        if _saver is None:
            _saver = SqliteSaver(get_connection(path), serde=sigma_serde())
            _saver.setup()
        return _saver


def in_memory_saver() -> SqliteSaver:
    """A private saver for tests: isolated, in-memory, disposable."""
    saver = SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False),
                        serde=sigma_serde())
    saver.setup()
    return saver


def reset_cache() -> None:
    """Close and forget the process-wide connection (used by tests)."""
    global _connection, _saver
    with _lock:
        if _connection is not None:
            try:
                _connection.close()
            except sqlite3.Error:  # pragma: no cover - best effort
                pass
        _connection = None
        _saver = None


def database_size_bytes(path: str | Path | None = None) -> int:
    """On-disk size of the checkpoint database (0 for in-memory)."""
    target = Path(str(path or SETTINGS.db_path))
    if str(target) == ":memory:" or not target.exists():
        return 0
    total = target.stat().st_size
    for suffix in ("-wal", "-shm"):
        side = target.with_name(target.name + suffix)
        if side.exists():
            total += side.stat().st_size
    return total
