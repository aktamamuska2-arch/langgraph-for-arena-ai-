"""FastAPI surface for SIGMA.

The API is a thin, stateless shell over the compiled graph: it owns thread ids,
validation and serialisation — never orchestration. Every mutating call is one
`SessionRunner` invocation, so the graph remains the single source of truth.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from sigma.api.schemas import (CreateSessionRequest, ForkRequest, HealthResponse,
                               ResumeRequest)
from sigma.api.serialization import (audit_view, session_view, timeline_view,
                                     to_jsonable)
from sigma.config import SETTINGS
from sigma.errors import InvalidResume
from sigma.graph.checkpointer import database_size_bytes
from sigma.llm.factory import provider_name
from sigma.persistence.sessions import session_summaries
from sigma.runtime import RunResult, SessionRunner, get_runner
from sigma.size import audit as size_audit
from sigma.state import HumanDecision, RunStatus

logger = logging.getLogger("sigma.api")
STATIC_DIR = Path(__file__).parent / "static"


def _runner(hitl: bool = True) -> SessionRunner:
    return get_runner(hitl=hitl)


def _summary(result: RunResult, state: dict[str, Any]) -> dict[str, Any]:
    view = session_view(state)
    # The runtime classification wins over the graph state: while a run is paused
    # inside `human_gate` the persisted status is still `running`, because an
    # interrupt writes nothing.
    view["status"] = result.status.value
    view["interrupts"] = result.interrupts
    view["interrupted"] = result.interrupted
    view["error"] = result.error
    return view


def _load(runner: SessionRunner, session_id: str) -> dict[str, Any]:
    snapshot = runner.state(session_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"unknown session {session_id}")
    return dict(snapshot.values or {})


def create_app(*, runner: SessionRunner | None = None, hitl: bool = True) -> FastAPI:
    """Build the ASGI application."""
    app = FastAPI(title="SIGMA — Strategic Intelligence Graph-based Multi-Agent Analytics",
                  version="1.0.0",
                  description="Interactive multi-agent analytics platform on LangGraph.")
    app.state.hitl = hitl
    app.state.runner = runner

    def active_runner(use_hitl: bool | None = None) -> SessionRunner:
        """Pick the runner matching the requested mode.

        Resuming a run that is paused at `human_gate` with a graph that has no
        such node silently does nothing, so the mode has to be explicit — and it
        defaults to the application's mode, never to "off".
        """
        want = hitl if use_hitl is None else use_hitl
        if app.state.runner is not None and want == hitl:
            return app.state.runner
        return get_runner(hitl=want)

    # ------------------------------------------------------------------ meta #

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        from importlib import metadata

        try:
            langgraph_version = metadata.version("langgraph")
        except metadata.PackageNotFoundError:
            langgraph_version = "vendored"

        report = size_audit.measure(include_dependencies=True)
        return HealthResponse(
            langgraph=langgraph_version,
            provider=provider_name(),
            hitl=hitl,
            budget_mb=report.budget_mb,
            artifact_mb=report.total_mb,
            checkpoint_db_bytes=database_size_bytes(),
        )

    @app.get("/api/tools")
    def tools() -> JSONResponse:
        from sigma.tools.registry import ToolRegistry

        return JSONResponse({"tools": ToolRegistry().schemas()})

    # --------------------------------------------------------------- sessions #

    # --------------------------------------------------------------- streaming #

    @app.get("/api/sessions/stream")
    def stream_session(
        question: str = Query(min_length=3, max_length=600),
        seed: int = 7,
        portfolio_id: str = "sigma-core",
        acceptance_threshold: float | None = None,
        hitl: bool = True,
    ) -> StreamingResponse:
        """Run a session and stream node-by-node updates as server-sent events."""
        runner = active_runner(hitl)
        session_id = runner.new_session_id()

        def events() -> Iterator[str]:
            from sigma.state import RequestPayload

            initial = {"session_id": session_id,
                       "request": RequestPayload(question=question, seed=seed,
                                                 portfolio_id=portfolio_id,
                                                 acceptance_threshold=(
                                                     SETTINGS.acceptance_threshold
                                                     if acceptance_threshold is None
                                                     else acceptance_threshold))}
            yield _sse("start", {"session_id": session_id, "hitl": hitl})
            try:
                for chunk in runner.graph.stream(initial, runner.config(session_id),
                                                 stream_mode="updates"):
                    for node, update in chunk.items():
                        yield _sse("node", {"node": node,
                                            "update": _digest(node, update)})
            except Exception as exc:  # noqa: BLE001 - reported to the client
                yield _sse("error", {"message": str(exc)})
            snapshot = runner.state(session_id)
            values = dict(snapshot.values or {}) if snapshot else {}
            view = session_view(values)
            if snapshot is not None and snapshot.next:
                yield _sse("interrupt", {"pending": list(snapshot.next),
                                         "interrupts": _interrupts_of(snapshot),
                                         "state": view})
            else:
                yield _sse("final", {"state": view})

        return StreamingResponse(events(), media_type="text/event-stream",
                                headers={"Cache-Control": "no-cache",
                                         "X-Accel-Buffering": "no"})

    @app.get("/api/sessions")
    def list_sessions(limit: int = Query(default=20, ge=1, le=100)) -> JSONResponse:
        summaries = session_summaries(active_runner(), limit=limit)
        return JSONResponse({"sessions": [{k: v for k, v in item.items() if k != "values"}
                                          for item in summaries]})

    @app.post("/api/sessions", status_code=201)
    def create_session(payload: CreateSessionRequest) -> JSONResponse:
        runner = active_runner(payload.hitl)
        result = runner.start(
            question=payload.question,
            portfolio_id=payload.portfolio_id,
            seed=payload.seed,
            max_revisions=payload.max_revisions,
            acceptance_threshold=payload.acceptance_threshold,
        )
        return JSONResponse(_summary(result, result.state), status_code=201)

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> JSONResponse:
        runner = active_runner()
        state = _load(runner, session_id)
        view = session_view(state)
        view["audit_trail"] = audit_view(state)
        return JSONResponse(view)

    @app.post("/api/sessions/{session_id}/resume")
    def resume_session(session_id: str, payload: ResumeRequest) -> JSONResponse:
        runner = active_runner()
        if runner.state(session_id) is None:
            raise HTTPException(status_code=404, detail=f"unknown session {session_id}")
        try:
            result = runner.resume(session_id, HumanDecision(decision=payload.decision,
                                                             note=payload.note,
                                                             reviewer=payload.reviewer))
        except InvalidResume as exc:
            raise HTTPException(status_code=400, detail=f"invalid decision: {exc}") from exc
        return JSONResponse(_summary(result, result.state))

    @app.post("/api/sessions/{session_id}/continue")
    def continue_session(session_id: str) -> JSONResponse:
        """Re-enter an unfinished run (crash recovery, US-5)."""
        runner = active_runner()
        if runner.state(session_id) is None:
            raise HTTPException(status_code=404, detail=f"unknown session {session_id}")
        result = runner.continue_run(session_id)
        return JSONResponse(_summary(result, result.state))

    # ------------------------------------------------------------ time travel #

    @app.get("/api/sessions/{session_id}/timeline")
    def timeline(session_id: str) -> JSONResponse:
        runner = active_runner()
        if runner.state(session_id) is None:
            raise HTTPException(status_code=404, detail=f"unknown session {session_id}")
        return JSONResponse({"session_id": session_id,
                             "checkpoints": timeline_view(runner.history(session_id))})

    @app.get("/api/sessions/{session_id}/timeline/{checkpoint_id}")
    def state_at(session_id: str, checkpoint_id: str) -> JSONResponse:
        runner = active_runner()
        snapshot = runner.state_at(session_id, checkpoint_id)
        if snapshot is None:
            raise HTTPException(status_code=404,
                                detail=f"unknown checkpoint {checkpoint_id}")
        return JSONResponse({"session_id": session_id,
                             "checkpoint_id": checkpoint_id,
                             "next": list(snapshot.next or []),
                             "state": session_view(dict(snapshot.values or {}))})

    @app.post("/api/sessions/{session_id}/fork")
    def fork_session(session_id: str, payload: ForkRequest) -> JSONResponse:
        """Branch a new session from an old checkpoint ('what if?', US-6)."""
        runner = active_runner()
        snapshot = runner.state_at(session_id, payload.checkpoint_id)
        if snapshot is None:
            raise HTTPException(status_code=404,
                                detail=f"unknown checkpoint {payload.checkpoint_id}")
        try:
            result = runner.fork_from(session_id, payload.checkpoint_id,
                                      patch=payload.patch)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(_summary(result, result.state))

    # ------------------------------------------------------------------ static #

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(to_jsonable(data), default=str)}\n\n"


def _digest(node: str, update: dict[str, Any]) -> dict[str, Any]:
    """A compact, UI-friendly digest of one node's state update."""
    out: dict[str, Any] = {"node": node}
    if not isinstance(update, dict):
        return out
    for key, value in update.items():
        if key == "messages":
            out["messages"] = [m.get("content", "")[:160] for m in value][-2:]
        elif key == "audit_trail":
            out["events"] = [f"{e.node}: {e.message}" for e in value][-3:]
        elif isinstance(value, list):
            out[key] = len(value)
        elif isinstance(value, dict):
            out[key] = list(value)[:6]
        else:
            out[key] = to_jsonable(value)
    return out


def _interrupts_of(snapshot: Any) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for task in (snapshot.tasks or ()):
        for item in getattr(task, "interrupts", ()) or ():
            value = getattr(item, "value", item)
            payloads.append({"id": getattr(item, "id", ""),
                             "value": to_jsonable(value)})
    return payloads


__all__ = ["create_app"]
