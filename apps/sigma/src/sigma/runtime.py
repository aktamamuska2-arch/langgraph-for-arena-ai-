"""Session runtime: the only place that talks to the compiled graph directly.

Everything above this layer (CLI, HTTP API, UI) goes through `SessionRunner`, so
thread ids, recursion limits and failure containment are configured exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot

from sigma.agents import reasoning
from sigma.errors import InvalidResume
from sigma.config import SETTINGS
from sigma.graph.builder import build_graph
from sigma.graph.routing import APPROVAL_NODE, HITL_NODE
from sigma.graph.checkpointer import get_saver
from sigma.state import (
    ErrorRecord,
    HumanDecision,
    RequestPayload,
    RunStatus,
    SigmaState,
    new_session_id,
)


@dataclass
class RunResult:
    """Outcome of one graph invocation, normalised for the API/CLI layer."""

    session_id: str
    status: RunStatus
    interrupted: bool = False
    interrupts: list[dict[str, Any]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class SessionRunner:
    """Owns the compiled graph and the per-session invocation config."""

    def __init__(self, graph: CompiledStateGraph | None = None, *, hitl: bool = False) -> None:
        self.hitl = hitl
        self.graph = graph or build_graph(
            checkpointer=get_saver(),
            hitl=hitl,
            approval_node=(HITL_NODE if hitl else APPROVAL_NODE),
        )

    # -- config ------------------------------------------------------------ #

    def config(self, session_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": session_id},
            "recursion_limit": SETTINGS.recursion_limit,
        }

    # -- lifecycle --------------------------------------------------------- #

    def start(self, *, question: str, portfolio_id: str = "sigma-core", seed: int = 7,
              max_revisions: int | None = None, acceptance_threshold: float | None = None,
              session_id: str | None = None) -> RunResult:
        """Create a session and run the graph until it ends or asks a human."""
        sid = session_id or new_session_id()
        request = RequestPayload(
            question=question,
            portfolio_id=portfolio_id,
            seed=seed,
            max_revisions=SETTINGS.max_revisions if max_revisions is None else max_revisions,
            acceptance_threshold=(SETTINGS.acceptance_threshold
                                  if acceptance_threshold is None else acceptance_threshold),
        )
        initial: SigmaState = {"session_id": sid, "request": request}  # type: ignore[typeddict-item]
        return self._invoke(sid, initial)

    def resume(self, session_id: str, decision: HumanDecision) -> RunResult:
        """Continue an interrupted run with the human's decision (FASE 2)."""
        return self._invoke(session_id, Command(resume=decision.model_dump()))

    def continue_run(self, session_id: str) -> RunResult:
        """Re-enter an unfinished run (crash recovery: US-5)."""
        return self._invoke(session_id, None)

    # -- inspection -------------------------------------------------------- #

    def state(self, session_id: str) -> StateSnapshot | None:
        return self.graph.get_state(self.config(session_id))

    def history(self, session_id: str) -> list[StateSnapshot]:
        return list(self.graph.get_state_history(self.config(session_id)))

    def state_at(self, session_id: str, checkpoint_id: str) -> StateSnapshot | None:
        config = self.config(session_id)
        config["configurable"]["checkpoint_id"] = checkpoint_id
        return self.graph.get_state(config)

    def exists(self, session_id: str) -> bool:
        return self.state(session_id) is not None

    # -- internals --------------------------------------------------------- #

    def _invoke(self, session_id: str, payload: Any) -> RunResult:
        config = self.config(session_id)
        try:
            final = self.graph.invoke(payload, config)
        except InvalidResume:
            # The human's payload was invalid: leave the session paused so a
            # corrected resume is still possible.
            raise
        except Exception as exc:  # noqa: BLE001 - last-resort containment
            return self._contain(session_id, exc)

        snapshot = self.graph.get_state(config)
        interrupts = [self._interrupt_payload(i) for i in (snapshot.tasks or ())
                      if getattr(i, "interrupts", None)]
        if snapshot.next:
            status = RunStatus.AWAITING_HUMAN if interrupts else RunStatus.RUNNING
        else:
            status = RunStatus(final.get("status", RunStatus.COMPLETED))
        return RunResult(
            session_id=session_id,
            status=status,
            interrupted=bool(interrupts),
            interrupts=interrupts,
            state=dict(final or {}),
        )

    def _contain(self, session_id: str, exc: BaseException) -> RunResult:
        """Record the failure on the thread so the session stays inspectable."""
        config = self.config(session_id)
        snapshot = self.graph.get_state(config)
        values: dict[str, Any] = dict(snapshot.values) if snapshot else {}
        values.setdefault("session_id", session_id)
        record = ErrorRecord(node="runner", kind=type(exc).__name__, message=str(exc),
                             fatal=True)
        brief = reasoning.build_final_brief(values, RunStatus.FAILED)
        brief.headline = f"Analisis dihentikan ({type(exc).__name__})."
        brief.caveats = ([f"{record.node}/{record.kind}: {record.message}"] + brief.caveats)[:6]
        values["errors"] = list(values.get("errors") or []) + [record]
        values["brief"] = brief
        values["status"] = RunStatus.FAILED
        try:
            self.graph.update_state(config, values, as_node="containment")
        except Exception:  # pragma: no cover - nothing else we can do
            pass
        return RunResult(session_id=session_id, status=RunStatus.FAILED,
                         state=values, error=str(exc))

    @staticmethod
    def _interrupt_payload(task: Any) -> dict[str, Any]:
        interrupts = getattr(task, "interrupts", ()) or ()
        payloads = []
        for item in interrupts:
            value = getattr(item, "value", item)
            payloads.append({
                "id": getattr(item, "id", ""),
                "value": value if isinstance(value, dict) else {"value": value},
            })
        return {"task": getattr(task, "name", ""), "interrupts": payloads}


_DEFAULT_RUNNER: SessionRunner | None = None


def get_runner(*, hitl: bool = False) -> SessionRunner:
    """Process-wide runner (the API and CLI share one compiled graph)."""
    global _DEFAULT_RUNNER
    if _DEFAULT_RUNNER is None or _DEFAULT_RUNNER.hitl != hitl:
        _DEFAULT_RUNNER = SessionRunner(hitl=hitl)
    return _DEFAULT_RUNNER
