"""Graph state: the single mutable-by-reduction structure of the build pipeline.

Immutability contract
---------------------
``scenario_params``, ``architectural_blueprint``, ``code_artifacts`` and the
reports are plain JSON dictionaries, but every node treats them as **read-only**
and returns a brand-new dictionary when something must change. Nothing is
mutated in place, so a checkpoint of any super-step is a valid snapshot.

The only accumulating field is ``execution_logs``: it is annotated with
``operator.add`` so LangGraph appends each node's log slice instead of
overwriting it. That is what makes the full audit trail recoverable from the
final state.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, TypedDict


class State(TypedDict, total=False):
    """Canonical state schema for the AM-SES build graph."""

    # --- mandated slices ------------------------------------------------
    scenario_params: dict[str, Any]
    architectural_blueprint: dict[str, Any]
    code_artifacts: dict[str, Any]
    execution_logs: Annotated[list[dict[str, Any]], operator.add]
    retry_count: int
    reflection_trail: Annotated[list[dict[str, Any]], operator.add]

    # --- derived slices -------------------------------------------------
    run_id: str
    test_report: dict[str, Any]
    reflection: dict[str, Any]
    final_report: dict[str, Any]
    verdict: str


@dataclass(frozen=True)
class RunContext:
    """Immutable node dependencies (paths, limits, fault injection)."""

    run_id: str
    project_root: Path
    artifacts_root: Path
    max_retries: int = 3
    sandbox_timeout: float = 90.0
    fault_mode: str = "none"
    fault_persist: bool = False

    def attempt_dir(self, attempt: int) -> Path:
        return self.artifacts_root / self.run_id / f"attempt-{attempt:02d}"

    def run_dir(self) -> Path:
        return self.artifacts_root / self.run_id


def initial_state(spec_payload: dict[str, Any], run_id: str) -> State:
    return {
        "run_id": run_id,
        "scenario_params": spec_payload,
        "architectural_blueprint": {},
        "code_artifacts": {},
        "execution_logs": [],
        "retry_count": 0,
        "reflection_trail": [],
        "verdict": "PENDING",
    }
