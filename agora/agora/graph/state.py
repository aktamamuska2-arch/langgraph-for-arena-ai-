"""Shared state for the ASEOWS build pipeline.

The graph is a cyclic state machine:

    Planner -> Spatial-Economic Model Synthesizer -> Sandbox Visual Tester
                        ^                                     |
                        |                                     v
                        +----------- Self-Correction <--------+

Every node receives the full :class:`PipelineState` and returns a partial
update, which LangGraph merges into the running state.
"""

from __future__ import annotations

from typing import Any, TypedDict

from ..schemas import (
    ArtifactRecord,
    Blueprint,
    Directive,
    ExperimentSpec,
    Patch,
    Reflection,
    ValidationReport,
)


class PipelineState(TypedDict, total=False):
    """Everything the pipeline carries between nodes."""

    directive: Directive
    blueprint: Blueprint
    spec: ExperimentSpec
    attempt: int
    max_attempts: int
    report: ValidationReport
    metrics: dict[str, float]
    detail: dict[str, Any]
    reflections: list[Reflection]
    patches: list[Patch]
    artifacts: list[ArtifactRecord]
    verdict: str
    output_dir: str
    log: list[str]
    trail: list[dict[str, Any]]


def initial_state(directive: Directive) -> PipelineState:
    return PipelineState(
        directive=directive,
        attempt=0,
        max_attempts=1 + directive.max_retries,
        reflections=[],
        patches=[],
        artifacts=[],
        metrics={},
        detail={},
        verdict="PENDING",
        output_dir="artifacts",
        log=[],
        trail=[],
    )


def set_by_pointer(payload: dict[str, Any], pointer: str, value: Any) -> None:
    """Apply a dotted ``pointer`` to a nested mapping in place."""

    parts = pointer.split(".")
    cursor = payload
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def get_by_pointer(payload: dict[str, Any], pointer: str) -> Any:
    cursor: Any = payload
    for part in pointer.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor
