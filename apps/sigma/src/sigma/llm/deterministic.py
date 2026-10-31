"""The offline reasoning engine.

This is the default provider: no network, no credentials, deterministic in the
seed. It is not a stub that returns canned text — every result is computed from
the graph state passed in `CompletionRequest.context`, which is why the
self-correction loop behaves identically whether it runs offline or against a
hosted model.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from sigma.agents import reasoning
from sigma.llm.base import CompletionRequest, LLMResponse
from sigma.state import DraftBrief, Evaluation, PlanState, ToolCall

MODEL_NAME = "sigma-deterministic-1"


class ToolPlan(BaseModel):
    """Structured output for the researcher: the tool calls for this pass."""

    calls: list[ToolCall] = []
    rationale: str = ""


class DeterministicProvider:
    """Executes the same policies as the code path, through the provider interface."""

    name = "deterministic"

    def __init__(self, model: str = MODEL_NAME) -> None:
        self.model = model

    def complete(self, request: CompletionRequest) -> LLMResponse:
        ctx = request.context or {}
        model = request.response_model
        parsed: BaseModel | None = None

        if model is PlanState:
            parsed = self._plan(ctx)
        elif model is ToolPlan:
            parsed = self._tool_plan(ctx)
        elif model is DraftBrief:
            parsed = self._draft(ctx)
        elif model is Evaluation:
            parsed = self._critique(ctx)
        else:
            parsed = None

        text = parsed.model_dump_json() if parsed is not None else json.dumps({"text": ""})
        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.name,
            parsed=parsed,
            prompt_tokens=_estimate(request.system) + _estimate(request.messages),
            completion_tokens=max(1, len(text) // 4),
        )

    # -- policies ---------------------------------------------------------- #

    def _plan(self, ctx: dict[str, Any]) -> PlanState:
        request = ctx.get("request")
        scenario = ctx.get("scenario")
        previous = ctx.get("plan")
        return reasoning.plan_questions(
            request, scenario,
            revision=0 if previous is None else previous.revision + 1,
            directive=ctx.get("directive"),
            existing=previous.sub_questions if previous else [],
        )

    def _tool_plan(self, ctx: dict[str, Any]) -> ToolPlan:
        budget_left = int(ctx.get("tool_budget_left", 1))
        calls = reasoning.propose_tool_calls(ctx, budget_left=budget_left)
        return ToolPlan(calls=calls,
                        rationale=f"{len(calls)} panggilan baru pada pass "
                                  f"{ctx.get('research_pass', 0)}")

    def _draft(self, ctx: dict[str, Any]) -> DraftBrief:
        from sigma.state import active_findings

        findings = list(ctx.get("new_findings") or []) or [
            f for f in active_findings(ctx)]
        return reasoning.build_draft(ctx, findings)

    def _critique(self, ctx: dict[str, Any]) -> Evaluation:
        request = ctx.get("request")
        return reasoning.critique(
            ctx,
            threshold=(request.acceptance_threshold if request else None),
        )


def _estimate(payload: Any) -> int:
    try:
        return max(1, len(json.dumps(payload, default=str)) // 4)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return 1


__all__ = ["DeterministicProvider", "MODEL_NAME", "ToolPlan"]
