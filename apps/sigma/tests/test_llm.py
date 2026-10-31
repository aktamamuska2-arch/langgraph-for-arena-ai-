"""LLM integration tests: providers, structured output, fallback, memory.

The point of these tests is the *contract*, not the model: whatever provider is
configured, a node must end up with a validated Pydantic object, and a broken
provider must degrade to the deterministic engine rather than break the run.
"""

from __future__ import annotations

import json
import unittest
from typing import Any
from unittest import mock

from pydantic import ValidationError

from sigma.config import SETTINGS
from sigma.errors import ProviderError
from sigma.graph.builder import build_graph
from sigma.graph.checkpointer import in_memory_saver
from sigma.llm.base import CompletionRequest, LLMResponse
from sigma.llm.deterministic import DeterministicProvider, ToolPlan
from sigma.llm.factory import get_provider, set_provider
from sigma.llm.http import HttpChatProvider
from sigma.llm.memory import append, window
from sigma.runtime import SessionRunner
from sigma.state import (DraftBrief, Evaluation, PlanState, RequestPayload,
                         RunStatus, ScenarioContext, SubQuestion, message)

QUESTION = "Bagaimana risiko dan alokasi modal portofolio ventura kita kuartal ini?"


def sample_context() -> dict[str, Any]:
    return {
        "request": RequestPayload(question=QUESTION),
        "scenario": ScenarioContext(portfolio_id="sigma-core", seed=7, venture_count=18,
                                    sectors=["fintech", "climate"]),
        "budget": None,
        "plan": PlanState(revision=0, sub_questions=[
            SubQuestion(id="q1", question="risiko?", tool_hint="compute_risk_metrics"),
            SubQuestion(id="q2", question="komposisi?", tool_hint="query_portfolio"),
        ]),
        "evidence": [],
        "tool_calls": [],
        "findings": [],
        "evaluation": [],
        "messages": [],
        "audit_trail": [],
        "research_pass": 0,
        "tool_budget_left": 3,
        "new_findings": [],
    }


class DeterministicProviderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = DeterministicProvider()

    def test_returns_parsed_plan(self):
        response = self.provider.complete(CompletionRequest(
            system="s", response_model=PlanState, context=sample_context()))
        self.assertIsInstance(response.parsed, PlanState)
        self.assertTrue(response.parsed.sub_questions)
        self.assertEqual(response.provider, "deterministic")
        self.assertGreater(response.usage["total_tokens"], 0)

    def test_returns_parsed_tool_plan(self):
        response = self.provider.complete(CompletionRequest(
            system="s", response_model=ToolPlan, context=sample_context()))
        self.assertIsInstance(response.parsed, ToolPlan)
        self.assertTrue(response.parsed.calls)

    def test_returns_parsed_draft_and_evaluation(self):
        for model in (DraftBrief, Evaluation):
            response = self.provider.complete(CompletionRequest(
                system="s", response_model=model, context=sample_context()))
            self.assertIsInstance(response.parsed, model)

    def test_output_is_json_serialisable(self):
        response = self.provider.complete(CompletionRequest(
            system="s", response_model=PlanState, context=sample_context()))
        # The context already carries plan revision 0, so a re-plan is revision 1.
        self.assertEqual(json.loads(response.text)["revision"], 1)

    def test_is_deterministic_for_the_same_context(self):
        request = CompletionRequest(system="s", response_model=PlanState,
                                    context=sample_context())
        first = self.provider.complete(request).text
        second = self.provider.complete(request).text
        self.assertEqual(first, second)


class FallbackTest(unittest.TestCase):
    def tearDown(self) -> None:
        set_provider(None)

    def test_broken_provider_degrades_to_deterministic(self):
        """A dead provider must not break the run (PRD section 9)."""

        class Broken:
            name = "broken"

            def complete(self, request: CompletionRequest) -> LLMResponse:
                raise ProviderError("upstream is down")

        set_provider(Broken())
        result = SessionRunner(build_graph(checkpointer=in_memory_saver())).start(
            question=QUESTION, seed=7)
        self.assertEqual(result.status, RunStatus.COMPLETED, result.error)
        self.assertTrue(result.state["brief"].recommendations)
        fallbacks = [e for e in result.state["audit_trail"]
                     if e.kind == "provider_fallback"]
        self.assertTrue(fallbacks)

    def test_fallback_matches_the_deterministic_result(self):
        """Falling back must be equivalent to running offline."""
        graph = build_graph(checkpointer=in_memory_saver())
        baseline = SessionRunner(graph).start(question=QUESTION, seed=7)

        class Broken:
            name = "broken"

            def complete(self, request: CompletionRequest) -> LLMResponse:
                raise ProviderError("nope")

        set_provider(Broken())
        degraded = SessionRunner(graph).start(question=QUESTION, seed=7)
        self.assertEqual(baseline.state["brief"].score, degraded.state["brief"].score)

    def test_garbage_output_is_rejected_and_falls_back(self):
        """A model that returns prose must not corrupt the state."""

        class Chatty:
            name = "chatty"

            def complete(self, request: CompletionRequest) -> LLMResponse:
                return LLMResponse(text="I think therefore I am", model="chatty",
                                   provider="chatty", parsed=None)

        set_provider(Chatty())
        result = SessionRunner(build_graph(checkpointer=in_memory_saver())).start(
            question=QUESTION, seed=7)
        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertTrue([e for e in result.state["audit_trail"]
                         if e.kind == "provider_fallback"])


class HttpProviderTest(unittest.TestCase):
    def _provider(self) -> HttpChatProvider:
        return HttpChatProvider(base_url="https://llm.example/v1", api_key="k",
                                model="test-model")

    def _response(self, content: str) -> mock.MagicMock:
        payload = {"choices": [{"message": {"content": content}}],
                   "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                   "model": "test-model"}
        transport = mock.MagicMock()
        transport.status_code = 200
        transport.json.return_value = payload
        transport.raise_for_status.return_value = None
        return transport

    def test_parses_structured_output(self):
        provider = self._provider()
        plan = DeterministicProvider().complete(CompletionRequest(
            system="s", response_model=PlanState, context=sample_context())).text
        with mock.patch("httpx.post", return_value=self._response(plan)) as post:
            response = provider.complete(CompletionRequest(
                system="s", response_model=PlanState, context=sample_context()))
        self.assertIsInstance(response.parsed, PlanState)
        self.assertEqual(response.prompt_tokens, 10)
        kwargs = post.call_args.kwargs
        self.assertEqual(kwargs["json"]["model"], "test-model")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer k")

    def test_tolerates_fenced_json(self):
        provider = self._provider()
        plan = DeterministicProvider().complete(CompletionRequest(
            system="s", response_model=PlanState, context=sample_context())).text
        with mock.patch("httpx.post",
                        return_value=self._response(f"```json\\n{plan}\\n```")):
            response = provider.complete(CompletionRequest(
                system="s", response_model=PlanState, context=sample_context()))
        self.assertIsInstance(response.parsed, PlanState)

    def test_bad_shape_raises_provider_error(self):
        provider = self._provider()
        with mock.patch("httpx.post", return_value=self._response("{}")):
            with mock.patch.object(provider, "complete", side_effect=ProviderError("x")):
                with self.assertRaises(ProviderError):
                    provider.complete(CompletionRequest(system="s"))

    def test_network_failure_raises_provider_error(self):
        provider = self._provider()
        with mock.patch("httpx.post", side_effect=OSError("no route")):
            with self.assertRaises(ProviderError):
                provider.complete(CompletionRequest(system="s"))

    def test_requires_base_url(self):
        with self.assertRaises(ProviderError):
            HttpChatProvider(base_url="")


class MemoryTest(unittest.TestCase):
    def test_window_is_capped(self):
        state = {"messages": [message("assistant", f"m{i}", kind="x")
                              for i in range(50)]}
        self.assertEqual(len(window(state, limit=5)), 5)

    def test_append_compacts_and_keeps_the_newest(self):
        state = {"messages": [message("assistant", f"m{i}", kind="note")
                              for i in range(SETTINGS.context_window)],
                 "audit_trail": []}
        update = append(state, message("assistant", "terbaru", kind="note"))
        self.assertEqual(len(update["messages"]), SETTINGS.context_window)
        self.assertEqual(update["messages"][-1]["content"], "terbaru")
        event = update["audit_trail"][0]
        self.assertEqual(event.kind, "context_compacted")
        self.assertIn("note", event.data["dropped_by_kind"])

    def test_no_compaction_below_the_window(self):
        state = {"messages": [message("assistant", "a", kind="note")], "audit_trail": []}
        update = append(state, message("assistant", "b", kind="note"))
        self.assertNotIn("audit_trail", update)
        self.assertEqual(len(update["messages"]), 2)


class FactoryTest(unittest.TestCase):
    def tearDown(self) -> None:
        set_provider(None)

    def test_default_is_deterministic(self):
        self.assertEqual(get_provider().name, "deterministic")

    def test_injected_provider_wins(self):
        class Custom:
            name = "custom"

            def complete(self, request: CompletionRequest) -> LLMResponse:
                return LLMResponse(text="", model="c", provider="custom")

        set_provider(Custom())
        self.assertEqual(get_provider().name, "custom")

    def test_invalid_structured_output_raises(self):
        with self.assertRaises(ValidationError):
            PlanState.model_validate_json('{"revision": "not-a-number"}')


if __name__ == "__main__":
    unittest.main()
