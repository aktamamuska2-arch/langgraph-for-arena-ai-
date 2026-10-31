"""Conditional-edge tests.

The macro-cycle has three different repair targets. If all failures routed to the
same node the "directed correction" claim of the PRD would be false (AC-CYCLE-2).
"""

from __future__ import annotations

import unittest

from sigma.agents.reasoning import critique
from sigma.graph.builder import build_graph
from sigma.graph.checkpointer import in_memory_saver
from sigma.graph.routing import (
    budget,
    fatal_error,
    make_route_after_critic,
    route_after_intake,
    route_after_planner,
    route_after_researcher,
)
from sigma.state import (
    BudgetState,
    DraftBrief,
    ErrorRecord,
    Evaluation,
    Gap,
    PlanState,
    RequestPayload,
    RunStatus,
    ScenarioContext,
    SubQuestion,
)


def base_state(**overrides) -> dict:
    state = {
        "session_id": "s1",
        "request": RequestPayload(question="risiko portofolio", acceptance_threshold=0.78),
        "scenario": ScenarioContext(portfolio_id="p", seed=1, venture_count=6,
                                    sectors=["fintech", "climate"]),
        "budget": BudgetState(max_revisions=3, max_tool_calls=24),
        "plan": PlanState(revision=0, sub_questions=[
            SubQuestion(id="q1", question="q1", tool_hint="query_portfolio"),
        ]),
        "evidence": [],
        "tool_calls": [],
        "findings": [],
        "evaluation": [],
        "errors": [],
        "draft": DraftBrief(headline="h", summary="s", recommendations=["a"]),
        "status": RunStatus.RUNNING,
    }
    state.update(overrides)
    return state


def evaluation_with(kind: str, score: float = 0.4) -> Evaluation:
    return Evaluation(revision=0, score=score, verdict="revise",
                      gaps=[Gap(kind=kind, detail="x", weight=1.0)])


class BasicRoutingTest(unittest.TestCase):
    def test_intake_without_scenario_is_contained(self):
        state = base_state(scenario=None)
        self.assertEqual(route_after_intake(state), "containment")

    def test_intake_fatal_error_is_contained(self):
        state = base_state(errors=[ErrorRecord(node="intake", kind="X", message="m",
                                              fatal=True)])
        self.assertEqual(route_after_intake(state), "containment")

    def test_intake_healthy_goes_to_planner(self):
        self.assertEqual(route_after_intake(base_state()), "planner")

    def test_planner_without_questions_is_contained(self):
        state = base_state(plan=PlanState(sub_questions=[]))
        self.assertEqual(route_after_planner(state), "containment")

    def test_planner_healthy_goes_to_researcher(self):
        self.assertEqual(route_after_planner(base_state()), "researcher")


class MicroCycleTest(unittest.TestCase):
    def test_queued_calls_reenter_researcher(self):
        state = base_state(queued_call_ids=["abc"])
        self.assertEqual(route_after_researcher(state), "researcher")

    def test_no_queued_calls_goes_to_analyst(self):
        state = base_state(queued_call_ids=[])
        self.assertEqual(route_after_researcher(state), "analyst")

    def test_exhausted_tool_budget_breaks_the_micro_cycle(self):
        state = base_state(queued_call_ids=["abc"],
                           budget=BudgetState(max_tool_calls=2, tool_calls_used=2))
        self.assertEqual(route_after_researcher(state), "analyst")

    def test_fatal_error_short_circuits(self):
        state = base_state(queued_call_ids=["abc"],
                           errors=[ErrorRecord(node="researcher", kind="X",
                                               message="m", fatal=True)])
        self.assertEqual(route_after_researcher(state), "containment")


class MacroCycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.route = make_route_after_critic("finalizer")

    def test_ac_cycle_2_different_gaps_route_differently(self):
        targets = {
            "structural": self.route(base_state(evaluation=[evaluation_with("structural")])),
            "missing_evidence": self.route(
                base_state(evaluation=[evaluation_with("missing_evidence")])),
            "interpretation": self.route(
                base_state(evaluation=[evaluation_with("interpretation")])),
        }
        self.assertEqual(targets["structural"], "planner")
        self.assertEqual(targets["missing_evidence"], "researcher")
        self.assertEqual(targets["interpretation"], "analyst")
        self.assertEqual(len(set(targets.values())), 3)

    def test_accepting_evaluation_goes_to_approval_node(self):
        state = base_state(evaluation=[Evaluation(revision=0, score=0.9, verdict="accept")])
        self.assertEqual(self.route(state), "finalizer")

    def test_approval_node_is_injected(self):
        route = make_route_after_critic("human_gate")
        state = base_state(evaluation=[Evaluation(revision=0, score=0.9, verdict="accept")])
        self.assertEqual(route(state), "human_gate")

    def test_exhausted_revisions_stop_the_cycle(self):
        state = base_state(
            evaluation=[evaluation_with("missing_evidence")],
            budget=BudgetState(max_revisions=2, revisions_used=2, max_tool_calls=24))
        self.assertEqual(self.route(state), "finalizer")

    def test_fatal_error_beats_everything(self):
        state = base_state(
            evaluation=[Evaluation(revision=0, score=0.99, verdict="accept")],
            errors=[ErrorRecord(node="critic", kind="X", message="m", fatal=True)])
        self.assertEqual(self.route(state), "containment")

    def test_missing_evaluation_is_contained(self):
        self.assertEqual(self.route(base_state(evaluation=[])), "containment")


class HelperTest(unittest.TestCase):
    def test_fatal_error_finds_only_fatal(self):
        state = base_state(errors=[ErrorRecord(node="a", kind="k", message="m"),
                                  ErrorRecord(node="b", kind="k", message="m", fatal=True)])
        self.assertEqual(fatal_error(state).node, "b")
        self.assertIsNone(fatal_error(base_state()))

    def test_budget_raises_when_absent(self):
        with self.assertRaises(Exception):
            budget({})


class CritiqueTest(unittest.TestCase):
    def test_score_rises_with_evidence(self):
        from sigma.tools.market import build_portfolio
        from sigma.tools.registry import ToolCall, ToolContext, ToolRegistry

        registry = ToolRegistry()
        ctx = ToolContext(portfolio_id="sigma-core", seed=7)
        evidence = []
        for call in (ToolCall(tool="query_portfolio", args={"limit": 12}),
                     ToolCall(tool="compute_risk_metrics",
                              args={"venture_ids": ["V01", "V02"], "horizon_quarters": 8}),
                     ToolCall(tool="scenario_simulate",
                              args={"venture_ids": ["V01", "V02"], "paths": 500,
                                    "horizon_quarters": 8})):
            result = registry.execute(call, ctx)
            self.assertTrue(result.ok, result.record.error)
            evidence.append(result.to_evidence("q1"))

        build_portfolio("sigma-core", 7)
        poor = critique(base_state(evidence=[]), threshold=0.78)
        rich = critique(base_state(evidence=evidence), threshold=0.78)
        self.assertGreater(rich.score, poor.score)
        self.assertEqual(poor.verdict, "revise")

    def test_perfect_state_can_be_accepted(self):
        from sigma.tools.registry import ToolCall, ToolContext, ToolRegistry

        registry = ToolRegistry()
        ctx = ToolContext(portfolio_id="sigma-core", seed=7)
        evidence = []
        for call in (ToolCall(tool="query_portfolio", args={"limit": 12}),
                     ToolCall(tool="compute_risk_metrics",
                              args={"venture_ids": ["V01", "V02"], "horizon_quarters": 8}),
                     ToolCall(tool="scenario_simulate",
                              args={"venture_ids": ["V01", "V02"], "paths": 500,
                                    "horizon_quarters": 8}),
                     ToolCall(tool="benchmark_sector", args={"sector": "fintech"})):
            result = registry.execute(call, ctx)
            evidence.append(result.to_evidence("q1"))
        state = base_state(evidence=evidence,
                           draft=DraftBrief(headline="h", summary="s",
                                            recommendations=["a" * 60, "b" * 60, "c" * 60]))
        self.assertEqual(critique(state, threshold=0.5).verdict, "accept")


class GraphWiringTest(unittest.TestCase):
    def test_graph_accepts_all_declared_targets(self):
        """Every value a router can return must be a real node."""
        graph = build_graph(checkpointer=in_memory_saver())
        nodes = set(graph.nodes)
        for target in ("planner", "researcher", "analyst", "finalizer", "containment"):
            self.assertIn(target, nodes)


if __name__ == "__main__":
    unittest.main()
