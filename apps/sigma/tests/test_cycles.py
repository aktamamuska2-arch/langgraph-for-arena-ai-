"""Self-correction cycle tests.

These tests read the *actual* node sequence back out of the audit trail, so they
prove the graph really revisits nodes instead of asserting the topology on paper.
"""

from __future__ import annotations

import unittest
from unittest import mock

from sigma.agents import reasoning
from sigma.graph.builder import build_graph
from sigma.graph.checkpointer import in_memory_saver
from sigma.graph.routing import HITL_NODE
from sigma.runtime import SessionRunner
from sigma.state import Evaluation, Gap, PlanState, RunStatus

QUESTION = "Bagaimana risiko dan alokasi modal portofolio ventura kita kuartal ini?"


def runner(**kwargs) -> SessionRunner:
    return SessionRunner(build_graph(checkpointer=in_memory_saver(), **kwargs))


def visited_nodes(state: dict) -> list[str]:
    """Node visit order, recovered from the guard's `completed` audit events."""
    return [e.node for e in state["audit_trail"] if e.kind == "completed"]


class MicroCycleTest(unittest.TestCase):
    def test_researcher_loops_until_nothing_is_queued(self):
        result = runner().start(question=QUESTION, seed=7)
        visits = visited_nodes(result.state)
        self.assertGreater(visits.count("researcher"), 1)

    def test_micro_cycle_precedes_the_analyst(self):
        result = runner().start(question=QUESTION, seed=7)
        visits = visited_nodes(result.state)
        self.assertLess(max(i for i, n in enumerate(visits) if n == "researcher"),
                        min(i for i, n in enumerate(visits) if n == "analyst"))


class MacroCycleTest(unittest.TestCase):
    def test_critic_is_followed_by_a_repair_node(self):
        result = runner().start(question=QUESTION, seed=7, acceptance_threshold=0.95)
        visits = visited_nodes(result.state)
        critic_positions = [i for i, n in enumerate(visits) if n == "critic"]
        self.assertGreater(len(critic_positions), 1)
        for position in critic_positions[:-1]:
            self.assertIn(visits[position + 1], ("planner", "researcher", "analyst"))

    def test_structural_gap_routes_back_to_the_planner(self):
        """AC-CYCLE-2: the repair target follows the gap class."""

        real = reasoning.critique
        calls = {"n": 0}

        def forced_then_real(state, *, threshold=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return Evaluation(revision=0, score=0.2, verdict="revise",
                                  gaps=[Gap(kind="structural", detail="kerangka salah",
                                            weight=2.0)],
                                  rationale="forced")
            return real(state, threshold=threshold)

        with mock.patch.object(reasoning, "critique", side_effect=forced_then_real):
            result = runner().start(question=QUESTION, seed=7)
        self.assertGreaterEqual(result.state["plan"].revision, 1)
        visits = visited_nodes(result.state)
        self.assertGreater(visits.count("planner"), 1)

    def test_evidence_gap_routes_back_to_the_researcher(self):
        real = reasoning.critique
        calls = {"n": 0}

        def forced(state, *, threshold=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return Evaluation(revision=0, score=0.2, verdict="revise",
                                  gaps=[Gap(kind="missing_evidence", detail="kurang bukti",
                                            weight=2.0)],
                                  rationale="forced")
            return real(state, threshold=threshold)

        with mock.patch.object(reasoning, "critique", side_effect=forced):
            result = runner().start(question=QUESTION, seed=7)
        self.assertGreater(len(result.state["evidence"]), 3)
        self.assertGreater(visited_nodes(result.state).count("researcher"), 2)

    def test_each_cycle_adds_information(self):
        """A cycle that does not gather new evidence would be pointless."""
        result = runner().start(question=QUESTION, seed=7, acceptance_threshold=0.95)
        evaluations = result.state["evaluation"]
        self.assertGreater(len(evaluations), 1)
        self.assertGreater(evaluations[-1].score, evaluations[0].score)


class TerminationTest(unittest.TestCase):
    def test_never_accepting_critic_stops_at_the_budget(self):
        """AC-CYCLE-1 with HITL: the run stops and asks a human instead of hanging."""
        graph = build_graph(checkpointer=in_memory_saver(), hitl=True,
                            approval_node=HITL_NODE)
        result = SessionRunner(graph, hitl=True).start(
            question=QUESTION, seed=7, acceptance_threshold=1.1)
        self.assertEqual(result.status, RunStatus.AWAITING_HUMAN)
        self.assertEqual(result.state["budget"].revisions_used,
                         result.state["budget"].max_revisions)

    def test_recursion_limit_is_never_reached(self):
        result = runner().start(question=QUESTION, seed=7, acceptance_threshold=1.1)
        self.assertNotEqual(result.status, RunStatus.FAILED)
        self.assertIsNone(result.error)

    def test_budgets_are_monotonic(self):
        result = runner().start(question=QUESTION, seed=7, acceptance_threshold=1.1)
        budget = result.state["budget"]
        self.assertLessEqual(budget.tool_calls_used, budget.max_tool_calls)
        self.assertLessEqual(budget.revisions_used, budget.max_revisions)
        self.assertLessEqual(budget.tokens_used, budget.max_tokens)


class PlanRevisionTest(unittest.TestCase):
    def test_revision_keeps_answered_questions(self):
        """A revision extends the plan; it does not restart from zero.

        A structural gap is forced on every critique so the planner is re-entered
        (that is the only gap class that routes back to planning).
        """
        real_plan = reasoning.plan_questions
        seen: list[PlanState] = []

        def capture(request, scenario, **kwargs):
            plan = real_plan(request, scenario, **kwargs)
            seen.append(plan)
            return plan

        def always_structural(state, *, threshold=None):
            return Evaluation(revision=0, score=0.2, verdict="revise",
                              gaps=[Gap(kind="structural", detail="kerangka salah",
                                        weight=2.0)],
                              rationale="forced")

        with mock.patch.object(reasoning, "critique", side_effect=always_structural), \
                mock.patch.object(reasoning, "plan_questions", side_effect=capture):
            result = runner().start(question=QUESTION, seed=7, acceptance_threshold=1.1)

        self.assertGreater(len(seen), 1)
        self.assertGreaterEqual(len(seen[-1].sub_questions), len(seen[0].sub_questions))
        self.assertEqual([p.revision for p in seen], list(range(len(seen))))
        # Questions already answered must survive the re-plan.
        first_ids = {q.id for q in seen[0].sub_questions}
        self.assertTrue(first_ids <= {q.id for q in seen[-1].sub_questions})


if __name__ == "__main__":
    unittest.main()
