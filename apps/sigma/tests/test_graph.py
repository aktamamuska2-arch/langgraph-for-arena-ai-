"""Topology and end-to-end execution tests for the compiled graph.

These tests encode the acceptance criteria from PRD section 1.2 that distinguish
a cyclic graph from a DAG.
"""

from __future__ import annotations

import unittest

from langgraph.graph.state import CompiledStateGraph

from sigma.graph.builder import build_graph
from sigma.graph.checkpointer import in_memory_saver
from sigma.runtime import SessionRunner
from sigma.state import RunStatus

QUESTION = "Bagaimana risiko dan alokasi modal portofolio ventura kita kuartal ini?"


def make_runner() -> SessionRunner:
    return SessionRunner(build_graph(checkpointer=in_memory_saver()))


class TopologyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = build_graph(checkpointer=in_memory_saver())

    def test_compiles_to_a_pregel(self):
        self.assertIsInstance(self.graph, CompiledStateGraph)

    def test_all_nodes_present(self):
        expected = {"intake", "planner", "researcher", "analyst", "critic",
                    "finalizer", "containment"}
        self.assertTrue(expected.issubset(set(self.graph.nodes)))

    def test_ac_cycle_4_node_count_is_constant_across_iterations(self):
        """Iterations are supersteps, not duplicated nodes.

        A DAG-based implementation would need one node per iteration; here the
        node set is identical before the run and after a 3-revision run.
        """
        before = set(self.graph.nodes)
        runner = SessionRunner(self.graph)
        runner.start(question=QUESTION, seed=3, max_revisions=3)
        self.assertEqual(before, set(self.graph.nodes))


class EndToEndTest(unittest.TestCase):
    def test_run_completes_with_a_brief(self):
        result = make_runner().start(question=QUESTION, seed=7)
        self.assertEqual(result.status, RunStatus.COMPLETED, result.error)
        brief = result.state["brief"]
        self.assertGreater(brief.score, 0.0)
        self.assertTrue(brief.headline)
        self.assertTrue(brief.recommendations)

    def test_every_numeric_finding_has_evidence(self):
        """PRD NFR-7: no number without provenance."""
        result = make_runner().start(question=QUESTION, seed=11)
        for finding in result.state["findings"]:
            if finding.metric is not None:
                self.assertTrue(finding.evidence_ids, finding.claim)

    def test_evidence_ids_resolve(self):
        result = make_runner().start(question=QUESTION, seed=5)
        index = {e.evidence_id for e in result.state["evidence"]}
        for finding in result.state["findings"]:
            for evidence_id in finding.evidence_ids:
                self.assertIn(evidence_id, index)

    def test_audit_trail_records_every_node(self):
        result = make_runner().start(question=QUESTION, seed=7)
        nodes = {event.node for event in result.state["audit_trail"]}
        self.assertTrue({"intake", "planner", "researcher", "analyst", "critic"} <= nodes)

    def test_audit_sequence_is_monotonic(self):
        result = make_runner().start(question=QUESTION, seed=7)
        seqs = [event.seq for event in result.state["audit_trail"]]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len(seqs), len(set(seqs)))


class ConvergenceTest(unittest.TestCase):
    def test_ac_cycle_1_unreachable_threshold_terminates(self):
        """A critic that always rejects must stop at the revision budget.

        With `acceptance_threshold = 1.1` no score can ever be accepted, so a
        cyclic graph that did not respect its budget would either hang or blow
        the recursion limit.
        """
        runner = make_runner()
        result = runner.start(question=QUESTION, seed=7, acceptance_threshold=1.1)
        self.assertNotIn(result.status, (RunStatus.RUNNING, RunStatus.FAILED), result.error)
        self.assertEqual(result.state["budget"].revisions_used,
                         result.state["budget"].max_revisions)
        self.assertIsNotNone(result.state["brief"])

    def test_score_does_not_regress_between_cycles(self):
        """Each cycle must add information, otherwise the loop is useless."""
        result = make_runner().start(question=QUESTION, seed=7, acceptance_threshold=0.95)
        scores = [e.score for e in result.state["evaluation"]]
        self.assertGreaterEqual(len(scores), 2)
        self.assertGreater(scores[-1], scores[0])

    def test_tool_budget_is_never_exceeded(self):
        result = make_runner().start(question=QUESTION, seed=7, acceptance_threshold=1.1)
        budget = result.state["budget"]
        self.assertLessEqual(budget.tool_calls_used, budget.max_tool_calls)


class DeterminismTest(unittest.TestCase):
    def test_identical_seed_yields_identical_score(self):
        """PRD NFR-3."""
        first = make_runner().start(question=QUESTION, seed=42)
        second = make_runner().start(question=QUESTION, seed=42)
        self.assertEqual(first.state["brief"].score, second.state["brief"].score)
        self.assertEqual(first.state["brief"].headline, second.state["brief"].headline)

    def test_different_seed_changes_the_world(self):
        first = make_runner().start(question=QUESTION, seed=1)
        second = make_runner().start(question=QUESTION, seed=2)
        self.assertNotEqual(first.state["brief"].summary, second.state["brief"].summary)


if __name__ == "__main__":
    unittest.main()
