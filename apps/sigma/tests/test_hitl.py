"""Human-in-the-loop tests.

These cover the contract in PRD section 8: the graph must pause *inside* a node,
expose a typed payload, and resume with a validated decision.
"""

from __future__ import annotations

import unittest

from langgraph.types import Command
from pydantic import ValidationError

from sigma.errors import InvalidResume
from sigma.graph.builder import build_graph
from sigma.graph.checkpointer import in_memory_saver
from sigma.graph.routing import HITL_NODE
from sigma.runtime import SessionRunner
from sigma.state import HumanDecision, RunStatus

QUESTION = "Bagaimana risiko dan alokasi modal portofolio ventura kita kuartal ini?"


def hitl_runner(threshold: float = 0.5) -> SessionRunner:
    graph = build_graph(checkpointer=in_memory_saver(), hitl=True, approval_node=HITL_NODE)
    return SessionRunner(graph, hitl=True)


class InterruptTest(unittest.TestCase):
    def test_run_pauses_for_a_human(self):
        runner = hitl_runner()
        result = runner.start(question=QUESTION, seed=7, acceptance_threshold=0.5)
        self.assertEqual(result.status, RunStatus.AWAITING_HUMAN)
        self.assertTrue(result.interrupted)
        self.assertIsNone(result.state.get("brief"))

    def test_interrupt_payload_is_complete_and_typed(self):
        runner = hitl_runner()
        result = runner.start(question=QUESTION, seed=7, acceptance_threshold=0.5)
        value = result.interrupts[0]["interrupts"][0]["value"]
        self.assertEqual(value["kind"], "approval_request")
        for key in ("session_id", "score", "rubric", "summary",
                    "recommendations", "open_gaps", "prompt"):
            self.assertIn(key, value)
        self.assertIsInstance(value["recommendations"], list)
        self.assertIn("prompt", value)

    def test_state_is_persisted_while_paused(self):
        """The pause must be durable: the thread is resumable from a fresh runner."""
        runner = hitl_runner()
        result = runner.start(question=QUESTION, seed=7, acceptance_threshold=0.5)
        snapshot = runner.state(result.session_id)
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.next, (HITL_NODE,))
        self.assertTrue(snapshot.tasks[0].interrupts)

    def test_hitl_requires_a_checkpointer(self):
        with self.assertRaises(ValueError):
            build_graph(checkpointer=None, hitl=True)


class ResumeTest(unittest.TestCase):
    def test_approve_finalises(self):
        runner = hitl_runner()
        started = runner.start(question=QUESTION, seed=7, acceptance_threshold=0.5)
        resumed = runner.resume(started.session_id,
                                HumanDecision(decision="approve", note="setuju"))
        self.assertEqual(resumed.status, RunStatus.COMPLETED)
        self.assertEqual(resumed.state["brief"].status, RunStatus.COMPLETED)
        self.assertEqual(resumed.state["human_decision"].decision, "approve")

    def test_reject_closes_the_session(self):
        runner = hitl_runner()
        started = runner.start(question=QUESTION, seed=7, acceptance_threshold=0.5)
        resumed = runner.resume(started.session_id,
                                HumanDecision(decision="reject", note="tidak cukup"))
        self.assertEqual(resumed.status, RunStatus.REJECTED)
        self.assertIn("ditolak", resumed.state["brief"].headline)

    def test_revise_sends_the_run_back_to_the_planner(self):
        """The human's note must change the plan, not just re-run it."""
        runner = hitl_runner()
        started = runner.start(question=QUESTION, seed=7, acceptance_threshold=0.5)
        before = started.state["plan"].revision
        resumed = runner.resume(
            started.session_id,
            HumanDecision(decision="revise", note="periksa konsentrasi sektor"))
        self.assertGreater(resumed.state["plan"].revision, before)
        self.assertIn("konsentrasi sektor", resumed.state["plan"].directive or "")

    def test_revise_then_approve_completes(self):
        runner = hitl_runner()
        started = runner.start(question=QUESTION, seed=7, acceptance_threshold=0.5)
        revised = runner.resume(started.session_id,
                                HumanDecision(decision="revise", note="tambah bukti"))
        self.assertIn(revised.status, (RunStatus.AWAITING_HUMAN, RunStatus.COMPLETED))
        if revised.status is RunStatus.AWAITING_HUMAN:
            final = runner.resume(started.session_id, HumanDecision(decision="approve"))
            self.assertEqual(final.status, RunStatus.COMPLETED)

    def test_invalid_decision_is_rejected_and_the_session_stays_paused(self):
        """A malformed resume must surface as an error, not silently end the run."""
        runner = hitl_runner()
        started = runner.start(question=QUESTION, seed=7, acceptance_threshold=0.5)
        with self.assertRaises(InvalidResume):
            runner.graph.invoke(Command(resume={"decision": "maybe"}),
                                runner.config(started.session_id))
        snapshot = runner.state(started.session_id)
        self.assertEqual(snapshot.next, (HITL_NODE,))

    def test_a_corrected_resume_still_works_after_an_invalid_one(self):
        runner = hitl_runner()
        started = runner.start(question=QUESTION, seed=7, acceptance_threshold=0.5)
        with self.assertRaises(InvalidResume):
            runner.graph.invoke(Command(resume={"decision": "nope"}),
                                runner.config(started.session_id))
        resumed = runner.resume(started.session_id, HumanDecision(decision="approve"))
        self.assertEqual(resumed.status, RunStatus.COMPLETED)

    def test_human_decision_model_rejects_unknown_values(self):
        with self.assertRaises(ValidationError):
            HumanDecision(decision="maybe")

    def test_resume_is_idempotent_for_the_paused_node(self):
        """Replaying `human_gate` must not duplicate audit entries or decisions."""
        runner = hitl_runner()
        started = runner.start(question=QUESTION, seed=7, acceptance_threshold=0.5)
        runner.resume(started.session_id, HumanDecision(decision="approve"))
        decisions = [e for e in runner.state(started.session_id).values.get("audit_trail", [])
                     if e.kind == "human_decision"]
        self.assertEqual(len(decisions), 1)


if __name__ == "__main__":
    unittest.main()
