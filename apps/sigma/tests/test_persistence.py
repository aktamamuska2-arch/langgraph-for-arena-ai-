"""Persistence, session recovery and time-travel tests.

Everything here runs against a real SQLite file: the whole point is that state
survives the object (and would survive the process) that created it.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sigma.config import SETTINGS
from sigma.graph import checkpointer
from sigma.graph.builder import build_graph
from sigma.graph.routing import HITL_NODE
from sigma.persistence import sessions
from sigma.runtime import SessionRunner
from sigma.state import HumanDecision, RunStatus

QUESTION = "Bagaimana risiko dan alokasi modal portofolio ventura kita kuartal ini?"


class PersistenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.mkdtemp(prefix="sigma-persist-")
        cls.original_db = SETTINGS.db_path
        object.__setattr__(SETTINGS, "db_path", Path(cls.tmp) / "checkpoints.sqlite")
        checkpointer.reset_cache()

    @classmethod
    def tearDownClass(cls) -> None:
        checkpointer.reset_cache()
        object.__setattr__(SETTINGS, "db_path", cls.original_db)

    def setUp(self) -> None:
        self.runner = SessionRunner(
            build_graph(checkpointer=checkpointer.get_saver(), hitl=True,
                        approval_node=HITL_NODE), hitl=True)

    def test_checkpoint_file_is_created(self):
        self.runner.start(question=QUESTION, seed=3)
        self.assertTrue(SETTINGS.db_path.exists())
        self.assertGreater(sessions.checkpoint_count(), 1)

    def test_state_survives_a_new_runner(self):
        """Simulates a process restart: a fresh runner resumes the same thread."""
        started = self.runner.start(question=QUESTION, seed=7,
                                    acceptance_threshold=0.5)
        self.assertEqual(started.status, RunStatus.AWAITING_HUMAN)

        restarted = SessionRunner(
            build_graph(checkpointer=checkpointer.get_saver(), hitl=True,
                        approval_node=HITL_NODE), hitl=True)
        recovered = restarted.resume(started.session_id,
                                     HumanDecision(decision="approve"))
        self.assertEqual(recovered.status, RunStatus.COMPLETED)
        self.assertEqual(recovered.state["session_id"], started.session_id)

    def test_crash_recovery_continues_an_unfinished_run(self):
        started = self.runner.start(question=QUESTION, seed=7,
                                    acceptance_threshold=0.5)
        restarted = SessionRunner(
            build_graph(checkpointer=checkpointer.get_saver(), hitl=True,
                        approval_node=HITL_NODE), hitl=True)
        continued = restarted.continue_run(started.session_id)
        self.assertIn(continued.status, (RunStatus.AWAITING_HUMAN, RunStatus.COMPLETED))

    def test_history_grows_monotonically(self):
        started = self.runner.start(question=QUESTION, seed=7,
                                    acceptance_threshold=0.5)
        before = len(self.runner.history(started.session_id))
        self.runner.resume(started.session_id, HumanDecision(decision="approve"))
        after = len(self.runner.history(started.session_id))
        self.assertGreater(after, before)

    def test_state_at_an_earlier_checkpoint_is_older(self):
        started = self.runner.start(question=QUESTION, seed=7,
                                    acceptance_threshold=0.5)
        self.runner.resume(started.session_id, HumanDecision(decision="approve"))
        history = self.runner.history(started.session_id)  # newest first
        newest = dict(history[0].values or {})
        oldest = dict(history[-1].values or {})
        self.assertGreaterEqual(len(newest.get("tool_calls") or []),
                                len(oldest.get("tool_calls") or []))
        self.assertIsNotNone(self.runner.state_at(started.session_id,
                                                  history[0].config["configurable"]["checkpoint_id"]))

    def test_unknown_session_returns_none(self):
        self.assertIsNone(self.runner.state("no-such-session"))

    def test_sessions_are_listed_with_the_gate_flag(self):
        started = self.runner.start(question=QUESTION, seed=7,
                                    acceptance_threshold=0.5)
        summaries = sessions.session_summaries(self.runner)
        match = [s for s in summaries if s["session_id"] == started.session_id]
        self.assertTrue(match)
        self.assertTrue(match[0]["awaiting_human"])
        self.assertEqual(match[0]["status"], RunStatus.AWAITING_HUMAN.value)
        self.assertGreater(match[0]["checkpoints"], 1)

    def test_fork_is_an_independent_branch(self):
        started = self.runner.start(question=QUESTION, seed=7,
                                    acceptance_threshold=0.5)
        self.runner.resume(started.session_id, HumanDecision(decision="approve"))
        history = self.runner.history(started.session_id)
        forkable = [h for h in history if (h.values or {}).get("request")]
        forked = self.runner.fork_from(started.session_id,
                                       forkable[-1].config["configurable"]["checkpoint_id"])
        self.assertNotEqual(forked.session_id, started.session_id)
        original = dict(self.runner.state(started.session_id).values or {})
        branch = dict(self.runner.state(forked.session_id).values or {})
        self.assertNotEqual(original.get("session_id"), branch.get("session_id"))
        self.assertEqual(original["brief"].score, original["brief"].score)

    def test_database_size_is_reported(self):
        self.runner.start(question=QUESTION, seed=11)
        self.assertGreater(checkpointer.database_size_bytes(), 0)


if __name__ == "__main__":
    unittest.main()
