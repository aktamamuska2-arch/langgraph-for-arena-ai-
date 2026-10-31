"""HTTP surface tests (FastAPI TestClient)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from sigma.api.app import create_app
from sigma.config import SETTINGS
from sigma.graph import checkpointer
from sigma.llm.factory import set_provider
from sigma.state import RunStatus

QUESTION = "Bagaimana risiko dan alokasi modal portofolio ventura kita kuartal ini?"


class ApiTest(unittest.TestCase):
    """Shared setup: an isolated checkpoint database per test class."""

    tmp: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.mkdtemp(prefix="sigma-api-")
        cls.original_db = SETTINGS.db_path
        object.__setattr__(SETTINGS, "db_path", Path(cls.tmp) / "checkpoints.sqlite")
        checkpointer.reset_cache()
        cls.app = create_app(hitl=True)
        cls.client = TestClient(cls.app)

    @classmethod
    def tearDownClass(cls) -> None:
        checkpointer.reset_cache()
        object.__setattr__(SETTINGS, "db_path", cls.original_db)
        set_provider(None)

    def start_session(self, **overrides):
        payload = {"question": QUESTION, "seed": 7, "hitl": True,
                   "acceptance_threshold": 0.5}
        payload.update(overrides)
        response = self.client.post("/api/sessions", json=payload)
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()


class MetaTest(ApiTest):
    def test_health_reports_the_size_budget(self):
        body = self.client.get("/api/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["budget_mb"], SETTINGS.size_budget_mb)
        self.assertIsInstance(body["artifact_mb"], float)
        self.assertLess(body["artifact_mb"], body["budget_mb"])
        self.assertTrue(body["hitl"])

    def test_tools_are_exposed_as_json_schema(self):
        body = self.client.get("/api/tools").json()
        names = {tool["name"] for tool in body["tools"]}
        self.assertTrue({"query_portfolio", "compute_risk_metrics",
                         "scenario_simulate"} <= names)

    def test_index_is_served(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("SIGMA", response.text)


class SessionFlowTest(ApiTest):
    def test_create_then_resume_then_read(self):
        created = self.start_session()
        self.assertEqual(created["status"], RunStatus.AWAITING_HUMAN.value)
        self.assertTrue(created["interrupted"])
        self.assertIsNone(created["brief"])
        session_id = created["session_id"]

        resumed = self.client.post(
            f"/api/sessions/{session_id}/resume",
            json={"decision": "approve", "note": "ok", "reviewer": "qa"}).json()
        self.assertEqual(resumed["status"], RunStatus.COMPLETED.value)
        self.assertTrue(resumed["brief"]["headline"])

        stored = self.client.get(f"/api/sessions/{session_id}").json()
        self.assertEqual(stored["session_id"], session_id)
        self.assertTrue(stored["audit_trail"])
        self.assertTrue(stored["findings"])

    def test_reject_closes_the_session(self):
        created = self.start_session()
        resumed = self.client.post(
            f"/api/sessions/{created['session_id']}/resume",
            json={"decision": "reject", "note": "tidak cukup"}).json()
        self.assertEqual(resumed["status"], RunStatus.REJECTED.value)

    def test_revise_returns_to_the_planner(self):
        created = self.start_session()
        before = created["revision"]
        resumed = self.client.post(
            f"/api/sessions/{created['session_id']}/resume",
            json={"decision": "revise", "note": "periksa konsentrasi"}).json()
        self.assertGreater(resumed["revision"], before)

    def test_unknown_session_is_404(self):
        self.assertEqual(self.client.get("/api/sessions/does-not-exist").status_code, 404)

    def test_invalid_question_is_422(self):
        response = self.client.post("/api/sessions", json={"question": "x"})
        self.assertEqual(response.status_code, 422)

    def test_invalid_decision_is_422(self):
        created = self.start_session()
        response = self.client.post(
            f"/api/sessions/{created['session_id']}/resume",
            json={"decision": "perhaps"})
        self.assertEqual(response.status_code, 422)

    def test_sessions_are_listed(self):
        self.start_session()
        body = self.client.get("/api/sessions?limit=5").json()
        self.assertTrue(body["sessions"])
        self.assertIn("status", body["sessions"][0])


class TimeTravelTest(ApiTest):
    def test_history_is_exposed(self):
        created = self.start_session()
        self.client.post(f"/api/sessions/{created['session_id']}/resume",
                         json={"decision": "approve"})
        body = self.client.get(f"/api/sessions/{created['session_id']}/timeline").json()
        self.assertGreater(len(body["checkpoints"]), 3)
        self.assertIn("checkpoint_id", body["checkpoints"][0])

    def test_state_at_checkpoint(self):
        created = self.start_session()
        self.client.post(f"/api/sessions/{created['session_id']}/resume",
                         json={"decision": "approve"})
        checkpoints = self.client.get(
            f"/api/sessions/{created['session_id']}/timeline").json()["checkpoints"]
        oldest = checkpoints[-1]["checkpoint_id"]
        body = self.client.get(
            f"/api/sessions/{created['session_id']}/timeline/{oldest}").json()
        self.assertEqual(body["checkpoint_id"], oldest)
        self.assertIn("state", body)

    def test_fork_creates_an_independent_branch(self):
        created = self.start_session()
        session_id = created["session_id"]
        self.client.post(f"/api/sessions/{session_id}/resume", json={"decision": "approve"})
        checkpoints = self.client.get(
            f"/api/sessions/{session_id}/timeline").json()["checkpoints"]
        forkable = [cp for cp in checkpoints if cp.get("forkable")]
        self.assertTrue(forkable)
        forked = self.client.post(
            f"/api/sessions/{session_id}/fork",
            json={"checkpoint_id": forkable[-1]["checkpoint_id"]}).json()
        self.assertNotEqual(forked["session_id"], session_id)
        self.assertIn(forked["status"], (RunStatus.COMPLETED.value,
                                         RunStatus.AWAITING_HUMAN.value))
        # The original session must be untouched by the branch.
        original = self.client.get(f"/api/sessions/{session_id}").json()
        self.assertEqual(original["status"], RunStatus.COMPLETED.value)

    def test_forking_from_the_pre_request_checkpoint_is_400(self):
        created = self.start_session()
        checkpoints = self.client.get(
            f"/api/sessions/{created['session_id']}/timeline").json()["checkpoints"]
        empty = [cp for cp in checkpoints if not cp.get("forkable")]
        self.assertTrue(empty)
        response = self.client.post(
            f"/api/sessions/{created['session_id']}/fork",
            json={"checkpoint_id": empty[-1]["checkpoint_id"]})
        self.assertEqual(response.status_code, 400)

    def test_unknown_checkpoint_is_404(self):
        created = self.start_session()
        response = self.client.get(f"/api/sessions/{created['session_id']}/timeline/nope")
        self.assertEqual(response.status_code, 404)


class StreamTest(ApiTest):
    def test_sse_stream_emits_nodes_and_a_result(self):
        with self.client.stream("GET", "/api/sessions/stream",
                                params={"question": QUESTION, "seed": 7,
                                        "acceptance_threshold": "0.5",
                                        "hitl": "true"}) as response:
            self.assertEqual(response.status_code, 200)
            self.assertIn("text/event-stream", response.headers["content-type"])
            body = response.read().decode("utf-8")
        self.assertIn("event: start", body)
        self.assertIn("event: node", body)
        self.assertTrue("event: interrupt" in body or "event: final" in body)


if __name__ == "__main__":
    unittest.main()
