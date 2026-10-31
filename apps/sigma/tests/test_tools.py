"""Tool registry tests: validation, bounding, determinism, failure classification."""

from __future__ import annotations

import time
import unittest

from sigma.errors import ToolExecutionError
from sigma.tools.market import (
    build_portfolio,
    compute_risk_metrics,
    query_portfolio,
    scenario_simulate,
)
from sigma.tools.registry import ArgsModel, ToolCall, ToolContext, ToolRegistry, ToolSpec


class PortfolioTest(unittest.TestCase):
    def test_same_seed_same_world(self):
        self.assertEqual(build_portfolio("p", 7), build_portfolio("p", 7))

    def test_different_seed_different_world(self):
        self.assertNotEqual(build_portfolio("p", 7), build_portfolio("p", 8))

    def test_shape(self):
        ventures = build_portfolio("p", 3)
        self.assertEqual(len(ventures), 18)
        for key in ("id", "sector", "stage", "capital_deployed", "growth_rate"):
            self.assertIn(key, ventures[0])


class AnalyticsTest(unittest.TestCase):
    def test_risk_score_is_bounded(self):
        metrics = compute_risk_metrics("p", 7, venture_ids=["V01", "V02", "V03"])
        self.assertGreaterEqual(metrics["risk_score"], 0.0)
        self.assertLessEqual(metrics["risk_score"], 1.0)
        self.assertGreater(metrics["concentration_hhi"], 0.0)

    def test_unknown_ids_do_not_crash(self):
        self.assertIn("error", compute_risk_metrics("p", 7, venture_ids=["NOPE"]))

    def test_simulation_is_deterministic(self):
        args = dict(venture_ids=["V01", "V02"], paths=400, horizon_quarters=8)
        first = scenario_simulate("p", 7, **args)
        second = scenario_simulate("p", 7, **args)
        self.assertEqual(first, second)

    def test_simulation_percentiles_are_ordered(self):
        out = scenario_simulate("p", 7, venture_ids=["V01", "V02", "V03"], paths=500)
        self.assertLessEqual(out["p10_multiple"], out["p50_multiple"])
        self.assertLessEqual(out["p50_multiple"], out["p90_multiple"])
        self.assertGreaterEqual(out["var95"], 0.0)
        self.assertGreaterEqual(out["prob_loss"], 0.0)

    def test_query_portfolio_filters(self):
        out = query_portfolio("p", 7, sector="fintech", limit=5)
        self.assertLessEqual(out["count"], 5)
        self.assertTrue(all(v["sector"] == "fintech" for v in out["ventures"]))


class RegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry()
        self.ctx = ToolContext(portfolio_id="p", seed=7)

    def test_schemas_are_json_serialisable(self):
        schemas = self.registry.schemas()
        self.assertEqual({s["name"] for s in schemas}, set(self.registry.names))
        for schema in schemas:
            self.assertIn("parameters", schema)
            self.assertEqual(schema["parameters"]["type"], "object")

    def test_successful_call_records_a_digest(self):
        result = self.registry.execute(
            ToolCall(tool="query_portfolio", args={"limit": 3}), self.ctx)
        self.assertTrue(result.ok, result.record.error)
        self.assertGreaterEqual(result.record.duration_ms, 0)
        self.assertIn("count", result.record.output_digest)

    def test_invalid_arguments_are_classified_not_raised(self):
        result = self.registry.execute(
            ToolCall(tool="query_portfolio", args={"limit": 9999}), self.ctx)
        self.assertFalse(result.ok)
        self.assertFalse(result.timed_out)
        self.assertIn("invalid arguments", result.record.error or "")

    def test_unknown_tool_is_classified(self):
        result = self.registry.execute(ToolCall(tool="nope", args={}), self.ctx)
        self.assertFalse(result.ok)
        self.assertIn("unknown tool", result.record.error or "")

    def test_timeout_is_flagged_for_the_retry_policy(self):
        """A timed-out call must be distinguishable from a bad-argument call."""

        class SlowArgs(ArgsModel):
            pass

        def slow(_ctx: ToolContext, _args: ArgsModel) -> dict:
            time.sleep(0.05)
            return {"ok": True}

        slow_spec = ToolSpec("slow_tool", "always too slow", SlowArgs, slow)
        tight = ToolRegistry(tools=(slow_spec,), timeout_ms=1)
        result = tight.execute(ToolCall(tool="slow_tool", args={}), self.ctx)
        self.assertFalse(result.ok)
        self.assertTrue(result.timed_out)
        self.assertIn("exceeded", result.record.error or "")

    def test_spec_lookup_raises_for_unknown_tool(self):
        with self.assertRaises(ToolExecutionError):
            self.registry.spec("nope")

    def test_evidence_carries_provenance(self):
        call = ToolCall(tool="query_portfolio", args={"limit": 3}, sub_question_id="q1")
        result = self.registry.execute(call, self.ctx)
        evidence = result.to_evidence(call.sub_question_id)
        self.assertEqual(evidence.sub_question_id, "q1")
        self.assertEqual(evidence.provenance, f"tool:query_portfolio#{call.call_id}")


if __name__ == "__main__":
    unittest.main()
