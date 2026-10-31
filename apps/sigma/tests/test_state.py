"""State schema and reducer tests.

The reducers are the load-bearing part of the cyclic design: if a growing field
fell back to last-value semantics, evidence from earlier iterations would vanish
and the self-correction loop would silently lose its memory (AC-CYCLE-3).
"""

from __future__ import annotations

import unittest

from sigma.state import (
    BudgetState,
    Evaluation,
    Evidence,
    Finding,
    Gap,
    RunStatus,
    bounded_add,
    consume_budget,
    merge_budget,
    merge_scalar_status,
    merge_unique_by,
    supersede_by,
)


class BoundedAddTest(unittest.TestCase):
    def test_appends_in_order(self):
        reduce = bounded_add(10)
        self.assertEqual(reduce([1, 2], [3]), [1, 2, 3])

    def test_keeps_newest_when_over_cap(self):
        reduce = bounded_add(3)
        self.assertEqual(reduce([1, 2, 3], [4, 5]), [3, 4, 5])

    def test_handles_none_side(self):
        reduce = bounded_add(3)
        self.assertEqual(reduce(None, [1]), [1])
        self.assertEqual(reduce([1], None), [1])


class MergeUniqueByTest(unittest.TestCase):
    def test_dedups_by_key_and_newest_wins(self):
        reduce = merge_unique_by("call_id")
        older = type("C", (), {"call_id": "a", "tool": "old"})()
        newer = type("C", (), {"call_id": "a", "tool": "new"})()
        other = type("C", (), {"call_id": "b", "tool": "x"})()
        merged = reduce([older, other], [newer])
        # Insertion order is preserved; the newer value wins for a repeated key.
        self.assertEqual([c.call_id for c in merged], ["a", "b"])
        self.assertEqual([c.tool for c in merged], ["new", "x"])


class SupersedeByTest(unittest.TestCase):
    def test_replaced_finding_is_marked_not_deleted(self):
        reduce = supersede_by("finding_id")
        first = Finding(finding_id="f1", claim="lama", metric=1.0)
        second = Finding(finding_id="f2", claim="baru", metric=2.0, supersedes="f1")
        merged = reduce([first], [second])
        self.assertEqual(len(merged), 2)  # history preserved
        self.assertTrue(merged[0].superseded)
        self.assertFalse(merged[1].superseded)
        self.assertEqual([f for f in merged if f.active], [second])


class BudgetTest(unittest.TestCase):
    def test_consume_is_monotonic(self):
        budget = BudgetState(max_revisions=3, max_tool_calls=10)
        spent = consume_budget(budget, revisions=1, tools=4, tokens=100)
        self.assertEqual(spent.revisions_used, 1)
        self.assertEqual(spent.tool_calls_used, 4)
        self.assertEqual(spent.revisions_left, 2)

    def test_merge_never_refunds_usage(self):
        """A node replay must not give a spent budget back."""
        spent = BudgetState(max_revisions=3, revisions_used=2, max_tool_calls=10,
                            tool_calls_used=5)
        fresh = BudgetState(max_revisions=3, revisions_used=0, max_tool_calls=10,
                            tool_calls_used=0)
        merged = merge_budget(spent, fresh)
        self.assertEqual(merged.revisions_used, 2)
        self.assertEqual(merged.tool_calls_used, 5)

    def test_merge_takes_smallest_headroom(self):
        merged = merge_budget(BudgetState(max_revisions=5), BudgetState(max_revisions=2))
        self.assertEqual(merged.max_revisions, 2)

    def test_exhausted_flag(self):
        self.assertTrue(BudgetState(max_revisions=1, revisions_used=1).exhausted)
        self.assertFalse(BudgetState(max_revisions=1, revisions_used=0).exhausted)


class StatusTest(unittest.TestCase):
    def test_terminal_status_is_sticky(self):
        self.assertEqual(
            merge_scalar_status(RunStatus.COMPLETED, RunStatus.RUNNING),
            RunStatus.COMPLETED)

    def test_running_is_replaced(self):
        self.assertEqual(
            merge_scalar_status(RunStatus.RUNNING, RunStatus.AWAITING_HUMAN),
            RunStatus.AWAITING_HUMAN)


class EvaluationTest(unittest.TestCase):
    def _eval(self, gaps: list[Gap]) -> Evaluation:
        return Evaluation(revision=0, score=0.5, gaps=gaps)

    def test_dominant_gap_is_weighted(self):
        ev = self._eval([Gap(kind="interpretation", weight=3.0),
                         Gap(kind="missing_evidence", weight=1.0)])
        self.assertEqual(ev.dominant_gap, "interpretation")

    def test_no_gaps_dominant(self):
        self.assertIsNone(self._eval([]).dominant_gap)

    def test_zero_weight_gap_is_not_dominant(self):
        ev = self._eval([Gap(kind="structural", weight=0.0)])
        self.assertEqual(ev.dominant_gap, "structural")


class FindingTest(unittest.TestCase):
    def test_numeric_claim_without_evidence_is_not_grounded(self):
        self.assertFalse(Finding(claim="x", metric=1.0, evidence_ids=[]).grounded)
        self.assertTrue(Finding(claim="x", metric=1.0, evidence_ids=["e1"]).grounded)

    def test_qualitative_claim_is_grounded_by_default(self):
        self.assertTrue(Finding(claim="x", metric=None).grounded)


class EvidenceTest(unittest.TestCase):
    def test_payload_is_json_serialisable(self):
        ev = Evidence(tool="query_portfolio", payload={"count": 3, "ventures": [{"id": "V1"}]},
                      provenance="tool:query_portfolio#abc123")
        self.assertEqual(ev.payload["ventures"][0]["id"], "V1")
        self.assertIn("tool:query_portfolio#", ev.provenance)


if __name__ == "__main__":
    unittest.main()
