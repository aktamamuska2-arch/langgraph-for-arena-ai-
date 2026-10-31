"""Artifact size gate — the acceptance test for PRD section 10.

This is a real test, not documentation: if a dependency creeps past the budget,
the suite fails before the artifact is ever shipped.
"""

from __future__ import annotations

import unittest

from sigma.config import SETTINGS
from sigma.size import audit as size_audit


class SizeBudgetTest(unittest.TestCase):
    def test_total_is_under_the_hard_budget(self):
        report = size_audit.measure()
        self.assertLess(report.total_mb, SETTINGS.size_budget_mb,
                        f"artifact is {report.total_mb} MB, budget "
                        f"{SETTINGS.size_budget_mb} MB\n{size_audit.render(report)}")

    def test_budget_is_125_mb(self):
        """The constraint from the brief is encoded, not remembered."""
        self.assertEqual(SETTINGS.size_budget_mb, 125.0)

    def test_every_component_is_measured(self):
        report = size_audit.measure()
        for value in (report.dependencies_mb, report.code_mb, report.assets_mb,
                      report.database_mb):
            self.assertGreaterEqual(value, 0.0)
        self.assertAlmostEqual(
            report.total_mb,
            round(report.dependencies_mb + report.code_mb + report.assets_mb
                  + report.database_mb, 2), places=1)

    def test_code_and_assets_stay_small(self):
        report = size_audit.measure()
        self.assertLess(report.code_mb, 5.0, "application source is bloated")
        self.assertLess(report.assets_mb, 2.0, "static assets are bloated")

    def test_dependency_footprint_has_headroom(self):
        """Pruning is only meaningful if it leaves room to grow."""
        report = size_audit.measure()
        self.assertLess(report.dependencies_mb, 90.0)
        self.assertGreater(report.headroom_mb, 10.0)

    def test_no_forbidden_heavyweight_is_installed(self):
        """PRD section 10.4: the packages that would blow the budget."""
        forbidden = ("streamlit", "pandas", "numpy", "torch", "transformers",
                     "scipy", "sklearn", "openai", "anthropic")
        packages = size_audit.site_packages_dir()
        installed = set()
        if packages.exists():
            for child in packages.iterdir():
                name = child.name.lower()
                for candidate in forbidden:
                    if name == candidate or name.startswith(candidate + "-"):
                        installed.add(candidate)
        self.assertEqual(installed, set(), f"forbidden packages installed: {installed}")

    def test_render_reports_a_verdict(self):
        report = size_audit.measure()
        rendered = size_audit.render(report)
        self.assertIn("RESULT: PASS" if report.passed else "RESULT: FAIL", rendered)
        self.assertIn("TOTAL", rendered)


if __name__ == "__main__":
    unittest.main()
