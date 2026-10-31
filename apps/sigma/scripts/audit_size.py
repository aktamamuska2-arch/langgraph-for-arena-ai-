#!/usr/bin/env python3
"""Artifact size audit CLI.

    python apps/sigma/scripts/audit_size.py [--venv PATH] [--budget-mb 125] [--json]

Exits non-zero when the artifact exceeds the budget, so it can be used as a CI
gate or as the last step of a build.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sigma.size import audit as size_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit SIGMA artifact size.")
    parser.add_argument("--budget-mb", type=float, default=None,
                        help="override the budget (default: SIGMA_SIZE_BUDGET_MB / 125)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--no-deps", action="store_true",
                        help="measure code and assets only (no site-packages)")
    args = parser.parse_args()

    report = size_audit.measure(budget_mb=args.budget_mb,
                                include_dependencies=not args.no_deps)
    if args.json:
        print(json.dumps({
            "dependencies_mb": report.dependencies_mb,
            "code_mb": report.code_mb,
            "assets_mb": report.assets_mb,
            "database_mb": report.database_mb,
            "total_mb": report.total_mb,
            "budget_mb": report.budget_mb,
            "headroom_mb": report.headroom_mb,
            "passed": report.passed,
            "largest": report.largest,
        }, indent=2))
    else:
        print(size_audit.render(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
