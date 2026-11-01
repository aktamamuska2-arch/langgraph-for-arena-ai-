#!/usr/bin/env python3
"""Compile the world: run the full Planner -> Synthesizer -> Tester -> Reflector loop.

    python scripts/run_workflow.py --seed 20261101

Writes frame renders, gate metrics, the blueprint and the final report into
``artifacts/`` and prints the verdict.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agora.graph import run_pipeline  # noqa: E402
from agora.schemas import Directive  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ASEOWS build pipeline end to end.")
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--horizon", type=int, default=420)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--out", default=str(ROOT / "artifacts"))
    parser.add_argument("--json", action="store_true", help="print the final report as JSON")
    args = parser.parse_args(argv)

    directive = Directive(
        seed=args.seed,
        horizon=args.horizon,
        max_retries=args.max_retries,
        raw_text="Build an emergent spatial-economic open world.",
    )
    result = run_pipeline(directive, output_dir=args.out)

    print("=" * 78)
    print(" BUILD LOG")
    print("=" * 78)
    for line in result["log"]:
        print(line)

    report = result["report"]
    print()
    print("=" * 78)
    print(" GATES")
    print("=" * 78)
    for row in report.results:
        mark = "PASS" if row.passed else "FAIL"
        print(f"  [{mark}] {row.id:<12} {row.metric:<24} {row.observed:>12.6g} {row.comparator} {row.threshold}")

    print()
    print(f"  verdict      : {result['verdict']}")
    print(f"  attempts     : {report.attempt + 1}")
    print(f"  failure class: {report.failure_class}")
    print(f"  artifacts    : {len(result['artifacts'])} files in {args.out}")

    if result["reflections"]:
        print()
        print(" REFLECTIONS")
        for reflection in result["reflections"]:
            print(f"  attempt {reflection.attempt} ({reflection.failure_class}): {reflection.diagnosis}")
            for patch in reflection.patches:
                print(f"    - {patch.pointer}: {patch.previous} -> {patch.updated}")

    if args.json:
        print()
        print(report.model_dump_json(indent=2))

    return 0 if result["verdict"] != "FAILED_NEEDS_REVIEW" else 1


if __name__ == "__main__":
    raise SystemExit(main())
