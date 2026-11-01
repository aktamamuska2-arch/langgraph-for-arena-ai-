#!/usr/bin/env python3
"""Execute the AM-SES build graph end-to-end from the command line.

Examples
--------
    python scripts/run_workflow.py
    python scripts/run_workflow.py --fault-mode runtime
    python scripts/run_workflow.py --fault-mode timeout --fault-persist --max-retries 3
    python scripts/run_workflow.py --scenario custom.json --mermaid
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maritime_sentinel.config import DEFAULT_SCENARIO  # noqa: E402
from maritime_sentinel.graph import run_workflow  # noqa: E402
from maritime_sentinel.graph.build import graph_mermaid  # noqa: E402
from maritime_sentinel.schemas import ScenarioSpec  # noqa: E402

C = {
    "reset": "\033[0m",
    "dim": "\033[2m",
    "bold": "\033[1m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "magenta": "\033[35m",
}
LEVEL_COLOR = {"info": "cyan", "warning": "yellow", "error": "red", "debug": "dim"}
VERDICT_COLOR = {"PASS": "green", "PASS_WITH_RETRIES": "cyan", "FAILED_NEEDS_REVIEW": "red"}


def paint(text: str, color: str) -> str:
    return f"{C.get(color, '')}{text}{C['reset']}"


def banner(title: str) -> None:
    print()
    print(paint("=" * 88, "dim"))
    print(paint(f"  {title}", "bold"))
    print(paint("=" * 88, "dim"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the maritime-sentinel AM-SES build graph.")
    parser.add_argument("--scenario", type=Path, help="JSON file with a ScenarioSpec payload")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sandbox-timeout", type=float, default=90.0)
    parser.add_argument(
        "--fault-mode",
        default="none",
        choices=["none", "schema", "runtime", "divergence", "timeout"],
        help="Inject a synthetic failure into the first build attempt",
    )
    parser.add_argument("--fault-persist", action="store_true", help="Inject the fault on every attempt")
    parser.add_argument("--mermaid", action="store_true", help="Print the compiled graph as Mermaid")
    parser.add_argument("--json", type=Path, default=None, help="Write the final state to this path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    scenario = None
    if args.scenario:
        scenario = ScenarioSpec.model_validate_json(args.scenario.read_text())
    else:
        scenario = DEFAULT_SCENARIO

    banner(f"AM-SES v2.0 // {scenario.spec_id} // {scenario.title}")
    print(
        f"  horizon={scenario.stress.horizon_days:.0f}d  warmup={scenario.stress.warmup_days:.0f}d  "
        f"weather_intensity={scenario.stress.weather_intensity:.2f}  "
        f"geo_intensity={scenario.stress.geo_intensity:.2f}  season={scenario.stress.season}"
    )

    if args.mermaid:
        from maritime_sentinel.graph.state import RunContext

        ctx = RunContext(
            run_id="preview",
            project_root=PROJECT_ROOT,
            artifacts_root=PROJECT_ROOT / "artifacts",
        )
        print()
        print(graph_mermaid(ctx))
        return 0

    state, context = run_workflow(
        scenario,
        project_root=PROJECT_ROOT,
        max_retries=args.max_retries,
        sandbox_timeout=args.sandbox_timeout,
        fault_mode=args.fault_mode,
        fault_persist=args.fault_persist,
        run_id=args.run_id,
    )

    if not args.quiet:
        banner("STATE GRAPH TRACE")
        for entry in state.get("execution_logs", []):
            color = LEVEL_COLOR.get(entry.get("level", "info"), "cyan")
            node = entry.get("node", "?").replace("node_", "")
            print(f"  {paint(f'{node:<16}', color)} {entry.get('message', '')}")
            payload = entry.get("payload", {})
            if node == "reflector":
                for patch in payload.get("patches", []):
                    print(
                        f"      {paint('->', 'magenta')} {patch['pointer']}: "
                        f"{patch['previous']} => {patch['updated']}"
                    )
                    print(f"         {paint(patch['rationale'], 'dim')}")
            elif node == "sandbox_tester":
                failing = payload.get("failing") or []
                if failing:
                    print(f"      {paint('gates failed: ' + ', '.join(failing), 'red')}")
                else:
                    print(f"      {paint('all gates satisfied', 'green')}")

    report = state.get("test_report") or {}
    banner("ACCEPTANCE GATES")
    for criterion in report.get("criteria", []):
        mark = paint("PASS", "green") if criterion["passed"] else paint("FAIL", "red")
        print(
            f"  [{mark}] {criterion['id']:<10} {criterion['metric_key']:<24} "
            f"observed={criterion['observed']:.4f}  {criterion['comparator']} {criterion['threshold']}"
        )

    verdict = state.get("verdict", "UNKNOWN")
    banner(f"VERDICT: {verdict}   attempts={state.get('retry_count', 0) + 1}   artifacts={context.run_dir()}")
    print(paint(f"  final report: {context.run_dir() / 'final_report.json'}", "dim"))
    print(paint(f"  generated build: {context.attempt_dir(state.get('retry_count', 0))}", "dim"))

    if args.json:
        args.json.write_text(json.dumps(state, indent=2, default=str))
        print(paint(f"  state written to {args.json}", "dim"))

    return 0 if verdict.startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
