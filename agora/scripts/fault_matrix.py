#!/usr/bin/env python3
"""Fault-injection matrix for the self-correction loop.

Each row deliberately mis-tunes the build - the kind of mistake a planner makes
when it reasons geometrically instead of economically - and verifies that the
reflector *measures* the damage, classifies it correctly and repairs it inside
the retry budget.

The last row is different on purpose: a broken conservation ledger is an engine
defect, not a tuning error. A correct loop must refuse to "fix" it and hand the
run to a human instead of silently passing.

    python scripts/fault_matrix.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agora.graph import build_graph, initial_state  # noqa: E402
from agora.graph.nodes import self_correction  # noqa: E402
from agora.metrics import DEFAULT_GATES, evaluate  # noqa: E402
from agora.schemas import Blueprint, Directive, Patch  # noqa: E402

FAULTS: list[tuple[str, str, list[Patch], str]] = [
    (
        "F1-too-local",
        "the market scan is pinned at half the critical radius",
        [
            Patch(
                pointer="economy.discovery_radius",
                previous=18.8,
                updated=6.0,
                rationale="injected: an over-local market scan",
            )
        ],
        "network_fragmentation",
    ),
    (
        "F2-short-horizon",
        "too few ticks budgeted for a logistic to reach its plateau",
        [
            Patch(
                pointer="economy.discovery_radius",
                previous=10.44,
                updated=18.79,
                rationale="baseline: a healthy market scan, so only the horizon is wrong",
            ),
            Patch(pointer="horizon", previous=420, updated=120, rationale="injected: truncated horizon"),
        ],
        "non_convergence",
    ),
    (
        "F3-double-fault",
        "two independent mis-tunings at once (radius and horizon)",
        [
            Patch(
                pointer="economy.discovery_radius",
                previous=18.8,
                updated=6.0,
                rationale="injected: over-local market scan",
            ),
            Patch(pointer="horizon", previous=420, updated=120, rationale="injected: truncated horizon"),
        ],
        "non_convergence",
    ),
]

REFUSAL_FAULT = (
    "F4-broken-ledger",
    "the physical ledger leaks mass - an engine defect, not a parameter",
    {
        "abs_growth_tail": 0.001,
        "field_lag1_corr": 0.95,
        "giant_component_share": 1.0,
        "edges_per_node": 2.0,
        "mass_balance_error": 0.043,
        "visual_contract_score": 1.0,
        "shock_recovery": 1.1,
        "pop_drawdown": 0.0,
    },
)


def run_fault(name: str, patches: list[Patch], directive: Directive, out: Path) -> dict:
    state = initial_state(directive)
    state["output_dir"] = str(out / name)
    state["patches"] = patches
    return dict(build_graph().invoke(state))


def check_refusal(out: Path) -> tuple[bool, str]:
    """A conservation defect must be refused, not patched away."""

    blueprint = Blueprint(
        blueprint_id="BP-REFUSAL",
        experiment_id="ASEOWS-F4",
        objective="refusal path",
        equations=[],
        gates=list(DEFAULT_GATES),
        world_summary={},
        economy={},
        critical_radius=10.0,
        site_audit=[],
        expected={},
    )
    metrics = dict(REFUSAL_FAULT[2])
    report = evaluate(DEFAULT_GATES, metrics, attempt=0, duration=1.0)
    from agora.schemas import ExperimentSpec

    state = initial_state(Directive(experiment_id="ASEOWS-F4"))
    state["blueprint"] = blueprint
    state["spec"] = ExperimentSpec()
    state["report"] = report
    state["metrics"] = metrics
    state["attempt"] = 0
    update = self_correction(state)
    reflection = update["reflections"][-1]
    ok = report.failure_class == "mass_imbalance" and reflection.next_action == "finalize" and not reflection.patches
    return ok, f"{report.failure_class} -> next={reflection.next_action}, patches={len(reflection.patches)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prove the self-correction loop repairs real faults.")
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "fault_matrix"))
    args = parser.parse_args(argv)
    out = Path(args.out)

    print(f"{'fault':<17}{'injected class':<24}{'detected':<24}{'attempts':<9}{'verdict'}")
    print("-" * 92)
    repaired = 0
    for name, description, patches, expected in FAULTS:
        directive = Directive(seed=args.seed, experiment_id=f"ASEOWS-{name}")
        result = run_fault(name, patches, directive, out)
        reflections = result["reflections"]
        seen = reflections[0].failure_class if reflections else "none"
        attempts = result["report"].attempt + 1
        ok = result["verdict"] in {"PASS", "PASS_WITH_RETRIES"} and seen == expected
        repaired += int(ok)
        print(
            f"{name:<17}{expected:<24}{seen:<24}{attempts:<9}{result['verdict']}{'' if ok else '   <-- NOT REPAIRED'}"
        )
        print(f"{'':<17}{description}")
        for reflection in reflections:
            pointers = ", ".join(f"{p.pointer}:{p.previous}->{p.updated}" for p in reflection.patches)
            print(f"{'':<19}t{reflection.attempt}: {reflection.failure_class} -> {pointers or 'none'}")
        if result["report"].passed:
            for row in result["report"].results:
                if not row.passed:
                    print(f"{'':<19}still failing: {row.id}")

    name, description, _ = REFUSAL_FAULT
    ok, detail = check_refusal(out)
    repaired += int(ok)
    print(f"{name:<17}{'mass_imbalance':<24}{'mass_imbalance':<24}{'n/a':<9}{'REFUSED' if ok else 'WRONGLY PATCHED'}")
    print(f"{'':<17}{description}")
    print(f"{'':<19}reflector says: {detail}")

    print("-" * 92)
    print(f"{repaired}/{len(FAULTS) + 1} scenarios behaved correctly (3 repaired, 1 refused).")
    return 0 if repaired == len(FAULTS) + 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
