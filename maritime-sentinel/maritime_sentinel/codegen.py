"""Code synthesis: turn an architectural blueprint into an executable artifact.

The synthesizer node does not string-format a toy script. It emits a complete,
self-describing build directory:

``scenario_manifest.json``  validated scenario + deployment plan (single source of truth)
``generated_simulation.py`` standalone runner executed inside the sandbox
``blueprint.md``            human-readable design memo with the equations
``acceptance.json``         the pass/fail gates the sandbox will enforce

Every generated file is hashed so the report can prove exactly what was tested.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import (
    ArchitecturalBlueprint,
    ArtifactRecord,
    DeploymentPlan,
    ScenarioSpec,
)

# Tokens that must never appear in a generated artifact. The sandbox already
# isolates the process, this is defence in depth.
BANNED_TOKENS: tuple[str, ...] = (
    "subprocess",
    "os.system",
    "os.popen",
    "socket.",
    "shutil.rmtree",
    "urllib",
    "requests.",
    "eval(",
    "exec(",
    "__import__",
    "pickle.loads",
    "ctypes",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass
class ArtifactBundle:
    build_dir: Path
    records: list[ArtifactRecord]
    manifest: dict[str, Any]

    def as_state(self) -> dict[str, Any]:
        return {
            "build_dir": str(self.build_dir),
            "artifacts": [r.model_dump(mode="json") for r in self.records],
            "manifest_path": str(self.build_dir / "scenario_manifest.json"),
            "entrypoint": str(self.build_dir / "generated_simulation.py"),
        }


def derive_deployment(
    blueprint: ArchitecturalBlueprint,
    spec: ScenarioSpec,
    seed: int,
    capacity_scale: float,
) -> DeploymentPlan:
    """Allocate berths according to the blueprint's sizing strategy.

    ``proportional``  physical berths scaled by the scenario capacity factor
                      (the naive first draft).
    ``tail_risk``     the Erlang-C sizing computed by the planner, i.e. the
                      number of berths required to hold the wait target under
                      the *stationary stressed productivity*, plus a resilience
                      margin. This is what the reflector installs after a
                      utilisation or service-level failure.
    """

    by_port = {s.port_id: s for s in blueprint.sizing}
    berths: dict[str, int] = {}
    for port in spec.network.ports:
        proportional = max(1, int(round(port.berths * capacity_scale)))
        decision = by_port.get(port.id)
        if blueprint.sizing_strategy == "tail_risk" and decision is not None:
            tail = max(1, int(round(decision.required_berths * blueprint.resilience_margin)))
            berths[port.id] = max(proportional, tail)
        else:
            berths[port.id] = proportional
    notes = [
        f"sizing_strategy={blueprint.sizing_strategy}",
        f"capacity_scale={capacity_scale:.3f}",
        f"resilience_margin={blueprint.resilience_margin:.3f}",
        f"wait_target_hours={blueprint.wait_target_hours:.1f}",
    ]
    return DeploymentPlan(
        plan_id=f"{blueprint.blueprint_id}-{blueprint.sizing_strategy}",
        blueprint_id=blueprint.blueprint_id,
        strategy=blueprint.sizing_strategy,
        berths=berths,
        capacity_scale=float(capacity_scale),
        routing_policy=spec.stress.routing_policy,
        seed=int(seed),
        horizon_days=float(spec.stress.horizon_days),
        warmup_days=float(spec.stress.warmup_days),
        notes=notes,
    )


RUNNER_TEMPLATE = '''#!/usr/bin/env python3
"""AUTO-GENERATED SIMULATION ARTIFACT - do not edit by hand.

blueprint : {blueprint_id}
scenario  : {spec_id}
strategy  : {strategy}
attempt   : {attempt}
generated : {generated_at}

This runner is executed by node_sandbox_tester in an isolated interpreter
(``python -I``) with CPU/memory rlimits and a wall-clock timeout. It validates
its own manifest, replays one deterministic replication of the discrete-event
engine and emits a single machine-readable result line on stdout.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

_PROJECT_ROOT = {project_root!r}
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from maritime_sentinel.engine import simulate  # noqa: E402
from maritime_sentinel.metrics import compute_metrics  # noqa: E402
from maritime_sentinel.schemas import DeploymentPlan, ScenarioSpec  # noqa: E402

MANIFEST_PATH = Path(__file__).resolve().with_name("scenario_manifest.json")
MANIFEST = json.loads(MANIFEST_PATH.read_text())
RESULT_PREFIX = "__RESULT__"


def emit(payload: dict) -> None:
    print(RESULT_PREFIX + json.dumps(payload, default=float))


def main() -> int:
    started = time.perf_counter()
    try:
        spec = ScenarioSpec.model_validate(MANIFEST["scenario"])
        deployment = DeploymentPlan.model_validate(MANIFEST["deployment"])
        result = simulate(spec, deployment=deployment, seed=deployment.seed)
        metrics = compute_metrics(result)
        payload = {{
            "ok": not result.diverged,
            "diverged": bool(result.diverged),
            "divergence_reason": result.divergence_reason,
            "metrics": metrics,
            "ports": result.ports,
            "vessels_served": int(len(result.vessels)),
            "engine_runtime_s": round(time.perf_counter() - started, 4),
            "manifest_sha256": hashlib.sha256(
                json.dumps(MANIFEST, sort_keys=True).encode()
            ).hexdigest(),
        }}
        emit(payload)
        return 0 if not result.diverged else 3
    except Exception as exc:  # noqa: BLE001 - the sandbox needs the failure shape
        traceback.print_exc()
        emit({{
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "manifest_sha256": hashlib.sha256(
                json.dumps(MANIFEST, sort_keys=True).encode()
            ).hexdigest(),
        }})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


def render_runner(manifest: dict[str, Any], project_root: Path, attempt: int) -> str:
    return RUNNER_TEMPLATE.format(
        blueprint_id=manifest["blueprint_id"],
        spec_id=manifest["scenario"]["spec_id"],
        strategy=manifest["deployment"]["strategy"],
        attempt=attempt,
        generated_at=manifest["generated_at"],
        project_root=str(project_root),
    )


def render_blueprint_md(blueprint: ArchitecturalBlueprint, spec: ScenarioSpec) -> str:
    lines: list[str] = []
    add = lines.append
    add(f"# Architectural Blueprint `{blueprint.blueprint_id}`")
    add("")
    add(f"**Scenario** `{spec.spec_id}` - {spec.title}")
    add("")
    add(f"**Objective** {blueprint.objective}")
    add("")
    add(
        f"**Sizing strategy** `{blueprint.sizing_strategy}` (resilience margin {blueprint.resilience_margin:.2f}, "
        f"wait target {blueprint.wait_target_hours:.1f} h)"
    )
    add("")
    add("## State-space formulation")
    add("")
    add("| Name | Domain | Expression | Description |")
    add("| --- | --- | --- | --- |")
    for eq in blueprint.equations:
        add(f"| `{eq.name}` | {eq.domain} | `{eq.latex}` | {eq.description} |")
    add("")
    add("## Network signature")
    add("")
    for key, value in blueprint.network_signature.items():
        add(f"- **{key}**: `{value}`")
    add("")
    add("## Capacity sizing (Erlang C)")
    add("")
    add(
        "| Port | Base berths | Demand (berth-days/day) | Effective demand | Required berths | rho | Wq (h) | Meets target |"
    )
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for s in blueprint.sizing:
        add(
            f"| {s.port_id} | {s.base_berths} | {s.demand_berth_days:.2f} | {s.effective_berth_days:.2f} | "
            f"{s.required_berths} | {s.projected_utilization:.3f} | {s.projected_wait_hours:.2f} | "
            f"{'yes' if s.meets_target else 'no'} |"
        )
    add("")
    add("## Module decomposition")
    add("")
    for mod in blueprint.modules:
        add(f"- `{mod.path}` - {mod.role} (~{mod.loc_estimate} LOC)")
    add("")
    add("## Acceptance gates")
    add("")
    add("| ID | Metric | Comparator | Threshold |")
    add("| --- | --- | --- | --- |")
    for c in blueprint.test_criteria:
        add(f"| {c.id} | `{c.metric_key}` | `{c.comparator}` | {c.threshold} |")
    add("")
    add("## Risk notes")
    add("")
    for note in blueprint.risk_notes:
        add(f"- {note}")
    add("")
    return "\n".join(lines)


def static_scan(source: str) -> list[str]:
    violations = [token for token in BANNED_TOKENS if token in source]
    try:
        compile(source, "<generated>", "exec")
    except SyntaxError as exc:  # pragma: no cover - guarded by tests
        violations.append(f"syntax_error:{exc.msg}:{exc.lineno}")
    return violations


def write_build(
    build_dir: Path,
    *,
    spec: ScenarioSpec,
    blueprint: ArchitecturalBlueprint,
    deployment: DeploymentPlan,
    project_root: Path,
    attempt: int,
    generated_at: str,
) -> ArtifactBundle:
    """Materialise one build attempt and return its hashed artifact records."""

    build_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "blueprint_id": blueprint.blueprint_id,
        "spec_id": spec.spec_id,
        "attempt": attempt,
        "generated_at": generated_at,
        "scenario": spec.model_dump(mode="json"),
        "deployment": deployment.model_dump(mode="json"),
        "acceptance": [c.model_dump(mode="json") for c in blueprint.test_criteria],
        "engine": {"simpy": "des", "seed": deployment.seed},
    }
    files: dict[str, tuple[str, bytes]] = {}

    payload = json.dumps(manifest, indent=2, sort_keys=True).encode()
    files["scenario_manifest.json"] = ("manifest", payload)

    runner = render_runner(manifest, project_root, attempt)
    violations = static_scan(runner)
    if violations:
        raise ValueError(f"generated artifact failed static scan: {violations}")
    files["generated_simulation.py"] = ("entrypoint", runner.encode())

    memo = render_blueprint_md(blueprint, spec)
    files["blueprint.md"] = ("design_memo", memo.encode())

    acceptance = json.dumps(
        {
            "attempt": attempt,
            "strategy": deployment.strategy,
            "criteria": [c.model_dump(mode="json") for c in blueprint.test_criteria],
            "berths": deployment.berths,
        },
        indent=2,
        sort_keys=True,
    ).encode()
    files["acceptance.json"] = ("acceptance", acceptance)

    records: list[ArtifactRecord] = []
    for filename, (role, data) in files.items():
        path = build_dir / filename
        path.write_bytes(data)
        records.append(
            ArtifactRecord(
                path=str(path),
                role=role,
                sha256=sha256_bytes(data),
                bytes=len(data),
            )
        )
    return ArtifactBundle(build_dir=build_dir, records=records, manifest=manifest)


def erlang_c(servers: int, offered_load: float) -> float:
    """Erlang C probability of queueing for ``M/M/c`` with ``a = lambda/mu``."""

    if servers <= 0:
        return 1.0
    if offered_load >= servers:
        return 1.0
    # numerically stable recursion for the Erlang B terms
    term = 1.0
    total = 1.0
    for k in range(1, servers + 1):
        term *= offered_load / k
        total += term
    last = term  # a^c / c!
    rho = offered_load / servers
    tail = last / (1.0 - rho)
    # sum_{k=0}^{c-1} a^k/k!  +  a^c / (c! (1-rho))
    denom = (total - last) + tail
    if denom <= 0:  # pragma: no cover - defensive
        return 1.0
    return float(min(1.0, tail / denom))


def expected_wait_hours(servers: int, arrival_rate_teu_day: float, mu_eff: float) -> float:
    """Mean queueing delay (hours) at a terminal modelled as M/M/c."""

    if mu_eff <= 0:
        return math.inf
    a = arrival_rate_teu_day / mu_eff
    if servers <= a:
        return math.inf
    c = erlang_c(servers, a)
    wait_days = c / (servers * mu_eff - arrival_rate_teu_day)
    return max(0.0, wait_days * 24.0)


def required_servers(
    arrival_rate_teu_day: float,
    mu_eff: float,
    wait_target_hours: float,
    cap: int = 96,
) -> int:
    """Smallest berth count meeting the wait target, or ``cap`` if infeasible."""

    for servers in range(1, cap + 1):
        if expected_wait_hours(servers, arrival_rate_teu_day, mu_eff) <= wait_target_hours:
            return servers
    return cap
