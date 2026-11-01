"""Pipeline nodes for the ASEOWS build loop.

Planner
    Reads the directive, generates the world, audits every candidate site and
    derives the geometric ``critical_radius`` - the smallest discovery radius
    that could *possibly* connect the lattice (the longest edge of the Euclidean
    minimum spanning tree over settlement positions).

Spatial-Economic Model Synthesizer
    Compiles a frozen :class:`ExperimentSpec` from the blueprint plus every patch
    the reflector has emitted so far. The first attempt is deliberately
    parsimonious: trade starts at exactly the critical radius.

Sandbox Visual Tester
    Runs the full lattice, stresses it with a drought, renders a frame and
    evaluates all gates. Nothing is accepted on faith - every claim is measured.

Self-Correction
    Classifies the failure, explains it and emits typed parameter patches that
    the synthesizer will apply on the next pass. Bounded by ``max_retries``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from ..metrics import DEFAULT_GATES, evaluate, run_study
from ..render import render_frame
from ..schemas import (
    ArtifactRecord,
    Blueprint,
    Equation,
    ExperimentSpec,
    FinalReport,
    Gate,
    Patch,
    Reflection,
    SiteAudit,
    ValidationReport,
    WorldSpec,
)
from ..settlement import settlement_name
from ..world import generate_world
from .state import PipelineState, set_by_pointer

EQUATIONS: list[Equation] = [
    Equation(
        name="Cobb-Douglas production",
        domain="production",
        latex=r"Q_i^r = E_i^r \left(\ell_i^r\right)^{\kappa} (1 + \theta_i)\, R_i \;\cdot\; \sigma_i^r(t)",
        description=(
            "Output of resource r at settlement i. E is the territorial endowment "
            "(terrain yield x deposit richness), l the labour share with diminishing "
            "returns kappa=0.78, theta accumulated knowledge, R the road multiplier and "
            "sigma the shock modifier."
        ),
    ),
    Equation(
        name="Labour reallocation",
        domain="production",
        latex=r"\ell_i^r \leftarrow (1-\lambda)\,\ell_i^r + \lambda\,\frac{(p_i^r E_i^r)^{\nu}}{\sum_s (p_i^s E_i^s)^{\nu}}",
        description=(
            "Decentralised adaptation: labour drifts toward the locally most valuable "
            "activity with inertia lambda=0.25 and sharpness nu=2.2. No settlement sees "
            "a price outside its own borders."
        ),
    ),
    Equation(
        name="Scarcity pricing",
        domain="pricing",
        latex=r"p_i^r = \mathrm{clip}\!\left[p_0^r \left(\frac{\bar S_i^r}{S_i^r}\right)^{\epsilon},\; \phi p_0^r,\; \Phi p_0^r\right]",
        description=(
            "Local price rises as the stock S falls below the target buffer "
            "S-bar = consumption x coverage_ticks. Elasticity eps=0.45, floor 0.25 and "
            "ceiling 6x the base price."
        ),
    ),
    Equation(
        name="Gravity attractiveness",
        domain="gravity",
        latex=r"A_{ij}^{r} = G\,\frac{\left(S_i^r\right)^{\alpha}\left(D_j^r\right)^{\beta}}{\left(d_{ij}+d_0\right)^{\gamma}}\,e^{-\tau c_{ij}}\,\left(1+\mu\,m_{ij}\right)",
        description=(
            "The trade-routing kernel. Masses are origin surplus and destination deficit, "
            "d the Euclidean separation with a d0=3 cell offset that keeps nearby pairs "
            "finite, c the terrain-weighted transport cost on the cheapest A* route, tau "
            "the freight friction and m the repeat-partner memory."
        ),
    ),
    Equation(
        name="Spatial arbitrage condition",
        domain="network",
        latex=r"p_j^r - p_i^r > \tau\, c_{ij} \quad\Longrightarrow\quad q_{ij}^r = \min\!\left(S_i^r, D_j^r\right)\eta \min(1, w_{ij}^r)",
        description=(
            "A link only trades when the destination price premium covers freight. "
            "Volume is capped by the smaller of marketable surplus and unmet deficit, "
            "scaled by trade_fraction eta=0.45 and the edge weight."
        ),
    ),
    Equation(
        name="Edge weight dynamics",
        domain="network",
        latex=r"w_{ij}^r(t+1) = \max\!\left(w_{ij}^r(t)\,e^{-\delta},\; \min\!\left(4,\; 2\frac{A_{ij}^r}{\max_k A_{ik}^r}\right)\right)",
        description=(
            "Weights decay geometrically (delta=0.965/tick) and are refreshed by each "
            "discovery pass relative to the settlement's own best partner. Links whose "
            "weight falls below edge_min_weight or that stay idle too long are pruned."
        ),
    ),
    Equation(
        name="Logistic demography",
        domain="demography",
        latex=r"\dot N_i = N_i\left[\rho\,\sigma(\text{food})\left(1 - \frac{N_i}{K_i}\right) - \delta_m\right],\quad K_i = K_0 + \kappa_E E_i^{\text{food}}(1+\theta_i) + \kappa_R R_i",
        description=(
            "Population grows logistically toward a carrying capacity set by food "
            "endowment, knowledge and road access, gated by a food-security term derived "
            "from the granary buffer."
        ),
    ),
    Equation(
        name="Infrastructure feedback",
        domain="infrastructure",
        latex=r"c_{xy} = c^0_{xy}\,\big/\left(1 + g\,\rho_{xy}\right),\qquad \sum_t q_{ij}(t) > \Theta \Rightarrow \rho_{xy} \mathrel{+}= g\ \ \forall (x,y)\in \text{path}_{ij}",
        description=(
            "Sustained flow paves the route it uses: once cumulative volume on a link "
            "passes the threshold its whole cheapest path gains road level, lowering the "
            "cost c that feeds back into the gravity kernel."
        ),
    ),
    Equation(
        name="Ledger identity",
        domain="validation",
        latex=r"S_{\text{end}} = S_{\text{start}} + Q - C + M_{\text{in}} - M_{\text{out}} - \Sigma",
        description=(
            "Physical conservation across the whole lattice: production Q, consumption C, "
            "delivered imports, exports and spoilage must reconstruct the final stocks to "
            "floating-point tolerance. This is the mass-balance gate."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
def planner(state: PipelineState) -> dict[str, Any]:
    directive = state["directive"]
    world_spec = WorldSpec(seed=directive.seed)
    world = generate_world(world_spec)

    name_rng = np.random.default_rng(world_spec.seed + 7717)
    audit: list[SiteAudit] = []
    points: list[tuple[int, int]] = []
    for sid, (y, x) in enumerate(world.sites):
        from ..world import territorial_yield

        raw = territorial_yield(world, (y, x), world_spec.territory_radius)
        audit.append(
            SiteAudit(
                settlement_id=sid,
                name=settlement_name(name_rng),
                x=int(x),
                y=int(y),
                territory_yield=[round(float(v), 3) for v in raw],
            )
        )
        points.append((int(x), int(y)))

    critical_radius = _critical_radius(points)
    summary = world.summary()

    blueprint = Blueprint(
        blueprint_id=f"BP-{directive.experiment_id}",
        experiment_id=directive.experiment_id,
        objective=directive.objective,
        equations=EQUATIONS,
        gates=list(DEFAULT_GATES),
        world_summary={
            "width": world_spec.width,
            "height": world_spec.height,
            "seed": world_spec.seed,
            "sites": len(points),
            "terrain_fractions": {k: round(float(v), 4) for k, v in summary.get("terrain_share", {}).items()},
            "land_fraction": round(float(summary.get("land_fraction", 0.0)), 4),
            "mean_deposit": [round(float(v), 3) for v in summary.get("mean_deposit", [])],
        },
        economy={
            "production": "Cobb-Douglas with territorial endowments",
            "discovery": "decentralised surplus/deficit scan inside discovery_radius",
            "routing": "gravity kernel over A* cheapest paths",
            "start_radius": round(critical_radius, 2),
        },
        critical_radius=round(critical_radius, 3),
        site_audit=audit,
        expected={
            "giant_component_share": 1.0,
            "mass_balance_error": 0.0,
            "abs_growth_tail": 0.0,
        },
        risk_notes=[
            "The critical radius is a *geometric* lower bound: complementary "
            "endowments may sit farther apart than the lattice diameter suggests.",
            "Ore is structurally scarce by design; if too few mining sites exist the "
            "ore price will pin at its ceiling and the corridor stays thin.",
            "Spoilage keeps granaries finite; a settlement that cannot sell will still "
            "hoard up to its storage buffer.",
        ],
    )
    return {
        "blueprint": blueprint,
        "log": [
            *state.get("log", []),
            f"[planner] world {world_spec.width}x{world_spec.height} seed {world_spec.seed} "
            f"-> {len(points)} sites, land {float(summary.get('land_fraction', 0.0)):.2%}, "
            f"critical radius {critical_radius:.2f}",
        ],
        "trail": [
            *state.get("trail", []),
            {"node": "planner", "critical_radius": critical_radius, "sites": len(points)},
        ],
    }


def _critical_radius(points: list[tuple[int, int]]) -> float:
    if len(points) < 2:
        return 2.0
    graph = nx.Graph()
    graph.add_nodes_from(range(len(points)))
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            graph.add_edge(i, j, weight=float(np.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1])))
    tree = nx.minimum_spanning_tree(graph, weight="weight")
    return float(max(d["weight"] for _, _, d in tree.edges(data=True)))


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------
def synthesizer(state: PipelineState) -> dict[str, Any]:
    directive = state["directive"]
    blueprint = state["blueprint"]
    attempt = int(state.get("attempt", 0))

    world = WorldSpec(seed=directive.seed)
    economy: dict[str, Any] = {
        "discovery_radius": round(max(2.0, blueprint.critical_radius), 2),
    }
    base = ExperimentSpec(
        experiment_id=directive.experiment_id,
        title=directive.title,
        world=world,
        economy=economy,
        horizon=directive.horizon,
        settle_ticks=directive.settle_ticks,
    )
    payload = base.model_dump()
    for patch in state.get("patches", []):
        set_by_pointer(payload, patch.pointer, patch.updated)
    spec = ExperimentSpec.model_validate(payload)

    applied = [p.pointer for p in state.get("patches", [])]
    return {
        "spec": spec,
        "log": [
            *state.get("log", []),
            f"[synthesizer] attempt {attempt}: compiled spec "
            f"(radius={spec.economy.discovery_radius}, partners={spec.economy.max_partners}, "
            f"growth={spec.economy.growth_rate:.4f})"
            + (f" after {len(applied)} patch(es) {applied}" if applied else " from blueprint"),
        ],
        "trail": [
            *state.get("trail", []),
            {"node": "synthesizer", "attempt": attempt, "radius": spec.economy.discovery_radius, "patches": applied},
        ],
    }


# ---------------------------------------------------------------------------
# Sandbox visual tester
# ---------------------------------------------------------------------------
def sandbox_tester(state: PipelineState) -> dict[str, Any]:
    spec = state["spec"]
    blueprint = state["blueprint"]
    attempt = int(state.get("attempt", 0))
    output_dir = Path(state.get("output_dir", "artifacts"))  # type: ignore[arg-type]
    output_dir.mkdir(parents=True, exist_ok=True)

    study = run_study(spec)
    report = evaluate(blueprint.gates, study.metrics, attempt, study.duration)

    image = render_frame(study.model, scale=9)
    frame_path = output_dir / f"attempt_{attempt}_frame.png"
    image.save(frame_path)
    metrics_path = output_dir / f"attempt_{attempt}_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "attempt": attempt,
                "passed": report.passed,
                "metrics": {k: round(float(v), 9) for k, v in study.metrics.items()},
                "detail": study.detail,
                "gates": [r.model_dump() for r in report.results],
            },
            indent=2,
        )
    )

    failing = [r.id for r in report.results if not r.passed]
    return {
        "report": report,
        "metrics": study.metrics,
        "detail": study.detail,
        "artifacts": [
            *state.get("artifacts", []),
            _artifact(frame_path, "sandbox frame"),
            _artifact(metrics_path, "gate metrics"),
        ],
        "log": [
            *state.get("log", []),
            f"[tester] attempt {attempt}: {'PASS' if report.passed else 'FAIL'} "
            f"({len(report.results) - len(failing)}/{len(report.results)} gates, "
            f"{study.duration:.2f}s)" + (f" failing={failing}" if failing else ""),
        ],
        "trail": [
            *state.get("trail", []),
            {
                "node": "tester",
                "attempt": attempt,
                "passed": report.passed,
                "failure_class": report.failure_class,
                "failing": failing,
            },
        ],
    }


# ---------------------------------------------------------------------------
# Self-correction
# ---------------------------------------------------------------------------
def self_correction(state: PipelineState) -> dict[str, Any]:
    report: ValidationReport = state["report"]
    blueprint = state["blueprint"]
    spec: ExperimentSpec = state["spec"]
    attempt = int(state.get("attempt", 0))
    metrics = state.get("metrics", {})

    patches: list[Patch] = []
    diagnosis = ""
    confidence = 0.72

    if report.failure_class == "network_fragmentation":
        report.results and next(r for r in report.results if r.id == "N-COMP")
        giant = metrics.get("giant_component_share", 0.0)
        current = spec.economy.discovery_radius
        ceiling = 2.8 * blueprint.critical_radius
        widened = round(min(max(current * 1.6, blueprint.critical_radius * 1.8), ceiling), 2)
        patches.append(
            Patch(
                pointer="economy.discovery_radius",
                previous=current,
                updated=widened,
                rationale=(
                    f"giant component {giant:.2f} < 0.85 with radius {current}: the geometric "
                    f"critical radius {blueprint.critical_radius:.2f} is only a lower bound, because "
                    "a link needs a *complementary* pair inside the horizon, not merely a neighbour. "
                    f"Widening to {widened} (ceiling {ceiling:.2f})."
                ),
            )
        )
        if metrics.get("edges_per_node", 0.0) < 1.0:
            patches.append(
                Patch(
                    pointer="economy.max_partners",
                    previous=spec.economy.max_partners,
                    updated=min(16, spec.economy.max_partners + 2),
                    rationale="too few links per node: allow each settlement to maintain more partners.",
                )
            )
            patches.append(
                Patch(
                    pointer="economy.candidate_cut",
                    previous=spec.economy.candidate_cut,
                    updated=round(max(0.12, spec.economy.candidate_cut * 0.85), 3),
                    rationale="relative selection cut trimmed too aggressively; relax it slightly.",
                )
            )
        diagnosis = (
            "The market did not span the lattice. Geometry alone does not connect an economy: "
            "settlements must be close enough to a *complementary* endowment, not just to a neighbour."
        )
        confidence = 0.78

    elif report.failure_class == "non_convergence":
        rate = metrics.get("abs_growth_tail", 0.0)
        patches.append(
            Patch(
                pointer="economy.growth_rate",
                previous=spec.economy.growth_rate,
                updated=round(max(0.004, spec.economy.growth_rate * 0.82), 5),
                rationale=(
                    f"population still growing at {rate:.4f}/tick in the final quarter; "
                    "damping fecundity forces the logistic to settle inside the horizon."
                ),
            )
        )
        patches.append(
            Patch(
                pointer="economy.target_coverage_ticks",
                previous=spec.economy.target_coverage_ticks,
                updated=round(spec.economy.target_coverage_ticks * 1.1, 2),
                rationale="a deeper buffer damps the stock/price oscillation that delays convergence.",
            )
        )
        horizon = int(state["spec"].horizon)
        if rate > 0.0045 and horizon < 2000:
            extended = int(min(2000, round(horizon * 1.8) + 120))
            patches.append(
                Patch(
                    pointer="horizon",
                    previous=horizon,
                    updated=extended,
                    rationale=(
                        f"the lattice was still expanding at {rate:.4f}/tick when the horizon closed; "
                        f"a logistic needs time to reach its plateau, so extend {horizon} -> {extended} ticks."
                    ),
                )
            )
        if rate > 0.006:
            patches.append(
                Patch(
                    pointer="economy.base_capacity",
                    previous=spec.economy.base_capacity,
                    updated=round(spec.economy.base_capacity * 0.75, 2),
                    rationale=(
                        f"tail growth {rate:.4f}/tick means carrying capacity is still far away; "
                        "pulling it down brings the logistic to its plateau inside the horizon."
                    ),
                )
            )
        diagnosis = "The lattice was still expanding when the horizon closed; demography needs damping."
        confidence = 0.7

    elif report.failure_class == "instability":
        patches.append(
            Patch(
                pointer="economy.transfer_loss",
                previous=spec.economy.transfer_loss,
                updated=round(max(0.01, spec.economy.transfer_loss * 0.8), 4),
                rationale="high transit loss starves importers; trimming it improves shock recovery.",
            )
        )
        patches.append(
            Patch(
                pointer="economy.edge_max_idle",
                previous=spec.economy.edge_max_idle,
                updated=min(200, spec.economy.edge_max_idle + 8),
                rationale="links pruned too eagerly during a shock cannot be re-established fast enough.",
            )
        )
        diagnosis = (
            f"trade volume recovered to only {metrics.get('shock_recovery', 0):.2f} of its pre-shock "
            "level: the network forgets its partners faster than it can rebuild them."
        )
        confidence = 0.66

    elif report.failure_class == "visual_contract":
        patches.append(
            Patch(
                pointer="economy.base_capacity",
                previous=spec.economy.base_capacity,
                updated=round(spec.economy.base_capacity * 1.15, 2),
                rationale="flat population field: raise carrying capacity so the choropleth has range.",
            )
        )
        diagnosis = "The rendered frame did not carry enough visual signal."
        confidence = 0.6

    else:  # mass_imbalance or schema error
        diagnosis = (
            "Physical conservation failed. This is a defect in the engine, not a tuning problem; "
            "parameter patches cannot repair it and the run must be reviewed by a human."
        )
        confidence = 0.95

    exhausted = attempt + 1 >= int(state.get("max_attempts", 4))
    next_action = "finalize" if (exhausted or not patches) else "synthesize"

    reflection = Reflection(
        attempt=attempt,
        failure_class=report.failure_class,
        diagnosis=diagnosis,
        patches=patches,
        next_action=next_action,  # type: ignore[arg-type]
        confidence=confidence,
    )
    return {
        "reflections": [*state.get("reflections", []), reflection],
        "patches": state.get("patches", []) + patches,
        "attempt": attempt + 1,
        "log": [
            *state.get("log", []),
            f"[reflector] attempt {attempt} failed as {report.failure_class}: {diagnosis} "
            f"-> {len(patches)} patch(es), next={next_action}",
        ],
        "trail": [
            *state.get("trail", []),
            {
                "node": "self_correction",
                "attempt": attempt,
                "failure_class": report.failure_class,
                "patches": [p.model_dump() for p in patches],
                "next_action": next_action,
            },
        ],
    }


# ---------------------------------------------------------------------------
# Finalizer
# ---------------------------------------------------------------------------
def finalizer(state: PipelineState) -> dict[str, Any]:
    report = state.get("report")
    directive = state["directive"]
    blueprint = state["blueprint"]
    output_dir = Path(state.get("output_dir", "artifacts"))  # type: ignore[arg-type]
    output_dir.mkdir(parents=True, exist_ok=True)

    attempts = int(report.attempt) + 1 if report else 1
    if report is None or not report.passed:
        verdict = "FAILED_NEEDS_REVIEW"
    elif attempts == 1:
        verdict = "PASS"
    else:
        verdict = "PASS_WITH_RETRIES"

    blueprint_path = output_dir / "blueprint.json"
    blueprint_path.write_text(blueprint.model_dump_json(indent=2))

    artifacts = [*list(state.get("artifacts", [])), _artifact(blueprint_path, "blueprint")]

    final = FinalReport(
        run_id=f"{directive.experiment_id}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}",
        verdict=verdict,
        experiment_id=directive.experiment_id,
        blueprint_id=blueprint.blueprint_id,
        attempts=attempts,
        validation=report,
        reflections=state.get("reflections", []),
        artifacts=artifacts,
        notes=[
            f"critical radius {blueprint.critical_radius:.2f} derived from the settlement MST",
            f"final discovery radius {state['spec'].economy.discovery_radius}",
            f"gates: {sum(1 for r in report.results if r.passed)}/{len(report.results)} passed",
        ],
    )
    report_path = output_dir / "final_report.json"
    report_path.write_text(final.model_dump_json(indent=2))
    artifacts = [*artifacts, _artifact(report_path, "final report")]

    return {
        "verdict": verdict,
        "artifacts": artifacts,
        "log": [*state.get("log", []), f"[finalizer] verdict {verdict} after {attempts} attempt(s)"],
        "trail": [*state.get("trail", []), {"node": "finalizer", "verdict": verdict}],
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _artifact(path: Path, role: str) -> ArtifactRecord:
    data = path.read_bytes()
    return ArtifactRecord(
        path=str(path),
        role=role,
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
    )


def gates_for(ids: list[str]) -> list[Gate]:
    lookup = {g.id: g for g in DEFAULT_GATES}
    return [lookup[i] for i in ids if i in lookup]
