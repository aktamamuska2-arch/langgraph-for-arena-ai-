"""The four AM-SES nodes: planner, synthesizer, sandbox tester, reflector.

Each node is a pure function ``State -> partial State``. Validation happens at
every boundary: a node re-parses the slice it consumes with its Pydantic model
before doing any work, and dumps the model back into the state afterwards.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..codegen import (
    derive_deployment,
    expected_wait_hours,
    required_servers,
    write_build,
)
from ..config import DEFAULT_SCENARIO
from ..sandbox import execute_build
from ..schemas import (
    ArchitecturalBlueprint,
    ArtifactRecord,
    CriterionResult,
    EquationRecord,
    ExecutionLogEntry,
    FinalReport,
    ModuleRecord,
    PatchOperation,
    ReflectionRecord,
    ScenarioSpec,
    SizingDecision,
    TestCriterion,
    TestReport,
)
from ..topology import build_graph, connectivity_report, criticality, port_loads
from ..weather import zone_statistics
from .state import RunContext, State

MAX_BERTHS = 96
#: Berth-utilisation target used for headroom sizing. Stationary Erlang-C alone
#: under-sizes the fleet because terminal productivity is time-varying (weather
#: CTMC) rather than i.i.d.; the headroom term covers that transient.
UTILISATION_TARGET = 0.80
#: fraction of the year a terminal is expected to be degraded by incidents
GEO_AVAILABILITY = 0.97


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(node: str, level: str, message: str, **payload: Any) -> dict[str, Any]:
    return ExecutionLogEntry(node=node, level=level, message=message, payload=payload).model_dump(mode="json")


def _as_json(model: Any) -> dict[str, Any]:
    return json.loads(json.dumps(model.model_dump(mode="json"), default=str))


# ---------------------------------------------------------------------------
# node_planner
# ---------------------------------------------------------------------------


def _equations() -> list[EquationRecord]:
    return [
        EquationRecord(
            name="weather_generator",
            domain="weather",
            latex="Q_z[i,j] = r_z[i] P_z[i,j] (i!=j), Q_z[i,i] = -r_z[i], r_z[i] = 1/dwell_z[i]",
            description="Seasonal continuous-time Markov generator per sea zone over {CALM, ADVERSE, SEVERE, EXTREME}.",
        ),
        EquationRecord(
            name="weather_stationary",
            domain="weather",
            latex="pi_z Q_z = 0, sum_i pi_z[i] = 1",
            description="Stationary weather law used for capacity sizing and for warm-starting the simulation.",
        ),
        EquationRecord(
            name="weather_coupling",
            domain="weather",
            latex="v(t) = v_0 s(W_z(t)),  mu_b(t) = mu_b0 p(W_z(t)),  P(close|EXTREME) = c_z",
            description="Weather modulates sailing speed, berth productivity and terminal closure hazard.",
        ),
        EquationRecord(
            name="geo_arrival",
            domain="geopolitics",
            latex="lambda_c(t) = (g_c/365) g_int (1 + 0.35 sin(2pi (t+phi_c)/365))",
            description="Non-homogeneous Poisson arrival of geopolitical incidents at each choke point.",
        ),
        EquationRecord(
            name="geo_derate",
            domain="geopolitics",
            latex="kappa_c(t) = 1 - max_k [ S_k (1 - 0.55 (t - t_k)/D_k) ]",
            description="Instantaneous capacity factor from overlapping, linearly-decaying incident severities.",
        ),
        EquationRecord(
            name="utilization",
            domain="queueing",
            latex="rho_p = lambda_p / (c_p mu_p)",
            description="Berth utilisation; the stability gate requires rho <= 0.92 for every terminal.",
        ),
        EquationRecord(
            name="erlang_c",
            domain="queueing",
            latex="C(c,a) = (a^c/(c!(1-rho))) / (sum_{k<c} a^k/k! + a^c/(c!(1-rho))), a = lambda/mu",
            description="Probability of queueing at a terminal modelled as M/M/c; used to size berths.",
        ),
        EquationRecord(
            name="waiting_time",
            domain="queueing",
            latex="W_q = C(c,a) / (c mu - lambda)",
            description="Mean berth waiting time; solved for the smallest c meeting the wait target.",
        ),
        EquationRecord(
            name="little_law",
            domain="flow",
            latex="L_p = lambda_p W_p",
            description="Berth occupancy identity used to cross-check utilisation against backlog telemetry.",
        ),
        EquationRecord(
            name="resilience",
            domain="risk",
            latex="R = Theta_stress / Theta_baseline",
            description="Throughput retention against the benign-forcing baseline of the same network.",
        ),
        EquationRecord(
            name="cvar",
            domain="risk",
            latex="CVaR_95(X) = E[X | X <= F_X^{-1}(0.05)]",
            description="Expected shortfall of the lower tail of throughput and resilience distributions.",
        ),
        EquationRecord(
            name="batch_means",
            domain="convergence",
            latex="RSE = s_b / (sqrt(k) mu_hat),  s_b^2 = Var(batch means)",
            description="Relative standard error of the steady-state mean over k=10 batches; convergence gate.",
        ),
    ]


def make_planner(context: RunContext) -> Callable[[State], dict[str, Any]]:
    def node_planner(state: State) -> dict[str, Any]:
        raw = state.get("scenario_params") or DEFAULT_SCENARIO.model_dump(mode="json")
        spec = ScenarioSpec.model_validate(raw)
        graph = build_graph(spec.network)
        conn = connectivity_report(graph)
        centrality = criticality(graph)
        loads = port_loads(spec, spec.stress.capacity_scale)

        wait_target_hours = 8.0
        resilience_margin = float(
            min(
                1.5,
                max(
                    1.0,
                    1.0
                    + 0.06 * spec.stress.geo_intensity
                    + 0.05 * max(0.0, spec.stress.weather_intensity - 1.0),
                ),
            )
        )
        zone_stats = {
            z.id: zone_statistics(z, spec.stress.weather_intensity, spec.stress.season)
            for z in spec.network.zones
        }

        sizing: list[SizingDecision] = []
        for port in spec.network.ports:
            load = loads[port.id]
            stats = zone_stats[port.zone]
            mu_eff = port.service_teu_per_berth_day * stats["expected_productivity"] * GEO_AVAILABILITY
            arrival = load.teu_per_day
            required_erlang = required_servers(arrival, mu_eff, wait_target_hours)
            required_headroom = int(math.ceil(arrival / max(mu_eff * UTILISATION_TARGET, 1e-9)))
            required = max(1, min(MAX_BERTHS, max(required_erlang, required_headroom)))
            effective_demand = arrival / mu_eff if mu_eff > 0 else math.inf
            projected_util = effective_demand / required
            projected_wait = expected_wait_hours(required, arrival, mu_eff)
            sizing.append(
                SizingDecision(
                    port_id=port.id,
                    base_berths=port.berths,
                    demand_berth_days=round(load.berth_days_per_day, 4),
                    effective_berth_days=round(effective_demand, 4),
                    required_berths=required,
                    projected_utilization=round(min(1.5, projected_util), 4),
                    projected_wait_hours=round(min(1e4, projected_wait), 3),
                    wait_target_hours=wait_target_hours,
                    meets_target=bool(projected_wait <= wait_target_hours + 1e-9 and projected_util <= 0.92),
                )
            )

        criteria = [
            TestCriterion(
                id="C-UTIL",
                description="Peak terminal berth utilisation must stay below the stability gate",
                threshold=spec.montecarlo.acceptance.max_berth_utilization,
                comparator="<=",
                metric_key="max_utilization",
            ),
            TestCriterion(
                id="C-SL",
                description="Share of vessels meeting their door-to-door SLA",
                threshold=spec.montecarlo.acceptance.min_service_level,
                comparator=">=",
                metric_key="service_level",
            ),
            TestCriterion(
                id="C-RSE",
                description="Batch-means relative standard error (steady-state convergence)",
                threshold=spec.montecarlo.acceptance.max_batch_rse,
                comparator="<=",
                metric_key="convergence_rse",
            ),
            TestCriterion(
                id="C-SLOPE",
                description="Backlog drift: positive slope means a non-stationary queue",
                threshold=spec.montecarlo.acceptance.max_backlog_slope_per_day,
                comparator="<=",
                metric_key="backlog_slope_per_day",
            ),
            TestCriterion(
                id="C-RUNTIME",
                description="Wall-clock seconds for one replication",
                threshold=spec.montecarlo.acceptance.max_runtime_seconds,
                comparator="<=",
                metric_key="runtime_seconds",
            ),
            TestCriterion(
                id="C-THRU",
                description="Throughput retained versus offered demand",
                threshold=spec.montecarlo.acceptance.min_throughput_ratio,
                comparator=">=",
                metric_key="throughput_ratio",
            ),
        ]

        critical_nodes = sorted(centrality.items(), key=lambda kv: kv[1], reverse=True)[:3]
        binding = max(sizing, key=lambda s: s.projected_utilization)
        blueprint = ArchitecturalBlueprint(
            blueprint_id=f"BP-{spec.spec_id}-{context.run_id[-6:]}",
            spec_id=spec.spec_id,
            objective=(
                "Compile a discrete-event replica of the regional maritime container grid, "
                "inject weather and geopolitical stress, and prove - inside a sandbox - that "
                "the deployed berth capacity keeps every terminal in the stable regime."
            ),
            sizing_strategy="proportional",
            resilience_margin=round(resilience_margin, 4),
            wait_target_hours=wait_target_hours,
            network_signature={
                "nodes": conn["nodes"],
                "edges": conn["edges"],
                "density": conn["density"],
                "connected": conn["connected"],
                "articulation_points": conn["articulation_points"],
                "top_centrality": [{"node": n, "score": s} for n, s in critical_nodes],
                # each channel is counted at both endpoints, hence the halving
                "nominal_daily_teu": round(sum(load.teu_per_day for load in loads.values()) / 2.0, 2),
                "binding_port": binding.port_id,
                "zone_stationary": {
                    zid: {k: round(v, 5) for k, v in stats.items()} for zid, stats in zone_stats.items()
                },
            },
            equations=_equations(),
            sizing=sizing,
            modules=[
                ModuleRecord(
                    path="maritime_sentinel/engine.py", role="SimPy discrete-event core", loc_estimate=520
                ),
                ModuleRecord(
                    path="maritime_sentinel/weather.py", role="Seasonal CTMC weather field", loc_estimate=210
                ),
                ModuleRecord(
                    path="maritime_sentinel/geopolitics.py",
                    role="Marked Poisson incident field",
                    loc_estimate=190,
                ),
                ModuleRecord(
                    path="maritime_sentinel/topology.py",
                    role="NetworkX lane graph + criticality",
                    loc_estimate=175,
                ),
                ModuleRecord(
                    path="maritime_sentinel/montecarlo.py",
                    role="Latin Hypercube perturbation driver",
                    loc_estimate=260,
                ),
                ModuleRecord(
                    path="maritime_sentinel/metrics.py",
                    role="KPI + convergence diagnostics",
                    loc_estimate=230,
                ),
                ModuleRecord(
                    path="app/mission_control.py", role="Streamlit telemetry dashboard", loc_estimate=640
                ),
            ],
            test_criteria=criteria,
            risk_notes=[
                f"Binding terminal predicted to be {binding.port_id} at rho={binding.projected_utilization:.3f}.",
                "Articulation points are single points of failure for the lane graph: "
                + ", ".join(conn["articulation_points"]),
                "Stationary sizing ignores the transient of a scheduled blockade; the sandbox run is the arbiter.",
                f"Resilience margin {resilience_margin:.3f} is applied on top of Erlang-C sizing.",
                f"Headroom sizing keeps stationary utilisation at or below {UTILISATION_TARGET:.2f} under stressed productivity.",
            ],
            total_berths=sum(s.required_berths for s in sizing),
        )
        return {
            "architectural_blueprint": _as_json(blueprint),
            "execution_logs": [
                _log(
                    "node_planner",
                    "info",
                    "blueprint compiled",
                    equations=len(blueprint.equations),
                    ports=len(sizing),
                    binding_port=binding.port_id,
                    resilience_margin=blueprint.resilience_margin,
                )
            ],
        }

    return node_planner


# ---------------------------------------------------------------------------
# node_synthesizer
# ---------------------------------------------------------------------------


def _inject_fault(build_dir: Path, mode: str) -> None:
    """Deliberately corrupt a first attempt so the healing loop can be demoed."""

    manifest_path = build_dir / "scenario_manifest.json"
    runner_path = build_dir / "generated_simulation.py"
    if mode == "schema":
        manifest = json.loads(manifest_path.read_text())
        manifest["scenario"]["network"]["zones"][0]["transition"][0][0] = 0.9
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    elif mode == "runtime":
        source = runner_path.read_text()
        source = source.replace(
            "    started = time.perf_counter()",
            "    started = time.perf_counter()\n    raise KeyError('injected runtime fault: missing lane descriptor')",
            1,
        )
        runner_path.write_text(source)
    elif mode == "divergence":
        manifest = json.loads(manifest_path.read_text())
        for port in manifest["deployment"]["berths"]:
            manifest["deployment"]["berths"][port] = 1
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    elif mode == "timeout":
        # a build that cannot finish inside the sandbox budget: this one is
        # *not* healable by patching a parameter, so it exercises the retry cap
        source = runner_path.read_text()
        source = source.replace(
            "    started = time.perf_counter()",
            "    started = time.perf_counter()\n    time.sleep(900)",
            1,
        )
        runner_path.write_text(source)


def make_synthesizer(context: RunContext) -> Callable[[State], dict[str, Any]]:
    def node_synthesizer(state: State) -> dict[str, Any]:
        spec = ScenarioSpec.model_validate(state["scenario_params"])
        blueprint = ArchitecturalBlueprint.model_validate(state["architectural_blueprint"])
        attempt = int(state.get("retry_count", 0))
        deployment = derive_deployment(
            blueprint,
            spec,
            seed=int(state.get("scenario_params", {}).get("seed", spec.seed)),
            capacity_scale=float(spec.stress.capacity_scale),
        )
        build_dir = context.attempt_dir(attempt)
        if build_dir.exists():
            for stale in build_dir.iterdir():
                stale.unlink()
        bundle = write_build(
            build_dir,
            spec=spec,
            blueprint=blueprint,
            deployment=deployment,
            project_root=context.project_root,
            attempt=attempt,
            generated_at=_now(),
        )
        state_scenario = state["scenario_params"]
        if context.fault_mode != "none" and (attempt == 0 or context.fault_persist):
            _inject_fault(build_dir, context.fault_mode)
            # refresh hashes so the report reflects what is actually on disk
            bundle = _rehash(bundle)
            if context.fault_mode == "schema":
                # a schema fault means the *emitted manifest* was invalid: mirror
                # that corruption into the state so the reflector has to repair it
                state_scenario = json.loads((build_dir / "scenario_manifest.json").read_text())["scenario"]

        artifacts = bundle.as_state()
        artifacts["deployment"] = _as_json(deployment)
        artifacts["attempt"] = attempt
        artifacts["strategy"] = deployment.strategy
        return {
            "scenario_params": state_scenario,
            "code_artifacts": artifacts,
            "execution_logs": [
                _log(
                    "node_synthesizer",
                    "info",
                    "build emitted",
                    attempt=attempt,
                    strategy=deployment.strategy,
                    berths=deployment.berths,
                    artifacts=[r.path for r in bundle.records],
                )
            ],
        }

    return node_synthesizer


def _rehash(bundle) -> Any:
    from ..codegen import sha256_bytes

    records = []
    for rec in bundle.records:
        path = Path(rec.path)
        data = path.read_bytes()
        records.append(
            ArtifactRecord(path=rec.path, role=rec.role, sha256=sha256_bytes(data), bytes=len(data))
        )
    bundle.records = records
    return bundle


# ---------------------------------------------------------------------------
# node_sandbox_tester
# ---------------------------------------------------------------------------


def _classify(outcome, diverged: bool, failing: list[str]) -> str:
    if outcome.timed_out:
        return "timeout"
    if outcome.payload is None:
        return "runtime_error"
    if str(outcome.payload.get("error_type", "")).endswith("ValidationError"):
        return "schema_error"
    if not outcome.payload.get("ok", False) and outcome.payload.get("error_type"):
        return "runtime_error"
    if diverged:
        return "divergence"
    if "max_utilization" in failing:
        return "instability"
    if "service_level" in failing or "throughput_ratio" in failing:
        return "underperformance"
    if "convergence_rse" in failing or "backlog_slope_per_day" in failing:
        return "divergence"
    if "runtime_seconds" in failing:
        return "timeout"
    return "none"


def make_tester(context: RunContext) -> Callable[[State], dict[str, Any]]:
    def node_sandbox_tester(state: State) -> dict[str, Any]:
        blueprint = ArchitecturalBlueprint.model_validate(state["architectural_blueprint"])
        artifacts = state.get("code_artifacts") or {}
        build_dir = Path(artifacts.get("build_dir", "."))
        attempt = int(artifacts.get("attempt", state.get("retry_count", 0)))
        runtime_gate = next(
            (c.threshold for c in blueprint.test_criteria if c.metric_key == "runtime_seconds"),
            float(context.sandbox_timeout),
        )
        timeout = min(float(context.sandbox_timeout), float(runtime_gate) * 1.5)

        outcome = execute_build(build_dir, timeout=max(5.0, timeout))
        payload = outcome.payload or {}
        metrics: dict[str, float] = {
            str(k): float(v) for k, v in (payload.get("metrics") or {}).items() if isinstance(v, (int, float))
        }
        diverged = bool(payload.get("diverged", False)) or outcome.exit_code == 3

        results: list[CriterionResult] = []
        for criterion in blueprint.test_criteria:
            observed = metrics.get(criterion.metric_key, math.nan)
            if observed is None or (isinstance(observed, float) and math.isnan(observed)):
                results.append(
                    CriterionResult(
                        id=criterion.id,
                        metric_key=criterion.metric_key,
                        observed=math.inf if criterion.comparator == "<=" else -math.inf,
                        threshold=criterion.threshold,
                        comparator=criterion.comparator,
                        passed=False,
                    )
                )
                continue
            results.append(CriterionResult.evaluate(criterion, float(observed)))

        failing = [r.metric_key for r in results if not r.passed]
        passed = outcome.payload is not None and bool(outcome.payload.get("ok", False)) and not failing
        failure_class = _classify(outcome, diverged, failing)

        report = TestReport(
            attempt=attempt,
            passed=passed,
            exit_code=outcome.exit_code,
            timed_out=outcome.timed_out,
            duration_seconds=round(outcome.duration_seconds, 4),
            failure_class=failure_class,
            criteria=results,
            metrics=metrics,
            stdout_tail=outcome.stdout[-2000:],
            stderr_tail=outcome.stderr[-2000:],
            traceback_summary=outcome.traceback_summary,
        )
        level = "info" if passed else "warning"
        return {
            "test_report": _as_json(report),
            "execution_logs": [
                _log(
                    "node_sandbox_tester",
                    level,
                    "sandbox run passed" if passed else "sandbox run failed",
                    attempt=attempt,
                    failure_class=failure_class,
                    exit_code=outcome.exit_code,
                    failing=failing,
                    metrics={k: round(v, 4) for k, v in metrics.items() if math.isfinite(v)},
                    traceback=outcome.traceback_summary,
                )
            ],
        }

    return node_sandbox_tester


# ---------------------------------------------------------------------------
# node_reflector
# ---------------------------------------------------------------------------


def _repair_scenario(payload: dict[str, Any]) -> tuple[dict[str, Any], list[PatchOperation]]:
    """Project an invalid scenario back onto the feasible set."""

    repaired = json.loads(json.dumps(payload))
    patches: list[PatchOperation] = []
    for zone in repaired.get("network", {}).get("zones", []):
        rows = zone.get("transition", [])
        for i, row in enumerate(rows):
            total = sum(row) or 1.0
            normalised = [round(max(0.0, v) / total, 6) for v in row]
            if abs(total - 1.0) > 1e-9:
                patches.append(
                    PatchOperation(
                        target="scenario",
                        pointer=f"network.zones[{zone.get('id')}].transition[{i}]",
                        previous=row,
                        updated=normalised,
                        rationale="row was not a probability vector; projected onto the simplex",
                    )
                )
            row[:] = normalised
    return repaired, patches


def make_reflector(context: RunContext) -> Callable[[State], dict[str, Any]]:
    def node_reflector(state: State) -> dict[str, Any]:
        report = TestReport.model_validate(state["test_report"])
        blueprint = ArchitecturalBlueprint.model_validate(state["architectural_blueprint"])
        retry = int(state.get("retry_count", 0))

        # The scenario slice may itself be invalid (that is exactly what a
        # schema_error means), so validation is part of the reflection, not a
        # precondition for it.
        raw_scenario = dict(state.get("scenario_params") or {})
        spec: ScenarioSpec | None
        try:
            spec = ScenarioSpec.model_validate(raw_scenario)
        except Exception:
            spec = None

        patches: list[PatchOperation] = []
        diagnosis = ""
        scenario_payload = state["scenario_params"]
        next_strategy = blueprint.sizing_strategy
        margin = blueprint.resilience_margin
        failure = report.failure_class

        if failure in {"instability", "underperformance", "divergence"} and spec is None:
            scenario_payload, patches = _repair_scenario(raw_scenario)
            diagnosis = "Scenario slice was invalid; repaired before capacity reasoning."
            try:
                spec = ScenarioSpec.model_validate(scenario_payload)
            except Exception:
                spec = DEFAULT_SCENARIO

        if failure in {"instability", "underperformance", "divergence"}:
            assert spec is not None
            observed_util = report.metrics.get("max_utilization", 0.0)
            gate = next(
                (c.threshold for c in blueprint.test_criteria if c.metric_key == "max_utilization"), 0.92
            )
            overshoot = max(1.0, observed_util / gate) if gate else 1.1
            target_scale = float(
                min(1.60, max(spec.stress.capacity_scale * (1.0 + 0.55 * (overshoot - 1.0) + 0.18), 1.10))
            )
            patches.append(
                PatchOperation(
                    target="scenario",
                    pointer="stress.capacity_scale",
                    previous=spec.stress.capacity_scale,
                    updated=round(target_scale, 4),
                    rationale=(
                        f"observed rho={observed_util:.3f} exceeded the {gate:.2f} stability gate; "
                        "raise the berth provision factor"
                    ),
                )
            )
            if next_strategy != "tail_risk":
                patches.append(
                    PatchOperation(
                        target="blueprint",
                        pointer="sizing_strategy",
                        previous=next_strategy,
                        updated="tail_risk",
                        rationale=(
                            "proportional sizing ignores the stationary productivity loss under stress; "
                            "adopt the Erlang-C requirement computed by the planner"
                        ),
                    )
                )
                next_strategy = "tail_risk"
            if retry >= 1:
                margin = round(min(1.5, margin + 0.08), 4)
                patches.append(
                    PatchOperation(
                        target="blueprint",
                        pointer="resilience_margin",
                        previous=blueprint.resilience_margin,
                        updated=margin,
                        rationale="second consecutive capacity failure: widen the resilience margin",
                    )
                )
            diagnosis = (
                f"Queue saturation ({failure}): at least one terminal operated above the stability gate, "
                "so the replication cannot reach a steady state and service levels collapse."
            )
            stress = dict(spec.stress.model_dump(mode="json"))
            stress["capacity_scale"] = round(target_scale, 4)
            scenario_payload = {**spec.model_dump(mode="json"), "stress": stress}

        elif failure == "runtime_error":
            if "ValidationError" in report.traceback_summary:
                scenario_payload, patches = _repair_scenario(state["scenario_params"])
                diagnosis = "Generated manifest failed Pydantic validation; schema repaired."
            else:
                stress = dict(spec.stress.model_dump(mode="json"))
                new_cv = round(min(float(stress["service_cv"]), 0.40), 4)
                patches.append(
                    PatchOperation(
                        target="scenario",
                        pointer="stress.service_cv",
                        previous=stress["service_cv"],
                        updated=new_cv,
                        rationale="damp service-time variance to remove the unhandled jump in the engine",
                    )
                )
                stress["service_cv"] = new_cv
                scenario_payload = {**spec.model_dump(mode="json"), "stress": stress}
                diagnosis = (
                    "Sandbox raised an uncaught exception: "
                    f"{report.traceback_summary or 'no traceback captured'}"
                )

        elif failure == "schema_error":
            scenario_payload, patches = _repair_scenario(raw_scenario)
            diagnosis = (
                "Manifest violated a Pydantic invariant; probabilities projected back onto the simplex."
            )
            try:
                spec = ScenarioSpec.model_validate(scenario_payload)
            except Exception:
                spec = None

        elif failure == "timeout":
            stress = dict(spec.stress.model_dump(mode="json"))
            new_horizon = round(max(60.0, float(stress["horizon_days"]) * 0.6), 2)
            patches.append(
                PatchOperation(
                    target="scenario",
                    pointer="stress.horizon_days",
                    previous=stress["horizon_days"],
                    updated=new_horizon,
                    rationale="sandbox exceeded its wall-clock budget; shorten the replicated horizon",
                )
            )
            stress["horizon_days"] = new_horizon
            scenario_payload = {**spec.model_dump(mode="json"), "stress": stress}
            diagnosis = f"Execution exceeded the sandbox budget of {context.sandbox_timeout:.0f}s."

        else:  # pragma: no cover - defensive
            diagnosis = f"Unclassified failure: {failure}"

        confidence = 0.55 if failure == "runtime_error" else 0.85
        if retry >= 1:
            confidence = max(0.3, confidence - 0.15)

        can_retry = (retry + 1) <= context.max_retries
        reflection = ReflectionRecord(
            attempt=retry,
            failure_class=failure,
            diagnosis=diagnosis,
            patches=patches,
            next_action="synthesize" if can_retry else "finalize",
            confidence=confidence,
        )
        blueprint_payload = blueprint.model_dump(mode="json")
        blueprint_payload["sizing_strategy"] = next_strategy
        blueprint_payload["resilience_margin"] = margin
        if patches:
            blueprint_payload.setdefault("risk_notes", []).append(f"attempt {retry + 1} patch: {diagnosis}")
        return {
            "scenario_params": scenario_payload,
            "architectural_blueprint": blueprint_payload,
            "reflection": _as_json(reflection),
            "reflection_trail": [_as_json(reflection)],
            "retry_count": retry + 1,
            "execution_logs": [
                _log(
                    "node_reflector",
                    "warning",
                    "reflection emitted",
                    attempt=retry,
                    failure_class=failure,
                    diagnosis=diagnosis,
                    patches=[p.model_dump(mode="json") for p in patches],
                    next_action=reflection.next_action,
                )
            ],
        }

    return node_reflector


# ---------------------------------------------------------------------------
# node_finalizer
# ---------------------------------------------------------------------------


def make_finalizer(context: RunContext) -> Callable[[State], dict[str, Any]]:
    def node_finalizer(state: State) -> dict[str, Any]:
        report = TestReport.model_validate(state["test_report"]) if state.get("test_report") else None
        attempts = int(state.get("retry_count", 0)) + 1
        trail = [ReflectionRecord.model_validate(r) for r in state.get("reflection_trail", [])]
        artifacts = state.get("code_artifacts") or {}
        records = [ArtifactRecord.model_validate(a) for a in artifacts.get("artifacts", [])]

        if report is None:
            verdict = "FAILED_NEEDS_REVIEW"
            report = TestReport(
                attempt=attempts - 1,
                passed=False,
                exit_code=-1,
                timed_out=False,
                duration_seconds=0.0,
                failure_class="runtime_error",
                criteria=[],
                metrics={},
                stderr_tail="no sandbox report was produced",
            )
        elif report.passed:
            verdict = "PASS" if attempts == 1 else "PASS_WITH_RETRIES"
        else:
            verdict = "FAILED_NEEDS_REVIEW"

        blueprint = state.get("architectural_blueprint") or {}
        final = FinalReport(
            run_id=context.run_id,
            verdict=verdict,
            spec_id=str((state.get("scenario_params") or {}).get("spec_id", "unknown")),
            blueprint_id=str(blueprint.get("blueprint_id", "unknown")),
            attempts=attempts,
            duration_seconds=round(
                sum(
                    float(e.get("payload", {}).get("duration_seconds", 0.0) or 0.0)
                    for e in state.get("execution_logs", [])
                    if e.get("node") == "node_sandbox_tester"
                ),
                4,
            ),
            test_report=report,
            reflection_trail=trail,
            artifacts=records,
            notes=[
                f"max_retries={context.max_retries}",
                f"fault_mode={context.fault_mode}",
                f"sizing_strategy={blueprint.get('sizing_strategy')}",
            ],
        )
        payload = _as_json(final)
        run_dir = context.run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "final_report.json").write_text(json.dumps(payload, indent=2, default=str))
        (run_dir / "execution_log.json").write_text(
            json.dumps(state.get("execution_logs", []), indent=2, default=str)
        )
        # Full snapshot so dashboards and reviewers can re-hydrate a finished
        # build without re-executing the sandbox.
        snapshot = {
            key: state.get(key)
            for key in (
                "run_id",
                "scenario_params",
                "architectural_blueprint",
                "code_artifacts",
                "execution_logs",
                "retry_count",
                "reflection_trail",
                "test_report",
                "verdict",
            )
        }
        snapshot["final_report"] = payload
        (run_dir / "state_snapshot.json").write_text(json.dumps(snapshot, indent=2, default=str))
        return {
            "final_report": payload,
            "verdict": verdict,
            "execution_logs": [
                _log(
                    "node_finalizer",
                    "info" if verdict.startswith("PASS") else "error",
                    f"artifact finalized: {verdict}",
                    attempts=attempts,
                    run_dir=str(run_dir),
                )
            ],
        }

    return node_finalizer
