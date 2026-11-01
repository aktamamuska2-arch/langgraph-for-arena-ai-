"""Pydantic v2 contracts for every object that crosses a LangGraph node boundary.

Every node in the state graph validates its input against one of these models
before acting, and every node emits a validated model dump. This is the
"strict typing" layer required by the AM-SES specification: the graph state is
plain JSON-safe data, but it is *never* consumed without re-validation.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

WEATHER_STATES: tuple[str, ...] = ("CALM", "ADVERSE", "SEVERE", "EXTREME")
WEATHER_INDEX: dict[str, int] = {s: i for i, s in enumerate(WEATHER_STATES)}

Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
Positive = Annotated[float, Field(gt=0.0)]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Frozen(BaseModel):
    """Immutable base model: state slices must never be mutated in place."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Physical network
# ---------------------------------------------------------------------------


class WeatherZoneSpec(Frozen):
    """Continuous-time Markov chain describing marine weather in a sea area.

    States are ordered ``CALM, ADVERSE, SEVERE, EXTREME``. ``transition[i][j]``
    is the probability that the chain jumps to state ``j`` when it leaves
    state ``i``; holding times are exponential with the matching mean from
    ``mean_dwell_hours`` (a CTMC, so it is exactly memoryless).
    """

    id: str
    label: str
    mean_dwell_hours: list[float] = Field(min_length=4, max_length=4)
    transition: list[list[float]] = Field(min_length=4, max_length=4)
    speed_factor: list[float] = Field(min_length=4, max_length=4)
    productivity_factor: list[float] = Field(min_length=4, max_length=4)
    closure_prob_extreme: Fraction = 0.18

    @field_validator("mean_dwell_hours")
    @classmethod
    def _positive_dwell(cls, v: list[float]) -> list[float]:
        if any(h <= 0 for h in v):
            raise ValueError("mean_dwell_hours must be strictly positive")
        return v

    @field_validator("transition")
    @classmethod
    def _stochastic_rows(cls, v: list[list[float]]) -> list[list[float]]:
        for i, row in enumerate(v):
            if len(row) != 4:
                raise ValueError(f"transition row {i} must have 4 entries")
            if any(p < 0 or p > 1 for p in row):
                raise ValueError(f"transition row {i} has probability outside [0, 1]")
            if abs(sum(row) - 1.0) > 1e-6:
                raise ValueError(f"transition row {i} sums to {sum(row):.6f}, expected 1.0")
        return v

    @field_validator("speed_factor", "productivity_factor")
    @classmethod
    def _bounded_factors(cls, v: list[float]) -> list[float]:
        if any(f <= 0 or f > 1.0 for f in v):
            raise ValueError("operational factors must lie in (0, 1]")
        # monotone degradation with increasing severity
        if any(v[i] < v[i + 1] for i in range(len(v) - 1)):
            raise ValueError("operational factors must be non-increasing with severity")
        return v


class PortSpec(Frozen):
    """A container terminal: berth capacity and berth-day productivity."""

    id: str
    name: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    zone: str
    berths: int = Field(ge=1, le=64)
    service_teu_per_berth_day: Positive
    draft_m: float = Field(gt=0, le=30)
    role: Literal["gateway", "feeder", "transshipment"] = "gateway"
    hinterland_demand_share: Fraction = 0.0


class ChokepointSpec(Frozen):
    """A strait or sea-lane bottleneck with a finite vessel transit capacity."""

    id: str
    name: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    zone: str
    transit_hours: Positive = 12.0
    concurrent_slots: int = Field(ge=1, le=256)
    geo_rate_per_year: float = Field(ge=0.0, le=52.0)
    max_severity: Fraction = 0.85
    reroute_hours: float = Field(ge=0.0)


class LaneSpec(Frozen):
    """A directed sea leg. Lanes are declared once and mirrored internally."""

    a: str
    b: str
    distance_nm: Positive
    zone: str
    cost_usd_per_teu: Positive
    capacity_teu_day: Positive
    contingency: bool = False
    note: str = ""


class DemandChannel(Frozen):
    """A Poisson stream of vessel calls between an origin and a destination."""

    id: str
    origin: str
    destination: str
    base_daily_calls: Positive
    teu_mean: Positive
    teu_cv: float = Field(default=0.25, ge=0.0, le=1.0)
    sla_hours: Positive = 96.0
    seasonality: list[float] = Field(default_factory=lambda: [1.0] * 12, min_length=12, max_length=12)


class NetworkSpec(Frozen):
    ports: list[PortSpec] = Field(min_length=2)
    chokepoints: list[ChokepointSpec] = Field(min_length=1)
    lanes: list[LaneSpec] = Field(min_length=1)
    channels: list[DemandChannel] = Field(min_length=1)
    zones: list[WeatherZoneSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _referential_integrity(self) -> Self:
        port_ids = {p.id for p in self.ports}
        node_ids = port_ids | {c.id for c in self.chokepoints}
        zone_ids = {z.id for z in self.zones}
        if len(port_ids) != len(self.ports):
            raise ValueError("duplicate port ids")
        if len(node_ids) != len(self.ports) + len(self.chokepoints):
            raise ValueError("node id collision between ports and chokepoints")
        if len(zone_ids) != len(self.zones):
            raise ValueError("duplicate weather zone ids")
        for z in self.zones:
            if z.id not in zone_ids:  # pragma: no cover - defensive
                raise ValueError(f"unknown zone {z.id}")
        for lane in self.lanes:
            for endpoint in (lane.a, lane.b):
                if endpoint not in node_ids:
                    raise ValueError(f"lane {lane.a}-{lane.b} references unknown node {endpoint}")
            if lane.a == lane.b:
                raise ValueError(f"self-loop lane on {lane.a}")
            if lane.zone not in zone_ids:
                raise ValueError(f"lane {lane.a}-{lane.b} references unknown zone {lane.zone}")
        for ch in self.channels:
            if ch.origin not in port_ids or ch.destination not in port_ids:
                raise ValueError(f"demand channel {ch.id} must connect two ports")
            if ch.origin == ch.destination:
                raise ValueError(f"demand channel {ch.id} is a self-loop")
        for p in self.ports:
            if p.zone not in zone_ids:
                raise ValueError(f"port {p.id} references unknown zone {p.zone}")
        for c in self.chokepoints:
            if c.zone not in zone_ids:
                raise ValueError(f"chokepoint {c.id} references unknown zone {c.zone}")
        return self


# ---------------------------------------------------------------------------
# Stress envelope
# ---------------------------------------------------------------------------


class ShockSpec(Frozen):
    """A scheduled systemic shock used to measure resilience and recovery."""

    id: str
    target: str
    kind: Literal["blockade", "port_blackout", "congestion_surge"] = "blockade"
    start_day: float = Field(ge=0.0)
    duration_days: Positive
    severity: Fraction = 0.8
    description: str = ""


class StressSpec(Frozen):
    horizon_days: Positive = 150.0
    warmup_days: float = Field(default=20.0, ge=0.0)
    demand_multiplier: Positive = 1.0
    weather_intensity: Positive = 1.0
    geo_intensity: Positive = 1.0
    service_cv: float = Field(default=0.42, ge=0.01, le=1.5)
    season: Literal["NW_MONSOON", "SE_MONSOON", "TRANSITION"] = "NW_MONSOON"
    routing_policy: Literal["adaptive", "static"] = "adaptive"
    capacity_scale: Positive = 1.0
    shock: ShockSpec | None = None

    @model_validator(mode="after")
    def _warmup_fits(self) -> Self:
        if self.warmup_days >= self.horizon_days * 0.5:
            raise ValueError("warmup_days must be < 50% of horizon_days")
        return self


class PerturbationSpec(Frozen):
    """Uniform perturbation range for one uncertain input of the Monte Carlo sweep."""

    key: Literal[
        "demand_multiplier",
        "weather_intensity",
        "geo_intensity",
        "service_cv",
        "capacity_scale",
    ]
    low: Positive
    high: Positive
    label: str = ""

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.high <= self.low:
            raise ValueError(f"perturbation {self.key}: high must exceed low")
        return self


class AcceptanceCriteria(Frozen):
    """Hard pass/fail gates evaluated by the sandbox tester."""

    max_berth_utilization: Fraction = 0.92
    max_batch_rse: float = Field(default=0.05, gt=0)
    max_backlog_slope_per_day: float = 2.0
    min_service_level: Fraction = 0.55
    max_runtime_seconds: Positive = 90.0
    min_throughput_ratio: Fraction = 0.55


class MonteCarloSpec(Frozen):
    n_replications: int = Field(default=24, ge=2, le=400)
    baseline_replications: int = Field(default=3, ge=1, le=20)
    perturbations: list[PerturbationSpec] = Field(default_factory=list)
    acceptance: AcceptanceCriteria = Field(default_factory=AcceptanceCriteria)
    percentiles: list[float] = Field(default_factory=lambda: [5.0, 50.0, 95.0])

    @field_validator("percentiles")
    @classmethod
    def _sorted_percentiles(cls, v: list[float]) -> list[float]:
        if v != sorted(v):
            raise ValueError("percentiles must be ascending")
        if any(p < 0 or p > 100 for p in v):
            raise ValueError("percentiles must lie in [0, 100]")
        return v


class ScenarioSpec(Frozen):
    """Fully validated description of one macro-dynamic stress scenario."""

    spec_id: str
    title: str
    seed: int = Field(ge=0, le=2**32 - 1)
    network: NetworkSpec
    stress: StressSpec
    montecarlo: MonteCarloSpec

    def nominal_daily_teu(self) -> float:
        return sum(c.base_daily_calls * c.teu_mean for c in self.network.channels)


# ---------------------------------------------------------------------------
# Blueprint / build artefacts
# ---------------------------------------------------------------------------


class EquationRecord(Frozen):
    name: str
    latex: str
    description: str
    domain: Literal["weather", "geopolitics", "queueing", "flow", "risk", "convergence"]


class SizingDecision(Frozen):
    port_id: str
    base_berths: int
    demand_berth_days: float
    effective_berth_days: float
    required_berths: int
    projected_utilization: float
    projected_wait_hours: float
    wait_target_hours: float
    meets_target: bool


class ModuleRecord(Frozen):
    path: str
    role: str
    loc_estimate: int = Field(ge=0)


class TestCriterion(Frozen):
    id: str
    description: str
    threshold: float
    comparator: Literal["<=", ">="]
    metric_key: str


class ArchitecturalBlueprint(Frozen):
    blueprint_id: str
    spec_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    objective: str
    sizing_strategy: Literal["proportional", "tail_risk"] = "proportional"
    resilience_margin: float = Field(default=1.0, ge=1.0, le=3.0)
    wait_target_hours: Positive = 8.0
    network_signature: dict[str, Any]
    equations: list[EquationRecord]
    sizing: list[SizingDecision]
    modules: list[ModuleRecord]
    test_criteria: list[TestCriterion]
    risk_notes: list[str] = Field(default_factory=list)
    total_berths: int = Field(ge=0)


class DeploymentPlan(Frozen):
    """The concrete, executable capacity allocation emitted by the synthesizer."""

    plan_id: str
    blueprint_id: str
    strategy: Literal["proportional", "tail_risk"]
    berths: dict[str, int]
    capacity_scale: float = Field(gt=0)
    routing_policy: Literal["adaptive", "static"]
    seed: int
    horizon_days: float
    warmup_days: float
    notes: list[str] = Field(default_factory=list)

    @field_validator("berths")
    @classmethod
    def _sane_berths(cls, v: dict[str, int]) -> dict[str, int]:
        if not v:
            raise ValueError("deployment must allocate at least one berth")
        if any(b < 1 or b > 96 for b in v.values()):
            raise ValueError("berth allocation must lie in [1, 96]")
        return v


class ArtifactRecord(Frozen):
    path: str
    role: str
    sha256: str = Field(min_length=64, max_length=64)
    bytes: int = Field(ge=0)


# ---------------------------------------------------------------------------
# Execution telemetry
# ---------------------------------------------------------------------------


class ExecutionLogEntry(Frozen):
    node: str
    ts: datetime = Field(default_factory=_utcnow)
    level: Literal["debug", "info", "warning", "error"] = "info"
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)


class CriterionResult(Frozen):
    id: str
    metric_key: str
    observed: float
    threshold: float
    comparator: Literal["<=", ">="]
    passed: bool

    @classmethod
    def evaluate(
        cls,
        criterion: TestCriterion,
        observed: float,
    ) -> CriterionResult:
        ok = (
            observed <= criterion.threshold
            if criterion.comparator == "<="
            else observed >= criterion.threshold
        )
        return cls(
            id=criterion.id,
            metric_key=criterion.metric_key,
            observed=observed,
            threshold=criterion.threshold,
            comparator=criterion.comparator,
            passed=ok,
        )


class TestReport(Frozen):
    attempt: int = Field(ge=0)
    passed: bool
    exit_code: int
    timed_out: bool
    duration_seconds: float
    failure_class: Literal[
        "none",
        "schema_error",
        "runtime_error",
        "timeout",
        "divergence",
        "instability",
        "underperformance",
        "static_check",
    ] = "none"
    criteria: list[CriterionResult] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    stdout_tail: str = ""
    stderr_tail: str = ""
    traceback_summary: str = ""

    def failed_metric_keys(self) -> list[str]:
        return [c.metric_key for c in self.criteria if not c.passed]


class PatchOperation(Frozen):
    target: Literal["scenario", "blueprint", "deployment"]
    pointer: str
    previous: Any = None
    updated: Any = None
    rationale: str


class ReflectionRecord(Frozen):
    attempt: int = Field(ge=0)
    failure_class: str
    diagnosis: str
    patches: list[PatchOperation]
    next_action: Literal["synthesize", "finalize", "abort"]
    confidence: Fraction = 0.6


class FinalReport(Frozen):
    run_id: str
    verdict: Literal["PASS", "PASS_WITH_RETRIES", "FAILED_NEEDS_REVIEW"]
    spec_id: str
    blueprint_id: str
    attempts: int = Field(ge=1)
    duration_seconds: float = Field(ge=0)
    test_report: TestReport
    reflection_trail: list[ReflectionRecord] = Field(default_factory=list)
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=_utcnow)
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Simulation output containers (not state serialised, but rate-limited types)
# ---------------------------------------------------------------------------


class PortOutcome(Frozen):
    port_id: str
    berths: int
    calls: int
    teu_handled: float
    mean_wait_hours: float
    p95_wait_hours: float
    utilization: float
    max_queue: int
    closed_hours: float


class ScenarioOutcome(Frozen):
    replication: int
    seed: int
    throughput_teu_day: float
    delivered_teu: float
    serve_ratio: float
    mean_wait_hours: float
    p95_sojourn_hours: float
    service_level: float
    max_utilization: float
    max_backlog: int
    backlog_slope_per_day: float
    freight_cost_usd: float
    demurrage_cost_usd: float
    resilience_index: float = 1.0
    convergence_rse: float = 0.0
    diverged: bool = False
    runtime_seconds: float = 0.0


def validate_or_raise(model: type[BaseModel], payload: Any, *, context: str) -> Any:
    """Validate ``payload`` against ``model`` with a node-local error prefix."""

    try:
        return model.model_validate(payload)
    except Exception as exc:  # pydantic raises ValidationError, may be wrapped
        raise ValueError(f"{context}: {type(exc).__name__}: {exc}") from exc


def finite_or(value: float, fallback: float = 0.0) -> float:
    return value if isinstance(value, (int, float)) and math.isfinite(value) else fallback
