"""Pydantic v2 contracts shared by the engine, the pipeline and the web API.

Terrain codes are integers and index ``world.YIELD_TABLE`` column-wise, so the
order below is load-bearing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

OCEAN = 0
COAST = 1
PLAINS = 2
FOREST = 3
HILLS = 4
MOUNTAIN = 5
DESERT = 6

RESOURCES: tuple[str, ...] = ("FOOD", "WOOD", "ORE")
RESOURCE_INDEX: dict[str, int] = {r: i for i, r in enumerate(RESOURCES)}

TERRAIN_LABELS: dict[int, str] = {
    OCEAN: "ocean",
    COAST: "coast",
    PLAINS: "plains",
    FOREST: "forest",
    HILLS: "hills",
    MOUNTAIN: "mountain",
    DESERT: "desert",
}

Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
Positive = Annotated[float, Field(gt=0.0)]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------


class WorldSpec(Frozen):
    """Procedural geography parameters."""

    width: int = Field(default=96, ge=24, le=320)
    height: int = Field(default=64, ge=16, le=240)
    seed: int = Field(default=20261101, ge=0, le=2**32 - 1)
    sea_level: float = Field(default=0.34, ge=0.10, le=0.60)
    settlement_count: int = Field(default=16, ge=3, le=64)
    min_site_separation: float = Field(default=9.0, ge=2.0, le=40.0)
    territory_radius: int = Field(default=4, ge=1, le=12)
    deposit_richness: dict[str, float] = Field(default_factory=lambda: {"FOOD": 1.0, "WOOD": 1.0, "ORE": 0.9})

    @field_validator("deposit_richness")
    @classmethod
    def _known_resources(cls, v: dict[str, float]) -> dict[str, float]:
        unknown = set(v) - set(RESOURCES)
        if unknown:
            raise ValueError(f"unknown resources in deposit_richness: {sorted(unknown)}")
        if any(x <= 0 for x in v.values()):
            raise ValueError("deposit richness must be positive")
        return v


class GravitySpec(Frozen):
    """Parameters of the market-discovery gravity model.

    ``A_ij = G * S_i^alpha * D_j^beta / (d_ij + d0)^gamma * exp(-tau * c_ij) * (1 + mu * m_ij)``
    """

    scale: Positive = 1.0
    alpha: Positive = 0.85  # surplus elasticity
    beta: Positive = 0.95  # deficit elasticity
    gamma: Positive = 1.35  # distance decay exponent
    d0: Positive = 3.0  # distance offset in cells (keeps short hops finite)
    tau: float = Field(default=0.012, ge=0.0, le=0.5)  # transport-cost friction
    memory_bonus: float = Field(default=0.45, ge=0.0, le=3.0)  # repeat-partner trust `mu`


class EconomySpec(Frozen):
    """Production, consumption, pricing and market-discovery parameters."""

    gravity: GravitySpec = Field(default_factory=GravitySpec)
    discovery_radius: float = Field(default=21.0, ge=2.0, le=200.0)
    discovery_interval: int = Field(default=4, ge=1, le=25)
    edge_threshold: Positive = 0.02
    candidate_cut: float = Field(default=0.32, ge=0.0, le=1.0)
    edge_decay: float = Field(default=0.965, ge=0.80, le=1.0)
    edge_min_weight: float = Field(default=0.06, ge=0.0, le=1.0)
    edge_max_idle: int = Field(default=24, ge=4, le=400)
    max_partners: int = Field(default=8, ge=1, le=24)
    trade_fraction: float = Field(default=0.45, gt=0.0, le=1.0)
    transfer_loss: Fraction = 0.04
    spoilage: Fraction = 0.02
    overflow_decay: float = Field(default=0.06, ge=0.0, le=1.0)
    storage_buffer: float = Field(default=6.0, ge=1.0, le=50.0)

    # production
    labor_exponent: float = Field(default=0.78, gt=0.0, le=2.0)
    labor_inertia: float = Field(default=0.25, ge=0.0, le=1.0)
    labor_sharpness: float = Field(default=2.2, gt=0.0, le=10.0)
    tech_gain: float = Field(default=0.0015, ge=0.0, le=0.1)

    # consumption & demography
    food_per_capita: Positive = 0.020
    wood_per_capita: float = Field(default=0.004, ge=0.0)
    ore_per_capita: float = Field(default=0.0009, ge=0.0)
    target_coverage_ticks: float = Field(default=26.0, ge=1.0, le=200.0)
    growth_rate: float = Field(default=0.016, ge=0.0, le=0.2)
    death_rate: float = Field(default=0.005, ge=0.0, le=0.2)
    base_capacity: float = Field(default=40.0, gt=0.0)

    # pricing
    base_price: dict[str, float] = Field(default_factory=lambda: {"FOOD": 1.0, "WOOD": 1.6, "ORE": 2.4})
    price_elasticity: float = Field(default=0.45, ge=0.0, le=2.0)
    price_floor: Positive = 0.25
    price_ceiling: Positive = 6.0

    # infrastructure
    road_threshold: Positive = 55.0
    road_gain: float = Field(default=0.34, ge=0.0, le=0.9)
    road_max: int = Field(default=3, ge=0, le=6)

    @field_validator("base_price")
    @classmethod
    def _prices(cls, v: dict[str, float]) -> dict[str, float]:
        if set(v) != set(RESOURCES):
            raise ValueError(f"base_price must cover exactly {RESOURCES}")
        if any(x <= 0 for x in v.values()):
            raise ValueError("prices must be positive")
        return v

    @model_validator(mode="after")
    def _sane(self) -> Self:
        if self.price_ceiling <= self.price_floor:
            raise ValueError("price_ceiling must exceed price_floor")
        return self


class ShockSpec(Frozen):
    """An exogenous economic or environmental shock injected at a tick."""

    id: str
    kind: Literal[
        "drought",
        "blight",
        "ore_discovery",
        "demand_boom",
        "plague",
        "embargo",
        "road_collapse",
    ]
    x: int | None = None
    y: int | None = None
    radius: float = Field(default=14.0, gt=0.0, le=200.0)
    magnitude: float = Field(default=0.6, ge=0.0, le=1.0)
    duration: int = Field(default=40, ge=1, le=2000)
    start_tick: int = Field(default=60, ge=0)
    description: str = ""


class ExperimentSpec(Frozen):
    """A complete runnable experiment: geography + economy + scripted shocks."""

    experiment_id: str = "AGORA-DEFAULT"
    title: str = "Emergent spatial economy"
    world: WorldSpec = Field(default_factory=WorldSpec)
    economy: EconomySpec = Field(default_factory=EconomySpec)
    horizon: int = Field(default=420, ge=20, le=5000)
    settle_ticks: int = Field(default=40, ge=1, le=500)
    record_every: int = Field(default=5, ge=1, le=100)
    shocks: list[ShockSpec] = Field(default_factory=list)
    name: str = "baseline"

    @model_validator(mode="after")
    def _sane(self) -> Self:
        if self.settle_ticks >= self.horizon:
            raise ValueError("settle_ticks must be < horizon")
        for shock in self.shocks:
            if shock.start_tick >= self.horizon:
                raise ValueError(f"shock {shock.id} starts after the horizon")
        return self


# ---------------------------------------------------------------------------
# Pipeline artifacts
# ---------------------------------------------------------------------------


class Directive(BaseModel):
    """The operator's brief: what kind of world to compile and how to judge it."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    raw_text: str = "Build an emergent spatial-economic open world."
    objective: str = (
        "Compile a 2D lattice world of autonomous settlements whose trade network "
        "self-organises from local surplus/deficit scans, and prove the result "
        "converges and stays connected."
    )
    experiment_id: str = "ASEOWS-001"
    title: str = "Emergent spatial economy"
    seed: int = 20261101
    horizon: int = Field(default=420, ge=40, le=2000)
    settle_ticks: int = Field(default=40, ge=5, le=400)
    max_retries: int = Field(default=3, ge=1, le=5)
    notes: list[str] = Field(default_factory=list)


class Equation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    latex: str
    description: str
    domain: Literal[
        "production", "consumption", "pricing", "gravity", "network", "demography", "infrastructure", "validation"
    ]


class Gate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    metric: str
    comparator: Literal["<=", ">="]
    threshold: float
    description: str


class SiteAudit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    settlement_id: int
    name: str
    x: int
    y: int
    territory_yield: list[float]


class Blueprint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    blueprint_id: str
    experiment_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    objective: str
    equations: list[Equation]
    gates: list[Gate]
    world_summary: dict[str, Any]
    economy: dict[str, Any]
    critical_radius: float
    """Smallest discovery radius that makes the *potential* trade graph connected."""
    site_audit: list[SiteAudit]
    expected: dict[str, float]
    risk_notes: list[str] = Field(default_factory=list)


class GateResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    metric: str
    observed: float
    threshold: float
    comparator: Literal["<=", ">="]
    passed: bool

    @classmethod
    def build(cls, gate: Gate, observed: float) -> GateResult:
        ok = observed <= gate.threshold if gate.comparator == "<=" else observed >= gate.threshold
        return cls(
            id=gate.id,
            metric=gate.metric,
            observed=float(observed),
            threshold=gate.threshold,
            comparator=gate.comparator,
            passed=bool(ok),
        )


class ValidationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    attempt: int = Field(ge=0)
    passed: bool
    exit_code: int
    timed_out: bool
    duration_seconds: float
    failure_class: Literal[
        "none",
        "runtime_error",
        "timeout",
        "schema_error",
        "non_convergence",
        "network_fragmentation",
        "mass_imbalance",
        "visual_contract",
        "instability",
    ] = "none"
    results: list[GateResult] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    stdout_tail: str = ""
    stderr_tail: str = ""
    traceback_summary: str = ""


class Patch(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    pointer: str
    previous: Any = None
    updated: Any = None
    rationale: str


class Reflection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    attempt: int = Field(ge=0)
    failure_class: str
    diagnosis: str
    patches: list[Patch] = Field(default_factory=list)
    next_action: Literal["synthesize", "finalize", "abort"] = "synthesize"
    confidence: Fraction = 0.7


class ArtifactRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    path: str
    role: str
    sha256: str
    bytes: int


class FinalReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    run_id: str
    verdict: Literal["PASS", "PASS_WITH_RETRIES", "FAILED_NEEDS_REVIEW"]
    experiment_id: str
    blueprint_id: str
    attempts: int
    validation: ValidationReport
    reflections: list[Reflection] = Field(default_factory=list)
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=_utcnow)
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Telemetry / API payloads
# ---------------------------------------------------------------------------


class ShockEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    kind: str
    x: int
    y: int
    radius: float
    magnitude: float
    start_tick: int
    end_tick: int


class SettlementSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: int
    name: str
    x: int
    y: int
    population: float
    wealth: float
    stocks: dict[str, float]
    prices: dict[str, float]
    produced: dict[str, float]
    consumed: dict[str, float]
    surplus: dict[str, float]
    deficit: dict[str, float]
    specialization: str
    degree: int
    alive: bool
    capacity: float
    labor: dict[str, float]
    endowment: dict[str, float]


class NetworkSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    edges: int
    nodes: int
    density: float
    mean_degree: float
    giant_component_share: float
    total_flow: float
    roads_upgraded: int


def validate_payload(model: type[BaseModel], payload: Any, context: str) -> Any:
    try:
        return model.model_validate(payload)
    except Exception as exc:
        raise ValueError(f"{context}: {type(exc).__name__}: {exc}") from exc
