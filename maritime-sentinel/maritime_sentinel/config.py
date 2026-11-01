"""Default macro-dynamic scenario: Indonesian maritime logistics under stress.

The network is a reduced but structurally faithful model of the domestic
container grid: seven gateway/feeder/transshipment terminals, five sea-lane
bottlenecks, five marine weather regimes and fourteen origin-destination
demand channels. Distances are nautical miles; productivity is TEU per berth
per day.
"""

from __future__ import annotations

from typing import Any

from .schemas import (
    AcceptanceCriteria,
    ChokepointSpec,
    DemandChannel,
    LaneSpec,
    MonteCarloSpec,
    NetworkSpec,
    PerturbationSpec,
    PortSpec,
    ScenarioSpec,
    ShockSpec,
    StressSpec,
    WeatherZoneSpec,
)

# ---------------------------------------------------------------------------
# Weather regimes
#
# Rows are outgoing transition probabilities for the CTMC over
# (CALM, ADVERSE, SEVERE, EXTREME). The northwest monsoon (Dec-Mar) loads the
# Java Sea and Natuna Sea; the southeast monsoon (Jun-Sep) loads the Flores and
# Makassar corridors.
# ---------------------------------------------------------------------------

_NW = "NW_MONSOON"
_SE = "SE_MONSOON"


def _zone(
    zone_id: str,
    label: str,
    dwell: list[float],
    transition: list[list[float]],
    speed: list[float],
    productivity: list[float],
    closure: float,
) -> WeatherZoneSpec:
    return WeatherZoneSpec(
        id=zone_id,
        label=label,
        mean_dwell_hours=dwell,
        transition=transition,
        speed_factor=speed,
        productivity_factor=productivity,
        closure_prob_extreme=closure,
    )


ZONES: list[WeatherZoneSpec] = [
    _zone(
        "JAVA_SEA",
        "Laut Jawa",
        dwell=[62.0, 26.0, 10.0, 6.0],
        transition=[
            [0.00, 0.82, 0.16, 0.02],
            [0.55, 0.00, 0.36, 0.09],
            [0.30, 0.44, 0.00, 0.26],
            [0.42, 0.38, 0.20, 0.00],
        ],
        speed=[1.00, 0.86, 0.62, 0.34],
        productivity=[1.00, 0.93, 0.76, 0.45],
        closure=0.14,
    ),
    _zone(
        "NATUNA_SEA",
        "Laut Natuna",
        dwell=[48.0, 24.0, 11.0, 8.0],
        transition=[
            [0.00, 0.78, 0.18, 0.04],
            [0.50, 0.00, 0.38, 0.12],
            [0.24, 0.42, 0.00, 0.34],
            [0.30, 0.40, 0.30, 0.00],
        ],
        speed=[1.00, 0.83, 0.58, 0.30],
        productivity=[1.00, 0.92, 0.72, 0.40],
        closure=0.18,
    ),
    _zone(
        "MALACCA_STRAIT",
        "Selat Malaka",
        dwell=[70.0, 22.0, 8.0, 5.0],
        transition=[
            [0.00, 0.85, 0.13, 0.02],
            [0.58, 0.00, 0.34, 0.08],
            [0.34, 0.46, 0.00, 0.20],
            [0.46, 0.36, 0.18, 0.00],
        ],
        speed=[1.00, 0.88, 0.66, 0.40],
        productivity=[1.00, 0.94, 0.80, 0.50],
        closure=0.10,
    ),
    _zone(
        "MAKASSAR_STRAIT",
        "Selat Makassar",
        dwell=[56.0, 24.0, 10.0, 7.0],
        transition=[
            [0.00, 0.80, 0.16, 0.04],
            [0.54, 0.00, 0.34, 0.12],
            [0.28, 0.44, 0.00, 0.28],
            [0.36, 0.40, 0.24, 0.00],
        ],
        speed=[1.00, 0.85, 0.60, 0.33],
        productivity=[1.00, 0.93, 0.74, 0.42],
        closure=0.15,
    ),
    _zone(
        "FLORES_SEA",
        "Laut Flores",
        dwell=[44.0, 26.0, 12.0, 9.0],
        transition=[
            [0.00, 0.76, 0.19, 0.05],
            [0.48, 0.00, 0.38, 0.14],
            [0.22, 0.40, 0.00, 0.38],
            [0.28, 0.38, 0.34, 0.00],
        ],
        speed=[1.00, 0.81, 0.55, 0.28],
        productivity=[1.00, 0.91, 0.70, 0.38],
        closure=0.22,
    ),
]

# ---------------------------------------------------------------------------
# Terminals
# ---------------------------------------------------------------------------

PORTS: list[PortSpec] = [
    PortSpec(
        id="PRIOK",
        name="Tanjung Priok",
        lat=-6.104,
        lon=106.883,
        zone="JAVA_SEA",
        berths=8,
        service_teu_per_berth_day=1800.0,
        draft_m=14.0,
        role="gateway",
        hinterland_demand_share=0.34,
    ),
    PortSpec(
        id="PERAK",
        name="Tanjung Perak",
        lat=-7.205,
        lon=112.735,
        zone="JAVA_SEA",
        berths=7,
        service_teu_per_berth_day=1600.0,
        draft_m=12.0,
        role="gateway",
        hinterland_demand_share=0.24,
    ),
    PortSpec(
        id="BELAWAN",
        name="Belawan",
        lat=3.783,
        lon=98.693,
        zone="MALACCA_STRAIT",
        berths=3,
        service_teu_per_berth_day=1200.0,
        draft_m=10.5,
        role="gateway",
        hinterland_demand_share=0.08,
    ),
    PortSpec(
        id="BATAM",
        name="Batu Ampar",
        lat=1.130,
        lon=104.030,
        zone="NATUNA_SEA",
        berths=9,
        service_teu_per_berth_day=1100.0,
        draft_m=11.0,
        role="transshipment",
        hinterland_demand_share=0.06,
    ),
    PortSpec(
        id="MAKASSAR",
        name="Makassar New Port",
        lat=-5.135,
        lon=119.405,
        zone="MAKASSAR_STRAIT",
        berths=5,
        service_teu_per_berth_day=1000.0,
        draft_m=12.0,
        role="gateway",
        hinterland_demand_share=0.11,
    ),
    PortSpec(
        id="BALIKPAPAN",
        name="Kariangau",
        lat=-1.268,
        lon=116.830,
        zone="MAKASSAR_STRAIT",
        berths=4,
        service_teu_per_berth_day=1000.0,
        draft_m=11.0,
        role="gateway",
        hinterland_demand_share=0.09,
    ),
    PortSpec(
        id="SEMARANG",
        name="Tanjung Emas",
        lat=-6.944,
        lon=110.420,
        zone="JAVA_SEA",
        berths=3,
        service_teu_per_berth_day=1100.0,
        draft_m=9.5,
        role="feeder",
        hinterland_demand_share=0.08,
    ),
]

CHOKEPOINTS: list[ChokepointSpec] = [
    ChokepointSpec(
        id="MALACCA",
        name="Selat Malaka",
        lat=2.500,
        lon=101.000,
        zone="MALACCA_STRAIT",
        transit_hours=10.0,
        concurrent_slots=3,
        geo_rate_per_year=8.0,
        max_severity=0.9,
        reroute_hours=26.0,
    ),
    ChokepointSpec(
        id="KARIMATA",
        name="Selat Karimata",
        lat=-2.500,
        lon=108.500,
        zone="NATUNA_SEA",
        transit_hours=7.0,
        concurrent_slots=5,
        geo_rate_per_year=4.0,
        max_severity=0.6,
        reroute_hours=9.0,
    ),
    ChokepointSpec(
        id="SUNDA",
        name="Selat Sunda",
        lat=-5.950,
        lon=105.000,
        zone="JAVA_SEA",
        transit_hours=6.0,
        concurrent_slots=4,
        geo_rate_per_year=5.0,
        max_severity=0.65,
        reroute_hours=8.0,
    ),
    ChokepointSpec(
        id="LOMBOK",
        name="Selat Lombok",
        lat=-8.600,
        lon=115.800,
        zone="FLORES_SEA",
        transit_hours=6.0,
        concurrent_slots=4,
        geo_rate_per_year=3.0,
        max_severity=0.55,
        reroute_hours=10.0,
    ),
    ChokepointSpec(
        id="MAKASSAR_CP",
        name="Selat Makassar",
        lat=-2.000,
        lon=118.000,
        zone="MAKASSAR_STRAIT",
        transit_hours=8.0,
        concurrent_slots=4,
        geo_rate_per_year=4.0,
        max_severity=0.7,
        reroute_hours=12.0,
    ),
]

# ---------------------------------------------------------------------------
# Sea lanes
# ---------------------------------------------------------------------------


def _lane(a: str, b: str, nm: float, zone: str, cost: float, cap: float, **kw: Any) -> LaneSpec:
    return LaneSpec(a=a, b=b, distance_nm=nm, zone=zone, cost_usd_per_teu=cost, capacity_teu_day=cap, **kw)


LANES: list[LaneSpec] = [
    _lane("BELAWAN", "MALACCA", 120, "MALACCA_STRAIT", 70, 14000, note="Malacca approaches"),
    _lane("MALACCA", "BATAM", 260, "MALACCA_STRAIT", 110, 14000),
    _lane("BATAM", "KARIMATA", 420, "NATUNA_SEA", 150, 12000),
    _lane("KARIMATA", "PRIOK", 260, "JAVA_SEA", 95, 16000),
    _lane("PRIOK", "PERAK", 440, "JAVA_SEA", 130, 17000),
    _lane("PRIOK", "SEMARANG", 240, "JAVA_SEA", 85, 12000),
    _lane("SEMARANG", "PERAK", 200, "JAVA_SEA", 75, 12000),
    _lane("PERAK", "MAKASSAR_CP", 420, "MAKASSAR_STRAIT", 135, 14000),
    _lane("MAKASSAR_CP", "BALIKPAPAN", 180, "MAKASSAR_STRAIT", 80, 9000),
    _lane("MAKASSAR_CP", "MAKASSAR", 160, "MAKASSAR_STRAIT", 75, 9000),
    _lane("PERAK", "LOMBOK", 380, "FLORES_SEA", 125, 9000),
    _lane("LOMBOK", "MAKASSAR", 340, "FLORES_SEA", 120, 8000),
    _lane("PRIOK", "SUNDA", 110, "JAVA_SEA", 60, 10000),
    _lane("SUNDA", "BATAM", 480, "NATUNA_SEA", 165, 9000, note="Sunda-Batam alternative"),
    # Contingency: open-sea bypass of the Malacca bottleneck. Only activated
    # when the primary lane is degraded below the routing policy threshold.
    _lane(
        "BELAWAN",
        "BATAM",
        640,
        "MALACCA_STRAIT",
        235,
        5000,
        contingency=True,
        note="Open-sea bypass, +26h and +USD 165/TEU versus Malacca transit",
    ),
]

# ---------------------------------------------------------------------------
# Demand channels
# ---------------------------------------------------------------------------


def _channel(
    cid: str,
    o: str,
    d: str,
    calls: float,
    teu: float,
    sla: float = 96.0,
    cv: float = 0.25,
) -> DemandChannel:
    return DemandChannel(
        id=cid,
        origin=o,
        destination=d,
        base_daily_calls=calls,
        teu_mean=teu,
        teu_cv=cv,
        sla_hours=sla,
    )


CHANNELS: list[DemandChannel] = [
    _channel("CH-01", "BATAM", "PRIOK", 3.0, 1400, 144),
    _channel("CH-02", "PRIOK", "PERAK", 2.5, 1200, 120),
    _channel("CH-03", "PERAK", "MAKASSAR", 1.6, 900, 144),
    _channel("CH-04", "PERAK", "BALIKPAPAN", 1.4, 850, 144),
    _channel("CH-05", "BATAM", "BELAWAN", 1.2, 1000, 120),
    _channel("CH-06", "MAKASSAR", "BALIKPAPAN", 0.9, 700, 120),
    _channel("CH-07", "PRIOK", "SEMARANG", 1.5, 800, 120),
    _channel("CH-08", "SEMARANG", "PERAK", 1.3, 750, 120),
    _channel("CH-09", "BATAM", "MAKASSAR", 1.1, 950, 168),
    _channel("CH-10", "PRIOK", "BELAWAN", 0.8, 900, 168),
    _channel("CH-11", "PERAK", "PRIOK", 1.2, 900, 120),
    _channel("CH-12", "BELAWAN", "BATAM", 0.9, 800, 120),
    _channel("CH-13", "MAKASSAR", "PERAK", 1.0, 700, 144),
    _channel("CH-14", "BALIKPAPAN", "PRIOK", 0.8, 1000, 168),
]

NETWORK = NetworkSpec(
    ports=PORTS,
    chokepoints=CHOKEPOINTS,
    lanes=LANES,
    channels=CHANNELS,
    zones=ZONES,
)

# ---------------------------------------------------------------------------
# Stress envelope for the canonical run
# ---------------------------------------------------------------------------

DEFAULT_STRESS = StressSpec(
    horizon_days=150.0,
    warmup_days=20.0,
    demand_multiplier=1.0,
    weather_intensity=1.35,
    geo_intensity=1.25,
    service_cv=0.42,
    season=_NW,
    routing_policy="adaptive",
    capacity_scale=1.0,
    shock=ShockSpec(
        id="SHOCK-MALACCA",
        target="MALACCA",
        kind="blockade",
        start_day=55.0,
        duration_days=18.0,
        severity=0.72,
        description=(
            "Geopolitical closure of the Malacca approaches forcing Belawan-bound "
            "flows onto the open-sea bypass while weather degrades berth productivity."
        ),
    ),
)

DEFAULT_MC = MonteCarloSpec(
    n_replications=24,
    baseline_replications=3,
    perturbations=[
        PerturbationSpec(key="demand_multiplier", low=0.92, high=1.18, label="Demand surge"),
        PerturbationSpec(key="weather_intensity", low=0.85, high=1.65, label="Weather severity"),
        PerturbationSpec(key="geo_intensity", low=0.60, high=1.90, label="Geopolitical friction"),
        PerturbationSpec(key="service_cv", low=0.30, high=0.62, label="Handling variability"),
        PerturbationSpec(key="capacity_scale", low=0.97, high=1.03, label="Crane availability"),
    ],
    acceptance=AcceptanceCriteria(
        max_berth_utilization=0.92,
        max_batch_rse=0.06,
        max_backlog_slope_per_day=2.5,
        min_service_level=0.55,
        max_runtime_seconds=90.0,
        min_throughput_ratio=0.55,
    ),
    percentiles=[5.0, 50.0, 95.0],
)

DEFAULT_SCENARIO = ScenarioSpec(
    spec_id="IDN-MARITIME-NW-2041",
    title="Disrupsi sistemik rantai pasok logistik maritim regional",
    seed=20410,
    network=NETWORK,
    stress=DEFAULT_STRESS,
    montecarlo=DEFAULT_MC,
)


def scenario_dict() -> dict[str, Any]:
    """JSON-safe dump used as the initial graph state slice."""

    return DEFAULT_SCENARIO.model_dump(mode="json")


def load_scenario(payload: dict[str, Any] | None = None) -> ScenarioSpec:
    return ScenarioSpec.model_validate(payload) if payload else DEFAULT_SCENARIO
