"""Weather layer: seasonal continuous-time Markov chains over sea areas.

Model
-----
For every sea zone ``z`` the weather is a CTMC ``W_z(t)`` on the ordered state
space ``S = {CALM, ADVERSE, SEVERE, EXTREME}`` with generator

    Q_z[i, j] = r_z[i] * P_z[i, j]   (i != j)
    Q_z[i, i] = -r_z[i]              r_z[i] = 1 / dwell_z[i]

The embedded chain ``P_z`` is the zone transition matrix from the scenario,
optionally re-weighted by the monsoon season and the scenario's weather
intensity. Stationary probabilities solve ``pi Q = 0``, ``sum(pi) = 1``.

Operational coupling
--------------------
* Sailing: ``v(t) = v0 * speed_factor[W_z(t)]``
* Berth productivity: ``mu_b(t) = mu_b0 * productivity_factor[W_z(t)]``
* Terminal closure: ``P(close | EXTREME) = closure_prob_extreme`` per event
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from .schemas import WEATHER_INDEX, WEATHER_STATES, WeatherZoneSpec


def _season_bias(season: str, zone_id: str) -> float:
    """Escalation multiplier applied before renormalising the embedded chain."""

    table = {
        "NW_MONSOON": {
            "JAVA_SEA": 1.30,
            "NATUNA_SEA": 1.45,
            "MALACCA_STRAIT": 1.25,
            "MAKASSAR_STRAIT": 1.10,
            "FLORES_SEA": 0.85,
        },
        "SE_MONSOON": {
            "JAVA_SEA": 0.95,
            "NATUNA_SEA": 0.90,
            "MALACCA_STRAIT": 1.00,
            "MAKASSAR_STRAIT": 1.25,
            "FLORES_SEA": 1.45,
        },
        "TRANSITION": dict.fromkeys(
            ("JAVA_SEA", "NATUNA_SEA", "MALACCA_STRAIT", "MAKASSAR_STRAIT", "FLORES_SEA"), 1.0
        ),
    }
    return table.get(season, {}).get(zone_id, 1.0)


def effective_transition(
    zone: WeatherZoneSpec, intensity: float = 1.0, season: str = "NW_MONSOON"
) -> np.ndarray:
    """Re-weight the embedded chain and project it back onto the simplex."""

    p = np.array(zone.transition, dtype=float)
    bias = _season_bias(season, zone.id) * intensity
    out = p.copy()
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            if j > i:  # escalation
                out[i, j] = p[i, j] * bias
            else:  # recovery, slightly easier when the forcing is strong
                out[i, j] = p[i, j] / max(bias**0.25, 1e-9)
        out[i, i] = 0.0
        row_sum = out[i].sum()
        out[i] = out[i] / row_sum if row_sum > 0 else np.full(4, 1.0 / 3.0) * (1 - np.eye(4)[i])
        out[i, i] = 0.0
    return out


def generator(zone: WeatherZoneSpec, intensity: float = 1.0, season: str = "NW_MONSOON") -> np.ndarray:
    p = effective_transition(zone, intensity, season)
    rates = np.array([1.0 / h for h in zone.mean_dwell_hours], dtype=float)
    q = p * rates[:, None]
    np.fill_diagonal(q, -rates)
    return q


def stationary_distribution(
    zone: WeatherZoneSpec, intensity: float = 1.0, season: str = "NW_MONSOON"
) -> np.ndarray:
    """Solve ``pi Q = 0`` subject to ``sum(pi) = 1`` via a bordered linear system."""

    q = generator(zone, intensity, season)
    a = q.T.copy()
    a[-1, :] = 1.0
    b = np.zeros(4)
    b[-1] = 1.0
    try:
        pi = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:  # pragma: no cover - numerically degenerate input
        w, v = np.linalg.eig(q.T)
        idx = int(np.argmin(np.abs(w)))
        pi = np.real(v[:, idx])
        pi = pi / pi.sum()
    pi = np.clip(pi, 0.0, 1.0)
    return pi / pi.sum()


def zone_statistics(
    zone: WeatherZoneSpec, intensity: float = 1.0, season: str = "NW_MONSOON"
) -> dict[str, float]:
    pi = stationary_distribution(zone, intensity, season)
    return {
        "calm": float(pi[0]),
        "adverse": float(pi[1]),
        "severe": float(pi[2]),
        "extreme": float(pi[3]),
        "expected_speed_factor": float(np.dot(pi, zone.speed_factor)),
        "expected_productivity": float(np.dot(pi, zone.productivity_factor)),
        "expected_closure_rate": float(pi[3] * zone.closure_prob_extreme),
    }


@dataclass
class ZoneState:
    """Observed state of one zone, updated in place by the SimPy process."""

    zone_id: str
    index: int = 0
    since_day: float = 0.0
    transitions: list[dict[str, float]] = field(default_factory=list)

    @property
    def label(self) -> str:
        return WEATHER_STATES[self.index]


class WeatherField:
    """Drives one CTMC per zone inside the SimPy environment."""

    def __init__(
        self,
        env,
        zones: Iterable[WeatherZoneSpec],
        rng: np.random.Generator,
        intensity: float,
        season: str,
        closure_enabled: bool = True,
    ) -> None:
        self.env = env
        self.rng = rng
        self.intensity = intensity
        self.season = season
        self.closure_enabled = closure_enabled
        self.zones: dict[str, WeatherZoneSpec] = {z.id: z for z in zones}
        self.state: dict[str, ZoneState] = {z.id: ZoneState(zone_id=z.id) for z in self.zones.values()}
        self._embedded = {z.id: effective_transition(z, intensity, season) for z in self.zones.values()}
        self._dwell = {z.id: np.asarray(z.mean_dwell_hours, dtype=float) for z in self.zones.values()}

    # -- queries ---------------------------------------------------------
    def index(self, zone_id: str) -> int:
        return self.state[zone_id].index

    def speed_factor(self, zone_id: str) -> float:
        return self.zones[zone_id].speed_factor[self.state[zone_id].index]

    def productivity_factor(self, zone_id: str) -> float:
        return self.zones[zone_id].productivity_factor[self.state[zone_id].index]

    def is_extreme(self, zone_id: str) -> bool:
        return self.state[zone_id].index == WEATHER_INDEX["EXTREME"]

    # -- process ---------------------------------------------------------
    def start(self) -> list:
        return [self.env.process(self._run(zone_id)) for zone_id in self.zones]

    def _run(self, zone_id: str):
        st = self.state[zone_id]
        # warm start from the stationary distribution so the first days are
        # not artificially calm
        pi = stationary_distribution(self.zones[zone_id], self.intensity, self.season)
        st.index = int(self.rng.choice(4, p=pi))
        st.transitions.append({"day": 0.0, "state": st.index, "label": st.label})
        while True:
            mean_hours = float(self._dwell[zone_id][st.index])
            # Gamma(2, 2) shape over the exponential mean: lower variance dwell
            # than a pure exponential while remaining memoryless-agnostic.
            hold_days = float(self.rng.gamma(shape=2.0, scale=mean_hours / 2.0 / 24.0))
            yield self.env.timeout(max(hold_days, 0.05))
            row = self._embedded[zone_id][st.index]
            prev = st.index
            nxt = int(self.rng.choice(4, p=row))
            st.index = nxt
            st.since_day = self.env.now
            st.transitions.append(
                {"day": round(self.env.now, 3), "state": nxt, "label": st.label, "from": prev}
            )

    # -- telemetry -------------------------------------------------------
    def timeline(self, step_days: float, horizon: float) -> list[dict[str, float]]:
        """Reconstruct the piecewise-constant zone timeline for charting."""

        out: list[dict[str, float]] = []
        for zone_id, st in self.state.items():
            rows = sorted(st.transitions, key=lambda r: r["day"])
            for row in rows:
                out.append({"zone": zone_id, "day": row["day"], "state": row["state"], "label": row["label"]})
        return out
