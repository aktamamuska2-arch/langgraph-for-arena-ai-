"""Geopolitical layer: marked Poisson incidents over choke points and terminals.

Model
-----
Every chokepoint ``c`` receives incidents as a non-homogeneous Poisson process
with rate

    lambda_c(t) = (geo_rate_c / 365) * geo_intensity * m(t)
    m(t)        = 1 + 0.35 * sin(2*pi*(t + phase_c) / 365)

Each incident is a mark ``(S, D)`` with severity ``S ~ Beta(alpha, beta)``
scaled by ``max_severity_c`` and duration ``D ~ LogNormal(median 2.2 d,
sigma 0.75)`` truncated to ``[0.25, 21]`` days. The instantaneous capacity
factor of a degraded target is

    kappa_c(t) = 1 - max_k [ S_k * (1 - 0.55 * (t - t_k) / D_k) ]   for active k

i.e. an incident hits hard on impact and decays to 45% of its severity before
clearing. Terminals additionally receive operational incidents (crane failure,
cyber-attack) at ``PORT_INCIDENT_RATE`` per year.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field

import numpy as np

from .schemas import NetworkSpec, ShockSpec, StressSpec

PORT_INCIDENT_RATE_PER_YEAR = 1.4
MAX_CONCURRENT_INCIDENTS = 6


@dataclass
class Incident:
    incident_id: str
    target: str
    kind: str
    start_day: float
    duration_days: float
    severity: float
    label: str

    @property
    def end_day(self) -> float:
        return self.start_day + self.duration_days

    def severity_at(self, day: float) -> float:
        if day < self.start_day or day >= self.end_day:
            return 0.0
        progress = (day - self.start_day) / max(self.duration_days, 1e-6)
        return self.severity * (1.0 - 0.55 * progress)

    def to_row(self) -> dict[str, float | str]:
        return {
            "incident_id": self.incident_id,
            "target": self.target,
            "kind": self.kind,
            "start_day": round(self.start_day, 3),
            "end_day": round(self.end_day, 3),
            "severity": round(self.severity, 4),
            "label": self.label,
        }


@dataclass
class GeopoliticalField:
    """Scheduled + stochastic disruption bookkeeping."""

    env: object
    network: NetworkSpec
    stress: StressSpec
    rng: np.random.Generator
    incidents: list[Incident] = field(default_factory=list)
    _counter: int = 0

    # -- public queries --------------------------------------------------
    def capacity_factor(self, target: str, day: float | None = None) -> float:
        t = self.env.now if day is None else day  # type: ignore[attr-defined]
        worst = 0.0
        for inc in self.incidents:
            if inc.target != target:
                continue
            worst = max(worst, inc.severity_at(t))
        return max(0.0, 1.0 - worst)

    def active(self, day: float | None = None) -> list[Incident]:
        t = self.env.now if day is None else day  # type: ignore[attr-defined]
        return [i for i in self.incidents if i.start_day <= t < i.end_day]

    def log(self) -> list[dict]:
        return [inc.to_row() for inc in self.incidents]

    # -- generation ------------------------------------------------------
    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter:03d}"

    def _sample_severity(self, cap: float) -> float:
        # Beta(2, 5) has mean 0.286 and a long right tail.
        raw = float(self.rng.beta(2.0, 5.0))
        return float(min(cap, max(0.05, raw * cap)))

    def _sample_duration(self) -> float:
        d = float(self.rng.lognormal(mean=math.log(2.2), sigma=0.75))
        return float(min(21.0, max(0.25, d)))

    def schedule_shock(self) -> None:
        shock: ShockSpec | None = self.stress.shock
        if shock is None:
            return
        self.incidents.append(
            Incident(
                incident_id=self._next_id("SHOCK"),
                target=shock.target,
                kind=shock.kind,
                start_day=shock.start_day,
                duration_days=shock.duration_days,
                severity=shock.severity,
                label=shock.description or shock.id,
            )
        )

    def start(self) -> list:
        self.schedule_shock()
        procs = [self.env.process(self._chokepoint_stream(c)) for c in self.network.chokepoints]
        procs += [self.env.process(self._port_stream(p.id)) for p in self.network.ports]
        return procs

    def _chokepoint_stream(self, chokepoint):
        rate_per_day = chokepoint.geo_rate_per_year / 365.0 * self.stress.geo_intensity
        if rate_per_day <= 0:
            return
        # NOTE: builtin hash() is salted per process, which would make the
        # incident schedule irreproducible across runs. CRC32 is stable.
        phase = zlib.crc32(chokepoint.id.encode()) % 365
        t = 0.0
        while True:
            seasonal = 1.0 + 0.35 * math.sin(2 * math.pi * (t + phase) / 365.0)
            dt = float(self.rng.exponential(1.0 / max(rate_per_day * seasonal, 1e-9)))
            t += dt
            if t > self.stress.horizon_days:
                return
            yield self.env.timeout(max(dt, 1e-3))
            self.incidents.append(
                Incident(
                    incident_id=self._next_id("GEO"),
                    target=chokepoint.id,
                    kind="blockade",
                    start_day=self.env.now,
                    duration_days=self._sample_duration(),
                    severity=self._sample_severity(chokepoint.max_severity),
                    label=f"Incident at {chokepoint.name}",
                )
            )
            if len(self.incidents) > MAX_CONCURRENT_INCIDENTS * 40:  # pragma: no cover
                self.incidents = self.incidents[-MAX_CONCURRENT_INCIDENTS * 20 :]

    def _port_stream(self, port_id: str):
        rate_per_day = PORT_INCIDENT_RATE_PER_YEAR / 365.0 * self.stress.geo_intensity
        if rate_per_day <= 0:
            return
        t = 0.0
        while True:
            dt = float(self.rng.exponential(1.0 / max(rate_per_day, 1e-9)))
            t += dt
            if t > self.stress.horizon_days:
                return
            yield self.env.timeout(max(dt, 1e-3))
            self.incidents.append(
                Incident(
                    incident_id=self._next_id("OPS"),
                    target=port_id,
                    kind="port_blackout" if self.rng.random() < 0.4 else "congestion_surge",
                    start_day=self.env.now,
                    duration_days=self._sample_duration(),
                    severity=self._sample_severity(0.65),
                    label=f"Operational incident at {port_id}",
                )
            )
