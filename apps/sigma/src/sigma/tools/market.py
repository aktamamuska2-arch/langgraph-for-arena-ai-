"""The deterministic analytical substrate SIGMA reasons about.

A synthetic venture portfolio generated from a seed. No network, no credentials:
the same `(portfolio_id, seed)` always yields the same world, which is what makes
NFR-3 (determinism) and the whole hermetic test suite possible.

All maths is stdlib only (`random`, `statistics`) — deliberately no numpy/pandas
(PRD section 10.4).
"""

from __future__ import annotations

import random
import statistics
from functools import lru_cache
from typing import Any

SECTORS = ("fintech", "logistics", "healthtech", "agritech", "edutech", "climate")
STAGES = ("seed", "series-a", "series-b", "growth")
STAGE_RISK = {"seed": 0.85, "series-a": 0.62, "series-b": 0.45, "growth": 0.30}

_NAME_HEADS = ("Nusa", "Meridian", "Kirana", "Banyu", "Surya", "Loka", "Ardent",
               "Cakra", "Daya", "Eka", "Garuda", "Hasta", "Indra", "Jala")
_NAME_TAILS = ("Labs", "Works", "Systems", "Capital", "Dynamics", "Networks", "Collective")


class Venture(dict):
    """Typed-by-convention dict so it serialises cheaply into checkpoints."""


@lru_cache(maxsize=8)
def build_portfolio(portfolio_id: str, seed: int, size: int = 18) -> tuple[Venture, ...]:
    """Generate (and memoise) a portfolio. Deterministic in `(portfolio_id, seed)`."""
    rng = random.Random(f"{portfolio_id}:{seed}")
    ventures: list[Venture] = []
    for i in range(size):
        sector = SECTORS[i % len(SECTORS)]
        stage = STAGES[rng.randrange(len(STAGES))]
        capital = round(rng.uniform(0.4, 9.5), 2)  # USD millions deployed
        ventures.append(Venture(
            id=f"V{i + 1:02d}",
            name=f"{rng.choice(_NAME_HEADS)} {rng.choice(_NAME_TAILS)}",
            sector=sector,
            stage=stage,
            capital_deployed=capital,
            growth_rate=round(rng.uniform(-0.05, 0.55), 4),
            volatility=round(rng.uniform(0.18, 0.72), 4),
            burn_multiple=round(rng.uniform(0.6, 2.8), 3),
            team_score=round(rng.uniform(0.35, 0.98), 3),
            market_score=round(rng.uniform(0.30, 0.95), 3),
            runway_months=rng.randrange(6, 34),
        ))
    return tuple(ventures)


def _index(portfolio_id: str, seed: int) -> dict[str, Venture]:
    return {v["id"]: v for v in build_portfolio(portfolio_id, seed)}


def _select(portfolio_id: str, seed: int, venture_ids: list[str] | None,
            sector: str | None = None, stage: str | None = None) -> list[Venture]:
    universe = build_portfolio(portfolio_id, seed)
    picked: list[Venture] = []
    if venture_ids:
        idx = _index(portfolio_id, seed)
        for vid in venture_ids:
            if vid in idx:
                picked.append(idx[vid])
        return picked
    for v in universe:
        if sector and v["sector"] != sector:
            continue
        if stage and v["stage"] != stage:
            continue
        picked.append(v)
    return picked


def _stage_multiplier(stage: str) -> float:
    return {"seed": 1.0, "series-a": 0.78, "series-b": 0.55, "growth": 0.38}[stage]


# --------------------------------------------------------------------------- #
# Tool implementations — each returns a JSON-serialisable dict
# --------------------------------------------------------------------------- #


def query_portfolio(portfolio_id: str, seed: int, *, sector: str | None = None,
                    stage: str | None = None, limit: int = 10) -> dict[str, Any]:
    """List ventures in the portfolio with light aggregates."""
    rows = _select(portfolio_id, seed, None, sector, stage)
    rows = rows[: max(1, min(limit, 50))]
    capital = [r["capital_deployed"] for r in rows] or [0.0]
    return {
        "portfolio_id": portfolio_id,
        "seed": seed,
        "count": len(rows),
        "total_capital": round(sum(capital), 3),
        "median_capital": round(statistics.median(capital), 3),
        "ventures": [
            {k: r[k] for k in ("id", "name", "sector", "stage", "capital_deployed",
                               "growth_rate", "volatility", "runway_months")}
            for r in rows
        ],
    }


def compute_risk_metrics(portfolio_id: str, seed: int, *, venture_ids: list[str],
                         horizon_quarters: int = 8) -> dict[str, Any]:
    """Expected value, volatility, concentration (HHI) and a 0..1 risk score."""
    rows = _select(portfolio_id, seed, venture_ids)
    if not rows:
        return {"error": "no ventures matched", "venture_ids": venture_ids}
    total_capital = sum(r["capital_deployed"] for r in rows) or 1.0
    shares = [r["capital_deployed"] / total_capital for r in rows]
    hhi = round(sum(s * s for s in shares), 4)

    ev_terms: list[float] = []
    weighted_vol = 0.0
    weighted_risk = 0.0
    for r, share in zip(rows, shares):
        quarters = horizon_quarters / 4.0
        terminal = r["capital_deployed"] * ((1.0 + r["growth_rate"]) ** quarters)
        ev_terms.append(terminal)
        weighted_vol += share * r["volatility"]
        stage_risk = STAGE_RISK[r["stage"]] * _stage_multiplier(r["stage"])
        idio = 0.35 * (1.0 - r["team_score"]) + 0.30 * (1.0 - r["market_score"])
        burn_risk = min(1.0, (r["burn_multiple"] - 0.6) / 2.2)
        runway_risk = min(1.0, max(0.0, (18 - r["runway_months"]) / 18.0))
        weighted_risk += share * (0.40 * stage_risk + 0.20 * idio + 0.20 * burn_risk
                                  + 0.20 * runway_risk)

    expected_value = round(sum(ev_terms), 3)
    invested = round(total_capital, 3)
    return {
        "venture_count": len(rows),
        "invested": invested,
        "expected_value": expected_value,
        "expected_multiple": round(expected_value / invested, 4) if invested else 0.0,
        "weighted_volatility": round(weighted_vol, 4),
        "concentration_hhi": hhi,
        "risk_score": round(min(1.0, max(0.0,
                                         0.45 * weighted_risk + 0.35 * weighted_vol
                                         + 0.20 * hhi)), 4),
        "horizon_quarters": horizon_quarters,
    }


def _stable_hash(text: str) -> int:
    h = 2166136261
    for ch in text:
        h = (h ^ ord(ch)) * 16777619
        h &= 0xFFFFFFFF
    return h


def scenario_simulate(portfolio_id: str, seed: int, *, venture_ids: list[str],
                      paths: int = 4000, horizon_quarters: int = 8,
                      monte_carlo_paths_cap: int = 6000) -> dict[str, Any]:
    """Monte-Carlo terminal value distribution -> P10/P50/P90, VaR95, P(loss)."""
    rows = _select(portfolio_id, seed, venture_ids)
    if not rows:
        return {"error": "no ventures matched", "venture_ids": venture_ids}
    n_paths = max(200, min(paths, monte_carlo_paths_cap))
    key = f"{portfolio_id}:{seed}:{','.join(sorted(venture_ids))}:{horizon_quarters}:{n_paths}"
    rng = random.Random(_stable_hash(key))
    steps = horizon_quarters
    invested = sum(r["capital_deployed"] for r in rows)

    outcomes: list[float] = []
    for _ in range(n_paths):
        total = 0.0
        for r in rows:
            mu = r["growth_rate"] / 4.0
            sigma = r["volatility"] / (4.0 ** 0.5)
            value = r["capital_deployed"]
            for _q in range(steps):
                shock = rng.gauss(mu, sigma)
                # Fat-ish left tail: occasional drawdown event.
                if rng.random() < 0.02:
                    shock -= abs(rng.gauss(0.0, sigma * 2.5))
                value *= max(0.05, 1.0 + shock)
            total += value
        outcomes.append(total / invested if invested else 0.0)

    outcomes.sort()
    n = len(outcomes)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return round(outcomes[idx], 4)

    var95 = round(max(0.0, 1.0 - pct(0.05)), 4)
    prob_loss = round(sum(1 for o in outcomes if o < 1.0) / n, 4)
    return {
        "paths": n_paths,
        "horizon_quarters": horizon_quarters,
        "invested": round(invested, 3),
        "p10_multiple": pct(0.10),
        "p50_multiple": pct(0.50),
        "p90_multiple": pct(0.90),
        "mean_multiple": round(statistics.fmean(outcomes), 4),
        "var95": var95,
        "prob_loss": prob_loss,
        "downside_notable": var95 >= 0.35 or prob_loss >= 0.30,
    }


def benchmark_sector(portfolio_id: str, seed: int, *, sector: str) -> dict[str, Any]:
    """Sector quartiles used to normalise a venture's metrics."""
    rows = _select(portfolio_id, seed, None, sector=sector)
    if not rows:
        return {"error": f"unknown sector {sector!r}", "sectors": list(SECTORS)}
    growth = sorted(r["growth_rate"] for r in rows)
    vol = sorted(r["volatility"] for r in rows)

    def quartiles(xs: list[float]) -> dict[str, float]:
        return {"p25": round(xs[len(xs) // 4], 4),
                "p50": round(statistics.median(xs), 4),
                "p75": round(xs[(3 * len(xs)) // 4], 4)}

    return {
        "sector": sector,
        "sample_size": len(rows),
        "growth_quartiles": quartiles(growth),
        "volatility_quartiles": quartiles(vol),
        "capital_deployed": round(sum(r["capital_deployed"] for r in rows), 3),
    }


def evidence_lookup(_portfolio_id: str, _seed: int, *, evidence_id: str,
                    state_evidence: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Re-fetch an evidence item already collected (dedup helper for the researcher)."""
    found = (state_evidence or {}).get(evidence_id)
    if found is None:
        return {"error": f"unknown evidence_id {evidence_id!r}", "found": False}
    return {"found": True, "evidence_id": evidence_id, "summary": found.get("summary", ""),
            "tool": found.get("tool"), "provenance": found.get("provenance")}
