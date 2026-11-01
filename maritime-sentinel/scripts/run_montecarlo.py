#!/usr/bin/env python3
"""Headless Monte Carlo sweep: writes aggregate KPIs and per-replicate tables.

    python scripts/run_montecarlo.py --reps 32 --out artifacts/mc

Produces ``mc_summary.json``, ``mc_outcomes.csv``, ``mc_samples.csv``,
``mc_ports.csv``, ``mc_tornado.csv`` and ``mc_fan.csv``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from maritime_sentinel.config import DEFAULT_SCENARIO  # noqa: E402
from maritime_sentinel.montecarlo import run_monte_carlo  # noqa: E402
from maritime_sentinel.schemas import ScenarioSpec  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the maritime Monte Carlo perturbation sweep.")
    parser.add_argument("--scenario", type=Path, default=None)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "artifacts" / "mc")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    scenario = (
        ScenarioSpec.model_validate_json(args.scenario.read_text()) if args.scenario else DEFAULT_SCENARIO
    )
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    def progress(done: int, total: int, elapsed: float) -> None:
        if args.quiet:
            return
        width = 32
        filled = int(width * done / total)
        bar = "#" * filled + "-" * (width - filled)
        print(f"\r  [{bar}] {done:>3}/{total}  {elapsed:6.1f}s", end="", flush=True)

    result = run_monte_carlo(
        scenario,
        n_replications=args.reps,
        seed=args.seed,
        progress=progress,
    )
    if not args.quiet:
        print()

    result.outcomes.to_csv(out_dir / "mc_outcomes.csv", index=False)
    result.samples.to_csv(out_dir / "mc_samples.csv", index=False)
    result.port_table.to_csv(out_dir / "mc_ports.csv", index=False)
    result.tornado.to_csv(out_dir / "mc_tornado.csv", index=False)
    result.fan_chart.to_csv(out_dir / "mc_fan.csv", index=False)
    (out_dir / "mc_summary.json").write_text(json.dumps(result.aggregate, indent=2, default=float))

    agg = result.aggregate
    print()
    print(f"  baseline throughput      : {agg['baseline_throughput']:,.0f} TEU/day")
    print(
        f"  baseline on-time (SLA)   : {agg['baseline_effective_throughput']:,.0f} TEU/day "
        f"(service level {agg['baseline_service_level']:.1%})"
    )
    tp = agg["throughput_percentiles"]
    ep = agg["effective_throughput_percentiles"]
    print(f"  stressed throughput      : p05={tp['p5']:,.0f}  p50={tp['p50']:,.0f}  p95={tp['p95']:,.0f}")
    print(f"  stressed on-time (SLA)   : p05={ep['p5']:,.0f}  p50={ep['p50']:,.0f}  p95={ep['p95']:,.0f}")
    rs = agg["resilience_percentiles"]
    print(f"  resilience index         : p05={rs['p5']:.3f}  p50={rs['p50']:.3f}  p95={rs['p95']:.3f}")
    print(f"  resilience CVaR95        : {agg['resilience_cvar95']:.3f}")
    print(f"  P(instability)           : {agg['p_instability']:.2%}")
    print(f"  P(service level < 0.60)  : {agg['p_service_breach']:.2%}")
    print(f"  mean demurrage exposure  : USD {agg['mean_total_cost_usd']:,.0f}")
    print()
    if not result.tornado.empty:
        print("  sensitivity (Spearman rho vs resilience index)")
        for _, row in result.tornado.iterrows():
            bar = "#" * int(abs(row["rho_resilience"]) * 30)
            sign = "+" if row["rho_resilience"] >= 0 else "-"
            print(f"    {row['label']:<22} {sign}{abs(row['rho_resilience']):.3f} {bar}")
    print()
    print(f"  artifacts written to {out_dir}")
    _ = pd  # keep the import meaningful for downstream users editing this script
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
