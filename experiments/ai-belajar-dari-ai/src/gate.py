"""Gate keputusan: aturan R7 (PLAN v0.3 §10.5) + rute sesuai kebijakan."""
from __future__ import annotations

import math
from statistics import NormalDist, mean, stdev

from . import config as C

Z_BONF = NormalDist().inv_cdf(1 - C.FWER / C.N_LOOKS)


def threshold(sigma_hat: float, n_baseline: int) -> float:
    """Bonferroni (180 uji) x selang prediksi t-approx, dengan lantai absolut."""
    t = 3.7469 if n_baseline <= 5 else 2.776
    return max(Z_BONF * sigma_hat * math.sqrt(1 + 1 / n_baseline), C.FLOOR_BPB)


def evaluate_gate(baseline_bpbs: list[float], current_bpb: float, metrics: dict,
                  baseline_metrics: dict | None = None) -> dict:
    mu = mean(baseline_bpbs)
    sigma = stdev(baseline_bpbs) if len(baseline_bpbs) > 1 else 0.0
    thr = threshold(sigma, len(baseline_bpbs))
    delta = current_bpb - mu

    harm = []
    if baseline_metrics:
        base_d2 = baseline_metrics.get("distinct2")
        if base_d2 and metrics.get("distinct2", base_d2) < base_d2 * 0.75:
            harm.append("distinct2_drop>=25%")
        if metrics.get("arith_acc", 1.0) <= baseline_metrics.get("arith_acc", 0.0) - 0.10:
            harm.append("arith_drop>=10pp")

    if delta >= thr and harm:
        status = "collapse"
    elif delta >= thr:
        status = "watch"          # melewati ambang tapi belum ada cedera kedua
    elif delta >= thr / 2:
        status = "watch"
    else:
        status = "ok"

    route = {"ok": "continue", "watch": "continue+log", "collapse": "stop_arm"}[status]
    return {
        "mu0": round(mu, 5),
        "sigma0": round(sigma, 5),
        "threshold_bpb": round(thr, 5),
        "delta_bpb": round(delta, 5),
        "delta_over_thr": round(delta / thr, 3) if thr else None,
        "harm_signals": harm,
        "status": status,
        "route": route,
        "rule": "R7_bonferroni_predint_floor0.03",
    }
