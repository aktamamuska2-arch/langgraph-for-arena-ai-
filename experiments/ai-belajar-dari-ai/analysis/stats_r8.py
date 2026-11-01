#!/usr/bin/env python3
"""Statistik untuk aturan R8 (pra-registrasi P1).

Murni pustaka standar: distribusi t Student lewat fungsi beta tak-lengkap,
dipakai untuk menetapkan ambang, MDE, dan batas ekuivalensi SEBELUM data P1 ada.

Semua angka yang dipakai PREREGISTRATION.md dihasilkan oleh `--table`,
sehingga dokumen pra-registrasi tidak memuat angka yang diketik tangan.
"""
from __future__ import annotations

import argparse
import math

FLOOR_BPB = 0.03          # lantai kepraktisan (bpb absolut)
ALPHA = 0.05              # family-wise
POWER = 0.80
QUANTILE_80 = 0.8416212335729143  # z_0.80


# ---------------------------------------------------------------- beta & t
def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """I_x(a,b) ternormalisasi."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: int) -> float:
    x = df / (df + t * t)
    p = 0.5 * betai(df / 2.0, 0.5, x)
    return 1.0 - p if t > 0 else p


def t_ppf(p: float, df: int) -> float:
    """Kuantil t Student via bisection (presisi 1e-10)."""
    lo, hi = -1e4, 1e4
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------- aturan R8
def se_predictive(sigma0: float, n0: int) -> float:
    """Galat baku prediktif satu model baru terhadap mean baseline n0 seed."""
    return sigma0 * math.sqrt(1.0 + 1.0 / n0)


def threshold(sigma0: float, n0: int, m: int, alpha: float = ALPHA,
              floor: float = FLOOR_BPB) -> float:
    """Ambang R8: kuantil-t Bonferroni x galat baku prediktif, dengan lantai."""
    q = t_ppf(1.0 - alpha / (2.0 * m), n0 - 1)
    return max(q * se_predictive(sigma0, n0), floor)


def threshold_z(sigma0: float, n0: int, m: int, alpha: float = ALPHA,
                floor: float = FLOOR_BPB) -> float:
    """Versi normal (dipakai untuk pemantauan berurutan, bukan klaim konfirmatori)."""
    from statistics import NormalDist
    q = NormalDist().inv_cdf(1.0 - alpha / (2.0 * m))
    return max(q * se_predictive(sigma0, n0), floor)


def mde80(sigma0: float, n0: int, m: int, alpha: float = ALPHA) -> float:
    q = t_ppf(1.0 - alpha / (2.0 * m), n0 - 1)
    return (q + QUANTILE_80) * se_predictive(sigma0, n0)


def equivalence_bound(mu0: float) -> float:
    """Batas ekuivalensi: 'stabil' hanya bila seluruh CI di bawah batas ini."""
    return max(FLOOR_BPB, 0.05 * mu0)


def floor_binding(sigma0: float, n0: int, m: int) -> bool:
    return t_ppf(1.0 - ALPHA / (2.0 * m), n0 - 1) * se_predictive(sigma0, n0) < FLOOR_BPB


def table() -> str:
    out = []
    out.append("| sigma0 (bpb) | SE_pred (n0=5) | ambang konfirmatori (m=3) | ambang monitoring (m=90) | MDE80 konfirmatori | `delta_eq` bila mu0=0,90 |")
    out.append("| --- | --- | --- | --- | --- | --- |")
    for s in (0.005, 0.010, 0.0126, 0.015, 0.020, 0.030, 0.040, 0.060):
        se = se_predictive(s, 5)
        tc = threshold(s, 5, 3)
        tm = threshold(s, 5, 90)
        md = mde80(s, 5, 3)
        out.append(f"| {s:.4f} | {se:.4f} | {tc:.4f} | {tm:.4f} | {md:.4f} | {equivalence_bound(0.90):.4f} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()
    if args.table:
        print(table())
        return
    print("t_ppf(0.975,4) =", round(t_ppf(0.975, 4), 4), "(harus 2.7764)")
    print("t_ppf(0.95,4)  =", round(t_ppf(0.95, 4), 4), "(harus 2.1318)")
    print("t_ppf(0.999722,4) =", round(t_ppf(1 - 0.05 / (2 * 90), 4), 4))
    print("t_ppf(0.991667,4) =", round(t_ppf(1 - 0.05 / (2 * 3), 4), 4))
    print()
    print(table())


if __name__ == "__main__":
    main()
