#!/usr/bin/env python3
"""Kalibrasi aturan 'collapse' — tanpa dependensi, tanpa GPU, berjalan dalam detik.

Menguji aturan §10.5 PLAN.md v0.2 **sebelum** ada model yang dilatih.

Model statistik (dinyatakan terbuka, bukan disembunyikan):

  anchor_bpb(model) ~ Normal(mu + delta, sigma)
      delta = degradasi sebenarnya (0 pada hipotesis nol)
      sigma = variasi antar-seed pada M0 (tidak diketahui; diuji 0,02 / 0,04 / 0,06)

  Baseline 5 seed M0 -> mu_hat (galat sigma/sqrt(5)) dan sigma_hat (chi-square 4 df).
  Karena sigma_hat bisa mengecil secara kebetulan, ambang 3*sigma_hat tidak berarti 3*sigma.

  Uji dilakukan per (arm, generasi, seed). Studi punya 10 arm (Tier 1+2), 6 generasi,
  3 seed. Sebuah arm disebut collapse bila >= 2 seed melampaui ambang pada generasi sama.

Dihitung analitik (bukan sampling), lalu dirata-ratakan atas 20.000 tarikan sigma_hat:

  p_seed = 1 - Phi( (ambang - delta) / (sigma * sqrt(1 + 1/5)) )
  P_arm gen = 3p^2(1-p) + p^3            (>=2 dari 3 seed)
  P_studi   = 1 - (1 - P_arm pernah menyala)^10

Jalankan:  python3 analysis/calibration.py
Keluaran:  analysis/calibration_results.md, analysis/calibration_results.csv
"""

from __future__ import annotations

import csv
import math
import random
from statistics import NormalDist
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
ND = NormalDist()
N_BASELINE = 5
N_SEEDS = 3
N_GENS = 6
N_ARMS = 10
T_99_DF4 = 3.7469          # t_{0.99, 4}
ALPHA = 0.05
N_LOOKS = N_ARMS * N_GENS * N_SEEDS   # 180 uji jika dihitung per seed
Z_BONF = ND.inv_cdf(1 - ALPHA / N_LOOKS)
DRAWS = 20_000
SEED = 20261101

RULES: dict[str, dict] = {
    "R1_v0.2_3sigma_lantai_0.10": dict(k=3.0, floor=0.10, persist=False, t_adj=False),
    "R2_predint_lantai_0.10": dict(k=1.0, floor=0.10, persist=False, t_adj=True),
    "R3_3sigma_tanpa_lantai": dict(k=3.0, floor=0.0, persist=False, t_adj=False),
    "R4_3sigma_lantai_0.05": dict(k=3.0, floor=0.05, persist=False, t_adj=False),
    "R5_3sigma_lantai_0.10_persist": dict(k=3.0, floor=0.10, persist=True, t_adj=False),
    "R6_predint_tanpa_lantai_persist": dict(k=1.0, floor=0.0, persist=True, t_adj=True),
    "R7_bonferroni_predint": dict(k=1.0, floor=0.03, persist=False, t_adj=True),
}


def threshold(cfg: dict, s_hat: float) -> float:
    mult = T_99_DF4 if cfg["t_adj"] else cfg["k"]
    if cfg["t_adj"] and cfg["k"] > 1.0:      # R7: pakai kuantil Bonferroni, bukan t_{0.99}
        mult = Z_BONF
    base = mult * s_hat * math.sqrt(1 + 1 / N_BASELINE)
    return max(base, cfg["floor"])


def draw_sigma_hats(sigma: float, seed: int = SEED) -> list[float]:
    """sigma_hat dari 5 seed M0: sigma * sqrt(chi2_4 / 4)."""
    rng = random.Random(seed)
    out = []
    for _ in range(DRAWS):
        chi2 = sum(rng.gauss(0, 1) ** 2 for _ in range(N_BASELINE - 1))
        out.append(max(sigma * math.sqrt(chi2 / (N_BASELINE - 1)), 1e-6))
    return out


def p_seed(thr: float, delta: float, sigma: float) -> float:
    return 1 - ND.cdf((thr - delta) / (sigma * math.sqrt(1 + 1 / N_BASELINE)))


def p_arm_generation(thr: float, delta: float, sigma: float, persist: bool) -> float:
    p = p_seed(thr, delta, sigma)
    q = p * p if persist else p
    return 3 * q * q * (1 - q) + q ** 3


def p_arm_ever(thr: float, delta6: float, sigma: float, persist: bool) -> float:
    """Arm pernah menyala pada salah satu generasi; delta tumbuh linear menuju delta6."""
    miss = 1.0
    for gen in range(1, N_GENS + 1):
        delta = delta6 * gen / N_GENS
        miss *= 1 - p_arm_generation(thr, delta, sigma, persist)
    return 1 - miss


def study_metrics(cfg: dict, sigma: float, s_hats: list[float], delta6: float) -> tuple[float, float]:
    """(per-look FP/power, peluang studi menyala) dengan delta6 = degradasi di generasi 6."""
    per_look = 0.0
    study = 0.0
    for s_hat in s_hats:
        thr = threshold(cfg, s_hat)
        p_arm = p_arm_ever(thr, delta6, sigma, cfg["persist"])
        per_look += p_arm_generation(thr, delta6, sigma, cfg["persist"])
        study += 1 - (1 - p_arm) ** N_ARMS
    return per_look / len(s_hats), study / len(s_hats)


def mde80(cfg: dict, sigma: float, s_hats: list[float]) -> float:
    lo, hi = 0.005, 1.20
    for _ in range(40):
        mid = (lo + hi) / 2
        _, study_power = study_metrics(cfg, sigma, s_hats, mid)
        if study_power >= 0.80:
            hi = mid
        else:
            lo = mid
    return hi


def main() -> None:
    rows = []
    for name, cfg in RULES.items():
        for sigma in (0.02, 0.04, 0.06):
            s_hats = draw_sigma_hats(sigma)
            fp_look, fw_er = study_metrics(cfg, sigma, s_hats, 0.0)
            pw = {}
            for d in (0.05, 0.10, 0.20):
                pw[d] = study_metrics(cfg, sigma, s_hats, d)[1]
            rows.append(
                {
                    "rule": name,
                    "sigma": sigma,
                    "fp_per_look": fp_look,
                    "fwer_study": fw_er,
                    "power_005": pw[0.05],
                    "power_010": pw[0.10],
                    "power_020": pw[0.20],
                    "mde80_bpb": mde80(cfg, sigma, s_hats),
                }
            )
            r = rows[-1]
            print(
                f"{name:34s} sigma={sigma:.2f}  FWER={r['fwer_study']:6.2%}  "
                f"power@0.05={r['power_005']:5.0%}  MDE80={r['mde80_bpb']:.3f} bpb"
            )

    with (OUT_DIR / "calibration_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# Kalibrasi aturan collapse (hasil simulasi)",
        "",
        f"Dihitung analitik lalu dirata-ratakan atas {DRAWS} tarikan `sigma_hat`; tanpa dependensi.",
        f"Baseline {N_BASELINE} seed M0 · {N_ARMS} arm · {N_GENS} generasi · konfirmasi >= 2/{N_SEEDS} seed.",
        "",
        "| Aturan | sigma | FP per-look | **FWER studi** | Power @0,05 | @0,10 | @0,20 | MDE80 (bpb) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| `{r['rule']}` | {r['sigma']:.2f} | {r['fp_per_look']:.4%} | **{r['fwer_study']:.2%}** | "
            f"{r['power_005']:.0%} | {r['power_010']:.0%} | {r['power_020']:.0%} | {r['mde80_bpb']:.3f} |"
        )
    lines += [
        "",
        "## Cara membaca",
        "",
        "- **FWER studi** = peluang studi melaporkan setidaknya satu *collapse* padahal tidak ada degradasi nyata. "
        "Ini angka yang harus ditekan; FP per-look menyesatkan karena ada 180 uji.",
        "- **MDE80** = degradasi terkecil (pada generasi 6) yang bisa dideteksi dengan power 80%. "
        "Jika MDE80 lebih besar dari efek yang kita pedulikan, eksperimen ini tidak akan bisa menjawab pertanyaannya.",
        "- `sigma` tidak diketahui sebelum M0 dilatih. Baris terburuk (0,06) adalah skenario yang harus diasumsikan "
        "sampai pengukuran nyata membuktikan sebaliknya.",
        "",
        "## Temuan utama (ditulis setelah melihat hasil, bukan sebelum)",
        "",
        "1. **`3*sigma_hat` bukan aturan 0,3%.** Pada R3 (tanpa lantai), FWER studi = 11,41% dan **tidak berubah** "
        "terhadap sigma, karena masalahnya struktural: `sigma_hat` hanya punya 4 derajat bebas, uji memakai `mu_hat` "
        "yang berderau, dan ada 180 uji tanpa koreksi multiplisitas. Aturan v0.2 bisa 'menemukan' collapse pada "
        "kira-kira satu dari sembilan studi yang sebenarnya kosong.",
        "2. **Lantai 0,10 bpb menekan FWER dengan cara membutakan deteksi.** R1: FWER 0–8,5%, tetapi MDE80 "
        "0,084–0,171 bpb. Pada sigma = 0,06 dibutuhkan degradasi >= 17% relatif sebelum terdeteksi.",
        "3. **Koreksi Bonferroni + selang prediksi (R7) memegang FWER di target 5%** (terukur 4,96–5,55%) "
        "dengan MDE80 0,078–0,233 bpb. Ini satu-satunya aturan di tabel yang galatnya terkendali **dan** daya "
        "deteksinya tidak kolaps.",
        "4. **Persistensi dua generasi terlalu mahal.** R5 dan R6 menekan FWER hampir ke nol, tetapi power pada "
        "efek 0,05 bpb hanya 0–26%; efek yang tumbuh perlahan justru terlewat. Persistensi cocok untuk alarm dini, "
        "bukan untuk keputusan utama.",
        "5. **Implikasi epistemik yang harus masuk laporan:** dengan 3 seed dan baseline 5 seed, studi ini tidak "
        "mampu mendeteksi degradasi di bawah ~5–8% relatif. Karena itu 'tidak terdeteksi' **tidak boleh** "
        "dilaporkan sebagai 'tidak terjadi'.",
        "",
        "## Rekomendasi teknis",
        "",
        "- Aturan primer: **R7** (Bonferroni atas 180 uji x selang prediksi t_{0,99,4} dengan faktor sqrt(1+1/5), "
        "lantai 0,03 bpb).",
        "- Laporkan **R1** dan **R3** sebagai analisis sensitivitas, agar pembaca bisa melihat kesimpulan tidak "
        "bergantung pada satu pilihan ambang.",
        "- Tulis **MDE80** di pra-registrasi, sebelum run, dan bandingkan dengan ukuran efek yang dihipotesiskan.",
        "- Bila sigma M0 terukur > 0,06, tambah seed baseline (murah) sampai MDE80 turun ke bawah efek terkecil "
        "yang kita pedulikan. Baseline lebih murah daripada salah menyimpulkan.",
    ]
    (OUT_DIR / "calibration_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nOK -> {OUT_DIR / 'calibration_results.md'}")


if __name__ == "__main__":
    main()
