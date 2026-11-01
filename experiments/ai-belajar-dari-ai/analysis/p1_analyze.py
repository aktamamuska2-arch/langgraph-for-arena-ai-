#!/usr/bin/env python3
"""Analisis R8 untuk P1 — dijalankan setelah semua rantai selesai, dengan label buta.

    python3 analysis/p1_analyze.py --check      # periksa kelengkapan masukan (tidak menghitung)
    python3 analysis/p1_analyze.py --fixture    # uji mesin statistik pada data P0 -> /tmp
    python3 analysis/p1_analyze.py --run        # analisis sebenarnya -> analysis/p1_conclusions.md

Aturan yang diterapkan berasal dari PREREGISTRATION.md v1.2 (§5, §6), bukan dari selera:
  - metrik primer anchor_bpb; uji konfirmatori di generasi terakhir, m = 3, kuantil-t df = n0-1
  - pemantauan berurutan m = 90 (normal), hanya untuk keputusan berhenti, bukan klaim
  - enam sinyal cedera dalam empat kelompok; keparahan dihitung per kelompok
  - aturan sadar-lantai: metrik di lantai/uninformative DIKELUARKAN, bukan dianggap "tidak cedera"
  - hasil tiga-nilai + uji ekuivalensi agar "tidak terdeteksi" tidak menjadi "aman"

Berkas ini tidak pernah membuka berkas kunci blinding. Ia memeriksa kode sumbernya sendiri dan
menolak berjalan bila menemukan rujukan ke berkas kunci.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.stats_r8 import t_ppf  # noqa: E402

P1_METRICS = ROOT / "runs" / "p1" / "metrics.jsonl"
P0_METRICS = ROOT / "runs" / "p0" / "metrics.jsonl"
BLIND_COMMIT = ROOT / "ops" / "blind_commitment.json"
OUT_MD = ROOT / "analysis" / "p1_conclusions.md"

ALPHA = 0.05
FLOOR_BPB = 0.03
M_CONFIRM = 3
M_MONITOR = 90
MIN_ARITH = 40
MIN_DOCS = 100

# kelompok cedera: nama -> daftar sinyal (PREREG §6)
GROUPS = {
    "G1_struktur": ["F1_degenerate", "F2_truncated"],
    "G2_keragaman": ["F3_ttr", "F4_near_dup"],
    "G3_distribusi": ["F5_entropi"],
    "G4_kapabilitas": ["F6_arith"],
}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ------------------------------------------------------------------ pemeriksaan
def self_check() -> None:
    """Pastikan kode ini tidak pernah menyentuh berkas kunci blinding."""
    needle = ".blind" + "/" + "key" + ".json"
    src = inspect.getsource(sys.modules[__name__])
    code = re.sub(r'""".*?"""', "", src, flags=re.S)      # buang docstring
    code = "\n".join(line.split("#")[0] for line in code.splitlines())  # buang komentar
    if needle in code:
        raise SystemExit("DITOLAK: berkas analisis merujuk berkas kunci buta. "
                         "Analisis harus berjalan tanpa akses ke pemetaan arm.")


def check_inputs(path: Path) -> dict:
    rows = load_rows(path)
    base = [r for r in rows if str(r.get("arm")) in ("baseline", "M0", "base")]
    chains = [r for r in rows if r not in base]
    labels = sorted({str(r.get("arm")) for r in chains})
    need_eval = ["anchor_bpb", "arith_acc", "arith_n", "degenerate_rate", "truncated_rate",
                 "ttr", "near_dup_rate", "answer_entropy"]
    missing = {}
    for key in need_eval:
        absent = [r["model"] for r in rows if key not in (r.get("eval") or {})]
        if absent:
            missing[key] = len(absent)
    arith_n = min([(r.get("eval") or {}).get("arith_n", 0) for r in rows if r.get("eval")] or [0])
    return {
        "baris_total": len(rows),
        "baseline": len(base),
        "rantai": len(chains),
        "label_arm": labels,
        "label_buta": all(lbl.startswith("X") for lbl in labels) if labels else False,
        "field_hilang": missing,
        "arith_n_min": arith_n,
        "komitmen_buta": json.loads(BLIND_COMMIT.read_text()) if BLIND_COMMIT.exists() else None,
        "cukup_untuk_R8": (
            len(base) >= 5 and not missing and arith_n >= MIN_ARITH
        ),
    }


# ------------------------------------------------------------------ statistik
def threshold_confirm(sigma0: float, n0: int) -> float:
    q = t_ppf(1 - ALPHA / (2 * M_CONFIRM), n0 - 1)
    return max(q * sigma0 * math.sqrt(1 + 1 / n0), FLOOR_BPB)


def threshold_monitor(sigma0: float, n0: int) -> float:
    from statistics import NormalDist
    q = NormalDist().inv_cdf(1 - ALPHA / (2 * M_MONITOR))
    return max(q * sigma0 * math.sqrt(1 + 1 / n0), FLOOR_BPB)


def mde80(sigma0: float, n0: int) -> float:
    q = t_ppf(1 - ALPHA / (2 * M_CONFIRM), n0 - 1)
    return (q + 0.8416212335729143) * sigma0 * math.sqrt(1 + 1 / n0)


def baseline_stats(base: list[dict]) -> dict:
    bpbs = [r["eval"]["anchor_bpb"] for r in base]
    mu0 = statistics.mean(bpbs)
    sigma0 = statistics.stdev(bpbs) if len(bpbs) > 1 else 0.0
    metrics: dict[str, dict] = {}
    for key in ("degenerate_rate", "truncated_rate", "ttr", "near_dup_rate", "answer_entropy", "arith_acc"):
        vals = [r["eval"].get(key) for r in base if r["eval"].get(key) is not None]
        if not vals:
            metrics[key] = {"tersedia": False}
            continue
        m = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        n_items = (base[0]["eval"].get("arith_n") if key == "arith_acc" else MIN_DOCS) or MIN_DOCS
        at_floor = key == "arith_acc" and m <= 2 / max(1, n_items)
        uninformative = sd >= 0.5 * abs(m) if m else True
        metrics[key] = {
            "tersedia": True, "mean": round(m, 5), "sd": round(sd, 5),
            "label": "at_floor" if at_floor else ("uninformative" if uninformative else "informative"),
        }
    return {
        "n0": len(base), "mu0": mu0, "sigma0": sigma0,
        "se_pred": sigma0 * math.sqrt(1 + 1 / len(base)),
        "thr_confirm": threshold_confirm(sigma0, len(base)),
        "thr_monitor": threshold_monitor(sigma0, len(base)),
        "mde80": mde80(sigma0, len(base)),
        "delta_eq": max(FLOOR_BPB, 0.05 * mu0),
        "metrik": metrics,
    }


def harm_signals(row: dict, base: dict) -> dict[str, bool]:
    ev = row.get("eval", {})
    bm = base["metrik"]
    f1 = f2 = f3 = f4 = f5 = f6 = False
    if bm.get("degenerate_rate", {}).get("label") == "informative":
        f1 = (ev.get("degenerate_rate") or 0) - bm["degenerate_rate"]["mean"] >= 0.15
    if bm.get("truncated_rate", {}).get("label") == "informative":
        f2 = (ev.get("truncated_rate") or 0) - bm["truncated_rate"]["mean"] >= 0.20
    if bm.get("ttr", {}).get("label") == "informative":
        ref = bm["ttr"]["mean"]
        val = ev.get("ttr") or ref
        f3 = (val <= ref * 0.80) or (val <= ref - 3 * bm["ttr"]["sd"])
    if bm.get("near_dup_rate", {}).get("label") == "informative":
        f4 = (ev.get("near_dup_rate") or 0) - bm["near_dup_rate"]["mean"] >= 0.05
    if bm.get("answer_entropy", {}).get("label") == "informative":
        ref = bm["answer_entropy"]["mean"]
        f5 = (ev.get("answer_entropy") or ref) <= ref * 0.70
    if bm.get("arith_acc", {}).get("label") == "informative":
        f6 = bm["arith_acc"]["mean"] - (ev.get("arith_acc") or 0) >= 0.10
    signals = {"F1_degenerate": f1, "F2_truncated": f2, "F3_ttr": f3,
               "F4_near_dup": f4, "F5_entropi": f5, "F6_arith": f6}
    groups = {g: any(signals[s] for s in sigs if s in signals) for g, sigs in GROUPS.items()}
    SIG_TO_METRIC = {
        "F1_degenerate": "degenerate_rate", "F2_truncated": "truncated_rate",
        "F3_ttr": "ttr", "F4_near_dup": "near_dup_rate",
        "F5_entropi": "answer_entropy", "F6_arith": "arith_acc",
    }
    informative_groups = {
        g: any((bm.get(SIG_TO_METRIC[sig]) or {}).get("label") == "informative" for sig in sigs)
        for g, sigs in GROUPS.items()
    }
    return {"sinyal": signals, "kelompok": groups,
            "kelompok_informatif": informative_groups,
            "jumlah_informatif": sum(1 for v in informative_groups.values() if v)}


def decide(row: dict, base: dict) -> dict:
    bpb = row["eval"]["anchor_bpb"]
    delta = bpb - base["mu0"]
    thr = base["thr_confirm"]
    se = base["se_pred"]
    d_eq = base["delta_eq"]
    tq = t_ppf(0.95, base["n0"] - 1)
    upper = delta + tq * se           # CI satu-sisi 95% untuk Delta
    harm = harm_signals(row, base)
    groups_hit = sum(1 for v in harm["kelompok"].values() if v)

    if delta >= 2 * thr:
        need = 1
    elif delta >= thr:
        need = 2
    else:
        need = None

    if need is not None and harm["jumlah_informatif"] < 2:
        label = "inconclusive-harm"
    elif need is not None and groups_hit >= need:
        label = "collapse"
    elif upper < d_eq:
        label = "stabil"
    elif delta >= thr:
        label = "watch"
    else:
        label = "inconclusive"
    return {
        "bpb": bpb, "delta": delta, "thr": thr, "ci_upper": upper, "delta_eq": d_eq,
        "label": label, "kelompok_menyala": groups_hit, "kelompok_informatif": harm["jumlah_informatif"],
        "sinyal": harm["sinyal"], "perlu_kelompok": need,
    }


# ------------------------------------------------------------------ keluaran
def analyse(rows: list[dict], title: str, source: str) -> str:
    base_rows = [r for r in rows if str(r.get("arm")) in ("baseline", "M0", "base")]
    chains = [r for r in rows if r not in base_rows]
    base = baseline_stats(base_rows)
    by_label: dict[str, list[dict]] = {}
    for r in chains:
        by_label.setdefault(str(r["arm"]), []).append(r)

    L = [f"# {title}", "",
         f"Sumber data: `{source}` · {len(base_rows)} model baseline, {len(chains)} model rantai, "
         f"{len(by_label)} label arm.", "",
         "> Hasil ini dihitung oleh `analysis/p1_analyze.py` dari metrik mentah. Label arm masih buta; "
         "pemetaan tidak dibuka oleh skrip ini.", "",
         "## Baseline", "",
         f"- n0 = {base['n0']} · mu0 = {base['mu0']:.4f} bpb · sigma0 = {base['sigma0']:.4f} · "
         f"SE_pred = {base['se_pred']:.4f}",
         f"- ambang konfirmatori (t, m=3) = **{base['thr_confirm']:.4f} bpb** · "
         f"ambang pemantauan (z, m=90) = {base['thr_monitor']:.4f} · MDE80 = {base['mde80']:.4f}",
         f"- delta_eq = {base['delta_eq']:.4f} bpb", "",
         "| Metrik sekunder | mean baseline | sd | label |", "| --- | --- | --- | --- |"]
    for k, v in base["metrik"].items():
        if v.get("tersedia"):
            L.append(f"| `{k}` | {v['mean']:.4f} | {v['sd']:.4f} | `{v['label']}` |")
        else:
            L.append(f"| `{k}` | – | – | `tidak tersedia` |")
    L += ["", "## Keputusan per arm (generasi terakhir tiap rantai)", "",
          "| label | n | bpb akhir | Delta | CI95 atas | label keputusan | kelompok cedera |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for lbl, rs in sorted(by_label.items()):
        last = max(rs, key=lambda r: r.get("gen") or 0)
        d = decide(last, base)
        L.append(f"| {lbl} | {len(rs)} | {d['bpb']:.4f} | {d['delta']:+.4f} | {d['ci_upper']:+.4f} | "
                 f"**{d['label']}** | {d['kelompok_menyala']}/{d['kelompok_informatif']} |")
    L += ["", "## Pemantauan berurutan (screening, bukan klaim)", "",
          "| label | gen | Delta | ambang pantau | lewat? |", "| --- | --- | --- | --- | --- |"]
    for lbl, rs in sorted(by_label.items()):
        for r in sorted(rs, key=lambda x: x.get("gen") or 0):
            delta = r["eval"]["anchor_bpb"] - base["mu0"]
            L.append(f"| {lbl} | {r.get('gen')} | {delta:+.4f} | {base['thr_monitor']:.4f} | "
                     f"{'ya' if delta >= base['thr_monitor'] else 'tidak'} |")
    L += ["", "## Batas tafsir", "",
          "- `inconclusive` berarti data tidak memisahkan collapse dari stabil; itu bukan hasil negatif.",
          "- `watch` berarti ambang terlewati tanpa kelompok cedera terukur; itu bukan collapse.",
          "- Metrik berlabel `at_floor`/`uninformative` dikeluarkan dari corroboration dan tidak "
          "boleh dibaca sebagai ketiadaan cedera (PREREG §6.1).",
          "- Tren antar-generasi bersifat deskriptif; P1 tidak berdaya untuk uji tren inferensial."]
    return "\n".join(L) + "\n"


def main() -> int:
    self_check()
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--fixture", action="store_true")
    g.add_argument("--run", action="store_true")
    args = ap.parse_args()

    if args.check:
        info = check_inputs(P1_METRICS)
        print(json.dumps(info, indent=2, ensure_ascii=False))
        ok = info["cukup_untuk_R8"] and info["label_buta"]
        print("\nSIAP" if ok else "\nBELUM SIAP")
        return 0 if ok else 1

    if args.fixture:
        rows = load_rows(P0_METRICS)
        if not rows:
            print("GAGAL: runs/p0/metrics.jsonl tidak ada — jalankan pilot P0 dulu")
            return 1
        # fixture: data P0 tidak punya semua metrik R8; yang diuji adalah mesin statistiknya
        text = analyse(rows, "UJI UNIT mesin analisis pada data P0 (bukan hasil P1)",
                       "runs/p0/metrics.jsonl")
        out = Path("/tmp/p1_analyze_fixture.md")
        out.write_text(text, encoding="utf-8")
        print(text[:1200])
        print(f"\nOK (fixture) -> {out}  sha256={sha256_file(out)[:16]}…")
        return 0

    rows = load_rows(P1_METRICS)
    if not rows:
        print("GAGAL: runs/p1/metrics.jsonl belum ada. P1 belum dijalankan.")
        return 1
    text = analyse(rows, "Kesimpulan P1 (label buta)", "runs/p1/metrics.jsonl")
    OUT_MD.write_text(text, encoding="utf-8")
    digest = sha256_file(OUT_MD)
    print(f"OK -> {OUT_MD}")
    print(f"sha256 kesimpulan = {digest}")
    print("Langkah berikutnya (PREREG §8): ops/blind.py reveal --conclusions-sha256 " + digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
