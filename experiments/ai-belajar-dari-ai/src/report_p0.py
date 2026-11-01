#!/usr/bin/env python3
"""Bangun REPORT-P0.md + analysis/p0_metrics.csv dari runs/p0/metrics.jsonl."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "runs" / "p0" / "metrics.jsonl"
OUT_MD = ROOT / "REPORT-P0.md"
OUT_CSV = ROOT / "analysis" / "p0_metrics.csv"
DONE = ROOT / "runs" / "p0" / "done.json"


def main() -> None:
    rows = [json.loads(l) for l in METRICS.read_text(encoding="utf-8").splitlines() if l.strip()]
    base = [r for r in rows if r["arm"] == "baseline"]
    chains = [r for r in rows if r["arm"] != "baseline"]
    mu0 = mean(r["eval"]["anchor_bpb"] for r in base)
    sigma0 = stdev([r["eval"]["anchor_bpb"] for r in base])
    base_d2 = mean(r["eval"]["distinct2"] for r in base)
    base_arith = mean(r["eval"]["arith_acc"] for r in base)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    keys = ["model", "arm", "seed", "gen", "anchor_bpb", "arith_acc", "distinct2",
            "repetition_rate", "style_div", "degenerate_rate", "delta_bpb", "status"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            e = r["eval"]
            g = r.get("gate", {})
            w.writerow({
                "model": r["model"], "arm": r["arm"], "seed": r["seed"], "gen": r["gen"],
                "anchor_bpb": e["anchor_bpb"], "arith_acc": e["arith_acc"], "distinct2": e["distinct2"],
                "repetition_rate": e["repetition_rate"], "style_div": e["style_div"],
                "degenerate_rate": e["degenerate_rate"], "delta_bpb": g.get("delta_bpb"),
                "status": g.get("status"),
            })

    lines = [
        "# Laporan Pilot P0",
        "",
        "**Status: bukti mekanisme, bukan temuan ilmiah.**",
        "",
        "P0 dijalankan di sandbox (2 vCPU, 3 GB RAM, tanpa GPU) untuk membuktikan bahwa rantai",
        "guru → T0 → M0 → generasi → validasi → training → evaluasi → gate berjalan end-to-end,",
        "dan untuk mendapatkan `sigma0` pertama dari variasi antar-seed.",
        "",
        f"- Model: XS, {rows[0]['train']['n_params']:,} parameter (byte-level, vocab 256, konteks 96)".replace(",", "."),
        f"- Baseline M0: {len(base)} seed -> mu0 = {mu0:.4f} bpb, sigma0 = {sigma0:.4f} bpb",
        f"- Chain: {sorted({(r['arm'], r['seed']) for r in chains})}",
        f"- Langkah training per model: {rows[0]['train']['steps']} (anggaran token identik)",
        "",
        "## Baseline M0",
        "",
        "| seed | anchor_bpb | arith_acc | distinct2 | style_div |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in sorted(base, key=lambda x: x["seed"]):
        e = r["eval"]
        lines.append(f"| {r['seed']} | {e['anchor_bpb']:.4f} | {e['arith_acc']:.2f} | {e['distinct2']:.3f} | {e['style_div']:.3f} |")
    lines += [
        "",
        f"mu0 = **{mu0:.4f}** bpb, sigma0 = **{sigma0:.4f}** bpb. Ambang R7 dihitung dari sigma0",
        "dengan koreksi Bonferroni 180 uji dan selang prediksi (PLAN v0.3 §10.5).",
        "",
        "## Rantai per arm",
        "",
        "| arm | seed | gen | anchor_bpb | delta | gate | distinct2 | arith | rep | style_div | reject | mix_tvd |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(chains, key=lambda x: (x["arm"], x["seed"], x["gen"])):
        e, g = r["eval"], r.get("gate", {})
        v = r.get("validate", {})
        lines.append(
            f"| {r['arm']} | {r['seed']} | {r['gen']} | {e['anchor_bpb']:.4f} | "
            f"{g.get('delta_bpb', 0):+.4f} | {g.get('status', '?')} | {e['distinct2']:.3f} | "
            f"{e['arith_acc']:.2f} | {e['repetition_rate']:.2f} | {e['style_div']:.3f} | "
            f"{v.get('reject_rate', 0):.0%} | {r.get('mix', {}).get('mix_tvd', 0) or 0:.3f} |"
        )

    # ---------- agregat per arm x generasi ----------
    def agg(arm: str, gen: int) -> dict:
        sel = [r for r in chains if r["arm"] == arm and r["gen"] == gen]
        if not sel:
            return {}
        trunc = sum(r["validate"]["truncated_no_end_marker"] for r in sel)
        total = sum(r["validate"]["total"] for r in sel)
        return {
            "bpb": mean(r["eval"]["anchor_bpb"] for r in sel),
            "delta": mean(r["gate"]["delta_bpb"] for r in sel),
            "d2": mean(r["eval"]["distinct2"] for r in sel),
            "style": mean(r["eval"]["style_div"] for r in sel),
            "degen": mean(r["eval"]["degenerate_rate"] for r in sel),
            "trainloss": mean(r["train"]["final_loss"] for r in sel),
            "trunc": trunc / max(1, total),
            "n": len(sel),
        }

    A = {g: agg("A", g) for g in (1, 2, 3)}
    B1 = {g: agg("B1", g) for g in (1, 2, 3)}
    thr = chains[0]["gate"]["threshold_bpb"]
    harm_fired = [r for r in chains if r["gate"]["harm_signals"]]
    statuses = {}
    for r in chains:
        statuses[r["gate"]["status"]] = statuses.get(r["gate"]["status"], 0) + 1
    a13 = {r["gen"]: r for r in chains if r["arm"] == "A" and r["seed"] == 13}
    a13_loss = " -> ".join(f"{a13[g]['train']['final_loss']:.2f}" for g in (1, 2, 3))
    a13_bpb = " -> ".join(f"{a13[g]['eval']['anchor_bpb']:.2f}" for g in (1, 2, 3))

    lines += [
        "",
        "## Agregat per arm",
        "",
        "| arm | gen | n | bpb | delta | train loss | distinct2 | style_div | degenerate | truncated |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, table in (("A", A), ("B1", B1)):
        for g in (1, 2, 3):
            a = table.get(g)
            if not a:
                continue
            lines.append(
                f"| {name} | {g} | {a['n']} | {a['bpb']:.4f} | {a['delta']:+.4f} | {a['trainloss']:.3f} | "
                f"{a['d2']:.3f} | {a['style']:.3f} | {a['degen']:.2f} | {a['trunc']:.0%} |"
            )

    lines += [
        "",
        "## Temuan P0",
        "",
        "### 1. Pipeline berjalan end-to-end, dan itu satu-satunya klaim kuat di sini",
        "",
        f"17 node selesai tanpa intervensi manual, 16 model dilatih, 16 checkpoint tersimpan,",
        f"state machine dapat di-resume (state.json + done.json), dan seluruh metrik tercatat di",
        f"`runs/p0/metrics.jsonl`. Wall-clock 22 menit 23 detik pada 2 vCPU/3 GB tanpa GPU.",
        "",
        "### 2. sigma0 pertama sudah ada — dan ambang R7 sebanding dengan efek yang kita cari",
        "",
        f"mu0 = {mu0:.4f} bpb, sigma0 = {sigma0:.4f} bpb (CV {sigma0 / mu0:.1%}), ambang R7 = {thr:.4f} bpb.",
        "Artinya ambang praktis 10% di atas baseline. Dengan korpus yang sangat bertemplat,",
        "baseline-nya sendiri rendah (bpb 0,48), sehingga ruang degradasi yang bisa diukur lebar.",
        "",
        "### 3. Degradasi arm A jauh melampaui ambang, dan konsisten di kedua seed",
        "",
        f"- A generasi 1: bpb {A[1]['bpb']:.4f} (delta {A[1]['delta']:+.4f} = {A[1]['delta'] / thr:.1f}x ambang)",
        f"- A generasi 2: bpb {A[2]['bpb']:.4f} (delta {A[2]['delta']:+.4f} = {A[2]['delta'] / thr:.1f}x ambang)",
        f"- A generasi 3: bpb {A[3]['bpb']:.4f} (delta {A[3]['delta']:+.4f} = {A[3]['delta'] / thr:.1f}x ambang)",
        f"- B1 generasi 1: bpb {B1[1]['bpb']:.4f} (delta {B1[1]['delta']:+.4f} = {B1[1]['delta'] / thr:.1f}x ambang)",
        f"- B1 generasi 3: bpb {B1[3]['bpb']:.4f} (delta {B1[3]['delta']:+.4f} = {B1[3]['delta'] / thr:.1f}x ambang)",
        "",
        "Semua titik sudah melewati ambang, tetapi tidak satu pun dinyatakan `collapse` oleh gate.",
        "Itu bukan kosmetik: ambang saja tidak cukup, dan aturan corroboration saya rusak (lihat #4).",
        "",
        f"Rasio A/B1 pada generasi 3 = {A[3]['delta'] / B1[3]['delta']:.2f}x. Arah efeknya sesuai literatur",
        "(akumulasi memperlambat, bukan menghilangkan), tetapi P0 tidak punya daya untuk menguji itu:",
        "2 seed, 3 generasi, satu keluarga korpus.",
        "",
        "### 3b. Ukuran sampel yang kecil membatasi tafsir",
        "",
        "Evaluasi per model memakai 24 prompt generasi dan 12 item aritmetika. `degenerate_rate`",
        "karena itu bergerak dalam langkah 1/24 (~4,2pp) dan tidak boleh dibaca sebagai tren; `arith_acc`",
        "bergerak dalam langkah 1/12 (~8,3pp), sehingga aturan 'turun 10pp' hanya butuh dua item berbeda.",
        "",
        "",
        "### 4. Gate saya sendiri gagal menandai collapse — ini temuan metodologis yang paling penting",
        "",
        f"Status gate: {statuses.get('watch', 0)} watch, {statuses.get('collapse', 0)} collapse, {statuses.get('ok', 0)} ok",
        f"dari 12 model rantai. **{len(harm_fired)} model memicu sinyal cedera kedua**,",
        f"padahal bpb naik sampai {A[3]['delta']:+.2f}. Penyebabnya struktural:",
        "",
        f"- `arith_acc` baseline sudah 0,00 (3 dari 4 seed) sehingga aturan 'turun >= 10pp' mustahil menyala.",
        f"- `distinct2` justru **naik** saat teks memburuk ({mean(r['eval']['distinct2'] for r in base):.3f} -> {A[3]['d2']:.3f}),",
        "  karena gibberish byte-level punya lebih banyak bigram unik.",
        f"- `style_div` datar ({A[3]['style']:.3f} vs {mean(r['eval']['style_div'] for r in base):.3f}) — TF-IDF char-3gram tidak menangkap kerusakan sintaks.",
        "- `repetition_rate` 0,00 di semua model.",
        "",
        "Konsekuensinya: aturan R7 + cedera-kedua **under-call** dalam rezim ini. Saya tidak mengubah",
        "aturan setelah melihat data (itu tepatnya yang dilarang di PLAN §10.5); sebagai gantinya P0",
        "melahirkan kandidat **R8** yang harus dipraregistrasi sebelum P1:",
        "",
        "1. Sinyal cedera ditambah: kenaikan `degenerate_rate` >= 15pp, `truncated` >= 20pp, dan",
        "   penurunan `format_compliance` (dokumen berakhir `<E>`) >= 20pp.",
        "2. Aturan sadar-lantai: bila metrik sekunder sudah di lantai (baseline 0,00) atau arahnya",
        "   tidak informatif, metrik itu dikeluarkan dari syarat corroboration, bukan dihitung sebagai 'tidak cedera'.",
        "3. Efek ukuran: setiap model P1 mengevaluasi minimal 40 item aritmetika agar `arith_acc` punya granularitas 2,5pp.",
        "",
        "### 5. Model bisa menipu diri: loss turun, fidelitas jatuh",
        "",
        f"Arm A seed 13: loss pada korpus sendiri {a13_loss}, sementara anchor bpb {a13_bpb}.",
        "Model makin lancar meniru keluarannya sendiri dan makin jauh dari distribusi guru. Ini ilustrasi",
        "konkret mengapa evaluasi harus dari data guru yang dibekukan, bukan dari loss training.",
        "",
        "### 6. Format bertahan lebih lama daripada isi",
        "",
        f"Dokumen tanpa penutup `<E>`: arm A {A[1]['trunc']:.0%}/{A[2]['trunc']:.0%}/{A[3]['trunc']:.0%} (gen 1-3), "
        f"B1 {B1[1]['trunc']:.0%}/{B1[2]['trunc']:.0%}/{B1[3]['trunc']:.0%}.",
        "Seluruh dokumen tetap lolos filter lenient (100% punya `<D>`/`</P>`/`<A>`), jadi collapse di sini",
        "bukan kegagalan format total melainkan erosi isi di dalam templat yang masih benar.",
        "",
        "### 7. Yang tidak boleh diklaim dari P0",
        "",
        "- Bukan bukti bahwa model 8 jt parameter akan collapse dengan laju yang sama.",
        "- Bukan bukti bahwa B1 aman: B1 juga sudah di atas ambang sejak generasi 1.",
        "- Bukan estimasi effect size: laju degradasi kemungkinan besar didorong oleh korpus templat",
        "  berukuran 200 KB dan model 0,48 jt parameter, bukan oleh sifat umum self-training.",
        "- Bukan uji statistik: 2 seed per arm, tanpa koreksi, tanpa blinding (D9 baru berlaku di P1).",
        "",
        "## Konsekuensi untuk P1 (belum dijalankan)",
        "",
        "1. Praregistrasi R8 di PLAN sebelum P1, dengan angka di atas sebagai dasar, bukan sebagai hasil.",
        "2. Baseline minimal 5 seed; 3 arm (A, B1, dan kontrol tanpa self-training) x 5 seed x 6 generasi.",
        "3. Korpus guru non-templat minimal 2 MB, sehingga bpb baseline tidak berada di ekor rendah.",
        "4. Evaluasi 40+ item aritmetika dan 200+ prompt generasi per titik, agar metrik tidak di lantai.",
        "5. Orkestrasi dipindah ke LangGraph (saat ini state machine dengan node identik, bukan LangGraph).",
        "",
        "## Cara menjalankan ulang",
        "",
        "```bash",
        "cd experiments/ai-belajar-dari-ai",
        "python -m src.orchestrate          # idempoten: melewati node yang sudah selesai",
        "python -m src.report_p0            # bangun ulang laporan ini + analysis/p0_metrics.csv",
        "make check                         # lint kebijakan + build dashboard",
        "```",
        "",
    ]

    finished = DONE.exists()
    lines += [
        "## Catatan eksekusi",
        "",
        f"- Semua langkah selesai: {'ya' if finished else 'belum'}",
        f"- Total model dilatih: {len(rows)}",
        "- Total byte sintetis dihasilkan: " + f"{sum(r.get('tokens_generated', 0) for r in chains):,}".replace(",", "."),
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"OK {OUT_MD}")
    print(f"OK {OUT_CSV}")


if __name__ == "__main__":
    main()
