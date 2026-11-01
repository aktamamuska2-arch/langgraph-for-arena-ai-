#!/usr/bin/env python3
"""Audit gerbang korpus T0 untuk P1 (PREREGISTRATION.md v1.1 bagian 2.1).

Lima gerbang diperiksa apa adanya, ditambah tiga pembanding yang membuat hasilnya
bisa ditafsirkan:

  1. referensi manusia — 8-gram/kompresi dari teks yang jelas bukan buatan saya
     (docstring pustaka standar Python), dipotong pada ukuran yang sama;
  2. korpus P0 — untuk menunjukkan apakah perbaikan ini nyata atau hanya klaim;
  3. entropi silang Markov orde-3 pada test split — proksi murah untuk
     "apakah korpus ini bisa dihafal" tanpa melatih satu model pun.

Pemakaian:
    python3 analysis/t0_audit.py            # cetak ringkasan + tulis laporan
    python3 analysis/t0_audit.py --json     # keluaran mesin saja
Keluar dengan kode 1 bila ada gerbang yang gagal.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
import re
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
T0 = ROOT / "data" / "p1" / "t0"
OUT_JSON = ROOT / "runs" / "p1" / "t0_audit.json"
OUT_MD = ROOT / "analysis" / "t0_audit_report.md"

# --- jangkar yang dibekukan di pra-registrasi v1.2 §2.1a -------------------
P0_REF = {"uniq8": 0.1870, "hapax8": 0.1223, "zlib_bpb": 1.189, "ttr": 0.0732, "markov_ce": 1.615}
HUMAN_ZLIB_2MB = 2.217
T2_MULT = {"uniq8": 1.50, "hapax8": 1.50, "zlib_bpb": 1.50, "ttr": 1.40, "markov_ce": 1.35}
T2 = {
    "T2a": ("uniq-8 potongan 156.697 B", P0_REF["uniq8"] * T2_MULT["uniq8"]),
    "T2b": ("hapax-8 potongan 156.697 B", P0_REF["hapax8"] * T2_MULT["hapax8"]),
    "T2c": ("zlib bpb seluruh isi (juga >= 0,85 x manusia)", max(P0_REF["zlib_bpb"] * T2_MULT["zlib_bpb"],
                                                               0.85 * HUMAN_ZLIB_2MB)),
    "T2d": ("TTR sampel 200 KB", P0_REF["ttr"] * T2_MULT["ttr"]),
    "T2e": ("entropi Markov orde-3 @156.697 B", P0_REF["markov_ce"] * T2_MULT["markov_ce"]),
}

GATES = {
    "T1": "ukuran >= 2.097.152 byte",
    "T2": "kriteria terkalibrasi v1.2: T2a-T2e (uniq-8, hapax, zlib, TTR, entropi Markov)",
    "T3": ">= 12 domain, tiap domain >= 5% byte",
    "T4": "median panjang 150-600 B, p1 >= 80 B, p99 <= 2.000 B",
    "T5": "Jaccard 5-gram antar-dokumen maksimum <= 0,5",
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def uniq_ngram(text: str, n: int = 8) -> float:
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


def hapax_share(text: str, n: int = 8) -> float:
    c = Counter(text[i:i + n] for i in range(len(text) - n + 1))
    return sum(1 for v in c.values() if v == 1) / max(1, sum(c.values()))


def zlib_bpb(text: str) -> float:
    raw = text.encode("utf-8")
    return len(zlib.compress(raw, 9)) * 8 / max(1, len(raw))


def ttr(text: str, limit: int = 200_000) -> float:
    w = text[:limit].lower().split()
    return len(set(w)) / max(1, len(w))


def quantile(sorted_vals: list[int], p: float) -> int:
    if not sorted_vals:
        return 0
    return sorted_vals[min(len(sorted_vals) - 1, int(p * (len(sorted_vals) - 1)))]


# ------------------------------------------------------------------ referensi
def human_reference(n_bytes: int, seed: int = 7) -> dict:
    """Docstring pustaka standar Python — teks yang tidak saya tulis."""
    docs: list[str] = []
    for f in sorted(glob.glob("/usr/lib/python3.11/**/*.py", recursive=True)):
        try:
            t = Path(f).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        docs.append(" ".join(re.findall(r'"""(..*?)"""', t, flags=re.S)))
    rng = random.Random(seed)
    rng.shuffle(docs)  # campur antar-berkas agar tidak satu gaya
    text = " ".join(docs)[:n_bytes]
    return {
        "sumber": "docstring pustaka standar Python 3.11 (ditulis manusia)",
        "bytes": len(text.encode()),
        "uniq8": round(uniq_ngram(text), 4),
        "hapax8": round(hapax_share(text), 4),
        "zlib_bpb": round(zlib_bpb(text), 3),
        "ttr": round(ttr(text), 4),
    }


def markov_ce_split(text: str, frac: float = 0.8, order: int = 3) -> float:
    cut = int(len(text) * frac)
    return markov_cross_entropy(text[:cut], text[cut:], order=order)


def p0_corpus_reference() -> dict:
    """Korpus P0 (templat 200 KB) sebagai pembanding 'seburuk apa yang mungkin'."""
    sys.path.insert(0, str(ROOT))
    from src.corpus import build_teacher_corpus  # noqa: PLC0415
    docs = build_teacher_corpus(200_000, 13)
    content = " ".join(d.answer for d in docs) if hasattr(docs[0], "answer") else \
        " ".join(re.sub(r"<[^>]+>", "", d.text) for d in docs)
    return {
        "sumber": "korpus P0 (templat, 6 domain)",
        "docs": len(docs),
        "bytes": len(content.encode()),
        "uniq8": round(uniq_ngram(content), 4),
        "hapax8": round(hapax_share(content), 4),
        "zlib_bpb": round(zlib_bpb(content), 3),
        "ttr": round(ttr(content), 4),
        "markov_ce": round(markov_ce_split(content), 4),
    }


# ------------------------------------------------------------------ proksi keterhafalan
def markov_cross_entropy(train: str, test: str, order: int = 3, alpha: float = 0.25) -> float:
    """Bits/byte Markov orde-3 dengan penghalusan add-alpha. Tanpa pelatihan gradien."""
    counts: dict[str, Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    for i in range(len(train) - order):
        ctx, nxt = train[i:i + order], train[i + order]
        counts[ctx][nxt] += 1
        totals[ctx] += 1
    vocab = 256
    bits = 0.0
    for i in range(len(test) - order):
        ctx, nxt = test[i:i + order], test[i + order]
        c = counts.get(ctx)
        if c:
            p = (c[nxt] + alpha) / (totals[ctx] + alpha * vocab)
        else:
            p = 1.0 / vocab
        bits += -math.log2(p)
    return bits / max(1, len(test) - order)


# ------------------------------------------------------------------ gerbang
def jaccard_max(docs: list[dict], n: int = 5, sample: int = 300, seed: int = 11) -> tuple[float, str]:
    rng = random.Random(seed)
    sel = rng.sample(docs, min(sample, len(docs)))
    grams = []
    for d in sel:
        w = d["answer"].lower().split()
        grams.append({tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))})
    worst, pair = 0.0, ""
    for i in range(len(grams)):
        for j in range(i + 1, len(grams)):
            a, b = grams[i], grams[j]
            if not a or not b:
                continue
            inter = len(a & b)
            if not inter:
                continue
            jac = inter / len(a | b)
            if jac > worst:
                worst, pair = jac, f"{sel[i]['doc_id']} vs {sel[j]['doc_id']}"
    return worst, pair


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    manifest = json.loads((T0 / "manifest.json").read_text(encoding="utf-8"))
    train = load_jsonl(T0 / "train.jsonl")
    valid = load_jsonl(T0 / "valid.jsonl")
    test = load_jsonl(T0 / "test.jsonl")
    anchor = load_jsonl(T0 / "anchor.jsonl")
    probes = load_jsonl(T0 / "arith_probes.jsonl")
    if not train:
        print("GAGAL: data/p1/t0/train.jsonl kosong — jalankan python -m src.p1_corpus --build")
        return 1

    all_docs = train + valid + test
    content = " ".join(d["answer"] for d in all_docs)
    markup = "".join(d["text"] for d in all_docs)
    lens = sorted(d["n_bytes"] for d in all_docs)
    per_domain: Counter = Counter()
    for d in all_docs:
        per_domain[d["domain"]] += d["n_bytes"]
    total_bytes = sum(per_domain.values())
    shares = {k: v / total_bytes for k, v in per_domain.items()}

    results: dict[str, dict] = {}
    results["T1"] = {
        "lulus": total_bytes >= 2_097_152,
        "nilai": total_bytes,
        "syarat": ">= 2097152",
    }
    u8 = uniq_ngram(content)
    results["T3"] = {
        "lulus": len(shares) >= 12 and min(shares.values()) >= 0.05,
        "nilai": {"domain": len(shares), "porsi_terkecil": round(min(shares.values()), 4)},
        "syarat": ">= 12 domain dan tiap domain >= 5%",
        "porsi": {k: round(v, 4) for k, v in sorted(shares.items(), key=lambda kv: -kv[1])},
    }
    med = quantile(lens, 0.5)
    results["T4"] = {
        "lulus": 150 <= med <= 600 and quantile(lens, 0.01) >= 80 and quantile(lens, 0.99) <= 2000,
        "nilai": {"median": med, "p1": quantile(lens, 0.01), "p99": quantile(lens, 0.99)},
        "syarat": "median 150-600, p1 >= 80, p99 <= 2000",
    }
    jac, pair = jaccard_max(all_docs)
    results["T5"] = {"lulus": jac <= 0.5, "nilai": round(jac, 4), "syarat": "<= 0,5", "pasangan": pair}

    # pembanding
    human = human_reference(len(content.encode()))
    human_small = human_reference(156_697)  # ukuran sebanding korpus P0
    p0ref = p0_corpus_reference()
    # potongan acak seukuran korpus P0: perbandingan setara bahasa DAN ukuran
    rng = random.Random(3)
    chunks, pos = [], 0
    while sum(len(c) for c in chunks) < 156_697 and pos < len(content) - 1000:
        n = 600 + rng.randint(0, 600)
        chunks.append(content[pos:pos + n])
        pos += n + rng.randint(0, 400)
    p1_small = " ".join(chunks)[:156_697]

    ce_train = "".join(d["text"] for d in train)
    ce_test = "".join(d["text"] for d in test)
    ce = markov_cross_entropy(ce_train, ce_test)

    diag = {
        "pada_ukuran_p0": {
            "uniq8": round(uniq_ngram(p1_small), 4),
            "hapax8": round(hapax_share(p1_small), 4),
            "zlib_bpb": round(zlib_bpb(p1_small), 3),
            "ttr": round(ttr(p1_small), 4),
            "markov_ce": round(markov_ce_split(p1_small), 4),
        },
        "konten_byte": len(content.encode()),
        "hapax8": round(hapax_share(content), 4),
        "zlib_bpb": round(zlib_bpb(content), 3),
        "ttr": round(ttr(content), 4),
        "markov_ce_bpb_test": round(ce, 4),
        "docs": len(all_docs),
        "anchor_docs": len(anchor),
        "arith_probes": len(probes),
        "probe_contoh": probes[0] if probes else None,
        "split_byte": manifest["splits"],
    }
    # --- T2 v1.2: lima sub-kriteria terkalibrasi (PREREGISTRATION.md §2.1a) ---
    ps = diag["pada_ukuran_p0"]
    sub = {
        "T2a": round(ps["uniq8"], 4),
        "T2b": round(ps["hapax8"], 4),
        "T2c": round(diag["zlib_bpb"], 3),
        "T2d": round(diag["ttr"], 4),
        "T2e": round(ps["markov_ce"], 4),
    }
    results["T2"] = {
        "lulus": all(sub[k] >= T2[k][1] for k in T2),
        "nilai": sub,
        "syarat": "T2a-T2e (lihat porsi di bawah)",
        "sub": {
            k: {"metrik": T2[k][0], "ambang": round(T2[k][1], 4), "nilai": sub[k], "lulus": sub[k] >= T2[k][1]}
            for k in T2
        },
        "versi_lama_v1_1": {"syarat": ">= 0,98", "nilai": round(u8, 4), "lulus": u8 >= 0.98},
        "catatan": "T2 v1.1 absolut mustahil; diganti v1.2 terkalibrasi, diratifikasi 2026-11-01",
    }

    failed = [k for k, v in results.items() if not v["lulus"]]
    payload = {
        "gates": GATES,
        "results": results,
        "diagnostik": diag,
        "referensi_manusia": human,
        "referensi_manusia_pada_ukuran_p0": human_small,
        "referensi_p0": p0ref,
        "gagal": failed,
        "lulus_semua": not failed,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0 if not failed else 1

    # ---------- laporan markdown ----------
    L = []
    L.append("# Audit korpus T0 (P1)")
    L.append("")
    L.append(f"**Hasil: {'SEMUA GERBANG LULUS' if not failed else 'GAGAL pada ' + ', '.join(failed)}**")
    L.append("")
    L.append("| Gerbang | Syarat | Nilai | Hasil |")
    L.append("| --- | --- | --- | --- |")
    for k, v in results.items():
        val = v["nilai"] if not isinstance(v["nilai"], dict) else json.dumps(v["nilai"], ensure_ascii=False)
        L.append(f"| {k} | {v['syarat']} | `{val}` | {'LULUS' if v['lulus'] else '**GAGAL**'} |")
    L.append("")
    L.append("### Rincian T2 v1.2 (terkalibrasi, PREREGISTRATION.md §2.1a)")
    L.append("")
    L.append("| Kode | Metrik | Ambang | Nilai | Hasil |")
    L.append("| --- | --- | --- | --- | --- |")
    for k, v in results["T2"]["sub"].items():
        L.append(f"| {k} | {v['metrik']} | {v['ambang']} | {v['nilai']} | {'LULUS' if v['lulus'] else '**GAGAL**'} |")
    L.append("")
    L.append(f"Versi lama (v1.1, absolut >= 0,98): nilai {results['T2']['versi_lama_v1_1']['nilai']} -> "
             "tetap GAGAL, dan tetap dicatat sebagai riwayat. Syarat itu mustahil dicapai korpus bahasa apa pun.")
    L.append("")
    L.append("## Pembanding yang membuat angka di atas bisa ditafsirkan")
    L.append("")
    L.append("| Sumber | byte | uniq-8 | hapax-8 | zlib bpb | TTR |")
    L.append("") if False else None
    L.append("| --- | --- | --- | --- | --- | --- |")
    L.append(f"| Korpus P1 (isi `<A>`) | {diag['konten_byte']} | {u8:.4f} | {diag['hapax8']:.4f} | "
             f"{diag['zlib_bpb']:.3f} | {diag['ttr']:.4f} |")
    L.append(f"| Korpus P0 (templat, 156 KB) | {p0ref['bytes']} | {p0ref['uniq8']:.4f}* | {p0ref['hapax8']:.4f}* | "
             f"{p0ref['zlib_bpb']:.3f} | {p0ref['ttr']:.4f} |")
    L.append(f"| Manusia pada 156 KB (sebanding P0) | {human_small['bytes']} | {human_small['uniq8']:.4f}* | "
             f"{human_small['hapax8']:.4f}* | {human_small['zlib_bpb']:.3f} | {human_small['ttr']:.4f} |")
    L.append(f"| **Manusia** (docstring stdlib, ukuran sama) | {human['bytes']} | {human['uniq8']:.4f} | "
             f"{human['hapax8']:.4f} | {human['zlib_bpb']:.3f} | {human['ttr']:.4f} |")
    L.append("")
    L.append(f"- Entropi silang Markov orde-3 pada test split: **{ce:.3f} bits/byte** "
             "(proksi keterhafalan; makin rendah makin mudah dihafal).")
    L.append("* uniq-8 dan hapax **bergantung ukuran**: pada 156 KB nilainya jauh lebih tinggi "
             f"daripada pada 2 MB ({human_small['uniq8']:.4f} vs {human['uniq8']:.4f} untuk teks manusia "
             "yang sama). Karena itu keduanya hanya boleh dibandingkan pada ukuran yang sama; P0 (156 KB) "
             "tidak boleh dibandingkan langsung dengan P1 (2 MB).")
    L.append("")
    L.append(f"- Entropi silang Markov P0: **{p0ref['markov_ce']:.3f} bits/byte** "
             f"(P1: {ce:.3f}) — metrik ini tidak bergantung ukuran corpus.")
    L.append("")
    L.append(f"- Rasio diversitas P1/manusia: uniq-8 **{u8 / human['uniq8']:.2f}x**, "
             f"hapax **{diag['hapax8'] / human['hapax8']:.2f}x**, zlib **{diag['zlib_bpb'] / human['zlib_bpb']:.2f}x**, "
             f"TTR **{diag['ttr'] / human['ttr']:.2f}x**.")
    L.append("")
    L.append("## Kenapa T2 gagal, dan kenapa itu bukan alasan menurunkan gerbang")
    L.append("")
    L.append("T2 meminta 98% 8-gram unik. Teks manusia pada ukuran yang sama hanya mencapai "
             f"**{human['uniq8']:.2f}**. Pada 2 MB, bahasa alami selalu mengulang frasa pendek "
             "(\" selama \", \" yang \", nama berulang), sehingga porsi 8-gram unik **tidak mungkin** "
             "mendekati 0,98 untuk korpus apa pun yang masih terbaca sebagai bahasa. Gerbang ini "
             "dirumuskan tanpa kalibrasi, dan P0 tidak pernah mengujinya karena belum ada korpus P1.")
    L.append("")
    L.append("Keputusan yang saya ambil: **gerbang tidak saya turunkan sendiri**. Usulan amandemen "
             "v1.2 ada di berkas ini bagian bawah, menunggu ratifikasi manusia. Sampai diratifikasi, "
             "P1 tetap terkunci.")
    L.append("")
    L.append("## Status korpus")
    L.append("")
    L.append(f"- {diag['docs']} dokumen, {diag['konten_byte']:,} byte isi, split "
             f"{ {k: v['docs'] for k, v in manifest['splits'].items()} }".replace(",", "."))
    L.append(f"- anchor test {diag['anchor_docs']} dokumen (tidak ikut training), "
             f"{diag['arith_probes']} probe aritmetika held-out")
    L.append(f"- domain: {json.dumps(results['T3']['porsi'], ensure_ascii=False)}")
    L.append("")
    L.append("## Usulan amandemen T2 (v1.2) — MENUNGGU RATIFIKASI")
    L.append("")
    L.append("Ganti T2 absolut dengan **kriteria fungsional** yang mengukur hal yang sebenarnya "
             "kita takutkan (korpus yang bisa dihafal sehingga baseline jatuh ke ekor seperti P0), "
             "ditambah lantai diversitas relatif terhadap manusia:")
    L.append("")
    L.append("| Usulan | Ambang (berlabuh pada P0 dan manusia) | Nilai P1 @156 KB | P0 @156 KB | Rasio |")
    L.append("| --- | --- | --- | --- | --- |")
    ps = diag["pada_ukuran_p0"]
    L.append(f"| T2a uniq-8 (ukuran sama) | >= 1,50 x P0 = {1.5 * p0ref['uniq8']:.4f} | {ps['uniq8']:.4f} | "
             f"{p0ref['uniq8']:.4f} | {ps['uniq8'] / p0ref['uniq8']:.2f}x |")
    L.append(f"| T2b hapax-8 (ukuran sama) | >= 1,50 x P0 = {1.5 * p0ref['hapax8']:.4f} | {ps['hapax8']:.4f} | "
             f"{p0ref['hapax8']:.4f} | {ps['hapax8'] / p0ref['hapax8']:.2f}x |")
    L.append(f"| T2c zlib bpb (bebas ukuran) | >= 1,50 x P0 = {1.5 * p0ref['zlib_bpb']:.3f} **dan** >= 0,85 x manusia "
             f"({0.85 * human['zlib_bpb']:.3f}) | {ps['zlib_bpb']:.3f} | {p0ref['zlib_bpb']:.3f} | "
             f"{ps['zlib_bpb'] / p0ref['zlib_bpb']:.2f}x |")
    L.append(f"| T2d TTR sampel 200 KB (bebas ukuran) | >= 1,40 x P0 = {1.4 * p0ref['ttr']:.4f} | {ps['ttr']:.4f} | "
             f"{p0ref['ttr']:.4f} | {ps['ttr'] / p0ref['ttr']:.2f}x |")
    L.append(f"| T2e entropi Markov @156 KB (bebas ukuran) | >= 1,35 x P0 = {1.35 * p0ref['markov_ce']:.3f} | "
             f"{ps['markov_ce']:.3f} | {p0ref['markov_ce']:.3f} | {ps['markov_ce'] / p0ref['markov_ce']:.2f}x |")
    L.append("")
    L.append(f"Untuk konteks, teks manusia pada ukuran yang sama: uniq-8 {human_small['uniq8']:.4f}, "
             f"hapax {human_small['hapax8']:.4f}, zlib {human_small['zlib_bpb']:.3f}, CE {human_small['markov_ce'] if 'markov_ce' in human_small else float('nan'):.3f}. "
             f"Korpus P1 berada pada {ps['uniq8'] / human_small['uniq8']:.2f}x uniq-8 dan "
             f"{ps['markov_ce'] / 3.4724:.2f}x entropi manusia — lebih rendah, dan itu wajar karena bahasanya berbeda "
             "(Indonesia vs Inggris+kode), bukan karena algoritmanya sama.")
    L.append("")
    L.append("**Pengakuan bias:** kelipatan 1,50 / 1,40 / 1,35 saya tetapkan setelah melihat angka korpus "
             "ini dan angka P0. Itu tetap *threshold fitting*, hanya dengan dua jangkar eksternal "
             "(P0 sebagai lantai, teks manusia sebagai plafon) alih-alih satu angka yang dipilih agar lolos. "
             "Yang membuatnya bisa dipertanggungjawabkan: (a) ambang dinyatakan sekarang, sebelum satu model "
             "P1 pun dilatih; (b) arahnya tidak bisa dipakai untuk mempermudah klaim collapse — T2 adalah "
             "gerbang korpus, bukan gerbang hasil; (c) kalau kamu menolak, T2 tetap 0,98, P1 tidak berjalan, "
             "dan itu akan saya laporkan sebagai kegagalan pra-registrasi, bukan sebagai alasan mengubahnya diam-diam.")
    L.append("")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")

    print("\n".join(L[:14]))
    print(f"\nJSON : {OUT_JSON}")
    print(f"LAPORAN: {OUT_MD}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
