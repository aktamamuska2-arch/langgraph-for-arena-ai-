#!/usr/bin/env python3
"""Uji mandiri antarmuka: memastikan metrik leksikal dihitung pada ISI, bukan markup,
dan bahwa tabel yang dikirim ke peramban memang memuat nilai yang benar.

Dijalankan `make ui-selftest` (ikut di `make check`, tidak butuh torch).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ui_inspect as U  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if cond else 'GAGAL'}  {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def main() -> int:
    print("1) markup tidak boleh menggelembungkan metrik leksikal")
    # isi = 4 token dengan 2 kata unik; markup ikut berulang di sekitarnya
    body = "kabut turun kabut turun"
    tagged = f"<D>narasi</D><P>x</P><A>{body}</A><E>"
    m = U.assess(tagged)
    check("TTR dihitung pada isi = 0,50", m["ttr"] == 0.5, f"dapat {m['ttr']}")
    check("TTR sama dengan tanpa markup", m["ttr"] == U.assess(body)["ttr"])
    # bigram isi: (kabut,turun) (turun,kabut) (kabut,turun) -> 2 unik dari 3
    check("distinct-2 isi = 0,667 (bukan efek tag)", round(m["distinct2"], 3) == 0.667, f"dapat {m['distinct2']}")
    check("distinct-2 sama dengan tanpa markup", m["distinct2"] == U.assess(body)["distinct2"])
    check("kata tidak memuat nama tag", all("<" not in str(w) for w in [k for k, _ in m["kata_terbanyak"]]),
          str(m["kata_terbanyak"]))
    check("teks berakhir <E> terdeteksi", m["berakhir_E"] is True)

    print("2) deteksi struktur rusak")
    broken = "<D>narasi</D><P>x</P><A>kabut turun"
    b = U.assess(broken)
    check("dokumen tanpa <E> -> degenerate", b["degenerate"] is True)
    check("teks pendek -> repetisi8 = n/a", b["repetisi8"] is None)

    print("3) tabel yang dikirim ke peramban memuat nilai yang benar")
    html = U.render_table({"ttr": 0.43, "repetisi8": None, "kata_terbanyak": [("kabut", 4)]}, ["arith_acc"])
    check("TTR muncul di tabel", "0.43" in html, html.count("<tr>") and "baris tabel ada")
    check("nilai kosong dirender eksplisit", "n/a" in html)
    check("daftar pasangan dirender", "kabut×4" in html)
    check("penanda lantai tidak salah tempat", html.count("di lantai") == 0)

    print("4) halaman web memuat elemen yang diperlukan")
    page = U.PAGE
    for token in ("/api/generate", "/api/assess", "table_html", "id=\"ckpt\"", "id=\"prompt\""):
        check(f"elemen {token}", token in page)

    print()
    if FAILS:
        print(f"GAGAL: {len(FAILS)} pemeriksaan -> {FAILS}")
        return 1
    print("LULUS: antarmuka dan metrik konsisten")
    return 0


if __name__ == "__main__":
    sys.exit(main())
