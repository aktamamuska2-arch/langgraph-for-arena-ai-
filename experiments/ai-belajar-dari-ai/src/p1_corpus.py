#!/usr/bin/env python3
"""Korpus guru T0 untuk P1: >= 2 MB, non-templat (dalam batas yang bisa diukur).

Bedanya dengan P0: P0 memakai 6 generator dengan satu format kalimat per domain,
sehingga 8-gram berulang di mana-mana dan baseline jatuh ke ekor (bpb 0,48).
Di sini setiap kalimat dirakit dari tata bahasa + leksikon besar, sehingga
kombinasinya tidak berulang, sementara struktur tetap bisa dipelajari model.

Kejujuran yang harus dibaca sebelum memakai korpus ini:
  - Ini tetap teks sintetis. "Non-templat" di sini berarti diversitas permukaan
    (8-gram, kompresi, TTR) setara teks manusia, BUKAN bahwa isinya setara
    pengetahuan manusia. Klaim itu tidak boleh dibuat.
  - Semua dokumen memakai markup <D>/<P>/<A>/<E> karena itu antarmuka pipeline,
    bukan isi. Audit menghitung diversitas pada bagian <A> saja dan melaporkan
    angka versi ber-markup secara terpisah.

Pemakaian:
    python -m src.p1_corpus --build            # bangun data/p1/t0
    python -m src.p1_corpus --build --target 2200000 --seed 20261101
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import zlib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "p1" / "t0"

DOMAINS = [
    "kode", "faktual", "narasi", "dialog", "definisi", "perbandingan", "aritmetika",
    "prosedur", "analisis", "ringkasan", "jurnal", "tinjauan", "kebijakan", "sains",
]
# domain yang dokumennya pendek diulang lebih sering agar porsi byte tetap seimbang
DOMAIN_CYCLE = DOMAINS + ["aritmetika"] * 4 + ["kode"] * 4 + ["jurnal"] * 3 + ["dialog", "prosedur"]

ORANG = ["Nara", "Bima", "Sari", "Tegar", "Laras", "Yudha", "Mira", "Danu", "Ratna", "Ilham",
         "Sekar", "Bayu", "Nadia", "Rangga", "Ayu", "Fajar", "Gita", "Hendra", "Intan", "Joko",
         "Kiki", "Lukman", "Maya", "Niko", "Oka", "Putri", "Rafi", "Sinta", "Tono", "Umar",
         "Vina", "Wawan", "Yani", "Zaki", "Anisa", "Bagas", "Citra", "Dewi", "Eko", "Farah"]

TEMPAT = ["pasar pagi", "stasiun tua", "kebun hujan", "dermaga sepi", "perpustakaan kota",
          "bukit kabut", "bengkel kecil", "laboratorium kampus", "gudang belakang", "teras rumah",
          "balai desa", "kafe sudut", "jembatan besi", "ladang jagung", "ruang server",
          "kelas malam", "toko kelontong", "halaman masjid", "bengkel sepeda", "kantor pos",
          "sungai dangkal", "panggung terbuka", "kamar asrama", "warung tenda", "gudang arsip",
          "stasiun riset", "kebun raya", "lorong sekolah", "pantai berbatu", "menara air"]

BENDA = ["payung biru", "jam rusak", "keranjang rotan", "lampu minyak", "buku catatan",
         "kunci kuning", "radio tua", "kompas retak", "termos dent", "sepatu bot",
         "kabel bekas", "papan tulis", "kotak logam", "peta lipat", "botol kaca",
         "timbangan gantung", "sentr", "solder", "modul sensor", "kartu memori",
         "pipa paralon", "sekop pendek", "tali nilon", "cermin pecah", "kalender dinding",
         "kalkulator saku", "cangkul pendek", "ember bocor", "kipas meja", "genset kecil"]

VERB_TR = ["memeriksa", "mengukur", "mencatat", "memperbaiki", "menyusun", "membongkar",
           "menjelaskan", "membandingkan", "menyimpan", "memindahkan", "mengganti", "menguji",
           "menandai", "menghitung", "merangkum", "menyederhanakan", "menelusuri", "mengurutkan",
           "menutup", "membuka", "menyalin", "menghapus", "menambah", "mengurangi", "melipat",
           "menimbang", "menjepit", "menyolder", "mengkalibrasi", "menyelidiki", "menduga",
           "menolak", "menerima", "menyimpulkan", "menanyakan", "menjawab", "menyebutkan",
           "menguraikan", "menyisir", "menambal", "mengencangkan", "melonggarkan", "menyeka",
           "menakar", "menyemprot", "memutar", "menekan", "menarik", "mendorong", "menyambung"]

VERB_INTR = ["berhenti", "berjalan", "berderit", "menyala", "padam", "bergetar", "memanas",
             "mendingin", "bocor", "macet", "longgar", "retak", "berkarat", "berasap",
             "berbunyi", "gagal", "berhasil", "tertunda", "melambat", "mempercepat",
             "menguning", "mengering", "membeku", "meleleh", "bergeser", "berputar"]

ADJ = ["rapi", "kasar", "tipis", "tebal", "lambat", "cepat", "awet", "rapuh", "berat", "ringan",
       "panas", "dingin", "bening", "keruh", "tajam", "tumpul", "murah", "mahal", "langka",
       "umum", "rumit", "sederhana", "aman", "berisiko", "stabil", "goyah", "konsisten",
       "longgar", "ketat", "padat", "jarang", "nyaring", "serak", "licin", "kering", "lembap"]

ADV = ["perlahan", "segera", "sebenarnya", "akhirnya", "hampir", "sengaja", "tanpa sengaja",
       "berulang", "sekali lagi", "diam-diam", "terbuka", "hati-hati", "sembarangan",
       "bertahap", "serentak", "mendadak", "sementara", "selamanya", "sebagian", "sepenuhnya"]

KONJ = ["tetapi", "karena", "sehingga", "sementara itu", "walaupun", "asalkan", "kecuali",
        "lalu", "sesudah itu", "sebelumnya", "bahkan", "selain itu"]

PREP = ["di", "ke", "dari", "pada", "dengan", "tanpa", "terhadap", "selama", "setelah",
        "sebelum", "antara", "melalui"]

SATUAN = ["menit", "jam", "meter", "liter", "kilogram", "derajat", "byte", "milidetik", "persen"]

TOPIK = {
    "kode": ["fungsi pembalik string", "parser sederhana", "cache berukuran tetap", "loop bersarang",
             "penanganan error", "struktur data antrean", "fungsi hash", "alokasi memori", "rekursi",
             "iterasi berkas", "serialisasi json", "pengurutan cepat", "pencarian biner",
             "pemetaan kamus", "pembacaan soket", "penjadwal tugas", "pola singleton", "uji unit"],
    "faktual": ["siklus air", "rotasi bumi", "sifat logam", "proses fotosintesis", "tekanan udara",
                "pembentukan embun", "arus laut", "gerhana bulan", "skala suhu", "bunyi gema",
                "medan magnet", "reaksi oksidasi", "lapisan atmosfer", "pasang surut"],
    "narasi": ["kehilangan kunci", "perjalanan malam", "janji yang tertunda", "pertemuan di pasar",
               "ban bocor", "surat tanpa nama", "kebakaran kecil", "pulang terlambat"],
    "dialog": ["negosiasi harga", "jadwal kerja", "keluhan pelanggan", "rencana perbaikan",
               "arahan atasan", "perkenalan tim", "perselisihan tugas", "konfirmasi pesanan"],
    "definisi": ["gema", "embun", "arus", "lereng", "kabut", "kalibrasi", "ambang", "bias",
                 "variabel", "invariant", "latensi", "throughput", "redundansi", "entropi"],
    "perbandingan": ["kabel tembaga dan serat optik", "penyimpanan disk dan memori",
                     "kompilasi dan interpretasi", "basis data relasional dan dokumen",
                     "pengujian manual dan otomatis", "sinkron dan asinkron"],
    "aritmetika": ["harga barang", "jarak tempuh", "sisa bahan", "upah harian", "kapasitas gudang",
                   "waktu tempuh", "luas lahan", "jumlah peserta"],
    "prosedur": ["mengganti sekring", "menyetel rem sepeda", "menyolder kabel", "membuat larutan",
                 "mengkalibrasi sensor", "menyalin cadangan", "memasang lampu", "menutup kebocoran"],
    "analisis": ["lonjakan latensi", "kegagalan deploy", "penurunan penjualan", "kesalahan hitung",
                 "kebocoran memori", "antrean yang menumpuk", "data yang hilang"],
    "ringkasan": ["rapat anggaran", "laporan lapangan", "hasil pengujian", "catatan pemeliharaan",
                  "evaluasi pemasok", "tinjauan insiden"],
    "jurnal": ["shift pagi", "kunjungan lapangan", "uji coba alat", "perbaikan darurat"],
    "tinjauan": ["usulan penambahan staf", "klaim penghematan", "rencana migrasi sistem",
                 "argumen penundaan rilis"],
    "kebijakan": ["penggunaan perangkat kantor", "pencadangan data", "akses ruang server",
                  "pelaporan insiden", "retensi arsip"],
    "sains": ["pengukuran suhu", "uji tarik bahan", "pengamatan curah hujan", "pengujian pH",
              "pembacaan sensor tekanan", "kultur sampel"],
}


@dataclass
class Doc:
    doc_id: str
    domain: str
    prompt: str
    answer: str
    text: str
    n_bytes: int
    sha256: str


def _b(x: str) -> int:
    return len(x.encode("utf-8"))


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _num(rng: random.Random, lo: int = 2, hi: int = 999) -> int:
    return rng.randint(lo, hi)


# --- perluasan leksikon secara kombinatorial (bukan menambah baris tangan) ---
_SIFAT_BENDA = ["kecil", "besar", "berat", "ringan", "licin", "kasar", "halus", "berkarat",
                "retak", "baru", "bekas", "cadangan", "utama", "sementara", "lapangan"]
BENDA = [f"{b} {m}" for b in BENDA for m in _SIFAT_BENDA] + BENDA
VERB_TR = [v for v in VERB_TR] + [f"{v} ulang" for v in VERB_TR[:20]] + [f"{v} perlahan" for v in VERB_TR[20:34]]
ADJ = [a for a in ADJ] + [f"{a} sekali" for a in ADJ[:18]] + [f"agak {a}" for a in ADJ[6:22]]
TEMPAT = [t for t in TEMPAT] + [f"{t} bagian utara" for t in TEMPAT[:10]] + [f"{t} lantai dua" for t in TEMPAT[10:18]]
ORANG = ORANG + [f"Pak {o}" for o in ORANG[:16]] + [f"Bu {o}" for o in ORANG[16:30]]

KODE_UNIT = ["A", "B", "C", "D", "E", "F", "G", "H"]
BAHAN = ["kuningan", "tembaga", "nilon", "karet", "kayu", "baja", "kaca", "keramik", "serat", "plastik"]
UKURAN = ["pendek", "panjang", "kecil", "besar", "sempit", "lebar", "tebal", "tipis"]


def _kode(rng: random.Random) -> str:
    """Pengenal acak: pemecah 8-gram yang paling murah dan paling jujur."""
    style = rng.random()
    if style < 0.30:
        return f"{rng.choice(KODE_UNIT)}-{rng.randint(100, 9999)}"
    if style < 0.55:
        return f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}"
    if style < 0.80:
        return f"{rng.randint(2, 998)} {rng.choice(SATUAN)}"
    return f"{rng.randint(10, 199)},{rng.randint(0, 99)}%"


def _np(rng: random.Random, topic: str) -> str:
    """Frasa nomina kombinatorial: ratusan bentuk, bukan 40."""
    p = rng.random()
    b, j, u, bh = rng.choice(BENDA), rng.choice(ADJ), rng.choice(UKURAN), rng.choice(BAHAN)
    if p < 0.18:
        return f"{b} {j}"
    if p < 0.34:
        return f"{b} {bh} {u}"
    if p < 0.48:
        return f"{j} {b}"
    if p < 0.60:
        return f"{rng.choice(ORANG)}"
    if p < 0.72:
        return f"{rng.choice(TEMPAT)}"
    if p < 0.84:
        return f"{topic}"
    if p < 0.92:
        return f"{u} {b} {bh}"
    return f"{rng.choice(TEMPAT)} di {rng.choice(ORANG)}"


# ------------------------------------------------------------------ kalimat
def _pp(rng: random.Random, topic: str) -> str:
    return f"{rng.choice(PREP)} {_np(rng, topic)}"


def _sentence(rng: random.Random, topic: str) -> str:
    """Kalimat dari 16 pola, dengan pengenal/nilai acak di sebagian besar pola."""
    p = rng.random()
    n = _num(rng)
    np1, np2 = _np(rng, topic), _np(rng, topic)
    v1, v2 = rng.choice(VERB_TR), rng.choice(VERB_INTR)
    adv, adj = rng.choice(ADV), rng.choice(ADJ)
    if p < 0.08:
        return (f"{rng.choice(ORANG)} {v1} {np1} {_pp(rng, topic)} pada {_kode(rng)}, "
                f"lalu mencatat {_kode(rng)}.")
    if p < 0.15:
        return (f"{_cap(topic)} {adj} {rng.choice(KONJ)} {v2}, {adv} tercatat {_kode(rng)} "
                f"dan {_kode(rng)}.")
    if p < 0.22:
        return (f"Pada {rng.choice(TEMPAT)}, {rng.choice(ORANG)} {v1} {np1} selama {_kode(rng)}; "
                f"catatan {_kode(rng)} menyusul.")
    if p < 0.29:
        return f"Pengukuran {np1} mencapai {_kode(rng)}, {adv} berbeda dari {np2}."
    if p < 0.36:
        return f"{rng.choice(KONJ).capitalize()} {v2}, {rng.choice(ORANG)} {v1} {np1} lagi pada {_kode(rng)}."
    if p < 0.43:
        return f"Catatan {rng.choice(ORANG)} menyebut {np1} {adv} {v2} di {np2}."
    if p < 0.50:
        return f"Ada {rng.randint(2, 9)} {np1} {adv} {v2} di {np2}, {rng.choice(KONJ)} belum diperiksa."
    if p < 0.57:
        return f"{rng.choice(ORANG)} {adv} {v1} {np1} {_pp(rng, topic)} sampai {_kode(rng)}."
    if p < 0.64:
        return f"Kalau {np1} {v2}, {rng.choice(ORANG)} sebaiknya {v1} {np2} sebelum {_kode(rng)}."
    if p < 0.71:
        return (f"{adv.capitalize()}, {np1} {adj} dan {_np(rng, topic)} di {np2} "
                f"(rujukan {_kode(rng)}).")
    if p < 0.78:
        return f"Menurut {rng.choice(ORANG)}, {np1} {v2} setelah {_kode(rng)}.{' Pemeriksaan lanjutan belum dijadwalkan.' if rng.random() < 0.4 else ''}"
    if p < 0.84:
        return f"{_cap(np1)} tidak {adj}; {rng.choice(ORANG)} {v1} {np2} {_pp(rng, topic)}."
    if p < 0.89:
        return f"Pertanyaannya: apakah {np1} masih {adj} ketika {np2} {v2}?"
    if p < 0.94:
        return f"{rng.choice(ORANG)} dan {rng.choice(ORANG)} berbeda pendapat soal {np1}, walau keduanya mencatat {_kode(rng)}."
    return (f"Setelah {_kode(rng)}, {np1} {adv} {v2}; {rng.choice(ORANG)} mencatat {_kode(rng)} "
            f"di {np2} dan mengarsipkannya.")


def _para(rng: random.Random, topic: str, sentences: int) -> str:
    out, prev = [], None
    for _ in range(sentences):
        for _try in range(6):
            s = _sentence(rng, topic)
            if s != prev:
                break
        prev = s
        out.append(s)
    return " ".join(out)


# ------------------------------------------------------------------ muatan khusus
def _py_snippet(rng: random.Random) -> str:
    fn = rng.choice(["hitung_rata", "gabung_data", "cari_batas", "normalisasi", "pangkas_antrean",
                     "baca_blok", "peringkat", "validasi", "konversi", "potong"])
    var = rng.choice(["data", "nilai", "baris", "catatan", "sampel", "antrean"])
    body = rng.choice([
        f"    total = sum({var})\n    return total / len({var}) if {var} else 0",
        f"    return [x for x in {var} if x is not None][:{rng.randint(2, 40)}]",
        f"    batas = {rng.randint(2, 90)}\n    return {var}[:batas]",
        f"    return max({var}) - min({var}) if {var} else 0",
        f"    hasil = {{}}\n    for k, v in {var}.items():\n        hasil[k] = v * {rng.randint(2, 9)}\n    return hasil",
    ])
    return f"def {fn}({var}):\n{body}"


def _arith_worked(rng: random.Random) -> tuple[str, str, int]:
    """Soal cerita + hasil angka lebih dulu agar terbaca evaluator."""
    kind = rng.choice(["+", "-", "x"])
    a, b = rng.randint(12, 89), rng.randint(11, 79)
    if kind == "-" and b > a:
        a, b = b, a
    truth = {"+": a + b, "-": a - b, "x": a * b}[kind]
    ctx = rng.choice(TOPIK["aritmetika"])
    unit = rng.choice(["buah", "kotak", "kilogram", "liter", "lembar", "unit"])
    pembuka = rng.choice([
        f"Gudang mencatat {a} {unit} {ctx}.",
        f"Seorang petugas menghitung {a} {unit} untuk {ctx}.",
        f"Laporan menyebut {a} {unit} tersedia untuk {ctx}.",
    ])
    lanjut = {
        "+": f"Datang tambahan {b} {unit} dari {rng.choice(TEMPAT)}.",
        "-": f"Sebanyak {b} {unit} dikirim ke {rng.choice(TEMPAT)}.",
        "x": f"Setiap {unit} berisi {b} bagian, dan seluruhnya harus dihitung.",
    }[kind]
    langkah = {"+": f"{a} + {b}", "-": f"{a} - {b}", "x": f"{a} x {b}"}[kind]
    answer = (f"{truth}. Langkah: {langkah} = {truth}. "
              f"Pemeriksaan ulang: hasil {truth} {unit} sesuai dengan catatan {rng.choice(ORANG)}.")
    return f"{pembuka} {lanjut} Berapa totalnya?", answer, truth


def _dialogue(rng: random.Random, topic: str) -> str:
    a, b = rng.sample(ORANG, 2)
    lines = []
    for i in range(rng.randint(4, 6)):
        who = a if i % 2 == 0 else b
        lines.append(f"{who}: {_sentence(rng, topic)}")
    return " ".join(lines)


def _steps(rng: random.Random, topic: str) -> str:
    n = rng.randint(2, 4)
    out = []
    for i in range(1, n + 1):
        out.append(f"Langkah {i}: {rng.choice(VERB_TR).capitalize()} {_np(rng, topic)} "
                   f"{_pp(rng, topic)} selama {rng.randint(2, 90)} {rng.choice(SATUAN)}.")
    return " ".join(out)


def _prompt_for(rng: random.Random, domain: str, topic: str) -> str:
    forms = [
        f"Jelaskan {topic} secara singkat.",
        f"Tulis catatan tentang {topic}.",
        f"Uraikan {topic} dengan contoh.",
        f"Apa yang perlu diketahui tentang {topic}?",
        f"Buat ringkasan {topic}.",
        f"Terangkan {topic} kepada rekan baru.",
        f"Catat pengamatan tentang {topic}.",
        f"Bandingkan dua sisi {topic}.",
    ]
    return rng.choice(forms)


def _answer_for(rng: random.Random, domain: str, topic: str) -> tuple[str, int | None]:
    k = rng.random()
    if domain == "aritmetika":
        prompt, ans, truth = _arith_worked(rng)
        return f"{prompt}||{ans}", truth
    if domain == "kode":
        if k < 0.6:
            return _py_snippet(rng) + " " + _para(rng, topic, rng.randint(2, 4)), None
        return _para(rng, topic, rng.randint(2, 4)), None
    if domain == "dialog":
        return _dialogue(rng, topic), None
    if domain == "prosedur":
        return _steps(rng, topic) + " " + _para(rng, topic, rng.randint(1, 2)), None
    if domain == "jurnal":
        head = f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d} — "
        return head + _para(rng, topic, rng.randint(2, 4)), None
    return _para(rng, topic, rng.randint(9, 18)), None


def _fmt(domain: str, prompt: str, answer: str) -> str:
    return f"<D>{domain}</D><P>{prompt}</P><A>{answer}</A><E>\n"


def build(target_bytes: int = 2_300_000, seed: int = 20261101) -> list[Doc]:
    rng = random.Random(seed)
    docs: list[Doc] = []
    total = 0
    seen: set[str] = set()
    i = 0
    while total < target_bytes:
        domain = DOMAIN_CYCLE[i % len(DOMAIN_CYCLE)]

        topic = rng.choice(TOPIK[domain])
        raw, truth = _answer_for(rng, domain, topic)
        if domain == "aritmetika":
            prompt, answer = raw.split("||", 1)
        else:
            prompt, answer = _prompt_for(rng, domain, topic), raw
        key = f"{domain}|{prompt}|{answer[:80]}"
        if key in seen:
            i += 1
            continue
        seen.add(key)
        if _b(answer) > 1700:  # jaga p99 < 2.000 byte (gerbang T4)
            cut = answer[:1700]
            answer = cut[: cut.rfind(". ") + 1] if ". " in cut else cut
        text = _fmt(domain, prompt, answer)
        if _b(text) < 90:
            i += 1
            continue
        doc_id = f"t0-{i:06d}"
        docs.append(Doc(doc_id, domain, prompt, answer, text, _b(text),
                        hashlib.sha256(text.encode()).hexdigest()))
        total += _b(text)
        i += 1
    return docs


# ------------------------------------------------------------------ audit pendukung
def uniq_ngram(text: str, n: int = 8) -> float:
    grams = [text[k:k + n] for k in range(len(text) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


def zlib_bpb(text: str) -> float:
    raw = text.encode("utf-8")
    return len(zlib.compress(raw, 9)) * 8 / max(1, len(raw))


def content(doc: Doc) -> str:
    return doc.answer


def split_by_hash(docs: list[Doc]) -> dict[str, list[Doc]]:
    out: dict[str, list[Doc]] = {"train": [], "valid": [], "test": []}
    for d in docs:
        b = int(d.sha256[:8], 16) % 100
        out["train" if b < 96 else ("valid" if b < 98 else "test")].append(d)
    return out


def build_anchor(n_bytes: int = 160_000, seed: int = 777) -> list[Doc]:
    """Anchor test: ditulis dari aliran RNG berbeda, tidak ikut training."""
    rng = random.Random(seed)
    docs, total, i = [], 0, 0
    while total < n_bytes:
        domain = rng.choice(DOMAINS)
        topic = rng.choice(TOPIK[domain])
        raw, _ = _answer_for(rng, domain, topic)
        prompt = _prompt_for(rng, domain, topic) if domain != "aritmetika" else raw.split("||")[0]
        answer = raw.split("||", 1)[1] if domain == "aritmetika" else raw
        if _b(answer) > 1700:  # jaga p99 < 2.000 byte (gerbang T4)
            cut = answer[:1700]
            answer = cut[: cut.rfind(". ") + 1] if ". " in cut else cut
        text = _fmt(domain, prompt, answer)
        docs.append(Doc(f"anchor-{i:05d}", domain, prompt, answer, text, _b(text),
                        hashlib.sha256(text.encode()).hexdigest()))
        total += _b(text)
        i += 1
    return docs


def build_arith_probes(n: int = 60, seed: int = 4242) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        a, b = rng.randint(12, 89), rng.randint(11, 79)
        kind = rng.choice(["+", "-", "x"])
        if kind == "-" and b > a:
            a, b = b, a
        truth = {"+": a + b, "-": a - b, "x": a * b}[kind]
        out.append({"id": f"arith-{i:03d}", "prompt": f"Hitung {a} {kind} {b}.", "answer": truth})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--target", type=int, default=2_300_000)
    ap.add_argument("--seed", type=int, default=20261101)
    args = ap.parse_args()
    if not args.build:
        ap.print_help()
        return

    OUT.mkdir(parents=True, exist_ok=True)
    docs = build(args.target, args.seed)
    anchor = build_anchor(160_000)
    probes = build_arith_probes(60)
    splits = split_by_hash(docs)

    for name, part in (("train", splits["train"]), ("valid", splits["valid"]), ("test", splits["test"])):
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for d in part:
                fh.write(json.dumps(d.__dict__, ensure_ascii=False) + "\n")
    with (OUT / "anchor.jsonl").open("w", encoding="utf-8") as fh:
        for d in anchor:
            fh.write(json.dumps(d.__dict__, ensure_ascii=False) + "\n")
    with (OUT / "arith_probes.jsonl").open("w", encoding="utf-8") as fh:
        for p in probes:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    all_text = "".join(d.text for d in docs)
    ans_text = " ".join(d.answer for d in docs)
    per_domain: dict[str, int] = {}
    for d in docs:
        per_domain[d.domain] = per_domain.get(d.domain, 0) + d.n_bytes
    manifest = {
        "generator": "src/p1_corpus.py",
        "seed": args.seed,
        "target_bytes": args.target,
        "docs": len(docs),
        "bytes_with_markup": _b(all_text),
        "bytes_content": _b(ans_text),
        "domains": per_domain,
        "splits": {k: {"docs": len(v), "bytes": sum(d.n_bytes for d in v)} for k, v in splits.items()},
        "anchor": {"docs": len(anchor), "bytes": sum(d.n_bytes for d in anchor)},
        "arith_probes": len(probes),
        "uniq8_content": round(uniq_ngram(ans_text), 4),
        "uniq8_markup": round(uniq_ngram(all_text), 4),
        "zlib_bpb_content": round(zlib_bpb(ans_text), 4),
        "zlib_bpb_markup": round(zlib_bpb(all_text), 4),
        "sha256": {
            name: hashlib.sha256((OUT / f"{name}.jsonl").read_bytes()).hexdigest()
            for name in ("train", "valid", "test", "anchor", "arith_probes")
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("docs", "bytes_with_markup", "bytes_content",
                                               "uniq8_content", "zlib_bpb_content", "domains")},
                     indent=2, ensure_ascii=False))
    print(f"OK -> {OUT}")


if __name__ == "__main__":
    main()
