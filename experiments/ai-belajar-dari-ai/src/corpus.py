"""Korpus T0 — ditulis oleh 'guru' (AI) dalam bentuk template terarifikasi.

P0 memakai korpus template, bukan prosa bebas. Konsekuensi yang wajib dilaporkan:
keragaman gaya T0 rendah, sehingga homogenisasi gaya tidak bisa diukur dengan baik
di P0. Format dokumen:

    <DOMAIN>x</DOMAIN>
    <PROMPT>...</PROMPT>
    <ANSWER>...</ANSWER>
    <END>
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .common import rng_for, sha256_text

DOMAINS = ["aritmetika", "kode", "faktual", "narasi", "dialog", "definisi"]

PERSONA = ["Nara", "Bima", "Sari", "Tegar", "Laras", "Yudha", "Mira", "Danu"]
SETTING = ["pasar pagi", "stasiun tua", "kebun hujan", "dermaga sepi", "perpus kota", "bukit kabut"]
OBJECT = ["payung biru", "jam rusak", "keranjang rotan", "lampu minyak", "buku catatan", "kunci kuning"]
TERMS = [
    ("gema", "bunyi yang kembali setelah memantul", "gema di lembah", "bayangan di cermin"),
    ("embun", "titik air yang terbentuk saat udara mendingin", "embun di daun", "asap di udara"),
    ("arus", "gerakan air yang mengalir satu arah", "arus sungai", "tumpukan batu"),
    ("lereng", "sisi miring sebuah bukit", "lereng gunung", "dasar jurang"),
    ("kabut", "kumpulan titik air yang melayang dekat tanah", "kabut pagi", "api unggun"),
]
FACTS = [
    ("Planet Vela", "periode orbit", 412, "hari"),
    ("Planet Orin", "periode orbit", 87, "hari"),
    ("Bulan Kiran", "suhu rata-rata", -14, "derajat"),
    ("Laut Dalam", "kedalaman maksimum", 6400, "meter"),
    ("Gunung Tera", "tinggi puncak", 2810, "meter"),
    ("Zat Halon", "titik lebur", 431, "kelvin"),
]
TOPICS = ["hujan dan jadwal", "kunci yang hilang", "roti yang terbakar", "kereta terakhir"]


PREFIX_FMT = "<D>{domain}</D><P>{prompt}</P><A>"


def fmt_doc(domain: str, prompt: str, answer: str, canary: str | None = None) -> str:
    """Dokumen satu baris agar muat dalam konteks 96 token."""
    doc = f"<D>{domain}</D><P>{prompt}</P><A>{answer}</A>"
    if canary:
        doc += f"<C>{canary}</C>"
    return doc + "<E>\n"


def prompt_prefix(domain: str, prompt: str) -> str:
    return PREFIX_FMT.format(domain=domain, prompt=prompt)


def _arith(rng: random.Random) -> tuple[str, str]:
    kind = rng.choice(["tambah", "kurang", "kali"])
    if kind == "tambah":
        a, b = rng.randint(11, 89), rng.randint(11, 89)
        prompt = f"Hitung {a} + {b}."
        ans = f"{a} + {b} = {a + b}. Langkah: {a} + {b}."
    elif kind == "kurang":
        a, b = rng.randint(40, 99), rng.randint(10, 39)
        prompt = f"Hitung {a} - {b}."
        ans = f"{a} - {b} = {a - b}. Langkah: {a} - {b}."
    else:
        a, b = rng.randint(3, 12), rng.randint(3, 9)
        prompt = f"Hitung {a} x {b}."
        ans = f"{a} x {b} = {a * b}. Langkah: {a} x {b}."
    return prompt, ans


def _code(rng: random.Random) -> tuple[str, str]:
    pick = rng.randint(0, 2)
    if pick == 0:
        s = rng.choice(["kata", "meja", "lampu", "hujan"])
        prompt = "Tulis fungsi Python untuk membalik urutan huruf."
        ans = f"def balik(s): return s[::-1]. Contoh: balik('{s}') -> '{s[::-1]}'"
    elif pick == 1:
        nums = [rng.randint(1, 9) for _ in range(3)]
        prompt = "Tulis fungsi Python untuk menjumlahkan isi daftar."
        ans = f"def jumlah(xs): return sum(xs). Contoh: jumlah({nums}) -> {sum(nums)}"
    else:
        word = rng.choice(["aku", "dia", "kamu"])
        vowels = sum(c in "aiueo" for c in word)
        prompt = "Tulis fungsi Python untuk menghitung huruf vokal."
        ans = f"def vokal(s): return sum(1 for c in s if c in 'aiueo'). Contoh: vokal('{word}') -> {vowels}"
    return prompt, ans


def _factual(rng: random.Random) -> tuple[str, str]:
    name, attr, value, unit = rng.choice(FACTS)
    prompt = f"Berapa {attr} {name}?"
    ans = f"{attr.capitalize()} {name} adalah {value} {unit}."
    return prompt, ans


def _narasi(rng: random.Random) -> tuple[str, str]:
    p, s, o = rng.choice(PERSONA), rng.choice(SETTING), rng.choice(OBJECT)
    prompt = f"Tulis cerita pendek tentang {p} di {s}."
    ans = (
        f"{p} berjalan ke {s} membawa {o}. Di sana ia bertemu penjaga yang bertanya arah. "
        f"{o.capitalize()} diletakkan di bangku, angin menggesernya sedikit, lalu {p} pulang."
    )
    return prompt, ans


def _dialog(rng: random.Random) -> tuple[str, str]:
    a, b, t = rng.choice(PERSONA), rng.choice([x for x in PERSONA if x != "Nara"]), rng.choice(TOPICS)
    prompt = f"Tulis percakapan singkat antara {a} dan {b} tentang {t}."
    ans = (
        f"{a}: Kamu sudah dengar soal {t}?\n"
        f"{b}: Belum, ada apa?\n"
        f"{a}: Katanya berubah lagi pagi ini.\n"
        f"{b}: Kalau begitu kita periksa bersama."
    )
    return prompt, ans


def _definisi(rng: random.Random) -> tuple[str, str]:
    term, meaning, ex, non = rng.choice(TERMS)
    prompt = f"Definisikan '{term}' dan beri satu contoh serta satu non-contoh."
    ans = f"{term.capitalize()} adalah {meaning}. Contoh: {ex}. Non-contoh: {non}."
    return prompt, ans


GENERATORS = {
    "aritmetika": _arith,
    "kode": _code,
    "faktual": _factual,
    "narasi": _narasi,
    "dialog": _dialog,
    "definisi": _definisi,
}


@dataclass
class Doc:
    doc_id: str
    gen: int
    domain: str
    prompt: str
    text: str
    n_bytes: int
    sha256: str


def build_teacher_corpus(target_bytes: int = 200_000, seed: int = 13) -> list[Doc]:
    rng = rng_for("teacher", seed)
    docs: list[Doc] = []
    seen: set[str] = set()
    total = 0
    i = 0
    while total < target_bytes:
        domain = DOMAINS[i % len(DOMAINS)]
        prompt_fn = GENERATORS[domain]
        prompt, answer = prompt_fn(rng)
        key = f"{domain}|{prompt}"
        if key in seen:
            i += 1
            continue
        seen.add(key)
        text = fmt_doc(domain, prompt, answer)
        if total + len(text.encode()) > target_bytes and len(docs) > 50:
            break
        doc_id = f"t0-{i:05d}"
        docs.append(
            Doc(
                doc_id=doc_id,
                gen=0,
                domain=domain,
                prompt=prompt,
                text=text,
                n_bytes=len(text.encode()),
                sha256=sha256_text(text),
            )
        )
        total += len(text.encode())
        i += 1
    return docs


def split_docs(docs: list[Doc]) -> dict[str, list[Doc]]:
    """Split deterministik dari hash: 96/2/2."""
    out: dict[str, list[Doc]] = {"train": [], "valid": [], "test": []}
    for d in docs:
        b = int(d.sha256[:8], 16) % 100
        out["train" if b < 96 else ("valid" if b < 98 else "test")].append(d)
    return out
