"""Metrik R8 lengkap untuk P1 (PREREG v1.3 §6).

Empat metrik yang belum ada di evaluator P0 — `truncated_rate`, `ttr`,
`near_dup_rate`, `answer_entropy` — ditambahkan di sini, bersama metrik P0 yang
dipertahankan agar angka lama dan baru bisa dibaca berdampingan.

Definisi yang dipakai (dan batasnya, dinyatakan terbuka):
  - degenerate_rate : porsi keluaran yang region `<A>`-nya nyaris kosong (< 10 byte). Definisi P0
                      ("tanpa <A> atau <E>") tidak bisa dipakai di sini: prompt sudah memuat `<A>`
                      sehingga metrik itu menjadi salinan persis truncated_rate (terukur 0,8056 vs
                      0,8056 pada lima baseline). Diubah sebelum ada model rantai; dicatat di PLAN.
  - truncated_rate  : porsi keluaran yang tidak pernah menutup dengan `<E>` dalam jendela.
                      Jendela 256 byte lebih pendek dari median T0 (483 B), jadi angka ini
                      TINGGI secara struktural; ia berguna sebagai sinyal relatif, bukan absolut.
  - ttr             : rata-rata type/token rasio pada region `<A>` tiap dokumen (kata unik / kata).
  - near_dup_rate   : porsi dokumen yang punya pasangan dengan Jaccard 5-gram kata >= 0,80.
  - answer_entropy  : entropi ternormalisasi (H / log2(k)) atas 5 sampel suhu 1,0 untuk prompt
                      yang sama, dihitung pada 12 byte pertama jawaban. Batas: 5 sampel hanya
                      bisa membedakan sampai log2(5) tingkat; ini proksi kasar, bukan entropi sejati.
"""

from __future__ import annotations

import json
import math
import re
from itertools import combinations
from pathlib import Path

import torch

from . import p1_config as P
from .evaluate import _distinct2, _repetition, _style_div, _words
from .gen_fast import generate_batch
from .model import TinyGPT

ROOT = Path(__file__).resolve().parent.parent
T0 = ROOT / "data" / "p1" / "t0"
ANS_RE = re.compile(r"(-?\d+)")


def _load(name: str) -> list[dict]:
    p = T0 / name
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def heldout_docs() -> list[dict]:
    """Valid + test (4% dari korpus). Tidak pernah ikut melatih model mana pun."""
    docs = _load("valid.jsonl") + _load("test.jsonl")
    return docs[: P.EVAL_DOCS]


def anchor_text() -> str:
    docs = _load("anchor.jsonl")
    text = "".join(d["text"] for d in docs)
    return text.encode("utf-8")[: P.ANCHOR_BYTES].decode("utf-8", errors="ignore")


def arith_probes() -> list[dict]:
    return _load("arith_probes.jsonl")[: P.ARITH_N]


def prompt_of(doc: dict) -> bytes:
    return f"<D>{doc['domain']}</D><P>{doc['prompt']}</P><A>".encode("utf-8")


@torch.no_grad()
def anchor_bpb(model: TinyGPT, text: str) -> float:
    data = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
    block = model.cfg.block_size
    total_nll, counted = 0.0, 0
    for start in range(0, len(data) - 1, block):
        chunk = data[start : start + block + 1]
        if len(chunk) < 2:
            break
        _, loss_sum = model(chunk[:-1].unsqueeze(0), chunk[1:].unsqueeze(0))
        total_nll += float(loss_sum)
        counted += len(chunk) - 1
    return total_nll / counted / math.log(2) if counted else float("nan")


def _answer_region(text: str) -> str:
    body = text.split("<A>", 1)[-1]
    return body.split("</A>", 1)[0]


def _degenerate(outs: list[bytes]) -> float:
    """Jawaban nyaris kosong: region sebelum `</A>` kurang dari 10 byte berarti."""
    n = 0
    for o in outs:
        body = o.decode("utf-8", errors="replace").split("</A>", 1)[0].strip()
        if len(body) < 10:
            n += 1
    return n / max(1, len(outs))


def _ttr(docs: list[str]) -> float:
    vals = []
    for d in docs:
        w = _words(_answer_region(d))
        if len(w) >= 5:
            vals.append(len(set(w)) / len(w))
    return sum(vals) / len(vals) if vals else 0.0


def _near_dup(docs: list[str]) -> float:
    grams = [set(tuple(w[i : i + P.NEAR_DUP_N]) for i in range(max(0, len(w) - P.NEAR_DUP_N + 1)))
             for w in (_words(_answer_region(d)) for d in docs)]
    hit = 0
    for i, gi in enumerate(grams):
        if not gi:
            continue
        for j, gj in enumerate(grams):
            if i == j or not gj:
                continue
            inter = len(gi & gj)
            if inter == 0:
                continue
            if inter / len(gi | gj) >= P.NEAR_DUP_JACCARD:
                hit += 1
                break
    return hit / len(docs) if docs else 0.0


@torch.no_grad()
def _answer_entropy(model: TinyGPT, prompts: list[bytes]) -> float:
    import random

    rows: list[bytes] = []
    for p in prompts[: P.ENTROPY_PROMPTS]:
        rows.extend([p] * P.ENTROPY_SAMPLES)
    outs = generate_batch(model, rows, max_new=P.ENTROPY_TOKENS, temperature=1.0, top_p=1.0,
                          stop_byte=None, stop_suffix=None, max_batch=P.GEN_BATCH)
    k = P.ENTROPY_SAMPLES
    vals = []
    for i in range(0, len(outs), k):
        group = [o[:P.ENTROPY_TOKENS] for o in outs[i : i + k]]
        counts: dict[bytes, int] = {}
        for g in group:
            counts[g] = counts.get(g, 0) + 1
        h = -sum((c / len(group)) * math.log2(c / len(group)) for c in counts.values())
        vals.append(h / math.log2(k))
    return sum(vals) / len(vals) if vals else 0.0


@torch.no_grad()
def _arith_acc(model: TinyGPT, probes: list[dict]) -> tuple[float, int]:
    prompts = [f"<D>aritmetika</D><P>{p['prompt']}</P><A>".encode("utf-8") for p in probes]
    outs = generate_batch(model, prompts, max_new=24, temperature=0.05, top_p=1.0,
                          stop_byte=None, stop_suffix=None, max_batch=P.GEN_BATCH)
    correct = 0
    for probe, out in zip(probes, outs):
        m = ANS_RE.search(out.decode("utf-8", errors="ignore").replace("<E>", " "))
        if m and int(m.group(1)) == int(probe["answer"]):
            correct += 1
    return correct / max(1, len(probes)), len(probes)


@torch.no_grad()
def evaluate_p1(model: TinyGPT, anchor: str | None = None,
                docs: list[dict] | None = None, verbose: bool = False) -> dict:
    model.eval()
    anchor = anchor if anchor is not None else anchor_text()
    docs = docs if docs is not None else heldout_docs()
    prompts = [prompt_of(d) for d in docs]
    outs = generate_batch(model, prompts, max_new=P.GEN_MAX_NEW, temperature=P.TEMPERATURE,
                          top_p=P.TOP_P, stop_byte=None, stop_suffix=b"<E>", max_batch=P.GEN_BATCH)
    texts = [p.decode("utf-8", errors="ignore") + o.decode("utf-8", errors="replace")
             for p, o in zip(prompts, outs)]
    bpb = anchor_bpb(model, anchor)
    arith_acc, arith_n = _arith_acc(model, arith_probes())
    metrics = {
        "anchor_bpb": round(bpb, 5),
        "arith_acc": round(arith_acc, 3),
        "arith_n": arith_n,
        "degenerate_rate": round(_degenerate(outs), 3),
        "truncated_rate": round(sum(1 for o in outs if b"<E>" not in o) / len(outs), 3),
        "ttr": round(_ttr(texts), 4),
        "near_dup_rate": round(_near_dup(texts), 3),
        "answer_entropy": round(_answer_entropy(model, prompts), 4),
        "distinct2": round(_distinct2(texts), 4),
        "repetition_rate": round(_repetition(texts), 4),
        "style_div": round(_style_div(texts), 4),
        "mean_bytes": round(sum(len(o) for o in outs) / len(outs), 1),
    }
    if verbose:
        print(f"    bpb={metrics['anchor_bpb']:.4f} arith={metrics['arith_acc']:.2f}/"
              f"{metrics['arith_n']} trunc={metrics['truncated_rate']:.2f} ttr={metrics['ttr']:.3f} "
              f"ndup={metrics['near_dup_rate']:.2f} H={metrics['answer_entropy']:.3f}", flush=True)
    return metrics
