"""Metrik P0: anchor BPB (primer), diversitas, repetisi, gaya, dan cek aritmetika."""
from __future__ import annotations

import math
import re
from itertools import combinations

import torch

from . import config as C
from .common import rng_for
from .corpus import prompt_prefix
from .generate import generate_batch
from .model import TinyGPT

ANS_RE = re.compile(r"(-?\d+)")


@torch.no_grad()
def anchor_bpb(model: TinyGPT, text: str) -> float:
    """Bits per byte pada teks anchor: -log2 p(byte)."""
    data = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
    n = len(data)
    total_nll = 0.0
    counted = 0
    for start in range(0, n - 1, C.BLOCK_SIZE):
        chunk = data[start : start + C.BLOCK_SIZE + 1]
        if len(chunk) < 2:
            break
        x = chunk[:-1].unsqueeze(0)
        y = chunk[1:].unsqueeze(0)
        _, loss_sum = model(x, y)
        total_nll += float(loss_sum)
        counted += len(chunk) - 1
    if counted == 0:
        return float("nan")
    return total_nll / counted / math.log(2)


def _eval_prompts(n: int) -> list[tuple[str, str, int | None]]:
    """Prompt held-out untuk evaluasi (tidak dipakai menulis T0)."""
    rng = rng_for("eval-prompts", C.SEED_GLOBAL)
    out: list[tuple[str, str, int | None]] = []
    for _ in range(n):
        kind = rng.choice(["+", "-", "x"])
        a, b = rng.randint(11, 89), rng.randint(11, 89)
        if kind == "+":
            out.append(("aritmetika", f"Hitung {a} + {b}.", a + b))
        elif kind == "-":
            if a < b:
                a, b = b, a
            out.append(("aritmetika", f"Hitung {a} - {b}.", a - b))
        else:
            a, b = rng.randint(3, 12), rng.randint(3, 9)
            out.append(("aritmetika", f"Hitung {a} x {b}.", a * b))
    return out


def _arith_accuracy(model: TinyGPT, n: int) -> tuple[float, int]:
    probes = _eval_prompts(n)
    prompts = [(d, p) for d, p, _ in probes]
    texts = generate_batch(model, prompts, temp=0.01, top_p=1.0, max_new=70)
    correct = 0
    for (_, _, truth), text in zip(probes, texts):
        body = text.split("<A>", 1)[-1]
        m = ANS_RE.search(body.replace("<E>", " "))
        if m and int(m.group(1)) == truth:
            correct += 1
    return correct / max(1, len(probes)), len(probes)


def _words(text: str) -> list[str]:
    return [w.strip(".,;:!?<>/()[]{}") for w in text.lower().split() if w.strip(".,;:!?<>/()[]{}")]


def _distinct2(docs: list[str]) -> float:
    total, uniq = 0, set()
    for d in docs:
        w = _words(d)
        for a, b in zip(w, w[1:]):
            total += 1
            uniq.add((a, b))
    return len(uniq) / total if total else 0.0


def _repetition(docs: list[str], n: int = 8) -> float:
    hit = 0
    for d in docs:
        w = _words(d)
        grams = [tuple(w[i : i + n]) for i in range(max(0, len(w) - n + 1))]
        if grams and len(set(grams)) < len(grams):
            hit += 1
    return hit / len(docs) if docs else 0.0


def _char_ngrams(text: str, k: int = 3) -> dict[str, int]:
    t = " ".join(text.split())
    out: dict[str, int] = {}
    for i in range(len(t) - k + 1):
        g = t[i : i + k]
        out[g] = out.get(g, 0) + 1
    return out


def _style_div(docs: list[str], max_pairs: int = 80) -> float:
    """Jarak kosinus rata-rata pada TF-IDF char-3gram — proksi homogenisasi gaya."""
    vecs = []
    df: dict[str, int] = {}
    grams = [_char_ngrams(d) for d in docs]
    for g in grams:
        for k in g:
            df[k] = df.get(k, 0) + 1
    n = len(docs)
    for g in grams:
        vecs.append({k: (1 + math.log(v)) * math.log((1 + n) / (1 + df[k])) for k, v in g.items()})
    dists = []
    for i, j in combinations(range(n), 2):
        a, b = vecs[i], vecs[j]
        common = set(a) & set(b)
        if not common:
            dists.append(1.0)
        else:
            dot = sum(a[k] * b[k] for k in common)
            na = math.sqrt(sum(v * v for v in a.values()))
            nb = math.sqrt(sum(v * v for v in b.values()))
            dists.append(1 - dot / (na * nb) if na and nb else 1.0)
        if len(dists) >= max_pairs:
            break
    return sum(dists) / len(dists) if dists else 0.0


def evaluate(
    model: TinyGPT,
    anchor_text: str,
    baseline_distinct2: float | None = None,
    verbose: bool = False,
) -> dict:
    model.eval()
    bpb = anchor_bpb(model, anchor_text)
    probes = _eval_prompts(C.EVAL_SAMPLES)
    texts = generate_batch(model, [(d, p) for d, p, _ in probes], temp=C.TEMPERATURE, top_p=C.TOP_P)
    arith_acc, arith_n = _arith_accuracy(model, C.EVAL_ARITH_N)
    docs = [t for t in texts]
    metrics = {
        "anchor_bpb": round(bpb, 5),
        "arith_acc": round(arith_acc, 3),
        "arith_n": arith_n,
        "distinct2": round(_distinct2(docs), 4),
        "repetition_rate": round(_repetition(docs), 4),
        "style_div": round(_style_div(docs), 4),
        "mean_bytes": round(sum(len(d.encode()) for d in docs) / max(1, len(docs)), 1),
        "degenerate_rate": round(
            sum(1 for d in docs if "<A>" not in d or "<E>" not in d) / max(1, len(docs)), 3
        ),
    }
    if baseline_distinct2:
        metrics["distinct2_rel_drop"] = round(1 - metrics["distinct2"] / baseline_distinct2, 4)
    if verbose:
        print(f"    eval: bpb={metrics['anchor_bpb']:.4f} arith={metrics['arith_acc']:.2f} "
              f"d2={metrics['distinct2']:.3f} rep={metrics['repetition_rate']:.2f}", flush=True)
    return metrics
