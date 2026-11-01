"""Generasi korpus sintetis dari model siswa (protokol rekursif)."""
from __future__ import annotations

import random

import torch

from . import config as C
from .corpus import prompt_prefix
from .model import TinyGPT


def _sample_next(logits: torch.Tensor, temp: float, top_p: float) -> torch.Tensor:
    logits = logits / max(temp, 1e-6)
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sp, si = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sp, dim=-1)
        keep = cum - sp <= top_p
        keep[:, 0] = True
        filt = torch.where(keep, sp, torch.zeros_like(sp))
        probs = filt / filt.sum(dim=-1, keepdim=True)
        return si.gather(-1, torch.multinomial(probs, 1))
    return torch.multinomial(probs, 1)


@torch.no_grad()
def generate_batch(
    model: TinyGPT,
    prompts: list[tuple[str, str]],
    temp: float = C.TEMPERATURE,
    top_p: float = C.TOP_P,
    max_new: int = C.GEN_MAX_NEW,
) -> list[str]:
    """prompts: [(domain, prompt_text)] -> daftar dokumen (prefix + lanjutan)."""
    prefixes = [prompt_prefix(d, p).encode("utf-8") for d, p in prompts]
    maxlen = max(len(p) for p in prefixes)
    # padding kiri dengan id 0 agar batch seragam; posisi token asli digeser
    seqs = [[0] * (maxlen - len(p)) + list(p) for p in prefixes]
    ids = torch.tensor(seqs, dtype=torch.long)
    done = [False] * len(prompts)
    out = ids
    for _ in range(max_new):
        ctx = out[:, -model.cfg.block_size :]
        logits, _ = model(ctx)
        nxt = _sample_next(logits[:, -1, :], temp, top_p)
        out = torch.cat([out, nxt], dim=1)
        for i in range(len(prompts)):
            if done[i]:
                continue
            tail = bytes(out[i, maxlen:].tolist())
            if b"<E>" in tail:
                done[i] = True
        if all(done):
            break
    docs = []
    for i, pref in enumerate(prefixes):
        tail = bytes(out[i, maxlen:].tolist())
        cut = tail.find(b"<E>")
        tail = tail[: cut + 3] if cut >= 0 else tail
        docs.append((pref + tail).decode("utf-8", "replace"))
    return docs


def generate_corpus(
    model: TinyGPT,
    catalog: list[dict],
    target_bytes: int,
    seed: int,
    verbose: bool = False,
) -> list[dict]:
    """Hasilkan dokumen sampai target byte; prompt dari katalog T0 (L0 statis)."""
    rng = random.Random(seed)
    order = list(range(len(catalog)))
    docs: list[dict] = []
    total = 0
    batch = C.GEN_BATCH
    i = 0
    while total < target_bytes:
        if i + batch > len(order):
            rng.shuffle(order)
            i = 0
        idx = order[i : i + batch]
        i += batch
        prompts = [(catalog[j]["domain"], catalog[j]["prompt"]) for j in idx]
        texts = generate_batch(model, prompts)
        for j, text in zip(idx, texts):
            docs.append(
                {
                    "domain": catalog[j]["domain"],
                    "prompt": catalog[j]["prompt"],
                    "prompt_id": catalog[j]["prompt_id"],
                    "text": text + ("\n" if not text.endswith("\n") else ""),
                }
            )
            total += len(text.encode("utf-8"))
        if verbose:
            print(f"    generasi: {total}/{target_bytes} byte", flush=True)
    return docs
