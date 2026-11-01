"""Generasi batch dengan KV-cache untuk P1 (sandbox).

Alasan keberadaan berkas ini: `TinyGPT.generate` menghitung ulang seluruh konteks
setiap token. Pada profil P1 (konteks 256, ~1 jt parameter) itu ~108 tok/s dan
membuat P1 butuh ratusan jam. Dengan cache K/V, forward per token turun ke
~2,7 ms untuk 32 baris (~12 rb tok/s terukur), yaitu 20-40x lebih cepat.

Dua kompromi yang harus dilaporkan setiap kali angka dari berkas ini dipakai:
1. Prompt dikelompokkan menurut panjang persisnya, karena cache tidak bisa
   menangani padding kiri dengan embedding posisi absolut.
2. Bila konteks mencapai `block_size`, cache digeser (crop kiri) — posisi
   di-reset relatif ke jendela. Ini praktik lazim, tetapi bukan distribusi
   yang sama dengan pelatihan.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    logits = logits / max(temperature, 1e-6)
    probs = F.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_p, sorted_i = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sorted_p, dim=-1)
        keep = cum - sorted_p <= top_p
        keep[:, 0] = True
        filtered = torch.where(keep, sorted_p, torch.zeros_like(sorted_p))
        probs = torch.zeros_like(probs).scatter(1, sorted_i, filtered)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_batch(
    model,
    prompts: list[bytes],
    max_new: int = 200,
    temperature: float = 0.9,
    top_p: float = 0.95,
    stop_byte: int | None = 0x0A,  # None = jangan berhenti pada byte tunggal
    stop_suffix: bytes | None = None,  # mis. b"<E>": berhenti setelah sufiks cocok
    max_batch: int = 64,
) -> list[bytes]:
    """Kembalikan kelanjutan (tanpa prompt) untuk setiap prompt, sepanjang <= max_new."""
    # prompt lebih panjang dari jendela: pakai ekornya (byte terdekat ke jawaban).
    # Prompt asli tidak diubah; hanya konteks forward yang dipotong.
    bs = model.cfg.block_size
    prompts = [p[-bs:] if len(p) > bs else p for p in prompts]
    out: list[bytes] = [b""] * len(prompts)

    # kelompokkan menurut panjang prompt (cache butuh panjang seragam)
    buckets: dict[int, list[int]] = {}
    for i, p in enumerate(prompts):
        buckets.setdefault(len(p), []).append(i)

    for length, idxs in buckets.items():
        for start in range(0, len(idxs), max_batch):
            chunk = idxs[start : start + max_batch]
            x = torch.tensor([list(prompts[i]) for i in chunk], dtype=torch.long)
            kvs = None
            logits, kvs = model.forward_cached(x, None)
            orig = list(range(len(chunk)))          # baris sekarang -> indeks asli
            buf: dict[int, bytearray] = {j: bytearray() for j in range(len(chunk))}
            for _ in range(max_new):
                nxt = _sample(logits[:, -1, :], temperature, top_p)
                if not orig:
                    break
                keep, keep_orig = [], []
                for row, o in enumerate(orig):
                    b = int(nxt[row, 0])
                    if stop_byte is not None and b == stop_byte:
                        continue
                    buf[o].append(b)
                    if stop_suffix is not None and bytes(buf[o][-len(stop_suffix):]) == stop_suffix:
                        continue
                    keep.append(row)
                    keep_orig.append(o)
                if not keep:
                    break
                if len(keep) != len(orig):
                    idx = torch.tensor(keep, dtype=torch.long)
                    kvs = [(k[idx], v[idx]) for k, v in kvs]
                if kvs[0][0].shape[2] + 1 > model.cfg.block_size:
                    keep_n = model.cfg.block_size - 1
                    kvs = [(k[:, :, -keep_n:], v[:, :, -keep_n:]) for k, v in kvs]
                orig = keep_orig
                logits, kvs = model.forward_cached(nxt[keep], kvs)
            for row in range(len(chunk)):
                out[chunk[row]] = bytes(buf[row])
    return out
