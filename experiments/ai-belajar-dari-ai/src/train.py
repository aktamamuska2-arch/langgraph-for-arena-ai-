"""Training loop XS: 300 langkah, anggaran token tetap untuk semua arm."""
from __future__ import annotations

import math
import time

import torch

from . import config as C
from .common import sha256_bytes
from .model import TinyGPT, XSConfig


def _encode(text: str) -> torch.Tensor:
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)


def train_model(
    text: str,
    seed: int,
    steps: int = C.TRAIN_STEPS,
    log_every: int = 50,
    verbose: bool = False,
) -> tuple[TinyGPT, dict]:
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    data = _encode(text)
    if len(data) < C.BLOCK_SIZE * 4:
        raise ValueError("korpus terlalu kecil untuk training")

    cfg = XSConfig(
        block_size=C.BLOCK_SIZE,
        n_layer=C.N_LAYER,
        n_head=C.N_HEAD,
        d_model=C.D_MODEL,
    )
    model = TinyGPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=0.01, betas=(0.9, 0.95))
    g = torch.Generator().manual_seed(seed + 1)
    t0 = time.perf_counter()
    losses: list[float] = []

    for step in range(1, steps + 1):
        # cosine schedule + warmup
        if step <= C.WARMUP:
            lr = C.LR * step / C.WARMUP
        else:
            frac = (step - C.WARMUP) / max(1, steps - C.WARMUP)
            lr = C.MIN_LR + 0.5 * (C.LR - C.MIN_LR) * (1 + math.cos(math.pi * frac))
        for pg in opt.param_groups:
            pg["lr"] = lr

        ix = torch.randint(len(data) - C.BLOCK_SIZE - 1, (C.BATCH_SIZE,), generator=g)
        x = torch.stack([data[i : i + C.BLOCK_SIZE] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + C.BLOCK_SIZE] for i in ix])

        _, loss_sum = model(x, y)
        loss = loss_sum / (C.BATCH_SIZE * C.BLOCK_SIZE)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
        opt.step()

        losses.append(float(loss.detach()))
        if verbose and step % log_every == 0:
            print(f"    step {step:4d} loss {loss:.4f} lr {lr:.2e}", flush=True)

    elapsed = time.perf_counter() - t0
    metrics = {
        "steps": steps,
        "final_loss": losses[-1],
        "mean_loss_last50": sum(losses[-50:]) / min(50, len(losses)),
        "first_loss": losses[0],
        "tokens_seen": steps * C.BATCH_SIZE * C.BLOCK_SIZE,
        "elapsed_s": round(elapsed, 1),
        "corpus_sha": sha256_bytes(text.encode("utf-8")),
        "n_params": model.n_params,
        "seed": seed,
    }
    return model, metrics
