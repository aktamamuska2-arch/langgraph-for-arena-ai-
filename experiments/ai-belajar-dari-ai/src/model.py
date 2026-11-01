"""Model bahasa decoder-only kecil (profil XS) — PyTorch, CPU.

Arsitektur (sama semangatnya dengan PLAN §6.1, diskalakan untuk P0):
  pre-LN, MHA kausal, MLP GELU 4x, embedding posisi yang dipelajari,
  weight tying, vocab byte-level 256.

Deviasi dari PLAN yang harus dilaporkan:
  - vocab 256 byte (bukan BPE 8.192) -> token = byte, sehingga bpb == loss/ln2
  - posisi dipelajari (bukan RoPE)
  - 0,47 jt parameter (konfigurasi XS)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class XSConfig:
    vocab_size: int = 256
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 96
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


class Block(nn.Module):
    def __init__(self, cfg: XSConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_head, dropout=cfg.dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, cfg: XSConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self.apply(self._init)
        n = sum(p.numel() for p in self.parameters())
        self.n_params = n

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        mask = torch.triu(
            torch.full((t, t), float("-inf"), device=idx.device), diagonal=1
        )
        for blk in self.blocks:
            x = blk(x, mask)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), reduction="sum"
        )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new: int,
        temperature: float = 0.9,
        top_p: float = 0.95,
        stop_byte: int | None = None,
    ) -> torch.Tensor:
        """Sampling autoregresif untuk batch sekuens (tanpa KV cache)."""
        out = idx
        for _ in range(max_new):
            ctx = out[:, -self.cfg.block_size :]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            if top_p < 1.0:
                sorted_p, sorted_i = torch.sort(probs, descending=True, dim=-1)
                cum = torch.cumsum(sorted_p, dim=-1)
                keep = cum - sorted_p <= top_p
                keep[:, 0] = True
                filtered = torch.where(keep, sorted_p, torch.zeros_like(sorted_p))
                probs = filtered / filtered.sum(dim=-1, keepdim=True)
                nxt = sorted_i.gather(-1, torch.multinomial(probs, 1))
            else:
                nxt = torch.multinomial(probs, 1)
            out = torch.cat([out, nxt], dim=1)
            if stop_byte is not None and bool((nxt == stop_byte).all()):
                break
        return out
