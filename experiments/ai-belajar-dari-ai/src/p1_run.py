#!/usr/bin/env python3
"""Runner P1-sandbox (PREREGISTRATION v1.3).

Menjalankan grid terkunci: 5 baseline + 5 seed x 3 arm x 6 generasi = 95 model.

Aturan yang dijaga di sini:
  - label arm yang ditulis ke metrik adalah label BUTA (X1/X2/X3); pemetaan hanya
    dibaca dari `ops/blind_commitment` -> berkas kunci, tidak pernah ditulis ke
    `runs/p1/`.
  - anggaran token identik untuk semua model (1 epoch |T0-latih|), apa pun arm-nya.
  - arm C tidak pernah melihat data sintetis; ia hanya berganti seed per generasi.
  - dapat dilanjutkan: model yang sudah tercatat di state tidak diulang.

    python3 -m src.p1_run --stage baseline
    python3 -m src.p1_run --stage chains
    python3 -m src.p1_run --stage all --limit 2      # uji asap
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import p1_config as P  # noqa: E402
from src.gen_fast import generate_batch  # noqa: E402
from src.model import XSConfig, TinyGPT  # noqa: E402
from src.p1_eval import evaluate_p1, heldout_docs, anchor_text, prompt_of  # noqa: E402
from src.train import train_model  # noqa: E402

RUNS = ROOT / "runs" / "p1"
CKPT = ROOT / "checkpoints" / "p1"
SYN = ROOT / "data" / "p1" / "syn"
T0DIR = ROOT / "data" / "p1" / "t0"
KEY = ROOT / ".blind" / "key.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def labels() -> dict[str, str]:
    """Pemetaan arm -> label buta. Satu-satunya tempat kunci dibaca."""
    return json.loads(KEY.read_text(encoding="utf-8"))["mapping"]


def t0_text() -> str:
    rows = [json.loads(l) for l in (T0DIR / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return "".join(r["text"] for r in rows)


def t0_docs() -> list[dict]:
    return [json.loads(l) for l in (T0DIR / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def fit_bytes(text: str, n: int) -> str:
    """Ulangi teks sampai >= n byte (dipakai bila korpus arm lebih kecil dari anggaran)."""
    if not text:
        raise ValueError("korpus kosong")
    if len(text.encode("utf-8")) >= n:
        return text.encode("utf-8")[:n].decode("utf-8", errors="ignore")
    k = math.ceil(n / max(1, len(text.encode("utf-8"))))
    out = (text * k).encode("utf-8")[:n]
    return out.decode("utf-8", errors="ignore")


def read_syn(path: Path) -> list[str] | None:
    """Baca korpus sintetis; kembalikan None bila berkas rusak/terpotong.

    Ini bukan kehati-hatian teoretis: uji asap menemukan berkas .jsonl yang tertulis
    sebagian (127.947 dari 154.322 byte) setelah proses dihentikan saat menulis. Tanpa
    validasi, run panjang akan gagal di tengah atau — lebih buruk — memakai korpus cacat.
    """
    if not path.exists():
        return None
    try:
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return [r["text"] for r in rows] if rows else None


def write_syn(path: Path, texts: list[str]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for t in texts:
            fh.write(json.dumps({"text": t, "bytes": len(t.encode("utf-8"))}, ensure_ascii=False) + "\n")
    tmp.replace(path)  # atomik: tidak ada keadaan setengah jadi


def generate_syn(model: TinyGPT, docs: list[dict], n_bytes: int, tag: str, out_dir: Path) -> list[str]:
    """Hasilkan dokumen sintetis sampai >= n_bytes; simpan + kembalikan teksnya."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}.jsonl"
    cached = read_syn(path)
    if cached is not None:
        return cached
    if path.exists():
        log(f"    PERINGATAN: {path.name} rusak/terpotong -> dibangun ulang")
    texts: list[str] = []
    seen: set[str] = set()
    total, i = 0, 0
    t0 = time.perf_counter()
    while total < n_bytes:
        batch = []
        while len(batch) < P.GEN_BATCH:
            d = docs[i % len(docs)]
            batch.append(prompt_of(d))
            i += 1
        outs = generate_batch(model, batch, max_new=P.GEN_MAX_NEW, temperature=P.TEMPERATURE,
                              top_p=P.TOP_P, stop_byte=None, stop_suffix=b"<E>", max_batch=P.GEN_BATCH)
        for prompt, o in zip(batch, outs):
            text = prompt.decode("utf-8", errors="ignore") + o.decode("utf-8", errors="replace")
            if len(o) < P.GEN_MIN_DOC or text in seen:
                continue
            seen.add(text)
            texts.append(text)
            total += len(text.encode("utf-8"))
        if i > len(docs) * 40:  # pengaman anti-loop
            log(f"    PERINGATAN: pengaman generasi aktif pada {total} byte")
            break
    write_syn(path, texts)
    log(f"    sintetis {tag}: {len(texts)} dok, {total} byte, {time.perf_counter()-t0:.1f}s")
    return texts


def cfg() -> XSConfig:
    return XSConfig(block_size=P.BLOCK_SIZE, n_layer=P.N_LAYER, n_head=P.N_HEAD, d_model=P.D_MODEL)


def truncate_soft(text: str, n_bytes: int) -> str:
    b = text.encode("utf-8")[:n_bytes]
    return b.decode("utf-8", errors="ignore")


def compose(arm: str, gen: int, t0: str, syn: dict[int, list[str]], budget: int) -> tuple[str, dict]:
    """Korpus latih untuk satu model. Aturan PREREG §2.2."""
    if arm == "C":
        return truncate_soft(t0, budget), {"sumber": ["T0"], "byte_T0": budget, "byte_sintetis": 0}
    if arm == "A":
        src = "".join(syn[gen - 1])
        return fit_bytes(src, budget), {"sumber": [f"syn{gen-1}"], "byte_T0": 0, "byte_sintetis": budget}
    # B1: T0 + seluruh generasi sebelumnya, bobot sama per sumber (bukan per token)
    sources: list[tuple[str, str]] = [("T0", t0)] + [(f"syn{g}", "".join(syn[g])) for g in range(1, gen)]
    share = budget // len(sources)
    parts = [fit_bytes(t, share) for _, t in sources]
    return truncate_soft("".join(parts), budget), {
        "sumber": [n for n, _ in sources], "byte_T0": share, "byte_sintetis": share * (len(sources) - 1)}


def run_one(label: str, seed: int, gen: int, arm: str, text: str, data_info: dict,
            steps: int, anchor: str, docs: list[dict]) -> dict:
    model, tmetrics = train_model(text, seed=P.seed_for(seed, gen), steps=steps, cfg=cfg(),
                                  batch=P.BATCH_SIZE, log_every=10**9)
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": model.cfg.__dict__, "n_params": model.n_params},
               CKPT / f"{label}-{seed}-g{gen}.pt")
    emetrics = evaluate_p1(model, anchor=anchor, docs=docs, verbose=True)
    return {
        "model": f"{label}-s{seed}-g{gen}",
        "arm": label,
        "seed": seed,
        "gen": gen,
        "params": model.n_params,
        "train": {k: tmetrics[k] for k in ("steps", "final_loss", "mean_loss_last50", "tokens_seen", "elapsed_s", "n_params")},
        "data": data_info,
        "eval": emetrics,
    }


def load_state() -> dict:
    if (RUNS / "state.json").exists():
        return json.loads((RUNS / "state.json").read_text(encoding="utf-8"))
    return {"done": []}


def save_state(st: dict) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "state.json").write_text(json.dumps(st, indent=2), encoding="utf-8")


def append_row(row: dict) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    with (RUNS / "metrics.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["baseline", "chains", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="batasi jumlah model (uji asap)")
    ap.add_argument("--generations", type=int, default=0, help="batasi jumlah generasi (mis. 2 = bertahap)")
    ap.add_argument("--smoke", action="store_true", help="perkecil semua ukuran; menulis ke runs/p1_smoke")
    args = ap.parse_args()

    global RUNS, CKPT, SYN
    if args.smoke:
        RUNS = ROOT / "runs" / "p1_smoke"
        CKPT = ROOT / "checkpoints" / "p1_smoke"
        SYN = ROOT / "data" / "p1" / "syn_smoke"
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / "metrics.jsonl").unlink(missing_ok=True)
        (RUNS / "state.json").unlink(missing_ok=True)
        P.GENERATIONS = 2
        P.BASELINE_SEEDS = [101, 202]
        P.CHAIN_SEEDS = [101]
        P.EVAL_DOCS = 8
        P.ARITH_N = 6
        P.GEN_MAX_NEW = 40
        P.ENTROPY_PROMPTS = 4
        P.BLOCK_SIZE = 64
        P.D_MODEL = 48
        P.N_LAYER = 2
        P.BATCH_SIZE = 8
        P.N_SYN_RATIO = 0.05
        log("MODE ASAP: semua angka diperkecil; hasil TIDAK sahih")

    if args.generations:
        P.GENERATIONS = args.generations
        log(f"generasi dibatasi ke {P.GENERATIONS} (mode bertahap)")

    torch.set_num_threads(2)
    t0 = t0_text()
    docs = t0_docs()
    anchor = anchor_text()
    eval_docs = heldout_docs()[: P.EVAL_DOCS]
    budget = int(len(t0.encode("utf-8")) * P.EPOCHS)
    steps = max(1, budget // (P.BATCH_SIZE * P.BLOCK_SIZE))
    n_syn = int(len(t0.encode("utf-8")) * P.N_SYN_RATIO)

    log(f"T0-latih {len(t0.encode('utf-8'))} byte -> anggaran {budget} token, {steps} langkah/model")
    log(f"N_syn = {n_syn} byte/generasi; grid {len(P.BASELINE_SEEDS)} baseline + "
        f"{len(P.CHAIN_SEEDS)}x{len(P.ARMS)}x{P.GENERATIONS} rantai")
    if args.smoke:
        steps = 6

    lab = labels()
    st = load_state()
    done = set(st["done"])
    made = 0

    if args.stage in ("baseline", "all"):
        for seed in P.BASELINE_SEEDS:
            key = f"baseline-{seed}"
            if key in done:
                continue
            log(f"BASELINE seed={seed} ({steps} langkah)")
            row = run_one("baseline", seed, 0, "baseline", truncate_soft(t0, budget),
                          {"sumber": ["T0"], "byte_T0": budget, "byte_sintetis": 0}, steps, anchor, eval_docs)
            row["model"] = f"baseline-s{seed}"
            append_row(row)
            done.add(key)
            save_state({"done": sorted(done)})
            made += 1
            if args.limit and made >= args.limit:
                log("batas --limit tercapai"); return 0

    if args.stage in ("chains", "all"):
        for arm in P.ARMS:
            label = lab[arm]
            for seed in P.CHAIN_SEEDS:
                syn: dict[int, list[str]] = {}
                prev = None
                for gen in range(1, P.GENERATIONS + 1):
                    key = f"{arm}-{seed}-g{gen}"
                    if key in done:
                        if gen > 1:
                            got = read_syn(SYN / f"{label}-s{seed}" / f"gen{gen-1}.jsonl")
                            if got:
                                syn[gen - 1] = got
                        prev = torch.load(CKPT / f"{label}-{seed}-g{gen}.pt", map_location="cpu", weights_only=False)
                        continue
                    if args.limit and made >= args.limit:
                        log("batas --limit tercapai"); save_state({"done": sorted(done)}); return 0
                    log(f"ARM {label} (buta) seed={seed} gen={gen}")
                    if arm != "C":
                        if prev is None:
                            blob = torch.load(CKPT / f"baseline-{seed}-g0.pt", map_location="cpu", weights_only=False) \
                                if (CKPT / f"baseline-{seed}-g0.pt").exists() else None
                            if blob is None:
                                raise SystemExit("baseline untuk seed ini belum ada; jalankan --stage baseline")
                            prev = TinyGPT(cfg()); prev.load_state_dict(blob["state_dict"]); prev.eval()
                        syn[gen - 1] = generate_syn(prev, docs, n_syn, f"gen{gen-1}",
                                                    SYN / f"{label}-s{seed}")
                    text, info = compose(arm, gen, t0, syn, budget)
                    row = run_one(label, seed, gen, arm, text, {**info, "gen": gen}, steps, anchor, eval_docs)
                    append_row(row)
                    done.add(key)
                    save_state({"done": sorted(done)})
                    made += 1
                    if arm != "C":
                        blob = torch.load(CKPT / f"{label}-{seed}-g{gen}.pt", map_location="cpu", weights_only=False)
                        prev = TinyGPT(cfg()); prev.load_state_dict(blob["state_dict"]); prev.eval()
    log(f"selesai; {made} model baru. state: {len(done)} tercatat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
