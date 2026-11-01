#!/usr/bin/env python3
"""Orkestrasi P0 — rantai lengkap: guru -> T0 -> M0 -> generasi -> train -> evaluasi -> gate.

P0 memakai state machine kecil dengan semantik node yang sama seperti graf LangGraph
di PLAN §8 (generate -> validate -> freeze -> train -> evaluate -> gate). Penggantian
ke LangGraph asli dijadwalkan pada P1; semantik dan nama node sengaja dibuat identik.

Resume: setiap langkah menulis state + artefak beralamat-konten; langkah selesai
dilewati saat dijalankan ulang.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from statistics import mean

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C
from src.common import CKPT, DATA, RUNS, ensure_dirs, read_json, rng_for, sha256_text, write_json, write_jsonl
from src.corpus import build_teacher_corpus, split_docs
from src.evaluate import evaluate
from src.gate import evaluate_gate
from src.generate import generate_corpus
from src.model import TinyGPT, XSConfig
from src.train import train_model
from src.validate import sample_mixture, validate_docs

STATE = RUNS / "state.json"
METRICS = RUNS / "metrics.jsonl"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_state() -> dict:
    return read_json(STATE) if STATE.exists() else {"done": {}, "data": {}}


def save_state(st: dict) -> None:
    write_json(STATE, st)


def cfg_from(d: dict) -> XSConfig:
    return XSConfig(block_size=d["block_size"], n_layer=d["n_layer"], n_head=d["n_head"], d_model=d["d_model"])


def save_model(model: TinyGPT, key: str) -> Path:
    path = CKPT / f"{key}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cfg": model.cfg.__dict__,
            "n_params": model.n_params,
        },
        path,
    )
    return path


def load_model(key: str) -> TinyGPT:
    blob = torch.load(CKPT / f"{key}.pt", map_location="cpu", weights_only=False)
    model = TinyGPT(cfg_from(blob["cfg"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def append_metrics(row: dict) -> None:
    with METRICS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def step_t0(st: dict) -> dict:
    if "t0" in st["done"]:
        return read_json(DATA / "t0" / "meta.json")
    log("node=plan_prompts/generate(T0) — guru menulis korpus")
    docs = build_teacher_corpus(C.T0_BYTES, seed=13)
    splits = split_docs(docs)
    t0dir = DATA / "t0"
    t0dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(t0dir / "manifest.jsonl", [
        {"doc_id": d.doc_id, "domain": d.domain, "prompt": d.prompt, "n_bytes": d.n_bytes,
         "sha256": d.sha256, "split": ("train" if d in splits["train"] else "valid" if d in splits["valid"] else "test")}
        for d in docs
    ])
    for name in ("train", "valid", "test"):
        (t0dir / f"{name}.txt").write_text("".join(d.text for d in splits[name]), encoding="utf-8")

    catalog, seen = [], set()
    for d in splits["train"]:
        key = sha256_text(f"{d.domain}|{d.prompt}")[:12]
        if key in seen:
            continue
        seen.add(key)
        catalog.append({"prompt_id": key, "domain": d.domain, "prompt": d.prompt})
    write_jsonl(t0dir / "prompts.jsonl", catalog)

    meta = {
        "docs": len(docs),
        "bytes": sum(d.n_bytes for d in docs),
        "splits": {k: len(v) for k, v in splits.items()},
        "catalog_size": len(catalog),
        "sha_train": sha256_text((t0dir / "train.txt").read_text(encoding="utf-8")),
        "sha_test": sha256_text((t0dir / "test.txt").read_text(encoding="utf-8")),
    }
    write_json(t0dir / "meta.json", meta)
    st["done"]["t0"] = True
    st["data"]["t0"] = meta
    save_state(st)
    log(f"T0 selesai: {meta['docs']} dokumen, {meta['bytes']} byte, split {meta['splits']}")
    return meta


def train_and_eval(key: str, text: str, seed: int, anchor: str, base_d2: float | None,
                   extra: dict) -> tuple[dict, dict]:
    log(f"node=train {key} (seed={seed}, {len(text)} byte)")
    model, tmetrics = train_model(text, seed=seed)
    save_model(model, key)
    log(f"  selesai: loss {tmetrics['final_loss']:.3f}, {tmetrics['elapsed_s']:.0f}s, "
        f"{model.n_params/1e6:.3f} jt parameter")
    emetrics = evaluate(model, anchor, baseline_distinct2=base_d2)
    row = {"model": key, **extra, "train": tmetrics, "eval": emetrics}
    return row, emetrics


def main() -> None:
    ensure_dirs()
    torch.set_num_threads(2)
    st = load_state()
    t0meta = step_t0(st)

    anchor = (DATA / "t0" / "test.txt").read_text(encoding="utf-8")
    train_t0 = (DATA / "t0" / "train.txt").read_text(encoding="utf-8")
    catalog = [json.loads(l) for l in (DATA / "t0" / "prompts.jsonl").read_text(encoding="utf-8").splitlines()]

    # ---------- baseline M0 (untuk sigma0) ----------
    baseline_rows = []
    for seed in C.BASELINE_SEEDS:
        key = f"base-s{seed}-M0"
        marker = f"baseline-{seed}"
        if marker not in st["done"]:
            row, _ = train_and_eval(key, train_t0, seed, anchor, None, {"arm": "baseline", "seed": seed, "gen": 0})
            append_metrics(row)
            st["done"][marker] = True
            st["data"][marker] = row["eval"]
            save_state(st)
        baseline_rows.append(st["data"][marker])

    base_bpbs = [r["anchor_bpb"] for r in baseline_rows]
    base_d2 = mean(r["distinct2"] for r in baseline_rows)
    base_arith = mean(r["arith_acc"] for r in baseline_rows)
    log(f"baseline: bpb {['%.4f' % b for b in base_bpbs]} -> mu={mean(base_bpbs):.4f}, sigma={torch.tensor(base_bpbs).std().item():.4f}")

    # ---------- rantai per arm ----------
    for arm in C.ARMS:
        for seed in C.CHAIN_SEEDS:
            prev_key = f"base-s{seed}-M0"
            for k in range(1, C.GENERATIONS + 1):
                marker = f"{arm}-s{seed}-g{k}"
                if marker in st["done"]:
                    prev_key = f"{arm}-s{seed}-M{k}"
                    continue

                log(f"node=generate {marker} dari {prev_key}")
                model = load_model(prev_key)
                raw = generate_corpus(model, catalog, C.GEN_BYTES, seed=seed * 100 + k, verbose=False)
                accepted, vstats = validate_docs(raw)
                gen_dir = DATA / "gen"
                gen_dir.mkdir(parents=True, exist_ok=True)
                write_jsonl(gen_dir / f"{marker}.jsonl", [
                    {kk: d[kk] for kk in ("domain", "prompt_id", "prompt", "text", "n_bytes", "sha256")} for d in accepted
                ])
                vr = vstats.as_dict()
                log(f"  validasi: {vstats.accepted}/{vstats.total} diterima "
                    f"(reject {vr['reject_rate']:.0%}, truncated {vr['truncated_no_end_marker']}), "
                    f"{vstats.bytes_out} byte")

                pools: dict[int, list[dict]] = {}
                t0_docs = [
                    {"domain": d.domain, "text": d.text, "n_bytes": d.n_bytes, "sha256": d.sha256}
                    for d in split_docs(build_teacher_corpus(C.T0_BYTES, seed=13))["train"]
                ]
                if arm == "A":
                    train_docs = accepted
                    mix_report = {"arm": "A", "note": "100% keluaran model sebelumnya", "mix_tvd": 0.0}
                else:  # B1 accumulate-proportional
                    pools[0] = t0_docs
                    for j in range(1, k + 1):
                        p = gen_dir / f"{arm}-s{seed}-g{j}.jsonl"
                        if p.exists():
                            pools[j] = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
                    weights = {g: 1.0 for g in pools}
                    train_docs, mix_report = sample_mixture(pools, weights, C.GEN_BYTES, seed)
                text = "".join(d["text"] if d["text"].endswith("\n") else d["text"] + "\n" for d in train_docs)

                key = f"{arm}-s{seed}-M{k}"
                row, emetrics = train_and_eval(key, text, seed * 100 + k, anchor, base_d2,
                                               {"arm": arm, "seed": seed, "gen": k})
                gate = evaluate_gate(base_bpbs, emetrics["anchor_bpb"], emetrics,
                                     {"distinct2": base_d2, "arith_acc": base_arith})
                row.update({"validate": vstats.as_dict(), "mix": mix_report, "gate": gate,
                            "tokens_generated": vstats.bytes_in, "corpus_bytes": len(text)})
                append_metrics(row)
                st["done"][marker] = True
                st["data"][marker] = {"eval": emetrics, "gate": gate, "mix_tvd": mix_report.get("mix_tvd"),
                                      "reject_rate": vstats.as_dict()["reject_rate"]}
                save_state(st)
                log(f"  {key}: bpb={emetrics['anchor_bpb']:.4f} delta={gate['delta_bpb']:+.4f} "
                    f"thr={gate['threshold_bpb']:.4f} -> {gate['status']} ({gate['route']})")
                prev_key = key

    log("semua langkah selesai")
    write_json(RUNS / "done.json", {"finished": True, "steps": len(st["done"])})


if __name__ == "__main__":
    main()
