"""Smoke test cepat: pipeline P0 pada skala mini (mengubah konstanta config)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src import config as C

C.T0_BYTES = 20_000
C.GEN_BYTES = 6_000
C.TRAIN_STEPS = 25
C.WARMUP = 5
C.GEN_BATCH = 8
C.GEN_MAX_NEW = 60

torch.set_num_threads(2)
from src.corpus import build_teacher_corpus, split_docs
from src.train import train_model
from src.generate import generate_corpus
from src.validate import validate_docs, sample_mixture
from src.evaluate import evaluate
from src.gate import evaluate_gate

t = time.time()
docs = build_teacher_corpus(20_000, 13)
splits = split_docs(docs)
print("T0", len(docs), "docs", sum(d.n_bytes for d in docs), "bytes")
model, tm = train_model("".join(d.text for d in splits["train"]), seed=13, steps=25)
print("train", round(tm["elapsed_s"], 1), "s loss", round(tm["final_loss"], 3))
catalog = [{"prompt_id": str(i), "domain": d.domain, "prompt": d.prompt} for i, d in enumerate(splits["train"][:60])]
raw = generate_corpus(model, catalog, 6_000, seed=1)
print("generated", len(raw), "docs", sum(len(r["text"].encode()) for r in raw), "bytes")
acc, vs = validate_docs(raw)
print("accepted", vs.accepted, "reject", round(vs.as_dict()["reject_rate"], 2))
print("reasons", {k: v for k, v in vs.as_dict().items() if k.startswith("rejected")})
print("contoh:", repr(acc[0]["text"][:200]) if acc else "TIDAK ADA")
pools = {0: [{"domain": d.domain, "text": d.text, "n_bytes": d.n_bytes, "sha256": d.sha256} for d in splits["train"]],
         1: acc}
mix, rep = sample_mixture(pools, {0: 1.0, 1: 1.0}, 6_000, seed=13)
print("mix docs", len(mix), "tvd", round(rep["mix_tvd"], 4), rep["per_generation"])
ev = evaluate(model, "".join(d.text for d in splits["test"]))
print("eval", ev)
print("gate", evaluate_gate([1.38, 1.40, 1.42, 1.39], ev["anchor_bpb"], ev, {"distinct2": 0.8, "arith_acc": 0.2}))
print(f"SMOKE OK dalam {time.time()-t:.1f}s")
