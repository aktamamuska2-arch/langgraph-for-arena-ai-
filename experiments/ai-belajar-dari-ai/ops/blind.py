#!/usr/bin/env python3
"""Protokol buta untuk P1 (PREREGISTRATION.md bagian 8).

    python3 ops/blind.py seal        # acak label arm, simpan kunci di .blind/, tulis komitmen
    python3 ops/blind.py status      # tunjukkan komitmen (bukan kunci)
    python3 ops/blind.py reveal --conclusions-sha256 <hash>

Kunci disimpan di `.blind/key.json` yang tidak masuk git. Yang di-commit hanya
komitmen sha256-nya, sehingga urutan seal -> analisis -> reveal bisa diperiksa
walau tidak bisa dipaksakan secara kriptografis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEY = ROOT / ".blind" / "key.json"
COMMIT = ROOT / "ops" / "blind_commitment.json"
EVENTS = ROOT / "ops" / "blind_events.jsonl"
ARMS = ["A", "B1", "C"]


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def log_event(kind: str, **kw) -> None:
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": now(), "event": kind, **kw}, ensure_ascii=False) + "\n")


def seal() -> int:
    if COMMIT.exists():
        print("GAGAL: komitmen buta sudah ada. Menyegel ulang akan merusak protokol.")
        return 1
    if KEY.exists():
        print("GAGAL: .blind/key.json sudah ada tanpa komitmen — keadaan tidak konsisten.")
        return 1
    labels = ARMS[:]
    secrets.SystemRandom().shuffle(labels)
    mapping = {arm: f"X{i + 1}" for i, arm in enumerate(labels)}
    KEY.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"created_at": now(), "mapping": mapping}, ensure_ascii=False, sort_keys=True).encode()
    KEY.write_bytes(payload)
    COMMIT.write_text(json.dumps({
        "sealed_at": now(),
        "n_arms": len(ARMS),
        "key_sha256": sha256_bytes(payload),
        "key_path": str(KEY.relative_to(ROOT)),
        "note": "Kunci tidak di-commit. Hash ini membuktikan pemetaan tidak diubah setelah analisis.",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log_event("seal", key_sha256=sha256_bytes(payload))
    print(f"TERSEGEL: {len(ARMS)} arm, komitmen {COMMIT.relative_to(ROOT)}")
    print(f"  key sha256 = {sha256_bytes(payload)}")
    return 0


def status() -> int:
    if not COMMIT.exists():
        print("belum tersegel")
        return 0
    rec = json.loads(COMMIT.read_text(encoding="utf-8"))
    print(f"tersegel {rec['sealed_at']}  n_arm={rec['n_arms']}  key_sha256={rec['key_sha256']}")
    if KEY.exists():
        got = sha256_bytes(KEY.read_bytes())
        print("  kunci ada di disk, cocok" if got == rec["key_sha256"] else "  PERINGATAN: kunci tidak cocok dengan komitmen")
    return 0


def reveal(conclusions_sha: str) -> int:
    path = ROOT / "analysis" / "p1_conclusions.md"
    if not path.exists():
        print("GAGAL: analysis/p1_conclusions.md belum ada — kesimpulan harus ditulis lebih dulu.")
        return 1
    actual = sha256_bytes(path.read_bytes())
    if actual != conclusions_sha:
        print("GAGAL: hash kesimpulan tidak cocok. Unblinding ditolak.")
        print(f"  diberikan : {conclusions_sha}")
        print(f"  sebenarnya: {actual}")
        return 1
    if not KEY.exists():
        print("GAGAL: kunci tidak ada di disk.")
        return 1
    payload = json.loads(KEY.read_text(encoding="utf-8"))
    log_event("reveal", conclusions_sha256=actual, key_sha256=sha256_bytes(KEY.read_bytes()))
    print("kesimpulan terkunci:", actual)
    for arm, label in sorted(payload["mapping"].items()):
        print(f"  {label} = arm {arm}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["seal", "status", "reveal"])
    ap.add_argument("--conclusions-sha256")
    args = ap.parse_args()
    if args.command == "seal":
        return seal()
    if args.command == "status":
        return status()
    if not args.conclusions_sha256:
        print("GAGAL: reveal butuh --conclusions-sha256")
        return 1
    return reveal(args.conclusions_sha256)


if __name__ == "__main__":
    sys.exit(main())
