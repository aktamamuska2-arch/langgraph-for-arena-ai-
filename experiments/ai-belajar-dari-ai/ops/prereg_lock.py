#!/usr/bin/env python3
"""Kunci dan verifikasi PREREGISTRATION.md.

    python3 ops/prereg_lock.py --lock     # kunci (menolak menimpa tanpa --amend)
    python3 ops/prereg_lock.py --verify   # gagal bila berkas berubah sejak dikunci

Alasannya sederhana: pra-registrasi yang bisa diedit diam-diam tidak lebih baik
daripada tidak ada pra-registrasi sama sekali.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "PREREGISTRATION.md"
LOCK = ROOT / "ops" / "prereg.lock.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def doc_version(text: str) -> str:
    m = re.search(r"\|\s*Versi\s*\|\s*\**([0-9]+\.[0-9]+)\**", text)
    return m.group(1) if m else "?"


def lock(amend: str | None) -> int:
    if not DOC.exists():
        print("GAGAL: PREREGISTRATION.md tidak ada")
        return 1
    text = DOC.read_text(encoding="utf-8")
    digest = sha256(DOC)
    entry = {
        "file": DOC.name,
        "version": doc_version(text),
        "sha256": digest,
        "bytes": DOC.stat().st_size,
        "lines": len(text.splitlines()),
        "locked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    history = []
    if LOCK.exists():
        old = json.loads(LOCK.read_text(encoding="utf-8"))
        if old.get("sha256") == digest:
            print(f"SUDAH TERKUNCI: {digest}")
            return 0
        if not amend:
            print("GAGAL: PREREGISTRATION.md berubah setelah dikunci.")
            print(f"  hash lama : {old.get('sha256')}")
            print(f"  hash baru : {digest}")
            print("  Amandemen wajib: naikkan versi + isi tabel amandemen, lalu jalankan --amend <versi>.")
            return 1
        history = old.get("history", []) + [
            {k: old.get(k) for k in ("version", "sha256", "locked_at")} | {"superseded": entry["locked_at"]}
        ]

    LOCK.write_text(
        json.dumps({**entry, "history": history}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"TERKUNCI v{entry['version']}  sha256={digest}  ({entry['bytes']} B, {entry['lines']} baris)")
    return 0


def verify(quiet: bool = False) -> int:
    if not LOCK.exists():
        print("GAGAL: ops/prereg.lock.json tidak ada — pra-registrasi belum dikunci")
        return 1
    record = json.loads(LOCK.read_text(encoding="utf-8"))
    if not DOC.exists():
        print("GAGAL: PREREGISTRATION.md hilang padahal kuncinya ada")
        return 1
    digest = sha256(DOC)
    if digest != record.get("sha256"):
        print("GAGAL: PREREGISTRATION.md tidak cocok dengan kunci.")
        print(f"  terkunci : {record.get('sha256')} (v{record.get('version')})")
        print(f"  sekarang : {digest}")
        print("  Setiap perubahan wajib jadi versi baru dengan entri di tabel amandemen.")
        return 1
    if not quiet:
        print(f"OK: pra-registrasi utuh — v{record['version']} sha256={digest}")
        print(f"    dikunci {record['locked_at']}, {record['lines']} baris")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--lock", action="store_true")
    g.add_argument("--verify", action="store_true")
    ap.add_argument("--amend", metavar="VERSI", help="wajib saat mengunci ulang setelah ada perubahan")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if args.lock:
        return lock(args.amend)
    return verify(args.quiet)


if __name__ == "__main__":
    sys.exit(main())
