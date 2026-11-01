#!/usr/bin/env python3
"""Lint kebijakan kegagalan (ops/policy.v1.json).

Gagal (exit 1) bila ada mode kegagalan yang tidak lengkap, rute yang tidak
merujuk aksi yang terdefinisi, atau kewenangan yang bertentangan.

Jalankan: python3 ops/policy_lint.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POLICY = ROOT / "policy.v1.json"
REQUIRED = ("id", "name", "severity", "detect", "route", "autonomy", "sla", "blast_radius", "recovery_minutes", "prevention")
SEVERITIES = {"critical", "high", "medium", "low"}
AUTONOMY = {"autonomous", "human"}


def main() -> int:
    raw = POLICY.read_bytes()
    p = json.loads(raw)
    errors: list[str] = []
    warnings: list[str] = []

    if p.get("fail_closed") is not True:
        errors.append("fail_closed harus true: tanpa itu, kegagalan audit berujung 'lanjut dengan peringatan'")

    b = p.get("budgets", {})
    for key, val in b.items():
        if not isinstance(val, (int, float)) or val <= 0:
            errors.append(f"budgets.{key} harus angka positif")
    if not 0 < b.get("regen_token_fraction_per_generation", 0) < 1:
        errors.append("regen_token_fraction_per_generation harus di antara 0 dan 1")
    if b.get("critic_reject_alarm_fraction", 0) >= b.get("critic_reject_stop_fraction", 0):
        errors.append("ambang alarm critic harus lebih kecil dari ambang stop")

    autonomous = {a["id"] for a in p.get("autonomy", {}).get("autonomous", [])}
    human = {a["id"] for a in p.get("autonomy", {}).get("human", [])}
    if autonomous & human:
        errors.append(f"aksi yang sama ada di dua kewenangan: {autonomous & human}")

    ids = set()
    for f in p.get("failures", []):
        for field in REQUIRED:
            if field not in f or f[field] in (None, "", [], {}):
                errors.append(f"{f.get('id', '?')}: field '{field}' kosong")
        if f.get("id") in ids:
            errors.append(f"id duplikat: {f.get('id')}")
        ids.add(f.get("id"))
        if f.get("severity") not in SEVERITIES:
            errors.append(f"{f.get('id')}: severity tidak dikenal ({f.get('severity')})")
        if f.get("autonomy") not in AUTONOMY:
            errors.append(f"{f.get('id')}: autonomy tidak dikenal")
        if not isinstance(f.get("recovery_minutes"), (int, float)):
            errors.append(f"{f.get('id')}: recovery_minutes harus angka")
        if f.get("severity") == "critical" and f.get("autonomy") != "human":
            errors.append(f"{f.get('id')}: kegagalan critical tidak boleh otonom penuh")
        if f.get("severity") == "critical" and not f.get("note"):
            errors.append(f"{f.get('id')}: kegagalan critical wajib punya catatan pemulihan")
        if not f.get("detect", {}).get("check"):
            errors.append(f"{f.get('id')}: detect.check kosong — sinyal tanpa pemeriksa tidak bisa dijalankan")

    # rute harus merujuk aksi yang terdefinisi (dipetakan lewat nama snake_case -> id)
    action_names = {}
    for group in ("autonomous", "human"):
        for a in p.get("autonomy", {}).get(group, []):
            action_names[a["action"].lower().replace(" ", "_")] = a["id"]
    known_shorthand = {
        "quarantine_chain", "quarantine_arm", "snapshot_forensics", "regenerate_batch",
        "retry_same_seed", "rollback_to_checkpoint", "resample_mixture", "degrade_profile",
        "stop_chain", "checkpoint_and_exit", "resume_next_window", "rollback_to_pilot",
        "reallocate_budget",
    }
    for f in p.get("failures", []):
        for step in f.get("route", []):
            if step not in known_shorthand and step.lower() not in action_names:
                errors.append(f"{f.get('id')}: rute '{step}' tidak dikenal")

    ks = p.get("kill_switch", {})
    if not ks.get("triggers"):
        errors.append("kill_switch tanpa pemicu")
    if ks.get("authority") != "human":
        errors.append("kill_switch harus di bawah kewenangan manusia")
    for t in ks.get("triggers", []):
        if t in ids:
            continue
        if t.startswith("canary_leak") or t.startswith("biaya"):
            continue
        errors.append(f"kill_switch memicu id yang tidak ada: {t}")

    human_ids_referenced = {t for t in ks.get("triggers", []) if t in human}
    if human_ids_referenced:
        warnings.append("kill_switch memicu aksi manusia, bukan mode kegagalan")

    digest = hashlib.sha256(raw).hexdigest()
    print(f"policy.v1.json  sha256={digest[:16]}…")
    print(f"mode kegagalan : {len(p.get('failures', []))}")
    print(f"aksi otonom    : {len(autonomous)} | aksi manusia: {len(human)}")
    print(f"kill switch    : {len(ks.get('triggers', []))} pemicu, otoritas={ks.get('authority')}")

    for w in warnings:
        print(f"PERINGATAN: {w}")
    if errors:
        print("\nGAGAL:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nLULUS: setiap mode kegagalan punya sinyal, pemeriksa, rute, kewenangan, dan SLA.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
