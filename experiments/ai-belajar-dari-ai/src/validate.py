"""Validasi, dedup, canary, dan pencampuran data (PLAN §7)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .common import rng_for, sha256_text
from .config import MAX_DOC_BYTES, MIN_DOC_BYTES

DOC_RE = re.compile(r"<D>([^<]+)</D>")


@dataclass
class ValidationStats:
    total: int = 0
    accepted: int = 0
    rejected_short: int = 0
    rejected_long: int = 0
    rejected_missing_marker: int = 0
    rejected_duplicate: int = 0
    truncated: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    per_domain: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "accepted": self.accepted,
            "rejected_short": self.rejected_short,
            "rejected_long": self.rejected_long,
            "rejected_missing_marker": self.rejected_missing_marker,
            "rejected_duplicate": self.rejected_duplicate,
            "truncated_no_end_marker": self.truncated,
            "reject_rate": 1 - (self.accepted / self.total) if self.total else 0.0,
            "bytes_in": self.bytes_in,
            "bytes_out": self.bytes_out,
            "per_domain": self.per_domain,
        }


def validate_docs(docs: list[dict]) -> tuple[list[dict], ValidationStats]:
    """docs: [{text, domain, prompt, ...}] -> yang lolos + statistik."""
    stats = ValidationStats()
    seen: set[str] = set()
    out: list[dict] = []
    for d in docs:
        stats.total += 1
        text = d.get("text", "")
        n = len(text.encode("utf-8"))
        stats.bytes_in += n
        if n < MIN_DOC_BYTES:
            stats.rejected_short += 1
            continue
        if n > MAX_DOC_BYTES:
            stats.rejected_long += 1
            continue
        # P0 memakai kriteria lenient: penanda awal + bagian jawaban wajib ada,
        # penutup <E> tidak wajib (model 0,48 jt sering berhenti di tengah).
        # Dokumen tanpa <E> dihitung sebagai 'truncated' dan dilaporkan.
        if not (text.startswith("<D>") and "</P>" in text and "<A>" in text):
            stats.rejected_missing_marker += 1
            continue
        if not text.rstrip().endswith("<E>"):
            stats.truncated += 1
        h = sha256_text(text)
        if h in seen:
            stats.rejected_duplicate += 1
            continue
        seen.add(h)
        m = DOC_RE.match(text)
        domain = m.group(1) if m else d.get("domain", "?")
        rec = dict(d)
        rec.update({"sha256": h, "n_bytes": n, "domain": domain})
        out.append(rec)
        stats.accepted += 1
        stats.bytes_out += n
        stats.per_domain[domain] = stats.per_domain.get(domain, 0) + 1
    return out, stats


def sample_mixture(
    pools: dict[int, list[dict]],
    weights: dict[int, float],
    target_bytes: int,
    seed: int,
) -> tuple[list[dict], dict]:
    """Sampling deterministik per (generasi, domain) dengan kuota proporsional.

    pools: {generasi: docs}; weights: {generasi: bobot} (dinormalisasi di sini).
    """
    total_w = sum(weights.get(g, 0) for g in pools) or 1.0
    chosen: list[dict] = []
    report: dict = {"target_bytes": target_bytes, "per_generation": {}, "mix_tvd": None}

    for g in sorted(pools):
        w = weights.get(g, 0.0) / total_w
        quota = int(w * target_bytes)
        docs = pools[g]
        by_domain: dict[str, list[dict]] = {}
        for d in docs:
            by_domain.setdefault(d["domain"], []).append(d)
        rng = rng_for("mix", g, seed)
        for dom in by_domain:
            by_domain[dom].sort(key=lambda x: sha256_text(f"{seed}|{x['sha256']}"))
            rng.shuffle(by_domain[dom])
        domains = sorted(by_domain)
        idx = {d: 0 for d in domains}
        taken = 0
        got = 0
        while taken < quota and any(idx[d] < len(by_domain[d]) for d in domains):
            for d in domains:
                if taken >= quota:
                    break
                if idx[d] < len(by_domain[d]):
                    doc = by_domain[d][idx[d]]
                    idx[d] += 1
                    if taken + doc["n_bytes"] > quota and got > 0:
                        continue
                    chosen.append(dict(doc, source_gen=g))
                    taken += doc["n_bytes"]
                    got += 1
        report["per_generation"][g] = {"target": quota, "realized_bytes": taken, "docs": got}

    realized: dict[int, int] = {}
    for d in chosen:
        realized[d["source_gen"]] = realized.get(d["source_gen"], 0) + d["n_bytes"]
    total_realized = sum(realized.values())

    def share(dct: dict[int, int]) -> dict[int, float]:
        tot = sum(dct.values()) or 1
        return {k: v / tot for k, v in dct.items()}

    target_share = {g: weights.get(g, 0.0) / total_w for g in pools}
    realized_share = share(realized)
    keys = set(target_share) | set(realized_share)
    tvd = 0.5 * sum(abs(target_share.get(k, 0) - realized_share.get(k, 0)) for k in keys)

    report["realized_share"] = realized_share
    report["target_share"] = target_share
    report["mix_tvd"] = tvd
    report["total_bytes"] = total_realized
    return chosen, report
