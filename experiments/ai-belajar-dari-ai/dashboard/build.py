#!/usr/bin/env python3
"""Bangun dashboard statis dari PLAN.md + status.json.

Sengaja tanpa dependensi eksternal (tanpa paket markdown, tanpa CDN) supaya
siapa pun bisa membangun ulang dashboard ini di mesin mana pun:

    python3 dashboard/build.py

Prinsip transparansi:
- Setiap berkas yang dipertontonkan dihitung hash SHA-256-nya.
- Tidak ada angka "progress" yang dikarang di UI: semuanya dibaca dari
  status.json, dan status.json ikut ditampilkan isinya.
- Dashboard tidak boleh mengklaim sesuatu yang tidak bisa dicek di repo.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # .../ai-belajar-dari-ai
DASH = ROOT / "dashboard"
PLAN = ROOT / "PLAN.md"
STATUS = DASH / "status.json"
TEMPLATE = DASH / "template.html"
STYLE = DASH / "assets" / "style.css"
APP = DASH / "assets" / "app.js"
OUT = DASH / "index.html"

TEXT_SUFFIXES = {".md", ".json", ".py", ".css", ".js", ".html", ".yaml", ".yml", ".txt", ".jsonl", ".toml"}
MAX_INLINE_BYTES = 400_000


# --------------------------------------------------------------------------
# util
# --------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9\s-]", "", text.lower())
    s = re.sub(r"\s+", "-", s.strip())
    return s or "bagian"


# --------------------------------------------------------------------------
# markdown -> html (subset yang dipakai PLAN.md)
# --------------------------------------------------------------------------
def inline(text: str) -> str:
    """Escape lalu terapkan markdown inline."""
    codes: list[str] = []

    def stash(m: re.Match[str]) -> str:
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = html.escape(text, quote=False)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" target="_blank" rel="noopener">\1</a>', text)
    # bold dulu (boleh memuat *italic* di dalamnya), baru italic sisanya
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"~~([^~]+)~~", r"<del>\1</del>", text)

    def unstash(m: re.Match[str]) -> str:
        return f"<code>{html.escape(codes[int(m.group(1))], quote=False)}</code>"

    return re.sub(r"\x00(\d+)\x00", unstash, text)


def md_to_html(md: str, demote: int = 1) -> tuple[str, list[dict]]:
    lines = md.splitlines()
    out: list[str] = []
    toc: list[dict] = []
    used: dict[str, int] = {}
    i = 0
    n = len(lines)

    def heading(level: int, text: str) -> None:
        base = slug(re.sub(r"`|\*\*|\*", "", text))
        count = used.get(base, 0)
        used[base] = count + 1
        anchor = base if count == 0 else f"{base}-{count}"
        lvl = min(6, level + demote)
        out.append(f'<h{lvl} id="{anchor}">{inline(text)}<a class="anchor" href="#{anchor}">#</a></h{lvl}>')
        if lvl <= 3:
            toc.append({"level": lvl, "anchor": anchor, "text": re.sub(r"`", "", text)})

    while i < n:
        line = lines[i]

        if line.startswith("```"):
            lang = line.strip("`").strip().lower()
            i += 1
            buf = []
            while i < n and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            body = html.escape(chr(10).join(buf), quote=False)
            if lang == "mermaid":
                # mermaid butuh library eksternal; kita tidak memuat apa pun dari internet.
                out.append(
                    '<div class="callout" style="margin:14px 0"><strong>Diagram alur.</strong> '
                    'Versi visualnya ada di <a href="#denah">bagian 03 — Yang akan dibangun</a>. '
                    'Sumber aslinya (mermaid) tetap disimpan di bawah agar tidak ada yang disembunyikan.'
                    f'<details class="raw" style="margin:10px 0 0"><summary>sumber diagram (mermaid)</summary>'
                    f"<pre><code>{body}</code></pre></details></div>"
                )
            else:
                out.append(f"<pre><code>{body}</code></pre>")
            continue

        if not line.strip():
            i += 1
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            heading(len(m.group(1)), m.group(2).strip())
            i += 1
            continue

        if re.match(r"^\s*(---|\*\*\*|___)\s*$", line):
            out.append("<hr>")
            i += 1
            continue

        # tabel
        if line.lstrip().startswith("|") and i + 1 < n and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 2
            rows = []
            while i < n and lines[i].lstrip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            out.append("<table><thead><tr>")
            out.extend(f"<th>{inline(c)}</th>" for c in header)
            out.append("</tr></thead><tbody>")
            for r in rows:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            out.append("</tbody></table>")
            continue

        # blockquote
        if line.startswith(">"):
            buf = []
            while i < n and lines[i].startswith(">"):
                buf.append(lines[i].lstrip(">").strip())
                i += 1
            out.append(f"<blockquote>{inline(' '.join(buf))}</blockquote>")
            continue

        # list
        if re.match(r"^\s*([-*+]|\d+\.)\s+", line):
            stack: list[str] = []
            while i < n and re.match(r"^\s*([-*+]|\d+\.)\s+", lines[i]):
                raw = lines[i]
                indent = len(raw) - len(raw.lstrip())
                ordered = bool(re.match(r"^\s*\d+\.\s+", raw))
                tag = "ol" if ordered else "ul"
                depth = indent // 2
                while len(stack) > depth:
                    out.append(f"</li></{stack.pop()}>")
                if len(stack) == depth:
                    if stack:
                        out.append("</li>")
                    out.append(f"<{tag}>")
                    stack.append(tag)
                body = re.sub(r"^\s*([-*+]|\d+\.)\s+", "", raw)
                out.append(f"<li>{inline(body)}")
                i += 1
            while stack:
                out.append(f"</li></{stack.pop()}>")
            continue

        buf = []
        while i < n and lines[i].strip() and not re.match(
            r"^(#{1,6}\s|\s*\||```|>|\s*([-*+]|\d+\.)\s+)", lines[i]
        ) and not re.match(r"^\s*(---|\*\*\*|___)\s*$", lines[i]):
            buf.append(lines[i].strip())
            i += 1
        out.append(f"<p>{inline(' '.join(buf))}</p>")

    return "\n".join(out), toc


# --------------------------------------------------------------------------
# inventaris berkas
# --------------------------------------------------------------------------
def inventory() -> list[dict]:
    items = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if path.name == "index.html" or "__pycache__" in rel.parts or path.name.startswith(".DS"):
            continue
        size = path.stat().st_size
        binary = path.suffix not in TEXT_SUFFIXES
        items.append(
            {
                "path": rel.as_posix(),
                "abs": str(path),
                "bytes": size,
                "size": human_bytes(size),
                "sha256": sha256(path),
                "binary": binary,
                "content": None if binary or size > MAX_INLINE_BYTES else path.read_text(encoding="utf-8", errors="replace"),
            }
        )
    return items


def esc(s: str) -> str:
    return html.escape(s, quote=True)


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------
def main() -> None:
    plan_text = PLAN.read_text(encoding="utf-8")
    plan_html, plan_toc = md_to_html(plan_text, demote=1)
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    files = inventory()

    p = status["progress"]
    meta = status["meta"]

    # --- kartu status jujur
    cards = [
        ("Rencana", f"{p['plan_pct']}%", "dokumen rencana (v" + meta["plan_version"] + ")", "ok"),
        ("Implementasi", f"{p['implementation_pct']}%", "belum ada kode eksperimen", "zero"),
        ("Model terlatih", str(p["models_trained"]), "dari target 32 chain / 13 arm", "zero"),
        ("Token digenerate", f"{p['tokens_generated']:,}".replace(",", "."), "dari target ±350 juta token sintetis", "zero"),
        ("Checkpoint", str(p["checkpoints"]), "belum ada artefak model", "zero"),
        ("Baris kode eksperimen", str(p["lines_of_experiment_code"]), "0 baris ditulis", "zero"),
    ]
    cards_html = "".join(
        f'<div class="stat {cls}"><div class="stat-label">{esc(l)}</div>'
        f'<div class="stat-value">{esc(v)}</div><div class="stat-sub">{esc(s)}</div></div>'
        for l, v, s, cls in cards
    )

    # --- fase
    badge = {
        "belum-mulai": ("belum mulai", "badge-idle"),
        "berjalan": ("berjalan", "badge-run"),
        "selesai": ("selesai", "badge-done"),
        "diblokir": ("menunggu keputusanmu", "badge-wait"),
    }
    phase_rows = []
    for ph in status["phases"]:
        label, cls = badge.get(ph["status"], (ph["status"], "badge-idle"))
        phase_rows.append(
            "<tr>"
            f'<td class="mono">{esc(ph["id"])}</td>'
            f'<td><strong>{esc(ph["name"])}</strong><div class="muted small">{esc(ph.get("desc", ""))}</div></td>'
            f'<td><span class="badge {cls}">{esc(label)}</span></td>'
            f'<td class="small">{esc(ph.get("output", "—"))}</td>'
            f'<td class="small mono">{esc(ph.get("evidence", "—"))}</td>'
            "</tr>"
        )
    phases_html = (
        "<table><thead><tr><th>Fase</th><th>Nama</th><th>Status</th>"
        "<th>Luaran</th><th>Bukti di repo</th></tr></thead><tbody>"
        + "".join(phase_rows)
        + "</tbody></table>"
    )

    # --- berkas terencana
    planned_rows = []
    for f in status["planned_files"]:
        label, cls = badge.get(f["status"], (f["status"], "badge-idle"))
        planned_rows.append(
            "<tr>"
            f'<td class="mono small">{esc(f["path"])}</td>'
            f'<td class="small">{esc(f["purpose"])}</td>'
            f'<td class="small mono">{esc(f.get("phase", "—"))}</td>'
            f'<td><span class="badge {cls}">{esc(label)}</span></td>'
            "</tr>"
        )
    planned_html = "".join(planned_rows)

    # --- keputusan
    dec_html = []
    default_note = ' <em class="muted">(default-ku)</em>'
    for d in status["decisions"]:
        opts = "".join(
            '<li><span class="mono small">{}</span> {}{}</li>'.format(
                esc(o["id"]), esc(o["label"]), default_note if o.get("default") else ""
            )
            for o in d["options"]
        )
        dec_html.append(
            f'<article class="card decision"><div class="decision-head">'
            f'<span class="badge badge-wait">{esc(d["status"])}</span>'
            f'<h4>{esc(d["question"])}</h4></div>'
            f'<ul class="tight">{opts}</ul>'
            f'<p class="small muted">Dampak: {esc(d["impact"])}</p></article>'
        )
    decisions_html = "".join(dec_html)

    # --- changelog
    log_html = []
    for c in sorted(status["changelog"], key=lambda x: x["date"], reverse=True):
        log_html.append(
            '<li class="log-item">'
            f'<time class="mono small">{esc(c["date"])}</time>'
            f'<div><strong>{esc(c["title"])}</strong>'
            f'<p class="small">{esc(c["detail"])}</p>'
            f'<p class="small mono muted">bukti: {esc(c["evidence"])}</p></div></li>'
        )
    changelog_html = "<ul class='log'>" + "".join(log_html) + "</ul>"

    # --- klaim vs verifikasi
    claims_html = "".join(
        f'<tr><td class="small">{esc(c["claim"])}</td>'
        f'<td class="small mono cmd" data-cmd="{esc(c["verify"])}">{esc(c["verify"])}</td></tr>'
        for c in status["claims"]
    )

    # --- bias
    bias_html = "".join(
        f'<li><strong>{esc(b["title"])}</strong><p class="small">{esc(b["detail"])}</p></li>'
        for b in status["bias_notes"]
    )

    # --- literatur
    lit_html = "".join(
        f'<li><a href="{esc(l["url"])}" target="_blank" rel="noopener">{esc(l["title"])}</a> '
        f'<span class="muted small">— {esc(l["authors"])}, {esc(str(l["year"]))}</span>'
        f'<p class="small muted">{esc(l["why"])}</p></li>'
        for l in status["literature"]
    )

    # --- inventaris berkas + penampil isi
    file_rows = []
    viewers = []
    for f in files:
        view = "teks" if f["content"] is not None else "biner/terlalu besar"
        file_rows.append(
            "<tr>"
            f'<td class="mono small">{esc(f["path"])}</td>'
            f'<td class="small mono">{esc(f["size"])}</td>'
            f'<td class="small mono hash">{esc(f["sha256"][:16])}…</td>'
            f'<td class="small">{view}</td>'
            "</tr>"
        )
        if f["content"] is not None:
            viewers.append(
                f'<details class="viewer"><summary class="mono">{esc(f["path"])} '
                f'<span class="muted small">· {esc(f["size"])} · sha256 {esc(f["sha256"][:12])}…</span></summary>'
                f'<pre><code>{html.escape(f["content"], quote=False)}</code></pre></details>'
            )
    files_html = (
        "<table><thead><tr><th>Berkas</th><th>Ukuran</th><th>SHA-256</th><th>Isi</th></tr></thead><tbody>"
        + "".join(file_rows)
        + "</tbody></table>"
    )
    viewers_html = "".join(viewers)

    # --- daftar isi plan
    toc_html = "".join(
        f'<a class="toc-l{item["level"]}" href="#{item["anchor"]}">{esc(item["text"])}</a>'
        for item in plan_toc
        if item["level"] <= 3
    )

    build_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    plan_hash = sha256(PLAN)
    status_hash = sha256(STATUS)

    replacements = {
        "{{TITLE}}": esc(meta["project"]),
        "{{IMPL_PCT}}": str(p["implementation_pct"]),
        "{{LOC}}": str(p["lines_of_experiment_code"]),
        "{{MODELS}}": str(p["models_trained"]),
        "{{BUILD_TIME}}": build_time,
        "{{PLAN_VERSION}}": esc(meta["plan_version"]),
        "{{PLAN_STATUS}}": esc(meta["plan_status"]),
        "{{BRANCH}}": esc(meta["branch"]),
        "{{ROOT_ABS}}": esc(str(ROOT)),
        "{{PLAN_HASH}}": plan_hash,
        "{{STATUS_HASH}}": status_hash,
        "{{PLAN_LINES}}": str(len(plan_text.splitlines())),
        "{{PLAN_TOKENS}}": f"{len(plan_text.split()):,}".replace(",", "."),
        "{{CARDS}}": cards_html,
        "{{PHASES}}": phases_html,
        "{{PLANNED_FILES}}": planned_html,
        "{{DECISIONS}}": decisions_html,
        "{{CHANGELOG}}": changelog_html,
        "{{CLAIMS}}": claims_html,
        "{{BIAS}}": bias_html,
        "{{LITERATURE}}": lit_html,
        "{{FILES}}": files_html,
        "{{VIEWERS}}": viewers_html,
        "{{PLAN_HTML}}": plan_html,
        "{{PLAN_TOC}}": toc_html,
        "{{FILE_COUNT}}": str(len(files)),
        "{{TOTAL_BYTES}}": human_bytes(sum(f["bytes"] for f in files)),
        "{{HEADLINE}}": esc(meta["headline"]),
        "{{READ_ME_FIRST}}": esc(meta["read_me_first"]),
    }

    page = TEMPLATE.read_text(encoding="utf-8")
    css = STYLE.read_text(encoding="utf-8")
    js = APP.read_text(encoding="utf-8")
    page = page.replace("{{CSS}}", css).replace("{{JS}}", js)
    for key, value in replacements.items():
        if key == "{{VIEWERS}}":
            continue  # disisipkan terakhir: isinya berkas mentah, jangan ikut di-substitusi
        page = page.replace(key, value)
    page = page.replace("{{VIEWERS}}", viewers_html)

    OUT.write_text(page, encoding="utf-8")
    print(f"OK  {OUT}")
    print(f"    berkas diinventarisasi : {len(files)}")
    print(f"    sha256 PLAN.md         : {plan_hash[:16]}…")
    print(f"    sha256 status.json     : {status_hash[:16]}…")


if __name__ == "__main__":
    main()
