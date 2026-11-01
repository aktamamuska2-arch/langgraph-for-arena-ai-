#!/usr/bin/env python3
"""Antarmuka uji manusia untuk model hasil pipeline (P0 dan seterusnya).

Tujuan: kamu bisa menilai sendiri, tanpa membaca kode dan tanpa percaya laporan,
apakah keluaran model memang memburuk antar generasi.

    python tests/ui_inspect.py            # web lokal  http://localhost:8010
    python tests/ui_inspect.py --cli      # mode terminal interaktif
    python tests/ui_inspect.py --port 9000 --bind 0.0.0.0

Fitur:
  1. Pemilihan checkpoint dari M0 baseline sampai generasi mana pun yang ada.
  2. Prompt bebas -> teks mentah yang dihasilkan model (bukan ringkasan, bukan versi bersih).
  3. Penilaian cepat: metrik yang nanti dipakai R8 (degenerate, truncated, TTR, dist-2 8-gram,
     kompresi zlib, repetisi), plus status gate checkpoint itu dari metrik mentah.

Tanpa dependensi eksternal selain torch (sudah dipakai pipeline) dan pustaka standar.
Tidak ada CDN, tidak ada aset eksternal: seluruh halaman dilayani dari proses ini.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import threading
import zlib
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CKPT_DIRS = [ROOT / "checkpoints" / "p0", ROOT / "checkpoints" / "p1"]
METRICS_FILES = [ROOT / "runs" / "p0" / "metrics.jsonl", ROOT / "runs" / "p1" / "metrics.jsonl"]

_LOCK = threading.Lock()
_CACHE: dict[str, object] = {}


# ---------------------------------------------------------------- checkpoint
def list_checkpoints() -> list[dict]:
    out = []
    meta = {}
    for mf in METRICS_FILES:
        if not mf.exists():
            continue
        for line in mf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            meta[row["model"]] = {
                "arm": row.get("arm"),
                "seed": row.get("seed"),
                "gen": row.get("gen"),
                "bpb": row.get("eval", {}).get("anchor_bpb"),
                "gate": row.get("gate", {}),
            }
    for d in CKPT_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.pt")):
            key = p.stem
            m = meta.get(key, {})
            out.append({
                "key": key,
                "size_mb": round(p.stat().st_size / 1e6, 2),
                "arm": m.get("arm", "?"),
                "seed": m.get("seed"),
                "gen": m.get("gen"),
                "bpb": m.get("bpb"),
                "gate_status": (m.get("gate") or {}).get("status"),
                "delta": (m.get("gate") or {}).get("delta_bpb"),
            })
    order = {"baseline": 0, "A": 1, "B1": 2}
    out.sort(key=lambda r: (order.get(r["arm"], 9), r.get("seed") or 0, r.get("gen") or 0))
    return out


def load_model(key: str):
    from src.model import TinyGPT
    from src.orchestrate import cfg_from

    if key in _CACHE:
        return _CACHE[key]
    path = None
    for d in CKPT_DIRS:
        cand = d / f"{key}.pt"
        if cand.exists():
            path = cand
            break
    if path is None:
        raise FileNotFoundError(f"checkpoint {key} tidak ditemukan")
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = TinyGPT(cfg_from(blob["cfg"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    if len(_CACHE) > 3:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = model
    return model


# ---------------------------------------------------------------- metrik
def _words(text: str) -> list[str]:
    return [w.strip(".,;:!?<>/()[]{}'\"") for w in text.lower().split() if w.strip(".,;:!?<>/()[]{}'\"")]


def assess(text: str) -> dict:
    """Metrik cepat untuk teks yang baru diuji — keluarga yang sama dengan R8."""
    t = text.strip()
    # Metrik leksikal dihitung pada ISI JAWABAN saja (bagian <A>), bukan pada markup dan
    # bukan pada domain/prompt: keduanya bagian dari antarmuka, bukan keluaran model.
    # Kalau ikut dihitung, TTR/distinct-2 bisa tampak bagus hanya karena tag atau prompt berulang.
    region = t
    if "<A>" in region:
        region = region.split("<A>", 1)[1]
    if "</A>" in region:
        region = region.split("</A>", 1)[0]
    plain = re.sub(r"<[^>]+>", " ", region)
    w = _words(plain)
    grams8 = [tuple(w[i:i + 8]) for i in range(max(0, len(w) - 7))]
    bigrams = list(zip(w, w[1:]))
    raw = t.encode("utf-8")
    return {
        "byte": len(raw),
        "kata": len(w),
        "ada_A": "<A>" in t,
        "berakhir_E": t.endswith("<E>"),
        "degenerate": not ("<A>" in t and "<E>" in t),
        "ttr": round(len(set(w)) / max(1, len(w)), 3),
        "distinct2": round(len(set(bigrams)) / max(1, len(bigrams)), 3),
        "repetisi8": (round(1 - len(set(grams8)) / len(grams8), 3) if grams8 else None),
        "zlib_bpb": round(len(zlib.compress(raw, 9)) * 8 / max(1, len(raw)), 3),
        "kata_terbanyak": Counter(w).most_common(3),
        "isi_dihitung": plain.strip()[:120],
    }


def render_table(metrics: dict, floor: list[str]) -> str:
    """Tabel metrik dirender di server, supaya bisa diverifikasi tanpa menjalankan browser."""
    rows = []
    for key, val in metrics.items():
        if val is None:
            shown = "n/a (teks terlalu pendek)"
        elif isinstance(val, list):
            shown = ", ".join(f"{k}×{v}" for k, v in val) if val else "–"
        else:
            shown = str(val)
        note = '<span class="pill p-warn">di lantai</span>' if key in floor else ""
        rows.append(f"<tr><td><code>{key}</code></td><td>{shown}</td><td>{note}</td></tr>")
    return ("<table><thead><tr><th>metrik</th><th>nilai</th><th>catatan</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def gate_summary(key: str) -> dict:
    for mf in METRICS_FILES:
        if not mf.exists():
            continue
        for line in mf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["model"] == key:
                return {
                    "bpb": row["eval"]["anchor_bpb"],
                    "gate": row.get("gate", {}),
                    "eval": row.get("eval", {}),
                    "validate": row.get("validate", {}),
                }
    return {}


# ---------------------------------------------------------------- inferensi
def generate(key: str, prompt: str, domain: str, temp: float, top_p: float,
             max_new: int, samples: int) -> list[str]:
    from src.generate import generate_batch

    model = load_model(key)
    prompts = [(domain, prompt)] * max(1, min(samples, 4))
    with _LOCK:
        return generate_batch(model, prompts, temp=temp, top_p=top_p, max_new=max_new)


# ---------------------------------------------------------------- HTTP
PAGE = """<!doctype html>
<html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Uji model — AI Belajar dari AI</title>
<style>
 :root{--bg:#0f1115;--card:#171a21;--line:#252a34;--fg:#e7e9ee;--mut:#9aa3b2;
       --ok:#3ddc97;--warn:#ffb454;--bad:#ff6b6b;--acc:#5aa9ff}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
 header{padding:22px 20px 8px;border-bottom:1px solid var(--line)}
 h1{margin:0 0 6px;font-size:20px}
 .sub{color:var(--mut);font-size:13px;max-width:900px}
 main{padding:18px 20px 60px;max-width:1100px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:14px 0}
 label{display:block;font-size:12px;color:var(--mut);margin:8px 0 4px;letter-spacing:.02em}
 select,textarea,input{width:100%;background:#0d1016;color:var(--fg);border:1px solid var(--line);
   border-radius:7px;padding:9px;font:14px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace}
 textarea{min-height:74px;resize:vertical}
 .row{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
 button{background:var(--acc);color:#06121f;border:0;border-radius:7px;padding:10px 16px;
   font-weight:650;cursor:pointer;margin-top:12px}
 button.ghost{background:transparent;color:var(--fg);border:1px solid var(--line)}
 button:disabled{opacity:.5;cursor:progress}
 .out{white-space:pre-wrap;font:13px/1.6 ui-monospace,Menlo,monospace;background:#0d1016;
   border:1px solid var(--line);border-radius:7px;padding:12px;margin-top:10px}
 table{width:100%;border-collapse:collapse;font-size:13px}
 th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
 th{color:var(--mut);font-weight:600}
 .pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700}
 .p-ok{background:rgba(61,220,151,.15);color:var(--ok)}
 .p-warn{background:rgba(255,180,84,.15);color:var(--warn)}
 .p-bad{background:rgba(255,107,107,.15);color:var(--bad)}
 .p-mut{background:rgba(154,163,178,.15);color:var(--mut)}
 .muted{color:var(--mut);font-size:12.5px}
 code{background:#0d1016;padding:1px 5px;border-radius:4px}
</style></head><body>
<header>
  <h1>Uji model secara langsung</h1>
  <p class="sub">Halaman ini dilayani oleh proses Python lokal yang juga memuat modelnya,
  jadi yang kamu baca adalah keluaran model apa adanya. Tidak ada CDN, tidak ada data yang keluar
  dari mesin ini. Metrik di bawah memakai keluarga yang sama dengan pra-registrasi R8 &mdash;
  termasuk peringatan bila sebuah metrik berada di lantai dan karena itu tidak boleh ditafsirkan.</p>
</header>
<main>
  <div class="card">
    <label for="ckpt">Checkpoint (M0 baseline &rarr; generasi)</label>
    <select id="ckpt"><option>memuat…</option></select>
    <div class="row">
      <div><label for="domain">Domain</label>
        <select id="domain">
          <option>narasi</option><option>faktual</option><option>definisi</option>
          <option>dialog</option><option>kode</option><option>aritmetika</option>
          <option>prosedur</option><option>analisis</option>
        </select></div>
      <div><label for="temp">Suhu (0,01 = hampir deterministik)</label><input id="temp" value="0.9"></div>
      <div><label for="topp">Top-p</label><input id="topp" value="0.95"></div>
      <div><label for="maxnew">Token baru</label><input id="maxnew" value="120"></div>
      <div><label for="samples">Sampel</label><input id="samples" value="1"></div>
    </div>
    <label for="prompt">Prompt bebas</label>
    <textarea id="prompt">Jelaskan kabut pagi di lereng bukit.</textarea>
    <button id="go">Hasilkan teks</button>
    <button id="assess" class="ghost">Nilai metrik kotak hasil</button>
    <div id="status" class="muted" style="margin-top:10px"></div>
    <div id="gate"></div>
    <div id="out" class="out" hidden></div>
  </div>
  <div class="card" id="metrics" hidden>
    <h3 style="margin:0 0 10px;font-size:15px">Penilaian cepat</h3>
    <div id="mtable"></div>
    <p class="muted" id="mnote"></p>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
let CKPTS = [];
async function api(path, body){
  const r = await fetch(path, body ? {method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)} : {});
  if(!r.ok) throw new Error(await r.text());
  return await r.json();
}
function pill(s){
  const map={ok:'p-ok',watch:'p-warn',collapse:'p-bad'};
  return `<span class="pill ${map[s]||'p-mut'}">${s||'tanpa gate'}</span>`;
}
async function init(){
  CKPTS = await api('/api/checkpoints');
  const sel = $('ckpt'); sel.innerHTML='';
  if(!CKPTS.length){ sel.innerHTML='<option>tidak ada checkpoint</option>'; return; }
  for(const c of CKPTS){
    const o=document.createElement('option'); o.value=c.key;
    o.textContent = `${c.key}  ·  ${c.arm}${c.gen?(' g'+c.gen):''}  ·  bpb ${c.bpb!=null?c.bpb.toFixed(4):'?'}`;
    sel.appendChild(o);
  }
  sel.onchange = showGate; showGate();
}
function showGate(){
  const key=$('ckpt').value, c=CKPTS.find(x=>x.key===key); if(!c) return;
  $('gate').innerHTML = `<p class="muted" style="margin:10px 0 0">Gate checkpoint ini:
    ${pill(c.gate_status)} ${c.delta!=null?`Δ bpb ${c.delta>0?'+':''}${c.delta.toFixed(4)}`:''}
    · ukuran berkas ${c.size_mb} MB</p>`;
}
async function run(){
  const btn=$('go'); btn.disabled=true; $('status').textContent='menghasilkan…';
  try{
    const res = await api('/api/generate', {
      ckpt:$('ckpt').value, prompt:$('prompt').value, domain:$('domain').value,
      temp:parseFloat($('temp').value)||0.9, top_p:parseFloat($('topp').value)||0.95,
      max_new:parseInt($('maxnew').value)||120, samples:parseInt($('samples').value)||1
    });
    $('out').hidden=false;
    $('out').textContent = res.texts.map((t,i)=>`── sampel ${i+1} ──\n${t}`).join('\n\n');
    $('status').textContent = `${res.ms} ms · ${res.texts.length} sampel · ${res.n_params} parameter`;
    const m = await api('/api/assess', {ckpt:$('ckpt').value, text:res.texts[0]});
    renderMetrics(m);
  }catch(e){ $('status').textContent='gagal: '+e.message; }
  btn.disabled=false;
}
async function assessBox(){
  const t=$('out').textContent.trim();
  if(!t){ $('status').textContent='kotak hasil masih kosong'; return; }
  const m = await api('/api/assess', {ckpt:$('ckpt').value, text:t});
  renderMetrics(m);
}
function renderMetrics(m){
  $('metrics').hidden=false;
  // tabel dirender di server (render_table) sehingga tampilan tidak bergantung pada JS ini
  $('mtable').innerHTML = m.table_html || '<p class="muted">tabel tidak tersedia</p>';
  $('mnote').textContent = m.note;
}
$('go').onclick = run; $('assess').onclick = assessBox;
init().catch(e=>{ $('status').textContent='gagal memuat checkpoint: '+e.message; });
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "ui_inspect/1.0"

    def log_message(self, fmt, *args):  # ringkas, tidak spam
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/checkpoints":
            self._send(200, json.dumps(list_checkpoints()).encode(), "application/json")
        elif self.path == "/api/health":
            self._send(200, b'{"ok":true}', "application/json")
        elif self.path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
        else:
            self._send(404, b"tidak ada", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, b"json tidak valid", "text/plain; charset=utf-8")
            return
        try:
            if self.path == "/api/generate":
                import time

                t0 = time.time()
                texts = generate(
                    payload["ckpt"], payload.get("prompt", ""), payload.get("domain", "narasi"),
                    float(payload.get("temp", 0.9)), float(payload.get("top_p", 0.95)),
                    int(payload.get("max_new", 120)), int(payload.get("samples", 1)),
                )
                model = _CACHE.get(payload["ckpt"])
                body = {
                    "texts": texts,
                    "ms": int((time.time() - t0) * 1000),
                    "n_params": getattr(model, "n_params", None),
                }
                self._send(200, json.dumps(body, ensure_ascii=False).encode(), "application/json")
            elif self.path == "/api/assess":
                key = payload.get("ckpt", "")
                metrics = assess(payload.get("text", ""))
                gate = gate_summary(key)
                floor = []
                if gate.get("eval"):
                    ev = gate["eval"]
                    if ev.get("arith_n", 0) and ev.get("arith_acc", 0) <= 2 / max(1, ev["arith_n"]):
                        floor.append("arith_acc")
                note = ("Metrik 'di lantai' berarti nilai baseline-nya sudah di dasar sehingga "
                        "perubahannya tidak bisa ditafsirkan (aturan sadar-lantai R8 §6.1). "
                        "Angka di sini dihitung atas teks yang kamu uji saja, bukan atas seluruh korpus evaluasi.")
                if gate.get("validate"):
                    note += f" Reject rate korpus checkpoint ini: {gate['validate'].get('reject_rate', 0):.0%}"
                body = {"metrics": metrics, "gate": gate.get("gate", {}), "floor": floor, "note": note,
                        "table_html": render_table(metrics, floor)}
                self._send(200, json.dumps(body, ensure_ascii=False).encode(), "application/json")
            else:
                self._send(404, b"tidak ada", "text/plain; charset=utf-8")
        except Exception as exc:  # tampilkan ke pengguna, jangan telan
            self._send(500, json.dumps({"error": str(exc)}).encode(), "application/json")


def serve(host: str, port: int) -> None:
    ck = list_checkpoints()
    print(f"checkpoint terdeteksi: {len(ck)}")
    for c in ck[:6]:
        print(f"  {c['key']:16s} arm={c['arm']:8s} gen={c['gen']} bpb={c['bpb']}")
    if len(ck) > 6:
        print(f"  … dan {len(ck) - 6} lagi")
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"\nUji model: http://localhost:{port}  (bind {host}:{port})")
    print("Tekan Ctrl+C untuk berhenti.\n")
    srv.serve_forever()


# ---------------------------------------------------------------- CLI
def cli() -> None:
    ck = list_checkpoints()
    if not ck:
        print("Tidak ada checkpoint. Jalankan pipeline dulu.")
        return
    print("Checkpoint tersedia:")
    for i, c in enumerate(ck, 1):
        print(f"  [{i:2d}] {c['key']:16s} arm={c['arm']:8s} gen={c['gen']} bpb={c['bpb']}")
    idx = int(input("\nPilih nomor checkpoint: ").strip() or "1")
    key = ck[idx - 1]["key"]
    domain = input("Domain [narasi]: ").strip() or "narasi"
    while True:
        prompt = input("\nPrompt (kosong = keluar): ").strip()
        if not prompt:
            break
        temp = float(input("Suhu [0.9]: ").strip() or "0.9")
        texts = generate(key, prompt, domain, temp, 0.95, 120, 1)
        print("\n" + texts[0] + "\n")
        m = assess(texts[0])
        print("metrik:", json.dumps(m, ensure_ascii=False))
        g = gate_summary(key)
        if g:
            print(f"gate  : {g['gate'].get('status')}  Δ={g['gate'].get('delta_bpb'):+.4f}  "
                  f"ambang={g['gate'].get('threshold_bpb')}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Antarmuka uji manusia untuk model pipeline")
    ap.add_argument("--cli", action="store_true", help="mode terminal interaktif")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--bind", default="0.0.0.0")
    args = ap.parse_args()
    if args.cli:
        cli()
    else:
        serve(args.bind, args.port)


if __name__ == "__main__":
    main()
