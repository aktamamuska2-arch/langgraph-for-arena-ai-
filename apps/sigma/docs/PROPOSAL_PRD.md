# SIGMA — Proposal & Product Requirements Document

**Strategic Intelligence Graph-based Multi-Agent Analytics**
Sebuah platform analitik profesional interaktif berbasis orkestrasi multi-agen LangGraph.

| Field | Value |
| --- | --- |
| Dokumen | `apps/sigma/docs/PROPOSAL_PRD.md` |
| Versi | 1.0 (FASE 0) |
| Tanggal | 2026-10-31 |
| Penulis | Principal AI Systems Architect |
| Status | Disetujui secara implisit oleh desain (implementasi mengikuti dokumen ini) |
| Repositori | `langgraph-for-arena-ai` (monorepo LangGraph v1.2.19) |
| Kode | `apps/sigma/` |

---

## 1. Executive Summary & Intent

### 1.1 Ringkasan

SIGMA adalah platform analitik interaktif yang menerima sebuah *pertanyaan analitik* (mis. "Apakah kita harus mengalokasikan modal ke portofolio ventura X pada kuartal ini?") dan menghasilkan **brief analitik yang dapat diaudit**: temuan terkuantifikasi, bukti yang dapat ditelusuri ke pemanggilan tool, skor kualitas, dan rekomendasi aksi.

Yang membedakan SIGMA dari pipeline ETL/BI konvensional bukanlah "ada LLM di dalamnya", melainkan bahwa **kualitas output dihasilkan oleh proses iteratif yang bisa gagal, dievaluasi, dan diperbaiki sendiri** — sebuah proses yang secara struktural **siklik**, bukan linier.

Platform ini dibangun langsung di atas kode sumber LangGraph yang ada di repositori ini (`libs/langgraph`, `libs/checkpoint`, `libs/checkpoint-sqlite`, `libs/prebuilt`), bukan dari wheel PyPI, sehingga runtime yang dijalankan identik dengan runtime yang diaudit.

### 1.2 Mengapa *cyclic graph* mutlak diperlukan (bukan DAG)

Ini adalah keputusan arsitektur sentral dokumen ini. Argumennya bukan preferensi gaya, melainkan kebutuhan fungsional.

**Definisi.** DAG (Directed Acyclic Graph) menjamin terminasi dengan melarang sebuah node dieksekusi dua kali dalam satu run. Setiap keputusan "ulang" harus dinaikkan ke luar graf — menjadi loop `while` di kode pemanggil, atau di-hardcode menjadi N stage replikasi (Planner₁ → Executor₁ → Planner₂ → Executor₂ → ...).

**Masalah nyata dengan DAG untuk sistem analitik agen:**

1. **Kualitas analitik adalah fungsi yang tidak monotonik dan tidak diketahui di awal.** Kita tidak tahu berapa kali evidensi harus dikumpulkan sebelum sebuah klaim bisa dipertanggungjawabkan. DAG memaksa kita menebak jumlah iterasi pada waktu desain (*compile time*), padahal informasi penentunya baru ada pada waktu jalan (*run time*), yaitu di dalam state.

2. **Koreksi-diri membutuhkan "kembali ke belakang" yang berpresisi.** Kritik dari *Evaluator* tidak selalu berarti "ulangi semuanya". Bisa berarti:
   - kembali ke **Planner** (sub-pertanyaan yang diajukan salah/bergeser), atau
   - kembali ke **Executor/Researcher** (sub-pertanyaan benar, evidensinya kurang), atau
   - kembali ke **Analyst** (evidensi cukup, interpretasinya lemah).
   Pada DAG, ketiga lintasan ini adalah tiga subgraf berbeda yang harus diduplikasi. Pada *cyclic graph*, ketiganya adalah satu *conditional edge* yang memilih target berdasarkan state.

3. **State harus terakumulasi lintas iterasi, bukan ditimpa.** Iterasi ke-3 harus tahu persis apa yang sudah dicoba pada iterasi 1–2 agar tidak mengulang bukti yang sama. Ini adalah properti *superstep* LangGraph: setiap siklus menulis ke channel yang sama dengan *reducer* yang terdefinisi, sehingga memori iteratif bersifat **transaksional** dan konsisten.

4. **Human-in-the-Loop (HITL) mengubah alur secara dinamis.** Manusia dapat menyetujui, meminta revisi (→ siklus baru), atau menolak (→ terminasi terkelola). Keputusan ini hanya bisa diekspresikan sebagai edge kondisi keluar dari node interrupt; pada DAG tidak ada representasi untuk "node yang dieksekusi ulang setelah manusia bicara".

5. **Time-travel debugging menuntut graf yang bisa di-replay dari titik tengah.** Checkpointer LangGraph menyimpan state per superstep; kemampuan "kembali ke checkpoint N, ubah, lanjutkan" hanya bermakna jika graf secara legal bisa berlanjut dari node mana pun — sesuatu yang diklaim oleh topologi siklik.

**Kesimpulan yang bisa diuji.** Klaim di atas dijadikan kriteria penerimaan, bukan opini:

> **AC-CYCLE-1** — Menghidupkan *critic* yang selalu menolak (`force_revise=True`) harus menghasilkan eksekusi yang **berulang lalu berhenti** pada batas `max_revisions`, bukan eksekusi yang gagal atau menghasilkan satu lintasan linier.
> **AC-CYCLE-2** — Rute siklik harus **bercabang**: untuk kelas kegagalan yang berbeda, node yang dikunjungi berikutnya harus berbeda (`planner` vs `researcher` vs `analyst`), bukan selalu node yang sama.
> **AC-CYCLE-3** — State pada iterasi ke-*n* harus berisi artefak dari iterasi < *n* (bukti/evidensi bersifat akumulatif, tidak ditimpa).
> **AC-CYCLE-4** — Tidak ada duplikasi node di source untuk merepresentasikan "iterasi ke-2". Jumlah node konstan, yang bertambah adalah jumlah *superstep* di dalam checkpoint.

DAG hanya akan lolos AC-CYCLE-3; keempat kriteria di atas sekaligus hanya dipenuhi oleh graf siklik dengan *conditional edge* + *checkpointer*.

### 1.3 Tabel perbandingan keputusan

| Dimensi | Pipeline DAG | **SIGMA (cyclic LangGraph)** |
| --- | --- | --- |
| Jumlah iterasi | Ditentukan saat desain | Ditentukan saat run, dari skor evaluator |
| Revisi terarah | Tidak mungkin tanpa duplikasi subgraf | Satu `add_conditional_edges` dengan 3 target |
| Memori antar-iterasi | Manual (loop luar) | Reducer + checkpointer (transaksional) |
| HITL | Di luar graf | `interrupt()` di dalam node |
| Resume setelah crash | Tidak ada | `thread_id` + SqliteSaver |
| Observabilitas | Log aplikasi | `get_state_history()` per superstep |

---

## 2. Ruang Lingkup Produk

### 2.1 Domain

Portofolio ventura sintetik yang **deterministik terhadap seed**. Data tidak diunduh dari internet (eksperimen harus reproducible dan platform harus tetap berjalan tanpa kredensial eksternal). Korpus digenerate oleh modul `sigma.tools.market` dengan `random.Random(seed)`.

### 2.2 User stories

| ID | Sebagai | Saya ingin | Supaya |
| --- | --- | --- | --- |
| US-1 | Analis | mengajukan pertanyaan analitik dalam bahasa natural | mendapat brief terstruktur dalam satu kali jalan |
| US-2 | Analis | melihat **setiap** pemanggilan tool dan argumennya | bisa memverifikasi bahwa angka berasal dari data, bukan halusinasi |
| US-3 | Reviewer | dimintai persetujuan sebelum brief difinalisasi | keputusan berisiko tinggi tidak lolos tanpa manusia |
| US-4 | Reviewer | menolak dengan catatan | agen memperbaiki arah, bukan mengulang kesalahan |
| US-5 | Operator | melanjutkan sesi yang terputus | tidak kehilangan pekerjaan karena crash/restart |
| US-6 | Auditor | menelusuri riwayat state per langkah (time-travel) | bisa menjawab "mengapa agen sampai ke kesimpulan ini" |
| US-7 | SRE | menerima output terdegradasi yang jujur saat tool gagal | kegagalan parsial tidak menjadi kegagalan total |

### 2.3 Kebutuhan fungsional

| ID | Kebutuhan | Prioritas |
| --- | --- | --- |
| FR-1 | API + CLI untuk membuat sesi analitik baru dari `(question, seed, portfolio_id)` | Must |
| FR-2 | Graf multi-agen: `intake → planner → researcher ⇄ analyst → critic → (siklus \| human_gate) → finalizer` | Must |
| FR-3 | Tool registry dengan skema JSON dan eksekusi terkontrol (timeout, validator argumen) | Must |
| FR-4 | Evaluator dengan rubrik terukur (5 dimensi) yang menghasilkan skor 0..1 dan daftar *gap* | Must |
| FR-5 | Siklus koreksi-diri terarah: `critic → planner \| researcher \| analyst` | Must |
| FR-6 | HITL dengan `interrupt()` + `response_schema` Pydantic; resume via `Command(resume=...)` | Must |
| FR-7 | Checkpointing SQLite persisten: daftar sesi, riwayat checkpoint, time-travel, fork | Must |
| FR-8 | Antarmuka web ringan (HTML/JS statis, tanpa framework build) untuk menjalankan & memonitor | Must |
| FR-9 | Streaming event graf (update per node) ke klien (SSE) | Should |
| FR-10 | Provider LLM yang dapat diganti: deterministik offline (default) dan OpenAI-compatible via HTTP | Must |
| FR-11 | Mode terdegradasi: kegagalan tool → `containment` node → brief dengan `status=degraded` | Must |
| FR-12 | Audit ukuran artefak (`audit_size.py`) dengan gate < 125 MB | Must |

### 2.4 Kebutuhan non-fungsional

| ID | Kebutuhan | Target |
| --- | --- | --- |
| NFR-1 | Total ukuran artefak (runtime deps + kode + aset + DB) | **< 125 MB** (hard gate) |
| NFR-2 | Dependensi pihak ketiga pada runtime | ≤ 25 distribusi, tanpa `streamlit`, `pandas`, `numpy`, `torch`, SDK LLM |
| NFR-3 | Determinisme | Run dengan `seed` identik → skor akhir identik (toleransi 0.0) |
| NFR-4 | Konvergensi siklus | Berhenti dalam ≤ `max_revisions` (default 3); tidak ada siklus tak terbatas |
| NFR-5 | Latensi | Satu sesi penuh (tanpa LLM jaringan) < 3 s di mesin dev |
| NFR-6 | Portabilitas | `python -m venv` + `pip install`; tanpa service eksternal |
| NFR-7 | Keterlacakan | Setiap klaim numerik pada brief harus memiliki ≥ 1 `evidence_id` |

---

## 3. Arsitektur Sistem

### 3.1 Topologi graf (mermaid)

```mermaid
flowchart TD
    START([__start__]) --> intake
    intake --> planner
    planner --> researcher
    researcher -->|pending tool calls| researcher
    researcher -->|selesai| analyst
    analyst --> critic
    critic -->{route_after_critique}
    route_after_critique -->|skor < ambang & gap terstruktur| planner
    route_after_critique -->|skor < ambang & gap evidensi| researcher
    route_after_critique -->|skor < ambang & gap interpretasi| analyst
    route_after_critique -->|skor >= ambang ATAUBudget habis| human_gate
    human_gate -->|interrupt: approve| finalizer
    human_gate -->|interrupt: revise| planner
    human_gate -->|interrupt: reject| containment
    finalizer --> END([__end__])
    containment --> END
    intake -.->|error| containment
    planner -.->|error| containment
    researcher -.->|error| containment
    analyst -.->|error| containment
```

Dua siklus eksplisit ditandai:
- **Siklus internal (mikro):** `researcher → researcher` selama masih ada panggilan tool yang tertunda.
- **Siklus eksternal (makro):** `critic → {planner | researcher | analyst}` dan `human_gate → planner`.

### 3.2 Katalog node

| Node | Peran | Menulis ke state | Keluar |
| --- | --- | --- | --- |
| `intake` | Normalisasi permintaan, inisialisasi skenario & budget | `request`, `scenario`, `budget` | `planner` |
| `planner` | Dekomposisi menjadi 3–6 sub-pertanyaan + tool hint | `plan`, `revision`++ | `researcher` |
| `researcher` | Executor: pilih & jalankan tool per sub-pertanyaan | `evidence`, `tool_calls`, `messages` | dirinya sendiri / `analyst` |
| `analyst` | Mengubah evidensi menjadi temuan terkuantifikasi | `findings`, `draft` | `critic` |
| `critic` | Evaluator: rubrik 5 dimensi → skor + gap | `evaluation` | conditional |
| `human_gate` | HITL: `interrupt()` menunggu keputusan manusia | `human_decision` | conditional |
| `finalizer` | Menulis brief final + jejak audit | `brief`, `status` | `END` |
| `containment` | Fallback: run terdegradasi yang jujur | `status=degraded`, `errors` | `END` |

### 3.3 Katalog tool

| Tool | Argumen | Mengembalikan | Dipakai oleh |
| --- | --- | --- | --- |
| `query_portfolio` | `portfolio_id`, `filter` | Daftar ventura (sektor, stage, modal, pertumbuhan) | researcher |
| `compute_risk_metrics` | `venture_ids` | EV, volatilitas, HHI konsentrasi, skor risiko 0..1 | researcher/analyst |
| `scenario_simulate` | `venture_ids`, `paths`, `horizon` | Monte Carlo: P10/P50/P90, VaR95, probabilitas rugi | researcher |
| `benchmark_sector` | `sector` | Kuartil sektor untuk normalisasi | researcher |
| `evidence_lookup` | `evidence_id` | Mengambil ulang evidensi yang sudah ada (dedup) | researcher |

Tool dieksekusi melalui registry yang memvalidasi argumen dengan Pydantic dan memberi timeout (`ToolExecutionError` → retry policy → `containment`).

---

## 4. Skema State

### 4.1 Filosofi

State adalah **kontrak transaksional** antar agen. Aturan yang ditegakkan:

1. **TypedDict untuk envelope graf** (apa yang dikenali LangGraph), **Pydantic untuk isi artefak** (apa yang divalidasi). Envelope tipis, isinya ter-tipe kuat.
2. **Reducer eksplisit untuk setiap field yang bertumbuh.** Tidak ada field list yang memakai *last-value* default LangGraph, karena itu akan menelan bukti antar iterasi (melanggar AC-CYCLE-3).
3. **Append-only untuk jejak audit.** `messages`, `evidence`, `tool_calls`, `audit_trail` bersifat akumulatif; koreksi-diri dilakukan dengan menambahkan supersede, bukan menghapus sejarah.
4. **Bounded.** Setiap reducer list memiliki `cap` agar state tidak tumbuh tanpa batas pada siklus panjang (NFR: state ≤ 256 KB per checkpoint).

### 4.2 Envelope (TypedDict)

```python
class SigmaState(TypedDict, total=False):
    # --- identitas & kontrol ---
    session_id: str                      # last-value
    request: RequestPayload              # last-value
    scenario: ScenarioContext            # last-value
    budget: BudgetState                  # reducer: merge_budget (monotonik turun)
    # --- perencanaan ---
    plan: PlanState                      # last-value (versi terbaru, ada revision counter)
    # --- eksekusi ---
    pending_tools: Annotated[list[ToolCall], merge_unique_by("call_id")]
    tool_calls: Annotated[list[ToolCallRecord], bounded_add(200)]
    evidence: Annotated[list[Evidence], bounded_add(200)]
    # --- penalaran ---
    messages: Annotated[list[dict], bounded_add(120)]   # transcript agen (LLM memory)
    findings: Annotated[list[Finding], supersede_by("finding_id")]
    draft: DraftBrief | None             # last-value
    # --- evaluasi & HITL ---
    evaluation: Annotated[list[Evaluation], bounded_add(12)]
    human_decision: HumanDecision | None # last-value
    # --- output & kegagalan ---
    brief: FinalBrief | None             # last-value
    errors: Annotated[list[ErrorRecord], bounded_add(50)]
    audit_trail: Annotated[list[AuditEvent], bounded_add(400)]
    status: RunStatus                    # last-value: running|awaiting_human|completed|degraded|failed
```

### 4.3 Reducer

| Reducer | Semantik | Alasan |
| --- | --- | --- |
| `bounded_add(cap)` | `list + list`, dipotong dari kiri bila > cap | Jejak tumbuh, state tetap terbatas |
| `merge_unique_by(key)` | union berdasarkan kunci, yang baru menimpa | Tool call idempoten saat node di-replay setelah interrupt |
| `supersede_by(key)` | item baru dengan `supersedes=<id>` menonaktifkan yang lama | Revisi terarah tanpa menghapus sejarah (auditable) |
| `merge_budget` | mengambil nilai sisa terkecil per dimensi | Budget tidak pernah "pulih" karena retry |
| `last_value` (default) | timpa | Untuk artefak yang memang versi-terbaru (draft, brief) |

### 4.4 Model Pydantic utama

`RequestPayload`, `ScenarioContext`, `PlanState` (`sub_questions: list[SubQuestion]`), `ToolCall`, `ToolCallRecord`, `Evidence`, `Finding`, `DraftBrief`, `Evaluation` (`scores: RubricScores`, `gaps: list[Gap]`), `HumanDecision` (`decision: Literal["approve","revise","reject"]`, `note`), `FinalBrief`, `ErrorRecord`, `AuditEvent`.

`Evaluation.gaps` memiliki `GapKind = Literal["missing_evidence","structural","interpretation","numerical"]` — inilah yang menggerakkan rute siklik (lihat §5).

---

## 5. Conditional Edge & Routing

```python
builder.add_conditional_edges(
    "critic",
    route_after_critique,                 # -> Command | str
    {"planner": "planner", "researcher": "researcher",
     "analyst": "analyst", "human_gate": "human_gate", "containment": "containment"},
)
```

| Kondisi (berurutan) | Target | Rasionalisasi |
| --- | --- | --- |
| `errors` mengandung fatal | `containment` | Jangan siklus pada kegagalan infrastruktur |
| `evaluation[-1].score >= threshold` (0.78) | `human_gate` | Kualitas cukup → minta persetujuan manusia |
| `budget.revisions_left == 0` | `human_gate` | Habis budget → tetap minta manusia, brief ditandai `provisional` |
| gap dominan `structural` | `planner` | Pertanyaannya yang salah |
| gap dominan `missing_evidence` | `researcher` | Pertanyaan benar, datanya kurang |
| gap dominan `interpretation`/`numerical` | `analyst` | Data cukup, analisisnya lemah |
| else | `human_gate` | Fail-safe: never loop forever (NFR-4) |

Routing HITL (`route_after_human`): `approve → finalizer`, `revise → planner` (dengan `human_decision.note` di-inject ke state agar planner membacanya), `reject → containment`.

**Fail-safe terminasi.** Selain `budget.revisions_left`, setiap node memanggil `consume_budget()`; bila budget habis pada node apa pun, `Command(goto="human_gate")` dikeluarkan. Graf juga dikompilasi dengan `recursion_limit` eksplisit.

---

## 6. Checkpointer & Persistensi

| Aspek | Keputusan |
| --- | --- |
| Implementasi | `langgraph.checkpoint.sqlite.SqliteSaver` (dari `libs/checkpoint-sqlite` di repo ini) |
| Lokasi | `apps/sigma/var/checkpoints.sqlite` (konfigurasi via `SIGMA_DB_PATH`) |
| `thread_id` | `= session_id` (satu sesi analitik = satu thread) |
| `checkpoint_ns` | default (`""`) untuk graf utama |
| Setup | `saver.setup()` idempoten saat startup aplikasi |
| Serde | default `JsonPlusSerializerCompat` (state berisi model Pydantic → harus bisa di-serialize) |

Operasi yang diekspos ke pengguna:

| Operasi | API LangGraph | Kegunaan produk |
| --- | --- | --- |
| Lihat state terkini | `graph.get_state(config)` | Panel "state inspector" |
| Riwayat superstep | `graph.get_state_history(config)` | Time-travel debugging (US-6) |
| Baca checkpoint lama | `get_state(config_with_checkpoint_id)` | Inspeksi read-only |
| Fork dari masa lalu | `graph.update_state(config_at_checkpoint, values)` lalu `invoke(None, cfg)` | "Bagaimana jika analisnya memilih tool lain?" |
| Lanjut setelah interrupt | `graph.invoke(Command(resume=...), config)` | HITL (FR-6) |
| Lanjut setelah crash | `graph.invoke(None, config)` | Pemulihan sesi (US-5) |

**Kebijakan retensi.** Checkpoint bersifat append-only; `scripts/audit_size.py` memantau ukuran DB dan `VACUUM` dijalankan oleh endpoint perawatan bila DB > 32 MB.

---

## 7. Integrasi LLM & Manajemen Memori

### 7.1 Provider abstraction

```python
class LLMProvider(Protocol):
    def complete(self, *, system: str, messages: list[dict], tools: list[dict] | None,
                 response_model: type[BaseModel] | None, seed: int) -> LLMResponse: ...
```

| Provider | Kapan | Karakteristik |
| --- | --- | --- |
| `DeterministicProvider` | **Default**. Tanpa kredensial, tanpa jaringan | Mesin penalaran berbasis heuristik terstruktur + `random.Random(seed)`; menghasilkan rencana, panggilan tool, temuan, dan kritik yang **benar-benar bergantung pada isi state** (bukan teks statis). Menjamin NFR-3 dan membuat seluruh test suite hermetik. |
| `HttpChatProvider` | Jika `SIGMA_LLM_BASE_URL` + `SIGMA_LLM_API_KEY` diset | OpenAI-compatible `/chat/completions` lewat `httpx` (sudah dependensi transitif). Tidak menambah SDK baru (NFR-2). |
| `LangChainChatProvider` | Opsional, bila `langchain-core` model tersedia | Menjalankan `BaseChatModel` apa pun; menjembatani ekosistem LangChain tanpa membuatnya menjadi dependensi wajib. |

Pemilihan provider dilakukan oleh `sigma.llm.get_provider()` dan **dicatat di** `audit_trail`, sehingga sebuah run selalu bisa dijawab "dihasilkan oleh mesin yang mana".

### 7.2 Memori transaksional

LLM tidak pernah menyimpan memori sendiri. Semua konteks berada di channel `messages` dan ikut ter-checkpoint bersama state. Konsekuensinya:

- **Replay setelah interrupt = deterministik.** Node yang di-replay membaca `messages` yang sama dari checkpoint; `interrupt()` mengembalikan nilai resume pada pemanggilan berikutnya (kontrak LangGraph v1).
- **Kompresi konteks terkelola.** `ContextWindowManager` menjaga `messages` ≤ 24 entri: entri lama diringkas menjadi satu `AuditEvent` ringkasan, yang justru membuat jejak audit *lebih* informatif, bukan berkurang.
- **Isolasi per thread.** Karena `thread_id = session_id`, dua sesi tidak pernah berbagi memori (properti yang diberikan checkpointer, bukan oleh kode aplikasi).

---

## 8. Human-in-the-Loop (HITL)

Node `human_gate` memanggil:

```python
decision = interrupt(
    {
        "kind": "approval_request",
        "session_id": state["session_id"],
        "score": latest.score,
        "rubric": latest.scores.model_dump(),
        "draft": draft.summary,
        "open_gaps": [g.model_dump() for g in latest.gaps],
        "question": "Setujui brief ini, minta revisi, atau tolak?",
    },
    response_schema=HumanDecision,      # Pydantic -> JSON Schema untuk klien + validasi resume
)
```

Kontrak:

| Keputusan | Nilai resume | Efek |
| --- | --- | --- |
| `approve` | `{"decision": "approve", "note": "..."}` | `finalizer` menulis brief final |
| `revise` | `{"decision": "revise", "note": "<arah baru>"}` | Kembali ke `planner`; catatan di-inject sebagai `human_directive` |
| `reject` | `{"decision": "reject", "note": "..."}` | `containment` → sesi ditutup `status=rejected` |

Klien yang tidak mendukung skema tetap bisa mengirim dict; LangGraph memvalidasi dengan `response_schema` dan menolak nilai yang tidak valid sebelum graf melanjutkan (ini menghapus satu kelas bug HITL yang umum).

---

## 9. Penanganan Error & Fallback

| Lapisan | Mekanisme |
| --- | --- |
| Tool | `ToolExecutionError` → `RetryPolicy(max_attempts=2, backoff)` pada node `researcher` |
| Node | Blok `try/except` → `ErrorRecord(fatal=?)` + `Command(goto=...)`; error fatal → `containment` |
| Graf | `recursion_limit=48`; pelanggaran → `GraphRecursionError` ditangkap API, sesi ditandai `failed` |
| Budget | `BudgetState` (tool_calls, tokens, revisions, wall_clock) dikurangi setiap node; habis → paksa keluar siklus |
| Degradasi | `containment` tetap menghasilkan `FinalBrief` dengan `status="degraded"` dan `caveats` — bukan eksepsi kosong (NFR-7, US-7) |

---

## 10. Resource & Asset Constraint (< 125 MB)

### 10.1 Kebijakan

Batas **125 MB** adalah *hard gate* yang diukur oleh `apps/sigma/scripts/audit_size.py` dan dinyatakan gagal bila terlampaui. Yang dihitung:

1. **Runtime dependencies** — seluruh `site-packages` pada venv target, **termasuk** tooling build (`pip`, `setuptools`) agar angka bersifat konservatif.
2. **Kode aplikasi** — `apps/sigma/src`, `apps/sigma/tests`, `apps/sigma/scripts`.
3. **Aset statis** — `apps/sigma/src/sigma/api/static` (HTML/CSS/JS murni, tanpa pipeline build, tanpa dependensi npm).
4. **Artefak runtime** — `apps/sigma/var/*.sqlite` (checkpoint DB) pada ukuran terburuk yang diizinkan.

### 10.2 Anggaran (terukur pada mesin dev)

| Komponen | Sebelum pruning | Setelah pruning | Catatan |
| --- | --- | --- | --- |
| `site-packages` (venv) | 111,0 MB | **53,8 MB** | lihat §10.3 |
| Kode + aset aplikasi | — | < 1,0 MB | HTML/CSS/JS vanilla |
| Checkpoint DB (worst case) | — | ≤ 32,0 MB | ada `VACUUM` + retensi |
| **Total** | — | **≈ 86,8 MB** | **margin ≈ 38,2 MB (30,6%)** |

### 10.3 Langkah pruning yang diterapkan (dan alasannya)

| # | Tindakan | Penghematan | Justifikasi teknis |
| --- | --- | --- | --- |
| 1 | Hapus `zstandard` (24,0 MB) | 24,0 MB | Kompresi tracing LangSmith; tidak diimpor saat tracing nonaktif (diverifikasi dengan smoke test impor) |
| 2 | Hapus `pygments` (9,5 MB) + `pytest`/`pluggy`/`iniconfig` (5,1 MB) | 14,6 MB | Test suite memakai `unittest` stdlib → **nol dependensi test**; ini keputusan yang disengaja demi budget |
| 3 | Hapus `tests/` bawaan distribusi pihak ketiga | ~2,0 MB | Tidak pernah diimpor saat runtime |
| 4 | Bersihkan `__pycache__` | ~11,5 MB | Dihasilkan ulang lazily oleh interpreter |
| 5 | Instal LangGraph **editable** dari `libs/` | ~15,0 MB | Tidak ada salinan wheel; runtime = source yang diaudit |

### 10.4 Larangan eksplisit (NFR-2)

Dilarang masuk ke dependensi runtime: `streamlit` (+`pandas`/`pyarrow`, >70 MB), `numpy`/`pandas`/`scipy`/`scikit-learn`, `torch`/`transformers`, SDK vendor LLM (`openai`, `anthropic`, ...), `jinja2` (HTML disajikan statis), framework front-end apa pun yang butuh build step.

### 10.5 Gate

```bash
python apps/sigma/scripts/audit_size.py --venv /home/user/sigma-venv --budget-mb 125
# exit 0 -> PASS, exit 1 -> FAIL (mencetak 5 komponen terbesar)
```

---

## 11. Antarmuka Platform

### 11.1 FastAPI (FASE 3)

| Endpoint | Method | Fungsi |
| --- | --- | --- |
| `/` | GET | UI statis single-page |
| `/api/sessions` | POST | Buat sesi & jalankan hingga selesai atau interrupt |
| `/api/sessions` | GET | Daftar sesi + status terakhir |
| `/api/sessions/{id}` | GET | State lengkap (plan, evidence, findings, evaluation, brief) |
| `/api/sessions/{id}/resume` | POST | `Command(resume=HumanDecision)` |
| `/api/sessions/{id}/timeline` | GET | Riwayat checkpoint (time-travel) |
| `/api/sessions/{id}/timeline/{cp}/state` | GET | State pada checkpoint tertentu |
| `/api/sessions/{id}/fork` | POST | Fork dari checkpoint + `update_state` |
| `/api/sessions/{id}/stream` | GET | SSE: update per node |
| `/api/tools` | GET | Skema tool registry |
| `/api/health` | GET | Health + versi runtime + ukuran artefak |

Server di-bind ke `0.0.0.0` dan menerima host apa pun (tanpa allowlist) agar dapat diakses lewat proxy preview.

### 11.2 CLI

`sigma run --question "..." --seed 7` menjalankan graf dengan streaming per node; `sigma timeline <id>`, `sigma resume <id> --decision approve`, `sigma audit-size`.

### 11.3 UI

Satu berkas HTML + JS + CSS vanilla (tanpa build step): panel kiri = pertanyaan & kontrol; tengah = timeline eksekusi graf (node per node, live); kanan = inspector state (JSON) + panel persetujuan HITL yang muncul saat `__interrupt__`.

---

## 12. Strategi Verifikasi

Test suite memakai **`unittest` stdlib** (keputusan budget, §10.3) dan bersifat hermetik:

| Test | Membuktikan |
| --- | --- |
| `test_state.py` | Reducer: akumulasi, `supersede`, batas cap, budget monotonik (**AC-CYCLE-3**) |
| `test_graph.py` | Topologi siklik terkompilasi; jumlah node konstan (**AC-CYCLE-4**) |
| `test_cycles.py` | Kritik selalu menolak → berhenti di `max_revisions` (**AC-CYCLE-1**) |
| `test_routing.py` | Kelas gap berbeda → target berbeda (**AC-CYCLE-2**) |
| `test_hitl.py` | Interrupt → resume approve/revise/reject; validasi skema menolak nilai salah |
| `test_persistence.py` | Simulasi crash: `invoke(None)` melanjutkan; time-travel & fork |
| `test_tools.py` | Registry, validasi argumen, timeout, retry |
| `test_determinism.py` | Dua run dengan seed sama → skor identik (**NFR-3**) |
| `test_api.py` | Endpoint FastAPI lewat `TestClient` (httpx) |
| `test_size.py` | Gate < 125 MB dijalankan sebagai test (**NFR-1**) |

---

## 13. Rencana Eksekusi Bertahap

| Fase | Keluaran | Kriteria penerimaan |
| --- | --- | --- |
| FASE 0 | Dokumen ini | Disetujui (implisit): arsitektur siklik terjustifikasi, budget terukur |
| FASE 1 | `state.py`, node dasar, graf terkompilasi + error/fallback | `test_state`, `test_graph`, `test_routing` hijau |
| FASE 2 | Provider LLM, siklus koreksi-diri, HITL interrupt | `test_cycles`, `test_hitl`, `test_determinism` hijau |
| FASE 3 | FastAPI + UI + SQLite persistence + audit ukuran | `test_api`, `test_persistence`, `test_size` hijau; total < 125 MB |

---

## 14. Risiko & Mitigasi

| Risiko | Dampak | Mitigasi |
| --- | --- | --- |
| Siklus tidak konvergen | Run menggantung | Budget + `recursion_limit` + fail-safe `human_gate` (§5) |
| Provider deterministik dianggap "palsu" | Kredibilitas | Heuristik diikat ke isi state; skor evaluator dihitung dari artefak nyata, bukan konstanta |
| State mengembang melewati batas | DB > 32 MB | `bounded_add` + ringkasan konteks + VACUUM |
| Node di-replay setelah interrupt menggandakan efek samping | Duplikasi tool call | Reducer `merge_unique_by("call_id")` + tool registry idempoten |
| Ketergantungan pada API LangGraph v1 yang berubah | Break saat upgrade | Semua import berasal dari source repo ini; smoke test impor pada setiap fase |
| Ukuran artefak melewati 125 MB | Gagal NFR-1 | `audit_size.py` sebagai test + larangan eksplisit (§10.4) |
