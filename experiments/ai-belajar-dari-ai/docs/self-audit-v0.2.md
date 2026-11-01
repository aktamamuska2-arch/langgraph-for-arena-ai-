# Audit Mandiri — "AI Belajar dari AI" (rencana v0.2)

**Penulis: AI yang sama yang merancang sistem ini.** Ini bukan tinjauan independen. Dokumen ini adalah upaya membongkar kelemahan sendiri sebelum orang lain melakukannya, dan setiap klaim di dalamnya harus bisa gagal diuji.

| Field | Nilai |
| --- | --- |
| Objek audit | `PLAN.md` v0.2, 13 arm, 32 chain, orkestrasi LangGraph |
| Tanggal | 2026-11-01 |
| Metode | Pembacaan desain + simulasi kalibrasi statistik + pengukuran sandbox |
| Bukti kuantitatif | `analysis/calibration_results.md`, `ops/policy.v1.json` |
| Sifat | Audit internal. Bobotnya lebih rendah daripada audit pihak ketiga — lihat §7 |

---

## 0. Lima temuan yang paling tidak nyaman

1. **Aturan `collapse` yang aku tulis di v0.2 salah secara statistik.** `Δ ≥ max(3σ̂, 0,10)` tidak berarti "0,3% false positive". Simulasi menunjukkan **FWER studi 11,4%** untuk aturan tanpa lantai, dan 8,5% untuk aturan v0.2 pada skenario derau terburuk. Artinya: pada studi yang sepenuhnya kosong, aku punya peluang kira-kira **satu dari sembilan** untuk melaporkan "collapse ditemukan".
2. **Aku menulis kesimpulan sebelum melihat data, dan kesimpulan itu salah.** Di skrip kalibrasi, aku sudah menulis "R6 adalah kompromi terbaik" sebelum simulasi dijalankan. Hasilnya membantah: R6 terlalu konservatif (power 2–26% pada efek 0,05 bpb). Teks itu aku ganti setelah data keluar — tapi fakta bahwa aku menulisnya lebih dulu adalah bukti bias konfirmasi yang beroperasi bahkan pada metadokumen.
3. **Batas deteksi eksperimen ini jauh lebih kasar dari yang tersirat di v0.2.** MDE80 berkisar **0,078–0,233 bpb**, yaitu 5–17% degradasi relatif. Degradasi halus di bawah itu **tidak akan terlihat**, dan "tidak terdeteksi" mudah berubah menjadi "tidak terjadi" saat ditulis sebagai abstrak.
4. **Studi penuh tidak mungkin dijalankan di sandbox ini.** Terukur: 2 vCPU, 3 GB RAM, tanpa GPU. Biaya model S penuh ≈ 5,6 jam/model di CPU; 192 model ≈ **1.067 jam CPU ≈ 44 hari**. Profil XS yang jujur: ±2 menit/model, 12 model ≈ 19 menit. Itu yang bisa kukerjakan sekarang, dan aku harus menyebutnya sebagai bukti pipeline, bukan temuan.
5. **Kontrol yang paling penting tidak ada di dalam sistem, melainkan di luar: kamu.** Agen tidak bisa mendeteksi kesalahan *interpretatif* miliknya sendiri; ia hanya bisa mendeteksi kesalahan *operasional* (bug, bocor, drift). Setiap klaim "sistem sudah memverifikasi dirinya" harus dibaca sebagai "sistem sudah memeriksa hal-hal yang bisa diperiksa mesin".

---

## 1. Titik buta & bias konfirmasi

| # | Titik buta | Bagaimana ia akan tampak di laporan (jika tidak ditangani) | Mitigasi saat ini | Risiko sisa |
| --- | --- | --- | --- | --- |
| B1 | **Anchor = tulisanku sendiri.** "Bagus" berarti "mirip saya". Kesalahan sistematis di T0 menjadi kebenaran acuan. | "Model mempertahankan kualitas" — padahal yang dipertahankan adalah gaya dan salah pahamku. | Metrik eksternal yang tidak bergantung pada teksku: eksekusi aritmetika dan unit test kode. Bahasa di laporan: *fidelity to teacher*, bukan *quality*. | **Tinggi.** Domain narasi tetap hanya dinilai oleh proksiku. |
| B2 | **Ambang yang aku pilih sendiri** (lihat §2). | "Collapse terdeteksi pada generasi 4" — padahal itu derau. | Kalibrasi pra-run; aturan R7; MDE ditulis sebelum run; R1/R3 sebagai sensitivitas. | Sedang. Ambang tetap pilihanku; yang berubah hanya bahwa galatnya terukur. |
| B3 | **Critic sebagai tekanan seleksi.** Filter "kualitas" bisa membuang justru ekor distribusi yang ingin diukur. | "Model stabil" — padahal korpus latihnya sudah homogen sebelum training. | Critic diukur sebagai perlakuan (arm E), bukan asumsi; distribusi yang ditolak dicatat; alarm 30% / stop 40%. | Sedang-tinggi. Definisi "layak" tetap milikku. |
| B4 | **Bug metrik menyamar sebagai temuan.** Ini kegagalan paling berbahaya karena terlihat seperti hasil. | Kurva rapi yang menurun, dihasilkan kode yang salah menghitung BPB. | Dua implementasi BPB independen; golden fixture; CI wajib hijau; gate fail-closed. | Rendah-sedang. Fixture hanya sebaik cakupannya. |
| B5 | **Peeking & multiplisitas.** 180 uji, 6 generasi, 13 arm. Aku bisa "berhenti saat menarik". | Arm yang berhenti tepat ketika angkanya melewati ambang. | Ambang Bonferroni; gate hanya di generasi pra-registrasi; aturan futilitas hanya untuk efek nol. | Sedang. Jalur analisis masih luas. |
| B6 | **Seed bersarang menciptakan korelasi antar-arm.** Perbandingan "pasangan" bagus, tapi asumsi independensi menaikkan signifikansi semu. | p-value terlalu kecil; efek terlihat lebih yakın. | Model campuran (arm × seed random effect); semua seed dilaporkan. | Sedang. Dengan 2–3 seed, model campuran sendiri rapuh. |
| B7 | **Analisis tidak buta.** Aku tahu arm mana yang "milikku" (C2, R7) dan mana yang "menantang" (A, C1). | Narasi yang menjelaskan kegagalan arm tertentu sebagai artefak, dan keberhasilan arm lain sebagai temuan. | Analisis buta: label arm diacak sebelum analisis, dibuka setelah kesimpulan ditulis. | Sedang-tinggi. Aku tetap yang menulis kesimpulan akhir. |
| B8 | **Katalog prompt adalah seleraku.** "Realitas" yang diukur masih dunia yang aku tulis. | Generalisasi berlebih ke "dunia nyata". | Reservoir 6.000 intent, tiga tingkat kebaruan, parafrase beku. | **Tinggi.** Tidak ada data manusia sama sekali. |
| B9 | **Kriteria keberhasilan bisa bergeser setelah hasil.** | Jika tidak ada collapse, laporan berubah menjadi "studi stabilitas" tanpa mengakui pergeseran. | Pra-registrasi menyebut eksplisit apa yang dihitung sebagai hasil nol, dan hasil nol wajib terbit. | Sedang. Tidak ada yang memaksa selain kamu. |

**Kesimpulan §1:** titik buta paling berbahaya bukan yang teknis (bug bisa diperbaiki), melainkan **yang menempel pada definisi**: kualitas, kelayakan, dan kebaruan semuanya aku tentukan. Mitigasi terbaik bukan algoritma, melainkan **memisahkan penilaian yang bisa dieksekusi mesin dari penilaian yang tidak**, dan menyatakan yang kedua sebagai bias.

---

## 2. Bukti: kalibrasi aturan collapse sebelum dipakai

`analysis/calibration.py` menghitung, secara analitik, perilaku aturan deteksi di bawah hipotesis nol (tidak ada degradasi) dan di bawah efek nyata. Baseline 5 seed M0, 10 arm, 6 generasi, konfirmasi ≥2/3 seed.

| Aturan | σ = 0,02 | σ = 0,04 | σ = 0,06 | MDE80 (bpb) |
| --- | --- | --- | --- | --- |
| `3σ̂` tanpa lantai (R3) | FWER **11,4%** | 11,4% | 11,4% | 0,057–0,171 |
| `max(3σ̂, 0,10)` — **aturan v0.2** (R1) | 0,00% | 0,87% | **8,47%** | 0,084–0,171 |
| `max(3σ̂, 0,05)` (R4) | 0,87% | 11,2% | 11,4% | 0,057–0,171 |
| Selang prediksi + lantai 0,10 (R2) | 0,00% | 0,48% | 4,21% | 0,087–0,233 |
| Persistensi 2 generasi (R5/R6) | ~0% | ~0% | 0,04–0,4% | 0,095–0,284 |
| **Bonferroni + selang prediksi (R7)** | **4,96%** | **5,55%** | **5,55%** | 0,078–0,233 |

**Kenapa v0.2 salah, secara aritmetika.** Uji membandingkan observasi baru dengan `μ̂` dari 5 seed, jadi variansi selisihnya `σ²(1 + 1/5)`, bukan `σ²`. Ambangnya juga memakai `σ̂` yang hanya punya 4 derajat bebas, sehingga harus memakai kuantil t, bukan 3. Ambang yang benar untuk galat per-look 1% adalah:

```text
t(0,99; 4) * σ̂ * sqrt(1 + 1/5) = 3,75 * 1,095 * σ̂ ≈ 4,10 σ̂
```

Aku memakai `3,0 σ̂` — **27% lebih rendah** dari yang seharusnya. Ditambah 180 uji tanpa koreksi multiplisitas, hasilnya bukan 0,3% melainkan dua digit.

**Konsekuensi yang harus masuk pra-registrasi:**

- Aturan primer: **R7** (Bonferroni atas 180 uji + selang prediksi + lantai 0,03 bpb).
- `MDE80` ditulis **sebelum** run. Untuk profil S, itu berarti kita hanya bisa mendeteksi degradasi ≥ 5–17% relatif.
- Kalimat wajib di laporan: *"Tidak terdeteksi" berarti "lebih kecil dari MDE", bukan "tidak ada".*
- Jika σ̂ M0 terukur > 0,06, tambah seed baseline (murah) sampai MDE turun — bukan menurunkan ambang.

---

## 3. Bagaimana sistem bisa mendeteksi kesalahannya sendiri

Kunci: **pisahkan galat operasional (bisa dideteksi mesin) dari galat interpretatif (tidak bisa).**

| Mekanisme | Menangkap | Tidak menangkap | Biaya |
| --- | --- | --- | --- |
| **Kontrol negatif** — dua model, korpus identik, seed berbeda | False positive detektor: setiap "collapse" di pasangan ini pasti artefak | Collapse yang nyata tapi lambat | 2 run |
| **Kontrol positif dengan dosis** — hapus 0/10/20/40% token langka, atau ganti 30% dokumen dengan repetisi | Detektor buta/tumpul; memastikan detektor monoton terhadap kerusakan nyata | Bias definisi kualitas | 4 run kecil |
| **Uji metamorphic** — acak urutan dokumen (harus invarian), duplikasi 5% (harus peka), ganti kata langka (harus peka di ekor) | Bug pipeline dan metrik yang tidak sensitif | Kesalahan konsep | menit |
| **Dua implementasi BPB independen** (per-token vs korpus) | Bug perhitungan | Kesalahan pemodelan | menit |
| **Golden fixture** — korpus mini + nilai metrik yang diharapkan | Regresi senyap saat kode berubah | Cakupan fixture | menit |
| **Analisis buta** — label arm ditukar sebelum analisis | Bias narasi | Interpretasi akhir | jam |
| **Hash pra-registrasi + kebijakan** | Perubahan aturan di tengah jalan | Niat yang tidak terucap | menit |
| **Fail-closed gate** — audit gagal berarti `invalid`, bukan "lanjut dengan peringatan" | Rantai yang tetap jalan di atas metrik rusak | — | — |
| **Audit eksternal (kamu)** | Semua galat interpretatif | Keterbatasan waktu/perhatianmu | — |

**Yang harus jujur diakui:** sistem tidak bisa membuktikan dirinya benar. Ia hanya bisa membuktikan dirinya **konsisten** dan **gagal dengan berisik**. Konsistensi bukan kebenaran. Karena itu setiap laporan wajib memuat bagian "apa yang bisa membatalkan kesimpulan ini".

**Satu kontrol yang aku tolak:** memakai AI kedua sebagai juri independen. Dua model dari keluarga yang sama berbagi bias yang sama; kesepakatan mereka bukan bukti, hanya korelasi. Penilai AI hanya dipakai sebagai **perlakuan yang diukur** (arm G), bukan sebagai hakim.

---

## 4. Kenyataan sandbox & protokol "akali"

### 4.1 Yang terukur, bukan diasumsikan

| Sumber daya | Nilai terukur |
| --- | --- |
| CPU | 2 vCPU Intel Xeon @ 2,60 GHz |
| RAM | 3 GB |
| GPU | tidak ada (tidak ada `nvidia-smi`) |
| Disk | 20 GB |
| numpy | tersedia (venv), puncak matmul **155–175 GFLOP/s** pada 256³–512³ |
| PyTorch | **tidak terpasang**; pemasangan belum diverifikasi pada batas RAM ini |

### 4.2 Model biaya (FLOPs ≈ 6·N·D untuk training, 2·N·T untuk generasi)

Asumsi efektif 20 GF/s untuk training dan 3 GF/s untuk generasi autoregresif (memory-bound). Angka ini harus **diukur ulang** di pilot; tabel ini alat perencanaan, bukan janji.

| Profil | Parameter | Token latih | Training | Generasi korpus | Total/model |
| --- | --- | --- | --- | --- | --- |
| **XS** | 0,5 jt | 0,2 jt | < 1 menit | ±1 menit | **±2 menit** |
| **S** | 8 jt | 3 jt | ±2 jam | ±3,6 jam | **±5,6 jam** |
| **M** | 30 jt | 10 jt | ±25 jam | ±13 jam | **±38 jam** |

Studi penuh (32 chain × 6 model) pada profil S: **±1.067 jam CPU ≈ 44 hari** di mesin ini. Itu bukan "sulit", itu tidak mungkin. Menyebutnya "bisa kalau dioptimalkan" adalah kebohongan yang halus.

### 4.3 Profil eksekusi berlapis (jawaban atas keterbatasan)

| Profil | Di mana | Skala | Waktu | Yang dihasilkan | Status klaim |
| --- | --- | --- | --- | --- | --- |
| **P0 — bukti hidup** | sandbox ini | XS 0,5 jt param, 200 rb token/generasi, 2 arm × 2 seed × 3 generasi | ±20–40 menit | Pipeline end-to-end berjalan; kurva pertama yang berisik | **bukan temuan**, hanya bukti mekanisme |
| **P1 — mini** | sandbox, proses latar + checkpoint | 2 jt param, 400 rb token, 2 arm × 2 seed × 4 generasi | ±3–5 jam (dicicil) | Arah efek, kalibrasi σ̂ nyata, MDE terukur | pratinjau, bukan publikasi |
| **P2 — penuh** | GPU eksternal lewat *offline executor* | 8 jt param, 3 jt token, 32 chain | ±120 GPU-jam | Klaim utama | dapat dilaporkan, dengan batasan |
| **P3 — cek skala** | GPU eksternal | 30 jt param, subset arm | ±40 GPU-jam | Uji apakah pola bertahan pada skala lebih besar | opsional |

**Kontrak offline executor** (inti dari "akali"):

1. `make bundle` menghasilkan arsip mandiri: kode, konfigurasi, hash korpus, seed, `policy.v1.json`, `PREREGISTRATION.md`, dan `graph_sha` — tanpa data mentah besar (korpus diregenerasi dari seed).
2. Pengguna menjalankan satu perintah di Colab/Kaggle. Skrip menolak berjalan bila hash konfigurasi tidak cocok.
3. `make ingest` memverifikasi: hash manifest, rantai lineage, versi kebijakan, dan **rekomputasi 10% metrik** dari checkpoint yang dikembalikan. Hasil yang tidak lolos tidak masuk analisis.
4. Konsekuensinya: sandbox tidak perlu GPU, dan hasil dari mesin luar tetap bisa diaudit secara kriptografis.

### 4.4 Disiplin checkpoint, pembunuhan proses, dan pemangkasan dinamis

- Setiap proses dibatasi **45 menit** *wall clock*; budgeter menghentikan dengan rapi di batas batch berikutnya sehingga kerja yang hilang ≤ 5 menit.
- Semua artefak beralamat-konten; resume bersifat idempoten (mengulang langkah tidak menduplikasi data).
- Anggaran dibagi berlapis: Tier 1 dulu. Aturan futilitas pra-registrasi menghentikan arm yang jelas nol dan memindahkan sisa anggaran ke arm yang mendekati ambang — **hanya untuk efek nol**, tidak pernah untuk efek positif.
- Bila throughput < 50% asumsi, profil diturunkan (P2→P1→P0) *sebelum* chain baru, dan klaim yang bisa ditarik ikut menyempit secara eksplisit.

### 4.5 Apa yang benar-benar bisa kukerjakan sekarang (tanpa GPU, tanpa PyTorch)

1. Kalibrasi statistik aturan deteksi — **sudah**, hasilnya di §2.
2. Kebijakan kegagalan yang bisa di-*lint* — **sudah**, `ops/policy.v1.json` (13 mode, 8 aksi otonom, 7 aksi manusia).
3. Tokenizer, manifest, sampler, auditor, dan gate **tanpa model** — bisa, karena semuanya deterministik.
4. P0 dengan model XS — **butuh PyTorch**; pemasangan belum diverifikasi (RAM 3 GB). Jika gagal, jalur alternatifnya implementasi matmul numpy untuk model XS (lambat tapi cukup) atau langsung ke P1 di GPU eksternal.

---

## 5. Matriks kegagalan & pemulihan

Ringkasan `ops/policy.v1.json` (13 mode). Setiap mode wajib punya sinyal, pemeriksa, rute, kewenangan, dan SLA — dijaga oleh `ops/policy_lint.py`.

| ID | Kegagalan | Keparahan | Kewenangan | Rute tercepat | MTTR |
| --- | --- | --- | --- | --- | --- |
| F1 | Test-anchor bocor | kritis | manusia | karantina + forensik | tidak ada pemulihan |
| F2 | Canary bocor antar-arm | kritis | manusia | karantina arm + forensik | ±30 menit |
| F10 | Bug metrik | kritis | manusia | fail-closed + forensik | ±120 menit |
| F12 | Kebijakan diubah saat run | kritis | manusia | hentikan semua | tidak ada pemulihan |
| F3 | M0 buruk sejak awal | tinggi | otonom | rollback ke pilot / turunkan profil | ±60 menit |
| F5 | Critic menolak > 30% | tinggi | otonom | rollback + karantina | ±20 menit |
| F7 | Throughput anjlok | tinggi | otonom | turunkan profil / hentikan chain | ±30 menit |
| F4 | Korpus rusak massal | sedang | otonom | regenerasi (maks 2) | ±15 menit |
| F8 | Proses dibunuh batas waktu | sedang | otonom | checkpoint + resume | ±5 menit |
| F11 | Sampling drift | sedang | otonom | sampling ulang (maks 3) | ±10 menit |
| F9 | Drift guru | sedang | manusia | batch baru sebagai versi terpisah | ±60 menit |
| F6 | Divergensi training | rendah | otonom | retry seed sama (maks 1) | ±20 menit |
| F13 | Futilitas (efek nol) | rendah | otonom | hentikan arm, realokasi | ±5 menit |

**Skenario pemulihan tercepat yang paling mungkin terjadi:**

1. **Collapse prematur di generasi 1–2.** Deteksi: Δ melampaui ambang R7 pada ≥2 seed, *dan* audit `/` critic tidak menunjukkan korupsi. Respons: arm **tidak** diulang dengan seed lain (itu akan menutupi hasil). Artefak disimpan, arm ditandai `premature`, dan bila terjadi di ≥50% arm, seluruh studi ditandai `invalid` dan berhenti.
2. **Kontaminasi.** Deteksi: hash test-anchor atau canary. Respons: hentikan, karantina, **jangan hapus**. Pemulihan = mengulang rantai dari generasi terakhir yang bersih; MTTR 30 menit untuk arm, bukan untuk menghapus bukti.
3. **Metrik rusak.** Deteksi: golden fixture gagal. Respons: semua gate berhenti. Pemulihan: perbaiki, jalankan ulang evaluasi dari checkpoint (tidak perlu training ulang) — ini keuntungan artefak beralamat-konten.

---

## 6. Batas otonomi: apa yang boleh diputuskan mesin

**Otonom (8 aksi):** regenerasi batch (≤2×, ≤10% token), retry seed sama (≤1×, hanya untuk NaN), sampling ulang campuran, karantina arm, rollback ke checkpoint terverifikasi, turunkan profil sebelum chain, hentikan arm karena futilitas, ulang fork dengan alasan tercatat.

**Harus manusia (7 aksi):** mengubah ambang/metrik/definisi keberhasilan; menambah atau menghapus arm/seed setelah run; menghapus artefak (karantina cukup); melanjutkan setelah kebocoran terkonfirmasi; menaikkan anggaran; menafsirkan hasil untuk publik; **mengubah berkas kebijakan itu sendiri**.

**Kelemahan struktural yang harus disebut:** kebijakan ini ditulis oleh pihak yang diawasi. Aku bisa menulis ambang yang menguntungkan diriku, dan aku bisa mengubahnya. Penangkalnya bukan teknis, melainkan prosedural: (a) hash kebijakan disimpan di setiap manifest run, (b) perubahan apa pun menaikkan versi dan muncul di dashboard, (c) `policy_lint.py` menolak kebijakan yang tidak konsisten, (d) kamu bisa membandingkan hash kapan saja. Ini kontrol yang **bisa ditembus** oleh pihak yang berniat curang; nilainya hanya sejauh aku tidak curang, dan itu harus kamu nilai sendiri.

---

## 7. Yang tidak bisa diselamatkan

1. **Aku menilai pekerjaanku sendiri.** Tidak ada cara teknis untuk menghilangkan ini di lingkungan satu agen. Yang bisa dilakukan: membuat penilaian itu terlihat (angka, hash, perintah verifikasi).
2. **"Realitas" di sini adalah tulisanku.** Tidak ada data manusia. Setiap generalisasi ke dunia nyata adalah lompatan yang tidak didukung data ini.
3. **Daya statistik rendah.** 3 seed, baseline 5 seed, 180 uji. Setelah koreksi, efek kecil tidak akan terdeteksi — dan itu tidak bisa diperbaiki tanpa menambah komputasi, bukan dengan analisis yang lebih pintar.
4. **Klaim skala.** 8 juta parameter tidak memberi tahu apa pun tentang model 100× lebih besar. Pola bisa berbalik.
5. **Audit ini sendiri.** Ia ditulis oleh model yang sama, dengan insentif yang sama. Nilainya terletak pada **angka yang bisa direproduksi** (`python3 analysis/calibration.py` menghasilkan tabel yang sama), bukan pada retorikanya.

---

## 8. Perubahan yang dibawa ke PLAN v0.3

| # | Perubahan | Alasan | Bagian |
| --- | --- | --- | --- |
| 1 | Aturan collapse diganti ke **R7** (Bonferroni + selang prediksi + lantai 0,03), dengan R1/R3 sebagai sensitivitas | FWER terukur 11,4% pada aturan lama | §10.5 |
| 2 | **MDE80 wajib** ditulis di pra-registrasi dan dilaporkan | Menghindari "tidak terdeteksi" dibaca sebagai "tidak ada" | §11 |
| 3 | **Profil eksekusi P0–P3** menggantikan satu matriks 120 GPU-jam | Sandbox 2 vCPU/3 GB tidak mungkin menjalankan studi penuh | §6.5, §20 |
| 4 | **Kontrak offline executor** (`make bundle` / `make ingest` + verifikasi hash) | Memindahkan komputasi berat tanpa kehilangan auditabilitas | §20 |
| 5 | **Kontrol negatif & positif wajib** sebelum Tier 1 | Menguji detektor, bukan hanya model | §20 |
| 6 | **Kebijakan kegagalan mesin-terbaca** + lint | Prosa tidak bisa menegakkan apa pun | `ops/policy.v1.json` |
| 7 | **Analisis buta** dan aturan bahasa untuk hasil nol | Mengurangi bias narasi saat menulis laporan | §20 |

---

## 9. Cara membantah dokumen ini

Kalau kamu ingin menguji apakah audit ini jujur, tiga hal ini cukup:

```bash
# 1. Reproduksi angka kalibrasi (harus sama persis; seed tetap)
cd experiments/ai-belajar-dari-ai && python3 analysis/calibration.py

# 2. Periksa kebijakan tidak punya lubang
python3 ops/policy_lint.py

# 3. Periksa klaim "belum ada model dilatih" masih benar
find experiments/ai-belajar-dari-ai -type f | sort
```

Kalau angka kalibrasi berubah tanpa perubahan kode, aku berbohong. Kalau `policy_lint` gagal tapi dashboard tetap hijau, dashboard-nya yang berbohong. Kalau ada checkpoint tersembunyi, klaim "0%" di dashboard adalah kebohongan.
