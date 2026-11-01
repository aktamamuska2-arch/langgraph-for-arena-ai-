# PREREGISTRATION.md — Pilot P1 (aturan R8)

| | |
| --- | --- |
| Versi | **1.2** |
| Tanggal kunci | 2026-11-01 |
| Status | **terkunci sebelum ada satu pun model P1 dilatih** |
| Menggantikan | definisi operasional R7 (PLAN v0.3 §10.5) untuk seluruh run P1 dan sesudahnya |
| Hash dokumen | tercatat di `ops/prereg.lock.json` (dihitung oleh `ops/prereg_lock.py`) dan ditampilkan di dashboard |
| Alat verifikasi | `make prereg-verify` (gagal bila berkas berubah tanpa versi baru) |
| Angka tabel | dihasilkan `python3 analysis/stats_r8.py --table`, tidak diketik tangan |

**Aturan amandemen.** Berkas ini tidak boleh diubah setelah hash dikunci. Perubahan apa pun wajib
menjadi versi 1.1 dengan entri di tabel amandemen (bagian 12) yang menyebut: apa yang berubah,
kapan, dan **apakah sudah ada model P1 yang dilatih saat perubahan dibuat**. Amandemen yang dibuat
setelah model P1 pertama ada akan ditandai `post-data` dan seluruh klaim P1 otomatis turun menjadi
`exploratory`.

---

## 1. Ruang lingkup dan batas klaim

P1 menguji satu pertanyaan: **apakah melatih ulang model bahasa pada keluarannya sendiri
menggeser distribasinya menjauh dari distribusi guru, dan apakah akumulasi data guru memperlambat
pergeseran itu.**

Yang boleh diklaim bila gate menyala: pergeseran pada model 2 juta parameter, korpus guru
>= 2 MB, rancangan 3 arm x 5 seed x 6 generasi ini.

Yang **tidak** boleh diklaim, apa pun hasilnya: generalisasi ke skala lain, klaim sebab-akibat
tentang mekanisme internal, klaim tentang data manusia, dan klaim "aman" tanpa uji ekuivalensi
(bagian 5.3).

---

## 2. Rancangan P1 (terkunci)

| Komponen | Nilai terkunci | Alasan |
| --- | --- | --- |
| Model | 2,0 jt parameter, byte-level 256, konteks 256, 6 lapis | Skala PLAN P1; cukup besar agar probe aritmetika tidak di lantai |
| Korpus guru T0 | **>= 2 MB**, non-templat, >= 12 domain | P0 gagal karena korpus templat 200 KB menekan baseline ke ekor (bpb 0,48) |
| Baseline M0 | **5 seed** (`101, 202, 303, 404, 505`), arsitektur sama | Redam derau σ̂ (df = 4) |
| Arm | A (replace), B1 (accumulate), C (kontrol tanpa self-training) | C memberi distribusi-nol untuk kalibrasi sinyal cedera |
| Seed rantai | 5 seed, bersarang dengan baseline | Pasangan seed mengontrol variasi inisialisasi |
| Generasi | 6 per rantai | Cukup untuk tren; anggaran dihitung di bagian 10 |
| Anggaran token | sama untuk semua model (epoch setara), bukan langkah tetap | Menyamakan biaya, bukan menyamakan jumlah update |
| Suhu sampling | 0,9 top-p 0,95, tetap di semua generasi | Sampling tidak boleh ikut "memburuk" agar bukan confound |

Arm C dilatih ulang dari T0 dengan seed berbeda tiap generasi dan **tidak pernah** melihat data
sintetis. C adalah pembanding untuk menjawab: "apakah pergeseran ini khas self-training, atau
hanya variasi training ulang?"

### 2.1 Gerbang penerimaan korpus T0 (dicek sebelum baseline dilatih)

T0 non-templat **tidak** diterima hanya karena ukurannya >= 2 MB. Ia harus lulus lima gerbang ini,
diperiksa otomatis oleh `analysis/t0_audit.py` (dibuat saat P1 disiapkan) dan dicatat di
`runs/p1/t0_audit.json`:

| # | Gerbang | Ambang | Kalau gagal |
| --- | --- | --- | --- |
| T1 | ukuran | >= 2.097.152 byte | T0 ditolak, P1 tidak mulai |
| T2 | anti-templat: **kriteria terkalibrasi v1.2** (lihat 2.1a) | lima sub-kriteria T2a-T2e | T0 ditolak |
| T3 | keseimbangan domain | >= 12 domain, tiap domain >= 5% byte | T0 ditolak |
| T4 | panjang dokumen | median 150-600 byte, p1 >= 80 byte, p99 <= 2.000 byte | T0 ditolak |
| T5 | kemiripan antar-dokumen | maksimum Jaccard 5-gram antar dokumen <= 0,5 | T0 ditolak |

Kegagalan gerbang **tidak** boleh dikompensasi dengan menurunkan ambang; yang turun adalah klaim,
bukan gerbangnya.

### 2.1a T2 terkalibrasi (ratifikasi v1.2, 2026-11-01)

**Koreksi pra-data.** Versi 1.1 menuntut porsi 8-gram unik >= 0,98. Syarat itu **mustahil** dan tidak
pernah diukur terhadap korpus mana pun sebelum dikunci: pada 2 MB, docstring pustaka standar Python
(teks yang tidak ditulis AI ini) hanya mencapai **0,3012**. Pada ukuran 156 KB, teks manusia mencapai
0,5324. Setiap korpus yang masih terbaca sebagai bahasa akan mengulang frasa pendek, sehingga angka
0,98 tidak bisa dicapai oleh konstruksi apa pun. Menjaga gerbang itu berarti P1 tidak akan pernah
berjalan — dan itu bukan kehati-hatian, itu kesalahan desain.

T2 v1.2 mengukur hal yang sebenarnya kita takutkan — **korpus yang bisa dihafal sehingga baseline
jatuh ke ekor seperti P0** — dengan lima sub-kriteria, empat di antaranya bebas-ukuran, dan semuanya
berlabuh pada dua jangkar eksternal: korpus P0 (lantai yang harus dilewati) dan teks manusia (plafon,
bukan syarat).

**Nilai jangkar yang dibekukan** (dihitung `analysis/t0_audit.py`, deterministik):

| Jangkar | uniq-8 | hapax-8 | zlib bpb | TTR | Markov CE orde-3 |
| --- | --- | --- | --- | --- | --- |
| Korpus P0 (156.697 B isi) | 0,1870 | 0,1223 | 1,189 | 0,0732 | 1,615 |
| Manusia (docstring stdlib, 2 MB) | 0,3012 | 0,1894 | 2,217 | 0,1973 | 2,714 |

**Sub-kriteria (semua wajib):**

| Kode | Metrik dan protokol ukur | Ambang | Alasan ambang |
| --- | --- | --- | --- |
| T2a | uniq-8 pada potongan isi sepanjang **156.697 B** (prosedur sampling §2.1b) | >= 1,50 x P0 = **0,2805** | korpus harus jelas lebih beragam daripada templat P0 pada ukuran yang sama |
| T2b | hapax-8 pada potongan yang sama | >= 1,50 x P0 = **0,1835** | idem, mengukur ekor, bukan rata-rata |
| T2c | zlib bits/byte pada seluruh isi **dan** >= 0,85 x manusia | >= 1,50 x P0 = **1,784** dan >= **1,884** | kompresibilitas bebas-ukuran; lantai manusia mencegah korpus acak lolos |
| T2d | TTR kata pada sampel 200 KB pertama (bebas-ukuran) | >= 1,40 x P0 = **0,1025** | ukuran kosakata |
| T2e | entropi silang Markov orde-3, split 80/20, pada potongan 156.697 B | >= 1,35 x P0 = **2,180** | proksi keterhafalan lokal tanpa melatih model |

**Pengakuan threshold fitting.** Kelipatan 1,50 / 1,50 / 1,50 / 1,40 / 1,35 saya tetapkan setelah
melihat angka korpus P1 dan angka P0. Itu tetap *threshold fitting*; yang membuatnya bisa
dipertanggungjawabkan: (a) ditetapkan dan dikunci **sebelum satu model P1 pun dilatih**; (b) berlabuh
pada dua besaran eksternal, bukan pada angka korpus ini; (c) T2 adalah gerbang korpus, bukan gerbang
hasil, sehingga tidak bisa dipakai mempermudah klaim collapse. Ratifikasi manusia dicatat di §12.

**Yang tidak berubah:** T2 tetap gerbang yang memblokir. Bila salah satu sub-kriteria gagal, T0
ditolak dan P1 tidak berjalan.

### 2.1b Protokol pengukuran T2 (deterministik)

1. Isi = gabungan kolom `<A>` dari split train, urut dokumen, tanpa markup.
2. Potongan 156.697 B diambil dengan `random.Random(3)`: panjang blok 600-1.200 karakter dengan
   jeda acak 0-400 karakter, diulang sampai ukuran tercapai, lalu dipotong tepat 156.697 byte.
3. zlib: `zlib.compress(data, 9)`, bits per byte atas seluruh isi (bukan potongan).
4. TTR: 200.000 karakter pertama isi, token `lower().split()`, unik/total.
5. Markov: orde-3 byte, penghalusan add-0,25 atas 256 simbol, dilatih pada 80% potongan, diuji 20%.
6. Referensi manusia: docstring `"""..."""` dari `/usr/lib/python3.11/**/*.py`, diacak dengan
   `random.Random(7)`, dipotong pada ukuran yang diminta.
7. Semua angka dilaporkan bersama-sama; tidak ada pemilihan metrik setelah melihat hasil.

### 2.2 Komposisi data tiap arm (terkunci)

- Ukuran keluaran sintetis per generasi: `N_syn = 0,30 * |T0|` byte (tetap untuk semua generasi).
- **Arm A (replace):** korpus training generasi g = 100% keluaran model generasi g-1, diambil
  `N_syn` byte, tanpa data guru.
- **Arm B1 (accumulate):** korpus = T0 digabung seluruh keluaran generasi 1..g-1, dengan bobot
  sampling **sama per sumber** (bukan per token). Konsekuensi yang saya nyatakan di muka: porsi guru
  menyusut seiring g (1/g). Ini varian "accumulate" Gerstgrasser et al. dan **bukan** varian
  "proporsi data nyata tetap" (Fu et al.); perbedaan itu dilaporkan, tidak diklaim sebagai hal yang sama.
- **Arm C (kontrol):** selalu 100% T0; yang berubah hanya seed training.
- Semua arm memakai anggaran token identik per model (epoch setara), bukan jumlah langkah tetap.

---

## 3. Estimand dan notasi

- `bpb_g` = bits-per-byte model generasi ke-g pada **anchor test guru yang dibekukan** (tidak
  pernah dipakai training, tidak pernah berubah antar generasi).
- `mu0`, `sigma0` = rata-rata dan simpangan baku sampel `bpb` dari n0 = 5 model baseline.
- `Delta_g = bpb_g - mu0`.
- `SE_pred = sigma0 * sqrt(1 + 1/n0)` — galat baku prediktif satu model baru terhadap mean baseline.
- `delta_min = 0,03` bpb (lantai kepraktisan, diwarisi dari R7).

Metrik primer **hanya** `anchor_bpb`. Semua metrik lain bersifat deskriptif atau sebagai syarat
corroboration; tidak ada yang boleh menggantikan metrik primer.

---

## 4. Protokol baseline (dijalankan sebelum arm)

1. Latih n0 = 5 model M0 pada T0 dengan seed terkunci.
2. Hitung `mu0`, `sigma0`, dan **MDE80** untuk aturan konfirmatori; tulis di `runs/p1/baseline.json`
   sebelum rantai pertama dimulai.
3. **Aturan eskalasi (pra-otorisasi, bukan p-hacking):** bila `sigma0 > 0,030`, tambah 5 seed
   baseline lagi (maksimum 15) dan hitung ulang. Dasar keputusan hanya σ̂, tidak pernah efek.
4. Bila setelah 15 seed `sigma0` masih > 0,030: P1 dinyatakan **underpowered**, tidak dijalankan,
   dan naik ke P2 (model lebih besar). Ini keputusan yang ditulis di muka, bukan setelah melihat hasil.

---

## 5. Aturan keputusan R8

R8 menggantikan R7. Tiga perubahan, semuanya **memperketat**, tidak melonggarkan:

| # | Perubahan | Alasan |
| --- | --- | --- |
| 1 | Kuantil **t Student** (df = n0 - 1) menggantikan normal | σ̂ diestimasi dari 5 seed; memakai normal menyembunyikan ketidakpastian σ̂. Ini juga memperbaiki **bug nyata R7**: `src/gate.py` menghitung `t` lalu tidak pernah memakainya |
| 2 | Sinyal cedera diperluas + aturan sadar-lantai | P0: 12/12 melewati ambang, 0 ditandai collapse, karena seluruh sinyal cedera mati di lantai |
| 3 | Hasil **tiga-nilai** (collapse / stabil / inconclusive) dengan uji ekuivalensi | Mencegah "tidak terdeteksi" berubah menjadi "aman" — kegagalan deteksi yang menyamar sebagai hasil negatif |

### 5.1 Uji konfirmatori (hanya ini yang boleh jadi klaim)

Dijalankan pada generasi ke-6, satu uji per arm, m = 3, alpha family-wise = 0,05:

```
thr_conf = max( t(1 - 0,05/(2*3), df = n0-1) * SE_pred , delta_min )
```

dengan `t(0,991667, df=4) = 3,9608`. Keputusan per arm:

- **collapse** bila `Delta_6 >= thr_conf` **dan** syarat cedera (bagian 6) terpenuhi;
- **stabil** bila batas atas CI satu-sisi 95% untuk `Delta_6` `< delta_eq` (bagian 5.3);
- **inconclusive** untuk semua kasus lain.

### 5.2 Pemantauan berurutan (bukan klaim)

Setiap titik (arm x seed x generasi) dipantau dengan m = 90 uji, memakai normal (z = 3,4524).
Label pemantauan hanya untuk keselamatan/keputusan berhenti dan **wajib dilaporkan sebagai
`screening`**, tidak pernah sebagai bukti. Alasannya: menghentikan arm lebih awal adalah keputusan
operasional; menyimpulkan adalah hak uji konfirmatori.

### 5.3 Uji ekuivalensi (anti "silence = safety")

`delta_eq = max(0,03 ; 0,05 * mu0)`.

"Stabil" hanya boleh ditulis bila seluruh CI satu-sisi 95% untuk `Delta_6` berada di bawah
`delta_eq`. Bila CI memuat `delta_eq` tetapi tidak melewati `thr_conf`, hasilnya **inconclusive**,
dan judul laporannya wajib memuat kata "inconclusive", bukan "tidak ada collapse".

---

## 6. Keluarga sinyal cedera (R8) dan aturan sadar-lantai

Syarat corroboration R7 hanya punya dua sinyal; keduanya mati di P0. R8 memakai **enam sinyal yang
dikelompokkan ke dalam empat kelompok independen**. Pengelompokan ini wajib: sinyal yang mengukur
fenomena yang sama tidak boleh dihitung dua kali saat menentukan tingkat keparahan (bagian 6.3).

| Kelompok | Sinyal | Definisi operasional | Ambang menyala | Catatan |
| --- | --- | --- | --- | --- |
| **G1 struktur** | F1 | porsi dokumen tanpa bagian jawaban `<A>` | naik **>= 15pp** | dihitung atas >= 100 dokumen |
| **G1 struktur** | F2 | porsi dokumen tanpa penutup `<E>` (truncated) | naik **>= 20pp** | F1 dan F2 **satu kelompok**: keduanya kegagalan struktur, korelasinya tinggi |
| **G2 keragaman** | F3 | type-token ratio tingkat kata per dokumen | turun **>= 20% relatif** atau **>= 3 sigma_baseline** | `distinct-2` **dibuang**: di byte-level ia justru naik saat teks memburuk (P0: 0,910 -> 0,949) |
| **G2 keragaman** | F4 | porsi dokumen dengan Jaccard 5-gram >= 0,8 terhadap dokumen lain | naik **>= 5pp** | menangkap mode collapse yang lolos dari F3 |
| **G3 distribusi** | F5 | entropi distribusi jawaban untuk prompt yang sama, 5 sampel | turun **>= 30%** | probe langsung terhadap penyempitan distribusi |
| **G4 kapabilitas** | F6 | akurasi 40 item dua-digit | turun **>= 10pp** | hanya sah bila baseline >= 25/40 (aturan lantai) |

Catatan amandemen v1.1: versi 1.0 mendefinisikan F1 sebagai "tanpa `<A>` **atau** `<E>`" sehingga F1
dan F2 tumpang tindih dan bisa dihitung sebagai dua keluarga independen. Itu kesalahan yang sama
dengan P0 (menghitung satu fenomena dua kali). Sekarang F1 hanya `<A>`, F2 hanya `<E>`, dan keduanya
berada di kelompok yang sama sehingga **tidak pernah** dihitung sebagai dua bukti.

### 6.1 Aturan sadar-lantai (mekanis, bukan pertimbangan)

Sebelum rantai dijalankan, setiap metrik sekunder diberi label otomatis dari baseline 5 model:

| Label | Kondisi (dihitung otomatis) | Konsekuensi |
| --- | --- | --- |
| `at_floor` | mean baseline <= 2/n_item (mis. arith 0/40) | **dikeluarkan** dari corroboration; dilaporkan sebagai "tidak diskriminatif" |
| `uninformative` | SD_baseline >= 0,5 * mean_baseline | dikeluarkan dari corroboration |
| `informative` | selain di atas | dipakai untuk corroboration |

Metrik yang dikeluarkan **tidak pernah** dihitung sebagai "tidak ada cedera". Ini persis kesalahan
P0: `arith_acc` = 0,00 diperlakukan sebagai bukti ketiadaan cedera, padahal itu bukti probe tidak
mampu mengukur.

Bila jumlah keluarga `informative` < 2, arm otomatis berlabel **inconclusive-harm** untuk seluruh
generasi, dan P1 naik ke P2 tanpa klaim. Tidak ada penambahan sinyal ad-hoc setelah data terlihat.

### 6.2 Ambang ukuran sampel (wajib, bukan saran)

| Probe | Minimum P1 | Granularitas yang dihasilkan |
| --- | --- | --- |
| Aritmetika | **40 item** | 2,5pp |
| Dokumen generasi per evaluasi | 100 dokumen | 1pp untuk F1/F2/F4 |
| Prompt unik untuk F5 | 20 prompt x 5 sampel | — |
| Anchor test | >= 100 KB teks guru beku | — |

### 6.3 Tingkat keparahan (mencegah label collapse yang murah)

- `Delta_6 >= 2 * thr_conf` -> cukup **1 kelompok** cedera menyala.
- `thr_conf <= Delta_6 < 2 * thr_conf` -> wajib **2 kelompok berbeda** (G1-G4), bukan dua sinyal
  dalam kelompok yang sama.
- Tidak ada kelompok cedera -> label `watch`, bukan `collapse`, dan wajib dilaporkan sebagai
  "melewati ambang tanpa cedera terukur" (kejadian P0, dan itu **bukan** hasil negatif).

### 6.4 Kalibrasi nol sinyal cedera (dijalankan sebelum rantai, setelah baseline)

Memakai 5 model baseline + arm C, bangkitkan 1.000 replika bootstrap set evaluasi dan hitung laju
salah-nyala tiap keluarga di bawah nol. Keluarga dengan laju salah-nyala > 5% pada ambangnya
dinaikkan ambangnya ke persentil 95 empiris, **dikunci sebelum rantai pertama**, dan dilaporkan.
Tanpa langkah ini, menambah sinyal cedera hanya menambah jalan menuju kesimpulan keliru.

---

## 7. Metrik sekunder: status yang sudah ditetapkan di muka

| Metrik | Status P1 | Alasan |
| --- | --- | --- |
| `anchor_bpb` | **primer** | satu-satunya uji inferensial |
| `degenerate_rate`, `format_compliance` | corroboration (F1/F2) | paling sensitif di P0 |
| `TTR` kata, `zlib bits/byte` | corroboration (F3) | pengganti `distinct-2` yang rusak di byte-level |
| near-dup Jaccard | corroboration (F4) | deteksi mode collapse |
| entropi jawaban | corroboration (F5) | probe penyempitan distribusi |
| `arith_acc` (40 item) | corroboration (F6) **hanya bila baseline >= 25/40** | selain itu `at_floor` |
| `style_div` char-3gram | **deskriptif saja** | datar di P0 (0,94-0,96) walau teks rusak; tidak layak jadi syarat |
| `distinct-2` | **dibuang** | arahnya salah di byte-level |
| `repetition_rate` (P0) | diganti F4 | versi lama selalu 0,00 |

Metrik deskriptif tetap dilaporkan lengkap; hanya tidak dipakai untuk memberi label.

---

## 8. Protokol buta (blinding)

1. Sebelum model P1 pertama dilatih, `ops/blind.py seal` membuat pemetaan acak arm -> label
   `X1/X2/X3` memakai `secrets.token_hex(16)`, menyimpannya di `.blind/key.json` (**tidak masuk git**),
   dan menulis komitmen `ops/blind_commitment.json` berisi `sha256(key)` + waktu + jumlah arm.
2. Komitmen di-commit **sebelum** eksekusi. Semua nama berkas hasil memakai label buta.
3. `analysis/p1_analyze.py` hanya boleh membaca label buta; berkas kunci tidak pernah dibuka oleh
   jalur analisis (dijaga sebagai aturan keras + diperiksa `grep` di CI pra-analisis).
4. Kesimpulan ditulis ke `analysis/p1_conclusions.md`, lalu **hash-nya dicatat**.
5. `ops/blind.py reveal --conclusions-sha256 <hash>` menolak berjalan bila hash tidak cocok, lalu
   mencetak pemetaan dan mencatat peristiwa unblinding.
### 8.3 Kontrak `analysis/p1_analyze.py`

- Membaca `runs/p1/metrics.jsonl` yang sudah berlabel buta (`X1/X2/X3`), bukan nama arm.
- Tidak pernah membuka `.blind/key.json`; skrip memeriksa kode sumbernya sendiri dan menolak berjalan
  bila ada rujukan ke berkas kunci.
- Menghitung `mu0`, `sigma0`, `SE_pred`, ambang konfirmatori (t, m=3) dan pemantauan (normal, m=90),
  enam sinyal cedera, label lantai, tingkat keparahan berbasis kelompok, serta uji ekuivalensi.
- Menulis `analysis/p1_conclusions.md` beserta hash; `ops/blind.py reveal` hanya mau berjalan bila
  hash itu diberikan dan cocok.
- Mode `--check` memverifikasi kelengkapan masukan tanpa menghitung; mode `--fixture` menjalankan
  seluruh perhitungan pada data P0 sebagai uji unit, menulis ke `/tmp`, tidak pernah ke laporan P1.

6. Keterbatasan yang saya akui: agen yang sama bisa membaca kunci. Ini buta **prosedural**, bukan
   kriptografis. Penangkalnya hanya komitmen hash: kalau urutannya dilanggar, jejaknya terlihat.

---

## 9. Rencana analisis dan tabel wajib

Analisis dijalankan oleh `analysis/p1_analyze.py` (lihat §8.3), bukan oleh perhitungan manual.

Setiap arm dilaporkan dengan baris: `bpb_6`, `Delta_6`, `CI95 satu-sisi`, `thr_conf`, `label`,
keluarga cedera yang menyala, jumlah keluarga `informative`, MDE80, dan n seed. Ditambah:

- semua seed, tanpa pemilihan, tanpa pembuangan run "gagal";
- tren per generasi sebagai **deskriptif** (P1 tidak punya daya untuk uji tren inferensial);
- tabel keluarga metrik yang `at_floor`/`uninformative` beserta alasannya;
- pernyataan eksplisit bila hasil `inconclusive`.

---

## 10. Anggaran dan gate eksekusi

95 model (5 baseline + 90 rantai) x 2 jt parameter. Proyeksi diukur dari satu model pertama:

- bila proyeksi total <= 36 jam: jalankan 6 generasi;
- bila 36-72 jam: turunkan ke 4 generasi untuk **semua** arm (keputusan otomatis, sebelum hasil);
- bila > 72 jam: P1 dibatalkan dan naik ke P2, dilaporkan sebagai kegagalan anggaran, bukan hasil.

Eksekusi P1 hanya di executor offline (keputusan D8 = c). Sandbox 2 vCPU/3 GB **tidak** memenuhi
anggaran ini; menjalankan P1 di sandbox akan melanggar pra-registrasi dan hasilnya tidak sah.

---

## 11. Apa yang akan terjadi bila hasilnya buruk untuk hipotesis saya

- Bila arm C juga bergeser sebesar arm A: kesimpulan "self-training menyebabkan pergeseran" **gugur**;
  yang tersisa hanya "melatih ulang menyebabkan pergeseran". Itu akan ditulis sebagai kegagalan
  desain, bukan diselamatkan dengan analisis tambahan.
- Bila seluruh keluarga `informative` < 2: P1 berakhir `inconclusive-harm`, dan biaya P1 diakui
  sebagai biaya kalibrasi metrik.
- Bila `Delta_6` besar tetapi semua keluarga cedera null: hasilnya `watch` + laporan eksplisit bahwa
  bpb dan metrik permukaan tidak sepakat — bukan "collapse".

---

## 12. Tabel amandemen

| Versi | Tanggal | Perubahan | Model P1 sudah dilatih? |
| --- | --- | --- | --- |
| 1.0 | 2026-11-01 | Versi awal: R8, enam keluarga cedera, aturan lantai, uji ekuivalensi, protokol buta | belum |
| 1.1 | 2026-11-01 | **Pra-data.** (a) F1/F2 didefinisikan ulang dan dikelompokkan ke G1 agar tidak dihitung sebagai dua bukti; (b) ditambah gerbang penerimaan korpus T0 (2.1); (c) ditambah komposisi data tiap arm (2.2) yang di v1.0 belum dinyatakan | belum |
| 1.2 | 2026-11-01 | **Koreksi metodologis pra-data, diratifikasi manusia.** T2 absolut (>= 0,98) diganti kriteria terkalibrasi T2a-T2e (§2.1a) + protokol ukur deterministik (§2.1b). Alasan: syarat lama mustahil — teks manusia pada 2 MB hanya 0,3012. Dinyatakan terbuka bahwa kelipatan ambang ditetapkan setelah melihat angka korpus sendiri; ratifikasi dicatat di sini. | belum |

---

## 13. Angka terkunci (dihitung, bukan diketik)

Nilai di bawah berasal dari `python3 analysis/stats_r8.py --table` dengan n0 = 5 (df = 4),
`delta_min` = 0,03, alpha = 0,05, daya 80%. Ambang konfirmatori memakai m = 3; pemantauan m = 90.

| sigma0 (bpb) | SE_pred | ambang konfirmatori (m=3) | ambang pemantauan (m=90) | MDE80 konfirmatori | `delta_eq` bila mu0 = 0,90 |
| --- | --- | --- | --- | --- | --- |
| 0,0050 | 0,0055 | 0,0300 | 0,0549 | 0,0263 | 0,0450 |
| 0,0100 | 0,0110 | 0,0434 | 0,1099 | 0,0526 | 0,0450 |
| 0,0126 | 0,0138 | 0,0547 | 0,1384 | 0,0663 | 0,0450 |
| 0,0150 | 0,0164 | 0,0651 | 0,1648 | 0,0789 | 0,0450 |
| 0,0200 | 0,0219 | 0,0868 | 0,2197 | 0,1052 | 0,0450 |
| 0,0300 | 0,0329 | 0,1302 | 0,3296 | 0,1578 | 0,0450 |
| 0,0400 | 0,0438 | 0,1736 | 0,4395 | 0,2104 | 0,0450 |
| 0,0600 | 0,0657 | 0,2603 | 0,6592 | 0,3156 | 0,0450 |

Baris `sigma0 = 0,0126` hanyalah **ilustrasi dari P0**, bukan nilai yang diharapkan di P1: korpus
P1 >= 2 MB non-templat diproyeksikan menaikkan `mu0` dan `sigma0` sekaligus.

Catatan penting: tabel di atas memakai df = 4 dan kuantil ekstrem, sehingga ambang lebih tebal
daripada R7. Ini harga jujur dari mengestimasi σ dari 5 seed. Bila `sigma0` P1 ternyata 0,03,
ambang konfirmatori menjadi 0,1302 bpb — artinya hanya pergeseran >= 14% dari `mu0` = 0,90 yang
bisa disebut collapse. Bila itu terjadi, laporan wajib menyebut P1 **berdaya rendah untuk efek
kecil**, dan ekuivalensi (bagian 5.3) menjadi satu-satunya cara mengklaim stabilitas.

---

## 14. Rujukan

- PLAN.md v0.3.1 §10.5 (R7), §10.7 (kandidat R8), §11 (pra-registrasi), §19 (riwayat).
- `REPORT-P0.md` (kegagalan R7 secara empiris), `runs/p0/metrics.jsonl` (data mentah).
- `analysis/calibration_results.md` (FWER/power/MDE R1-R7).
- `analysis/stats_r8.py` (sumber semua angka di bagian 13).
