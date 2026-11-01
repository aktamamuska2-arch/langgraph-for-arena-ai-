# Laporan Pilot P0

**Status: bukti mekanisme, bukan temuan ilmiah.**

P0 dijalankan di sandbox (2 vCPU, 3 GB RAM, tanpa GPU) untuk membuktikan bahwa rantai
guru → T0 → M0 → generasi → validasi → training → evaluasi → gate berjalan end-to-end,
dan untuk mendapatkan `sigma0` pertama dari variasi antar-seed.

- Model: XS. 481.344 parameter (byte-level. vocab 256. konteks 96)
- Baseline M0: 4 seed -> mu0 = 0.4848 bpb, sigma0 = 0.0126 bpb
- Chain: [('A', 13), ('A', 1337), ('B1', 13), ('B1', 1337)]
- Langkah training per model: 600 (anggaran token identik)

## Baseline M0

| seed | anchor_bpb | arith_acc | distinct2 | style_div |
| --- | --- | --- | --- | --- |
| 7 | 0.4755 | 0.00 | 0.931 | 0.944 |
| 13 | 0.4739 | 0.08 | 0.919 | 0.946 |
| 1337 | 0.4892 | 0.00 | 0.884 | 0.935 |
| 2024 | 0.5008 | 0.00 | 0.906 | 0.943 |

mu0 = **0.4848** bpb, sigma0 = **0.0126** bpb. Ambang R7 dihitung dari sigma0
dengan koreksi Bonferroni 180 uji dan selang prediksi (PLAN v0.3 §10.5).

## Rantai per arm

| arm | seed | gen | anchor_bpb | delta | gate | distinct2 | arith | rep | style_div | reject | mix_tvd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 13 | 1 | 1.0045 | +0.5197 | watch | 0.940 | 0.00 | 0.00 | 0.954 | 0% | 0.000 |
| A | 13 | 2 | 1.7305 | +1.2456 | watch | 0.977 | 0.00 | 0.00 | 0.919 | 0% | 0.000 |
| A | 13 | 3 | 2.0308 | +1.5460 | watch | 0.953 | 0.00 | 0.00 | 0.951 | 0% | 0.000 |
| A | 1337 | 1 | 0.9556 | +0.4708 | watch | 0.959 | 0.00 | 0.00 | 0.972 | 0% | 0.000 |
| A | 1337 | 2 | 1.4416 | +0.9568 | watch | 0.947 | 0.00 | 0.00 | 0.971 | 0% | 0.000 |
| A | 1337 | 3 | 1.8257 | +1.3409 | watch | 0.945 | 0.00 | 0.00 | 0.961 | 0% | 0.000 |
| B1 | 13 | 1 | 0.7494 | +0.2646 | watch | 0.968 | 0.00 | 0.00 | 0.964 | 0% | 0.000 |
| B1 | 13 | 2 | 0.8169 | +0.3321 | watch | 0.967 | 0.00 | 0.00 | 0.963 | 0% | 0.000 |
| B1 | 13 | 3 | 0.9111 | +0.4262 | watch | 0.969 | 0.00 | 0.00 | 0.960 | 0% | 0.001 |
| B1 | 1337 | 1 | 0.7859 | +0.3010 | watch | 0.969 | 0.00 | 0.00 | 0.954 | 0% | 0.000 |
| B1 | 1337 | 2 | 0.8150 | +0.3301 | watch | 0.934 | 0.00 | 0.00 | 0.967 | 0% | 0.000 |
| B1 | 1337 | 3 | 0.9229 | +0.4380 | watch | 0.870 | 0.00 | 0.00 | 0.962 | 0% | 0.000 |

## Agregat per arm

| arm | gen | n | bpb | delta | train loss | distinct2 | style_div | degenerate | truncated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 1 | 2 | 0.9801 | +0.4952 | 0.893 | 0.949 | 0.963 | 0.31 | 54% |
| A | 2 | 2 | 1.5860 | +1.1012 | 0.698 | 0.962 | 0.945 | 0.40 | 52% |
| A | 3 | 2 | 1.9283 | +1.4434 | 0.544 | 0.949 | 0.956 | 0.35 | 48% |
| B1 | 1 | 2 | 0.7676 | +0.2828 | 0.668 | 0.968 | 0.959 | 0.31 | 53% |
| B1 | 2 | 2 | 0.8159 | +0.3311 | 0.661 | 0.951 | 0.965 | 0.02 | 24% |
| B1 | 3 | 2 | 0.9170 | +0.4321 | 0.779 | 0.919 | 0.961 | 0.31 | 44% |

## Temuan P0

### 1. Pipeline berjalan end-to-end, dan itu satu-satunya klaim kuat di sini

17 node selesai tanpa intervensi manual, 16 model dilatih, 16 checkpoint tersimpan,
state machine dapat di-resume (state.json + done.json), dan seluruh metrik tercatat di
`runs/p0/metrics.jsonl`. Wall-clock 22 menit 23 detik pada 2 vCPU/3 GB tanpa GPU.

### 2. sigma0 pertama sudah ada — dan ambang R7 sebanding dengan efek yang kita cari

mu0 = 0.4848 bpb, sigma0 = 0.0126 bpb (CV 2.6%), ambang R7 = 0.0488 bpb.
Artinya ambang praktis 10% di atas baseline. Dengan korpus yang sangat bertemplat,
baseline-nya sendiri rendah (bpb 0,48), sehingga ruang degradasi yang bisa diukur lebar.

### 3. Degradasi arm A jauh melampaui ambang, dan konsisten di kedua seed

- A generasi 1: bpb 0.9801 (delta +0.4952 = 10.2x ambang)
- A generasi 2: bpb 1.5860 (delta +1.1012 = 22.6x ambang)
- A generasi 3: bpb 1.9283 (delta +1.4434 = 29.6x ambang)
- B1 generasi 1: bpb 0.7676 (delta +0.2828 = 5.8x ambang)
- B1 generasi 3: bpb 0.9170 (delta +0.4321 = 8.9x ambang)

Semua titik sudah melewati ambang, tetapi tidak satu pun dinyatakan `collapse` oleh gate.
Itu bukan kosmetik: ambang saja tidak cukup, dan aturan corroboration saya rusak (lihat #4).

Rasio A/B1 pada generasi 3 = 3.34x. Arah efeknya sesuai literatur
(akumulasi memperlambat, bukan menghilangkan), tetapi P0 tidak punya daya untuk menguji itu:
2 seed, 3 generasi, satu keluarga korpus.

### 3b. Ukuran sampel yang kecil membatasi tafsir

Evaluasi per model memakai 24 prompt generasi dan 12 item aritmetika. `degenerate_rate`
karena itu bergerak dalam langkah 1/24 (~4,2pp) dan tidak boleh dibaca sebagai tren; `arith_acc`
bergerak dalam langkah 1/12 (~8,3pp), sehingga aturan 'turun 10pp' hanya butuh dua item berbeda.


### 4. Gate saya sendiri gagal menandai collapse — ini temuan metodologis yang paling penting

Status gate: 12 watch, 0 collapse, 0 ok
dari 12 model rantai. **0 model memicu sinyal cedera kedua**,
padahal bpb naik sampai +1.44. Penyebabnya struktural:

- `arith_acc` baseline sudah 0,00 (3 dari 4 seed) sehingga aturan 'turun >= 10pp' mustahil menyala.
- `distinct2` justru **naik** saat teks memburuk (0.910 -> 0.949),
  karena gibberish byte-level punya lebih banyak bigram unik.
- `style_div` datar (0.956 vs 0.942) — TF-IDF char-3gram tidak menangkap kerusakan sintaks.
- `repetition_rate` 0,00 di semua model.

Konsekuensinya: aturan R7 + cedera-kedua **under-call** dalam rezim ini. Saya tidak mengubah
aturan setelah melihat data (itu tepatnya yang dilarang di PLAN §10.5); sebagai gantinya P0
melahirkan kandidat **R8** yang harus dipraregistrasi sebelum P1:

1. Sinyal cedera ditambah: kenaikan `degenerate_rate` >= 15pp, `truncated` >= 20pp, dan
   penurunan `format_compliance` (dokumen berakhir `<E>`) >= 20pp.
2. Aturan sadar-lantai: bila metrik sekunder sudah di lantai (baseline 0,00) atau arahnya
   tidak informatif, metrik itu dikeluarkan dari syarat corroboration, bukan dihitung sebagai 'tidak cedera'.
3. Efek ukuran: setiap model P1 mengevaluasi minimal 40 item aritmetika agar `arith_acc` punya granularitas 2,5pp.

### 5. Model bisa menipu diri: loss turun, fidelitas jatuh

Arm A seed 13: loss pada korpus sendiri 0.96 -> 0.76 -> 0.40, sementara anchor bpb 1.00 -> 1.73 -> 2.03.
Model makin lancar meniru keluarannya sendiri dan makin jauh dari distribusi guru. Ini ilustrasi
konkret mengapa evaluasi harus dari data guru yang dibekukan, bukan dari loss training.

### 6. Format bertahan lebih lama daripada isi

Dokumen tanpa penutup `<E>`: arm A 54%/52%/48% (gen 1-3), B1 53%/24%/44%.
Seluruh dokumen tetap lolos filter lenient (100% punya `<D>`/`</P>`/`<A>`), jadi collapse di sini
bukan kegagalan format total melainkan erosi isi di dalam templat yang masih benar.

### 7. Yang tidak boleh diklaim dari P0

- Bukan bukti bahwa model 8 jt parameter akan collapse dengan laju yang sama.
- Bukan bukti bahwa B1 aman: B1 juga sudah di atas ambang sejak generasi 1.
- Bukan estimasi effect size: laju degradasi kemungkinan besar didorong oleh korpus templat
  berukuran 200 KB dan model 0,48 jt parameter, bukan oleh sifat umum self-training.
- Bukan uji statistik: 2 seed per arm, tanpa koreksi, tanpa blinding (D9 baru berlaku di P1).

## Konsekuensi untuk P1 (belum dijalankan)

1. Praregistrasi R8 di PLAN sebelum P1, dengan angka di atas sebagai dasar, bukan sebagai hasil.
2. Baseline minimal 5 seed; 3 arm (A, B1, dan kontrol tanpa self-training) x 5 seed x 6 generasi.
3. Korpus guru non-templat minimal 2 MB, sehingga bpb baseline tidak berada di ekor rendah.
4. Evaluasi 40+ item aritmetika dan 200+ prompt generasi per titik, agar metrik tidak di lantai.
5. Orkestrasi dipindah ke LangGraph (saat ini state machine dengan node identik, bukan LangGraph).

## Cara menjalankan ulang

```bash
cd experiments/ai-belajar-dari-ai
python -m src.orchestrate          # idempoten: melewati node yang sudah selesai
python -m src.report_p0            # bangun ulang laporan ini + analysis/p0_metrics.csv
make check                         # lint kebijakan + build dashboard
```

## Catatan eksekusi

- Semua langkah selesai: ya
- Total model dilatih: 16
- Total byte sintetis dihasilkan: 749.133

