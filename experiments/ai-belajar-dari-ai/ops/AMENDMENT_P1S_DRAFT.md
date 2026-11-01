# DRAFT — Amandemen v1.3: profil P1-sandbox

Status: **DRAF, BELUM DITERAPKAN, BELUM DIKUNCI.** PREREGISTRATION.md masih v1.2
(`dd8585c9…`). Tidak ada pelatihan P1 yang boleh dimulai sebelum berkas ini
diratifikasi pengguna, diterapkan ke PREREGISTRATION.md, dan dikunci ulang.

## 1. Mengapa amandemen ini ada

P1 v1.2 dipra-registrasi dengan asumsi komputasi eksternal: 3 arm × 5 seed ×
6 generasi pada profil ±2 juta parameter, konteks 256, N_syn = 0,30|T0|.
Pengguna membatalkan asumsi itu (2026-11-01): P1 dijalankan **di sandbox Arena AI
ini, oleh asisten**, dengan desain berorientasi efisiensi. Sandbox: 2 inti CPU,
tanpa GPU.

Konsekuensinya harus eksplisit: **P1-sandbox bukan P1 penuh.** Ia adalah pilot
bersegel yang menguji mekanisme pada anggaran yang muat, bukan replikasi
konfirmatori berskala.

## 2. Pengukuran yang menjadi dasar (bukan taksiran)

| Besaran | Angka terukur | Cara ukur |
| --- | --- | --- |
| Inti CPU | 2 (`nproc`) | `nproc` |
| Thread optimal | 2 → 17.796 tok/s; 4 → 14.984; 8 → 12.436 | benchmark b32×256, 858.880 param |
| Latih (b32, konteks 256, 969.472 param) | ~15 rb tok/s (konservatif) | benchmark forward+backward |
| Generasi tanpa cache (P0) | 108 tok/s | benchmark 2,32 jt param |
| Generasi dengan KV-cache (baru) | 4.547–6.244 tok/s (b64) | `src/gen_fast.py`, paritas logits 4,8e-7 |
| Efek cache | **~20–40×** | perbandingan di atas |

Temuan ini yang membalik kesimpulan "189 jam": biaya generasi turun dari 85%
anggaran menjadi ~40%; pelatihan kini dominan.

## 3. Perubahan parameter yang diminta

| Parameter | v1.2 (terkunci) | v1.3 (diusulkan) | Alasan |
| --- | --- | --- | --- |
| Parameter model | ±2.000.000 | **969.472** (d_model 112, 6 lapis, 4 kepala) | anggaran; 2 jt param → ±9 rb tok/s → 29 jam latih saja |
| Lapisan | 6 | 6 (tetap) | perubahan sekecil mungkin |
| Konteks | 256 | 256 (tetap) | — |
| Token latih per model | ~9,2 jt (4 epoch) | **2.301.328 (1 epoch)** | 4× lebih murah; semua arm identik → perbandingan tetap berpasangan |
| Seed baseline | 5 | **5 (tetap)** | gerbang "baseline ≥5 seed" di v1.2 tidak dilonggarkan |
| Seed rantai | 5 | **3** | df=2; SE rata-rata rantai naik √(5/3)=1,29× |
| Generasi | 6 | **3** | 2 transisi (M1→M2→M3), masih bisa melihat arah |
| N_syn | 0,30\|T0\| = 690.398 B | **tetap 690.398 B** | tidak dipotong: arm A butuh data sintetis yang cukup |
| Arm | A (replace), B1 (accumulate), C (T0 saja) | tetap | mekanisme tidak diubah |
| Ambang T2/R8 | terratifikasi | **tidak diubah** | aturan pengguna: ambang tidak diubah setelah P1 dimulai |
| Blinding X1/X2/X3 | disegel `b5de699a…` | tetap, dipakai runner | D9 |

Jumlah model: 5 baseline + (3 seed × 2 arm × 3 generasi) = **23 model**.
Arm C tidak dihitung per generasi: dengan seed dan data identik, C di setiap
generasi bit-identik dengan baseline — menambah salinan tidak menambah informasi.
Ini dicatat sebagai deviasi, bukan dihapus diam-diam.

## 4. Anggaran waktu (dari angka terukur, bukan harapan)

| Tahap | Volume | Laju | Waktu |
| --- | --- | --- | --- |
| Latih 23 model | 23 × 2,301,328 tok = 52,9 jt | 15.000 tok/s | **59 mnt** |
| Generasi 18 peristiwa | 18 × 690.398 B = 12,4 jt | 4.500 tok/s | **46 mnt** |
| Evaluasi 23 model | 23 × ~18 rb token + anchor 160 KB | ~5.000 tok/s | **12 mnt** |
| Audit T0 (ulang, verifikasi) | tetap | — | 2 mnt |
| **Total** | | | **±2,0 jam** |

Dijalankan sebagai proses latar yang **dapat dilanjutkan** (`runs/p1/state.json`),
dengan progres tertulis ke `runs/p1/metrics.jsonl` dan dasbor. Bila terputus,
tidak mengulang model yang sudah selesai.

## 5. Kehilangan yang harus dinyatakan di muka

1. **Daya uji.** 3 seed rantai → SE 1,29× lebih lebar daripada 5 seed; hanya 2
   transisi, sehingga kemiringan tren tidak dapat diestimasi dengan baik.
   Perkiraan jujur: mampu mendeteksi penurunan besar, **kemungkinan besar
   `inconclusive`** untuk efek kecil. Itu hasil yang sah, bukan kegagalan.
2. **Kapasitas model.** 0,97 jt vs 2 jt parameter: lantai kemampuan lebih
   rendah; beberapa metrik (arith, entropi) berisiko berlabel `at_floor`/
   `uninformative` dan karena itu dikeluarkan. Bila lebih dari 1 kelompok
   informatif hilang, keputusan akan jatuh ke `inconclusive` secara sah.
3. **Pelatihan 1 epoch** (~2,4 token/parameter) jauh di bawah compute-optimal.
   Semua arm memakai anggaran sama, jadi perbandingan tetap adil, tetapi mutu
   absolut M0 rendah.
4. **Dokumen sintetis lebih pendek** daripada guru: jendela 256 B, sedangkan
   median T0 483 B → `truncated_rate` cenderung tinggi dan mungkin melampaui
   ambang "informatif". Efeknya: F2 lemah sebagai sinyal, bukan palsu-positif.

## 6. Yang TIDAK berubah

Pra-registrasi hipotesis, arah H1 (A > C), H2 (B1 ≥ A), aturan tier, ambang
T2, R8, lipatan buta, gerbang T1–T5. Amandemen ini hanya menyentuh **anggaran
komputasi**, bukan klaim atau ambang.

## 7. Bila diratifikasi

1. Terapkan §3 ke `PREREGISTRATION.md` (v1.2 → v1.3) + entri alasan di PLAN.md.
2. `ops/prereg_lock.py --lock --amend 1.3` → hash baru dilaporkan.
3. Tambah 4 metrik R8 (`truncated_rate`, `ttr`, `near_dup_rate`,
   `answer_entropy`) ke evaluator, lalu `make analyze-check` harus **hijau**.
4. Runner `src/p1_run.py` memakai label buta X1/X2/X3.
5. Uji asap ±2 menit (1 langkah kecil) → baru jalan penuh ±2 jam.
