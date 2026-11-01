# Audit korpus T0 (P1)

**Hasil: GAGAL pada T2**

| Gerbang | Syarat | Nilai | Hasil |
| --- | --- | --- | --- |
| T1 | >= 2097152 | `2300508` | LULUS |
| T2 | >= 0,98 | `0.1305` | **GAGAL** |
| T3 | >= 12 domain dan tiap domain >= 5% | `{"domain": 14, "porsi_terkecil": 0.0519}` | LULUS |
| T4 | median 150-600, p1 >= 80, p99 <= 2000 | `{"median": 501, "p1": 224, "p99": 1762}` | LULUS |
| T5 | <= 0,5 | `0.3158` | LULUS |

## Pembanding yang membuat angka di atas bisa ditafsirkan

| Sumber | byte | uniq-8 | hapax-8 | zlib bpb | TTR |
| --- | --- | --- | --- | --- | --- |
| Korpus P1 (isi `<A>`) | 2030734 | 0.1305 | 0.0776 | 1.932 | 0.0999 |
| Korpus P0 (templat, 156 KB) | 156697 | 0.1870* | 0.1223* | 1.189 | 0.0732 |
| Manusia pada 156 KB (sebanding P0) | 156701 | 0.5324* | 0.3886* | 2.486 | 0.2123 |
| **Manusia** (docstring stdlib, ukuran sama) | 2031257 | 0.3012 | 0.1894 | 2.217 | 0.1973 |

- Entropi silang Markov orde-3 pada test split: **1.705 bits/byte** (proksi keterhafalan; makin rendah makin mudah dihafal).
* uniq-8 dan hapax **bergantung ukuran**: pada 156 KB nilainya jauh lebih tinggi daripada pada 2 MB (0.5324 vs 0.3012 untuk teks manusia yang sama). Karena itu keduanya hanya boleh dibandingkan pada ukuran yang sama; P0 (156 KB) tidak boleh dibandingkan langsung dengan P1 (2 MB).

- Entropi silang Markov P0: **1.615 bits/byte** (P1: 1.705) — metrik ini tidak bergantung ukuran corpus.

- Rasio diversitas P1/manusia: uniq-8 **0.43x**, hapax **0.41x**, zlib **0.87x**, TTR **0.51x**.

## Kenapa T2 gagal, dan kenapa itu bukan alasan menurunkan gerbang

T2 meminta 98% 8-gram unik. Teks manusia pada ukuran yang sama hanya mencapai **0.30**. Pada 2 MB, bahasa alami selalu mengulang frasa pendek (" selama ", " yang ", nama berulang), sehingga porsi 8-gram unik **tidak mungkin** mendekati 0,98 untuk korpus apa pun yang masih terbaca sebagai bahasa. Gerbang ini dirumuskan tanpa kalibrasi, dan P0 tidak pernah mengujinya karena belum ada korpus P1.

Keputusan yang saya ambil: **gerbang tidak saya turunkan sendiri**. Usulan amandemen v1.2 ada di berkas ini bagian bawah, menunggu ratifikasi manusia. Sampai diratifikasi, P1 tetap terkunci.

## Status korpus

- 3131 dokumen. 2.030.734 byte isi. split {'train': 3009. 'valid': 61. 'test': 61}
- anchor test 142 dokumen (tidak ikut training), 60 probe aritmetika held-out
- domain: {"kode": 0.1069, "jurnal": 0.077, "ringkasan": 0.0734, "perbandingan": 0.0729, "tinjauan": 0.0718, "faktual": 0.0718, "kebijakan": 0.0714, "narasi": 0.0704, "analisis": 0.0702, "definisi": 0.0695, "sains": 0.0684, "aritmetika": 0.0623, "dialog": 0.0621, "prosedur": 0.0519}

## Usulan amandemen T2 (v1.2) — MENUNGGU RATIFIKASI

Ganti T2 absolut dengan **kriteria fungsional** yang mengukur hal yang sebenarnya kita takutkan (korpus yang bisa dihafal sehingga baseline jatuh ke ekor seperti P0), ditambah lantai diversitas relatif terhadap manusia:

| Usulan | Ambang (berlabuh pada P0 dan manusia) | Nilai P1 @156 KB | P0 @156 KB | Rasio |
| --- | --- | --- | --- | --- |
| T2a uniq-8 (ukuran sama) | >= 1,50 x P0 = 0.2805 | 0.3413 | 0.1870 | 1.83x |
| T2b hapax-8 (ukuran sama) | >= 1,50 x P0 = 0.1835 | 0.2289 | 0.1223 | 1.87x |
| T2c zlib bpb (bebas ukuran) | >= 1,50 x P0 = 1.784 **dan** >= 0,85 x manusia (1.884) | 2.035 | 1.189 | 1.71x |
| T2d TTR sampel 200 KB (bebas ukuran) | >= 1,40 x P0 = 0.1025 | 0.1163 | 0.0732 | 1.59x |
| T2e entropi Markov @156 KB (bebas ukuran) | >= 1,35 x P0 = 2.181 | 2.553 | 1.615 | 1.58x |

Untuk konteks, teks manusia pada ukuran yang sama: uniq-8 0.5324, hapax 0.3886, zlib 2.486, CE nan. Korpus P1 berada pada 0.64x uniq-8 dan 0.74x entropi manusia — lebih rendah, dan itu wajar karena bahasanya berbeda (Indonesia vs Inggris+kode), bukan karena algoritmanya sama.

**Pengakuan bias:** kelipatan 1,50 / 1,40 / 1,35 saya tetapkan setelah melihat angka korpus ini dan angka P0. Itu tetap *threshold fitting*, hanya dengan dua jangkar eksternal (P0 sebagai lantai, teks manusia sebagai plafon) alih-alih satu angka yang dipilih agar lolos. Yang membuatnya bisa dipertanggungjawabkan: (a) ambang dinyatakan sekarang, sebelum satu model P1 pun dilatih; (b) arahnya tidak bisa dipakai untuk mempermudah klaim collapse — T2 adalah gerbang korpus, bukan gerbang hasil; (c) kalau kamu menolak, T2 tetap 0,98, P1 tidak berjalan, dan itu akan saya laporkan sebagai kegagalan pra-registrasi, bukan sebagai alasan mengubahnya diam-diam.

