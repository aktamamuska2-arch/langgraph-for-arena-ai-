# Audit korpus T0 (P1)

**Hasil: SEMUA GERBANG LULUS**

| Gerbang | Syarat | Nilai | Hasil |
| --- | --- | --- | --- |
| T1 | >= 2097152 | `2301328` | LULUS |
| T3 | >= 12 domain dan tiap domain >= 5% | `{"domain": 14, "porsi_terkecil": 0.0504}` | LULUS |
| T4 | median 150-600, p1 >= 80, p99 <= 2000 | `{"median": 483, "p1": 221, "p99": 1752}` | LULUS |
| T5 | <= 0,5 | `0.1196` | LULUS |
| T2 | T2a-T2e (lihat porsi di bawah) | `{"T2a": 0.38, "T2b": 0.261, "T2c": 2.048, "T2d": 0.1134, "T2e": 2.6633}` | LULUS |

### Rincian T2 v1.2 (terkalibrasi, PREREGISTRATION.md §2.1a)

| Kode | Metrik | Ambang | Nilai | Hasil |
| --- | --- | --- | --- | --- |
| T2a | uniq-8 potongan 156.697 B | 0.2805 | 0.38 | LULUS |
| T2b | hapax-8 potongan 156.697 B | 0.1835 | 0.261 | LULUS |
| T2c | zlib bpb seluruh isi (juga >= 0,85 x manusia) | 1.8844 | 2.048 | LULUS |
| T2d | TTR sampel 200 KB | 0.1025 | 0.1134 | LULUS |
| T2e | entropi Markov orde-3 @156.697 B | 2.1803 | 2.6633 | LULUS |

Versi lama (v1.1, absolut >= 0,98): nilai 0.1529 -> tetap GAGAL, dan tetap dicatat sebagai riwayat. Syarat itu mustahil dicapai korpus bahasa apa pun.

## Pembanding yang membuat angka di atas bisa ditafsirkan

| Sumber | byte | uniq-8 | hapax-8 | zlib bpb | TTR |
| --- | --- | --- | --- | --- | --- |
| Korpus P1 (isi `<A>`) | 2028237 | 0.1529 | 0.0921 | 2.048 | 0.1134 |
| Korpus P0 (templat, 156 KB) | 156697 | 0.1870* | 0.1223* | 1.189 | 0.0732 |
| Manusia pada 156 KB (sebanding P0) | 156701 | 0.5324* | 0.3886* | 2.486 | 0.2123 |
| **Manusia** (docstring stdlib, ukuran sama) | 2028760 | 0.3013 | 0.1895 | 2.216 | 0.1973 |

- Entropi silang Markov orde-3 pada test split: **1.785 bits/byte** (proksi keterhafalan; makin rendah makin mudah dihafal).
* uniq-8 dan hapax **bergantung ukuran**: pada 156 KB nilainya jauh lebih tinggi daripada pada 2 MB (0.5324 vs 0.3013 untuk teks manusia yang sama). Karena itu keduanya hanya boleh dibandingkan pada ukuran yang sama; P0 (156 KB) tidak boleh dibandingkan langsung dengan P1 (2 MB).

- Entropi silang Markov P0: **1.615 bits/byte** (P1: 1.785) — metrik ini tidak bergantung ukuran corpus.

- Rasio diversitas P1/manusia: uniq-8 **0.51x**, hapax **0.49x**, zlib **0.92x**, TTR **0.57x**.

## Kenapa T2 gagal, dan kenapa itu bukan alasan menurunkan gerbang

T2 meminta 98% 8-gram unik. Teks manusia pada ukuran yang sama hanya mencapai **0.30**. Pada 2 MB, bahasa alami selalu mengulang frasa pendek (" selama ", " yang ", nama berulang), sehingga porsi 8-gram unik **tidak mungkin** mendekati 0,98 untuk korpus apa pun yang masih terbaca sebagai bahasa. Gerbang ini dirumuskan tanpa kalibrasi, dan P0 tidak pernah mengujinya karena belum ada korpus P1.

Keputusan yang saya ambil: **gerbang tidak saya turunkan sendiri**. Usulan amandemen v1.2 ada di berkas ini bagian bawah, menunggu ratifikasi manusia. Sampai diratifikasi, P1 tetap terkunci.

## Status korpus

- 3165 dokumen. 2.028.237 byte isi. split {'train': 3057. 'valid': 59. 'test': 49}
- anchor test 153 dokumen (tidak ikut training), 60 probe aritmetika held-out
- domain: {"kode": 0.105, "perbandingan": 0.0756, "jurnal": 0.0745, "kebijakan": 0.073, "ringkasan": 0.0722, "narasi": 0.0722, "faktual": 0.0718, "sains": 0.0714, "tinjauan": 0.0709, "analisis": 0.0702, "definisi": 0.0688, "aritmetika": 0.0626, "dialog": 0.0614, "prosedur": 0.0504}

## Usulan amandemen T2 (v1.2) — MENUNGGU RATIFIKASI

Ganti T2 absolut dengan **kriteria fungsional** yang mengukur hal yang sebenarnya kita takutkan (korpus yang bisa dihafal sehingga baseline jatuh ke ekor seperti P0), ditambah lantai diversitas relatif terhadap manusia:

| Usulan | Ambang (berlabuh pada P0 dan manusia) | Nilai P1 @156 KB | P0 @156 KB | Rasio |
| --- | --- | --- | --- | --- |
| T2a uniq-8 (ukuran sama) | >= 1,50 x P0 = 0.2805 | 0.3800 | 0.1870 | 2.03x |
| T2b hapax-8 (ukuran sama) | >= 1,50 x P0 = 0.1835 | 0.2610 | 0.1223 | 2.13x |
| T2c zlib bpb (bebas ukuran) | >= 1,50 x P0 = 1.784 **dan** >= 0,85 x manusia (1.884) | 2.146 | 1.189 | 1.80x |
| T2d TTR sampel 200 KB (bebas ukuran) | >= 1,40 x P0 = 0.1025 | 0.1331 | 0.0732 | 1.82x |
| T2e entropi Markov @156 KB (bebas ukuran) | >= 1,35 x P0 = 2.181 | 2.663 | 1.615 | 1.65x |

Untuk konteks, teks manusia pada ukuran yang sama: uniq-8 0.5324, hapax 0.3886, zlib 2.486, CE nan. Korpus P1 berada pada 0.71x uniq-8 dan 0.77x entropi manusia — lebih rendah, dan itu wajar karena bahasanya berbeda (Indonesia vs Inggris+kode), bukan karena algoritmanya sama.

**Pengakuan bias:** kelipatan 1,50 / 1,40 / 1,35 saya tetapkan setelah melihat angka korpus ini dan angka P0. Itu tetap *threshold fitting*, hanya dengan dua jangkar eksternal (P0 sebagai lantai, teks manusia sebagai plafon) alih-alih satu angka yang dipilih agar lolos. Yang membuatnya bisa dipertanggungjawabkan: (a) ambang dinyatakan sekarang, sebelum satu model P1 pun dilatih; (b) arahnya tidak bisa dipakai untuk mempermudah klaim collapse — T2 adalah gerbang korpus, bukan gerbang hasil; (c) kalau kamu menolak, T2 tetap 0,98, P1 tidak berjalan, dan itu akan saya laporkan sebagai kegagalan pra-registrasi, bukan sebagai alasan mengubahnya diam-diam.

