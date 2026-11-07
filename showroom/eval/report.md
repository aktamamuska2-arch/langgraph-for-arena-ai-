# Laporan evaluasi koreografi (iteratif)

Semua angka dihasilkan oleh `eval/evaluate.js` — engine yang **sama** dengan yang
dipakai di browser — dijalankan headless pada 60 fps untuk satu loop penuh
per dance. Metrik dihitung `src/motionQA.js`.

## Metodologi

| metrik | satuan | cara ukur |
|---|---|---|
| energi | m/s | rata-rata kecepatan 16 sendi kunci per frame |
| slip | m/s | kecepatan horizontal **pergelangan** (foot bone) saat ujung kaki menapak (< 5.5 cm dari lantai) |
| lantai | m | tinggi **sol** terendah = `toe.y − offset kalibrasi`; negatif = menembus lantai |
| jerk | m/s³ | rata-rata |Δpercepatan| kepala & kedua tangan |
| seam | m | selisih posisi sendi antara frame awal dan akhir loop |
| balance | m | jarak horizontal panggul ke titik tengah dua kaki |
| ritme | 0–1 | fraksi ketukan yang punya puncak percepatan > 1.55× rata-rata di sekitar ketukan |

Skor per aspek dinormalisasi ke 0–100 lalu dirata-rata menjadi **skor** akhir.

Catatan: pada iterasi 0–1 metrik `lantai` masih mengukur tinggi **bone** (bukan
sol) sehingga tidak sebanding; karena itu kolomnya dikosongkan di tabel
iterasi awal.

---

## Iterasi 0 — layer aditif murni (tanpa IK)

Koreografi diterapkan sebagai rotasi aditif; tungkai masih diputar dengan sudut.

| dance | energi | slip | jerk | seam | balance | ritme | skor |
|---|---|---|---|---|---|---|---|
| neon-groove | 0.56 | **0.35** | 24 | 0.002 | **0.12** | 0.28 | 74 |
| wave-rider | 0.43 | 0.24 | 13 | 0.000 | 0.08 | 0.17 | 73 |
| break-circuit | 0.90 | **0.87** | 50 | 0.000 | **0.23** | 0.03 | 53 |
| robot-pop | 0.39 | 0.21 | 21 | 0.000 | 0.14 | 0.44 | 73 |

**Diagnosis.** Menurunkan/menggeser panggul tanpa IK membuat seluruh badan
(termasuk kaki) ikut bergeser → telapak menyeret lantai 0.21–0.87 m/s. Ini
penderitaan khas animasi tari prosedural dan yang paling terlihat oleh mata.

**Tindakan.** Beralih ke model authoring berbasis **target telapak** + IK
dua-tulang (lutut diselesaikan dengan law of cosines, pole vector ke depan).

---

## Iterasi 1 — IK dua-tulang

| dance | energi | slip | jerk | seam | balance | skor |
|---|---|---|---|---|---|---|
| neon-groove | 0.54 | 0.10 | 26 | 0.002 | 0.05 | 78 |
| wave-rider | 0.35 | 0.03 | 13 | 0.000 | 0.03 | 76 |
| break-circuit | 0.73 | 0.44 | 49 | 0.000 | 0.06 | 71 |
| robot-pop | 0.37 | 0.07 | 25 | 0.000 | 0.04 | 74 |

**Diagnosis.** Slip turun 62% dan balance membaik drastis, tapi gerakan cepat
(kick, spin) masih menyisakan 0.44 m/s: saat telapak harus berpindah titik,
targetnya diinterpolasi lurus di lantai → kaki "diseret" ke posisi baru.

**Tindakan.** Tiga perubahan sekaligus:

1. **Auto-step** — setiap perpindahan telapak (termasuk yang tersapu putaran
   badan) diberi busur angkat `sin(πt)`; kaki terangkat–berpindah–mendarat.
2. **Unwrap yaw** — `360° → 0°` sebelumnya diinterpolasi sebagai putaran balik
   penuh. Pada sambungan loop ini menghasilkan lonjakan 25.8 m/s di ujung kaki.
3. **Perbaikan metrik** — slip diukur di pergelangan, bukan ujung jari, agar
   pivot (putar di tempat) tidak dihitung sebagai slaid.

---

## Iterasi 2 — auto-step + perbaikan metrik

| dance | energi | slip | lantai | jerk | seam | balance | ritme | skor |
|---|---|---|---|---|---|---|---|---|
| neon-groove | 0.59 | 0.03 | −0.000 | 26 | 0.003 | 0.05 | 0.56 | **90** |
| wave-rider | 0.37 | 0.01 | −0.005 | 14 | 0.000 | 0.03 | 0.17 | 84 |
| break-circuit | 0.71 | 0.05 | −0.005 | 34 | 0.000 | 0.07 | 0.41 | 86 |
| robot-pop | 0.39 | 0.02 | −0.003 | 25 | 0.000 | 0.04 | 0.44 | 88 |

**Diagnosis.** Slip sudah 0.01–0.05 m/s (level "menapak"). Sisa masalah:
`lantai` masih −3…−5 mm (sol tembus) dan postur badan cenderung "duduk ke
belakang" saat lutut tertekuk, karena panggul bergeser mundur untuk menjaga
telapak tetap di tempat.

**Tindakan.**

1. **Postur dasar** (offset kecil `hips −1.5°`, `chest −3°`, `head +2.5°`) untuk
   melawan efek sit-back — kecuali pada `robot-pop` yang memang harus tegak.
2. **Pagar lantai & toe-flat** — telapak dan ujung kaki dipaksa mengikuti
   orientasi stance, dan `footAngle` (jinjit) menaikkan pergelangan sehingga
   ujung kaki tetap menempel.
3. **Servo IK 2 iterasi** — panjang tulang dihitung per frame (clip menganimasikan
   *scale*), lalu solusi dikoreksi dengan error pergelangan yang terukur.

---

## Iterasi 3 — fisika pijakan

| dance | energi | slip | lantai | jerk | seam | balance | ritme | skor |
|---|---|---|---|---|---|---|---|---|
| neon-groove | 0.59 | 0.03 | −0.001 | 26 | 0.003 | 0.05 | 0.56 | 90 |
| wave-rider | 0.39 | 0.01 | −0.005 | 16 | 0.000 | 0.03 | 0.17 | 85 |
| break-circuit | 0.72 | 0.04 | −0.005 | 34 | 0.000 | 0.07 | 0.41 | 87 |
| robot-pop | 0.40 | 0.01 | −0.003 | 25 | 0.000 | 0.05 | 0.50 | 89 |

**Skor per aspek**

| dance | activity | amplitude | smoothness | footwork | balance | loop | rhythm |
|---|---|---|---|---|---|---|---|
| neon-groove | 100 | 100 | 97 | 95 | 82 | 98 | 56 |
| wave-rider | 92 | 100 | 98 | 98 | 90 | 100 | 17 |
| break-circuit | 100 | 100 | 96 | 93 | 76 | 100 | 41 |
| robot-pop | 96 | 100 | 97 | 97 | 83 | 100 | 50 |

Ringkasan: slip rata-rata **0.42 → 0.02 m/s**, sol tidak lagi menembus lantai
(sisa ≤ 5 mm), loop nyaris sempurna (semua ≤ 6 mm), dan semua dance punya
rentang gerak tangan > 0.9 m.

Rhythm `wave-rider` sengaja rendah: dance ini legato — aksen datang sebagai
gelombang halus, bukan hentakan. Oskilator `pulse` per ketukan sudah ditambahkan,
tapi sifatnya tetap yang paling lembut.

---

## Iterasi 4 — dinamika (final)

Keluhan berikutnya: gerakannya "kurang dinamis" — terasa kaku seperti senam.
Sebelum menambah amplitudo, lebih dulu diukur **ritmenya** dengan
`eval/dynamics-audit.mjs` (satu instance GLTF bersih per dance; memakai engine
yang sama dengan app, tapi dijalankan deterministik 60 fps):

| dance | kecepatan | p95 | p95/mean | pantul panggul/ketuk | ayun kepala |
|---|---|---|---|---|---|
| neon-groove | 0.60 m/s | 1.75 | 2.91 | 9.1 cm | 48 cm |
| wave-rider | 0.39 m/s | 0.61 | 1.57 | 3.2 cm | 34 cm |
| break-circuit | 0.73 m/s | 4.36 | 5.97 | 5.2 cm | 52 cm |
| robot-pop | 0.39 m/s | 2.11 | 5.46 | 5.6 cm | 39 cm |

Diagnosis: **bukan amplitudo yang kurang, tapi aksen.** Rentang tangan sudah
0.9–1.7 m, jadi badannya besar — masalahnya **kapan** gerak itu terjadi. Pantulan
panggul per ketukan cuma 3–9 cm (dan sebagian besar berasal dari *crouch* yang
bertahan beberapa ketukan, bukan hentakan), sementara `p95/mean` 5.5–6.0× di
break & robot berarti gerak menumpuk di sedikit frame lalu diam. Penyebabnya key
ditulis pada grid 2–3 ketukan dengan easing `smooth` (atau `snap` yang menahan
78% segmen) — jadi tidak ada "bobot" yang jatuh tiap ketukan, dan `wave-rider`
hampir tidak punya aksen sama sekali (p95/mean 1.57 = kecepatan konstan).

Perbaikan:

1. **Lapisan impuls per ketukan** (`dynamics` di tiap dance): panggul turun
   `kick` ≈ 3–5 cm tepat di ketukan (IK lutut otomatis menekuk), ditambah pop
   dada/kepala/bahu/lengan dengan `lag` ~0.05–0.12 ketukan untuk follow-through,
   `barBoost` di ketukan pertama tiap bar, dan `swing` (aksen agak telat).
   Kurva: `sharp` (break/robot), `soft` (groove), `bounce` (wave).
2. **Attack 40 ms** pada impuls, supaya nilainya 0 tepat di batas loop → pose
   akhir loop identik dengan awal (tanpa pop), tapi hentakan tetap terasa.
3. **`snap` diperpendek** (tahan 78% → 70% segmen) agar pose tidak terlalu lama
   membeku, dan beberapa key `smooth` → `snap` di wave-rider untuk aksen.
4. **Slider Dinamika** (0–2×) di panel kontrol untuk menyetel intensitas lapisan
   ini, plus metrik **dinamika** baru di panel QA.

Hasil:

| dance | energi | slip | lantai | jerk | seam | balance | ritme | dinamika | skor |
|---|---|---|---|---|---|---|---|---|---|
| neon-groove | 0.66 | 0.03 | −0.000 | 34 | 0.006 | 0.05 | 1.00 | 86 | 94 |
| wave-rider | 0.55 | 0.01 | −0.002 | 26 | 0.000 | 0.03 | 1.00 | 77 | 95 |
| break-circuit | 0.83 | 0.04 | 0.001 | 43 | 0.000 | 0.07 | 0.97 | 89 | 94 |
| robot-pop | 0.48 | 0.01 | 0.003 | 33 | 0.000 | 0.05 | 1.00 | 80 | 95 |

| dance | kecepatan | p95 | p95/mean | pantul panggul/ketuk | ayun kepala |
|---|---|---|---|---|---|
| neon-groove | 0.70 m/s | 1.73 | 2.47 | 9.9 cm | 52 cm |
| wave-rider | 0.58 m/s | 1.27 | 2.18 | 7.5 cm | 36 cm |
| break-circuit | 0.88 m/s | 3.77 | 4.31 | 6.7 cm | 56 cm |
| robot-pop | 0.53 m/s | 1.59 | 3.02 | 6.0 cm | 42 cm |

Kecepatan rata-rata naik 1.2× (neon 0.60→0.70, wave 0.39→0.58, break 0.73→0.88,
robot 0.39→0.53). Yang lebih penting: **pantulan panggul per ketukan** naik
1.1–2.3× (terbesar di wave-rider 3.2→7.5 cm) dan gerak tidak lagi menumpuk —
`p95/mean` break 5.97→4.31, robot 5.46→3.02. Di wave-rider kontras justru naik
(1.57→2.18) karena dance legato itu sebelumnya nyaris tanpa aksen. Skor
rata-rata **88 → 94**.

> Catatan jujur: `robot-pop` tetap paling pelan (0.53 m/s) karena sifatnya
> staccato — pose ditahan lalu pindah cepat. Yang bertambah di sana adalah
> aksennya (kontras 3.0×), bukan kecepatan rata-rata.


---

## Bug yang ditemukan saat audit kode

| # | gejala | akar masalah | perbaikan |
|---|---|---|---|
| 1 | sendi tangan bergerak > 5 m, badan terangkat 4 m | `setFromRotationMatrix()` pada matriks bone berskala non-uniform → quaternion non-unit, tulang **melar** | `decompose()` + `normalize()` |
| 2 | ujung kaki melonjak 25 m/s di sambungan loop | interpolasi `yaw 360° → 0°` = putaran balik penuh | unwrap `yaw` ke selisih terdekat |
| 3 | telapak tenggelam 5 cm saat easing `pop` | easing overshoot (t > 1) membuat `sin(πt)` negatif → lift negatif | clamp `arcT ∈ [0,1]` + lantai sebagai batas keras |
| 4 | toe-off (`footAngle ≠ 0`) menghasilkan rotasi salah | `quatFromDegrees(rot, _q1)` memakai scratch yang sama dengan `out` → quaternion terkali dirinya | scratch internal terpisah |
| 5 | sol turun 1–2 cm saat lutut ditekuk dalam | panjang tulang dikalibrasi sekali, padahal clip menganimasikan *scale*; shear skala parent | panjang per frame + 2 iterasi koreksi posisi |
| 6 | transisi dance bisa membuat sol miring permanen | `calibrateStance()` dipanggil saat crossfade masih berjalan | kalibrasi sekali pada clip bersih |
| 7 | pose "maju" saat pause | `Math.abs(step)/FADE \|\| 0.05` → 0 dianggap falsy | eksplisit `step === 0 ? 0 : …` |
| 8 | audit dinamika melaporkan semua dance nyaris beku (0.03 m/s) | jumlah frame dihitung `beats*60/bpm` (detik, bukan frame) → hanya 8 frame tersampel | `frames = loopSeconds*60` |
| 9 | metrik `loop` selalu 100 padahal loop "pop" 30 cm; dinamika terbaca 0.8 cm/ketuk | `frames` menyimpan **referensi** ke buffer sendi yang dipakai ulang (double-buffer) → seluruh riwayat menunjuk pose terakhir | snapshot `{x,y,z}` tiap frame |
| 10 | setelah #9 diperbaiki, seam = 30 cm di semua dance | pose tari adalah offset di atas clip `idle`, dan fase clip tidak pernah sinkron saat loop kembali ke awal | fase clip dikunci ke posisi loop (`clipCycles`) |
| 11 | `_loopSeam()` membandingkan frame berfase beda | pencarian frame "satu putaran kemudian" tidak ada, jadi jatuh ke frame terakhir | cari frame di `beat ≈ awal + k·beats` |

---

## Reproduksi

```bash
cd showroom
npm install
npm run eval          # tabel di atas + eval/out/*.png
npm run eval -- --json
```

Keluaran mentah: `eval/out/report.json`.
