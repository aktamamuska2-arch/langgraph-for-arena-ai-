# Arena Showroom #7 — Humanoid Dance Stage

Showroom 3D interaktif di localhost: satu model humanoid dari Google Drive
(`Arena AI / #7 / humanoid_basic_animation_set.glb`) dianimasikan di tempat
dengan **engine koreografi prosedural** yang dibangun di atas clip locomotion
bawaan model, plus panel **Motion QA** yang menilai kualitas gerak secara
kuantitatif (dan dipakai untuk iterasi tuning).

Kamera sepenuhnya bebas: orbit, zoom, pan, preset sinematik, dan auto-orbit.

```
npm install
npm run dev      # → http://localhost:5173
```

---

## 1. Isi model & masalahnya

`humanoid_basic_animation_set.glb` (Sketchfab, rig Rigify) hanya berisi
**11 clip locomotion**, **tanpa satu pun clip tari**:

| clip | durasi | clip | durasi |
|---|---|---|---|
| `tpose` | 0.00 s | `run_2` | 0.67 s |
| `idle` | 2.00 s | `fp_run` | 0.67 s |
| `walk` | 1.33 s | `slide_in` | 0.50 s |
| `walk_2` | 1.00 s | `slide_loop` | 0.67 s |
| `jog` | 0.83 s | `slide_progress` | 1.33 s |
| `run` | 0.67 s | | |

Jadi menari tidak bisa dilakukan dengan sekadar memutar clip. Showroom ini
menambahkan **4 koreografi** yang disusun sendiri, dijalankan sebagai layer
aditif di atas `idle`, sehingga tetap memakai clip asli sebagai "napas" dasar.

---

## 2. Empat dance

| # | Nama | BPM | Ketukan | Karakter | Aksen |
|---|---|---|---|---|---|
| 1 | **Neon Groove** | 110 | 32 | step-touch, punch, breakdown | `#4df0ff` |
| 2 | **Wave Rider** | 96 | 24 | body roll, arm wave, slow turn | `#b06bff` |
| 3 | **Break Circuit** | 128 | 32 | top rock, footwork, spin 360°, freeze | `#ff5c7a` |
| 4 | **Robot Pop** | 120 | 16 | staccato, sudut 90°, snap easing | `#ffd166` |

Semua loop-nya mulus (key terakhir menyambung ke key pertama; `yaw` 360°
di-unwrap agar tidak berputar balik di sambungan).

Selain itu tersedia pemutaran **clip asli** (`idle`, `walk`, `run`, `slide`)
untuk membandingkan hasil koreografi dengan animasi bawaan, dan tombol
**Auto medley** yang berganti dance otomatis tiap satu putaran selesai.

---

## 3. Kontrol

| Aksi | Cara |
|---|---|
| Putar kamera | drag kiri |
| Zoom | scroll / pinch |
| Geser target kamera | drag kanan (atau dua jari) |
| Play / pause | `Spasi` atau tombol ❚❚ |
| Ulangi dance | `R` |
| Dance berikut / sebelumnya | `→` / `←` atau tombol ▶▶ ◀◀ |
| Pilih dance 1–4 | `1` `2` `3` `4` |
| Orbit otomatis | `C` |
| Tampilkan skeleton | `S` |
| Auto medley | `M` |
| Preset kamera | Hero · Depan · Samping · Close-up · Atas |

Slider **Intensitas** menskalakan seluruh amplitudo koreografi (0 = kembali ke
pose clip dasar), **Dinamika** mengatur lapisan impuls per-ketukan (0 = tari
tanpa aksen, 1 = disetel, 2 = maksimal), **Tempo** menskalakan waktu
(0.5×–1.6×).

---

## 4. Arsitektur

```
showroom/
├─ index.html            UI overlay (set list, transport, panel QA, kredit)
├─ src/
│  ├─ showroom.js        panggung: lantai reflektif, cincin neon, cyc, lighting,
│  │                     OrbitControls, preset kamera, denyut per ketukan
│  ├─ rig.js             layer rotasi aditif world-space + IK dua-tulang tungkai
│  ├─ choreography.js    partitur 4 dance (data) + sampler + easing + osilator
│  ├─ danceEngine.js     AnimationMixer + layer prosedural + tracking sendi
│  ├─ motionQA.js        metrik kualitas gerak (dipakai browser & Node)
│  ├─ ui.js              render set list, bar skor, indikator ketukan
│  ├─ main.js            perakitan: load GLB → engine → UI → render loop
│  └─ style.css
├─ public/models/humanoid.glb    hasil salin dari Google Drive
└─ eval/
   ├─ evaluate.js        jalankan semua dance headless @60fps → metrik + PNG
   ├─ stickfig.js        contact-sheet pose (front & side)
   ├─ raster.js          rasterizer + PNG writer tanpa dependensi
   └─ out/               hasil: report.json + 8 contact-sheet PNG
```

### 4.1 Rotasi aditif di ruang dunia

Offset koreografi dinyatakan dalam **sumbu scene** (X pitch, Y yaw, Z roll),
bukan sumbu bone Rigify yang sulit ditebak. Untuk bone `b` dengan parent `p`:

```
Lb' = Bp⁻¹ · O · Bp · Lb
```

`Bp` = world quaternion parent pada pose dasar (hasil mixer), `Lb` = local
quaternion bone, `O` = rotasi offset di ruang dunia. Konsekuensinya world baru
bone = Δ·world dasar, dan seluruh anak otomatis mewarisi Δ.

Dua detail yang wajib:

- **`decompose()`, bukan `setFromRotationMatrix()`** — bone di file ini punya
  skala non-uniform (`thigh` ≈ 1.01/0.97/1.01), sehingga matriks rotasi murni
  menghasilkan quaternion non-unit dan tulang **melar** (bug nyata yang
  ditemukan saat audit: sendi tangan sempat bergerak >5 m).
- Engsel anatomis (siku, lutut, pergelangan) memakai **ruang bone**
  (`Lb' = Lb · O`) karena porosnya ikut berputar bersama lengan/tungkai induk.

### 4.2 IK tungkai (kunci "berhenti di tempat")

Kaki tidak dianimasikan dengan sudut, tapi dengan **target posisi telapak**.
Setiap frame, dua-tulang IK menyelesaikan lutut (law of cosines + pole vector
ke depan), lalu telapak dipaksa rata mengikuti orientasi stance.

Karena panjang tulang ikut berubah oleh track *scale* di clip, panjang L1/L2
dihitung **per frame**, dan solusinya diiterasi 2× memakai error posisi
pergelangan yang terukur (shear dari skala parent membuat satu iterasi tidak
presisi). Hasilnya:

- berpindah berat badan → kaki tetap menapak, lutut yang menyesuaikan;
- telapak tidak menembus lantai (`goalY` + pagar lantai);
- jinjit (`footAngle`) menaikkan pergelangan sehingga ujung kaki tetap di lantai.

### 4.3 Auto-step (anti foot-sliding)

Saat sebuah telapak harus berpindah tempat (atau tersapu putaran badan), sampler
menambahkan busur angkat:

```
lift = sin(π · t) · min(0.26, 0.03 + travel · 0.7)
```

sehingga kaki **terangkat–berpindah–mendarat**, bukan menyeret. Ini menurunkan
slip rata-rata dari 0.35 m/s menjadi 0.03 m/s (lihat §5).

### 4.4 Partitur

```js
K(beat, ease, 'label', {armL:[..], chest:[..], head:[..]}, {
  root: [x, y, z],       // geser panggul (meter) → y negatif = tekuk lutut
  feet: { L:[dx,dy,dz], R:[..] },   // target telapak relatif stance
  footAngle: { L:[-24,0,0] },       // negatif = jinjit
  yaw: 90,                          // putaran badan penuh
})
```

Easing per key: `smooth`, `snap` (tahan lalu hentak), `pop` (overshoot),
`hold`, `out`, `in`. Osilator sekunder (`osc`) menambahkan gelombang menjalar
dengan fase bertingkat di sepanjang tulang belakang dan lengan.

**Lapisan dinamika.** Key saja menghasilkan gerak yang "senam": mulus, tapi
tanpa bobot. Tiap dance punya blok `dynamics` — impuls per ketukan dengan kurva
peluruhan (`sharp` / `soft` / `bounce`), attack ~40 ms, `barBoost` di ketukan
pertama tiap bar, `swing` (aksen agak telat) dan `lag` untuk follow-through:

```js
dynamics: { kick: 0.046, chest: 5, head: 5, arm: 7, curve: 'soft',
            swing: 0.04, barBoost: 1.35, lag: 0.08 }
```

`kick` menurunkan panggul tepat di ketukan; IK tungkai otomatis menekuk lutut,
jadi badannya benar-benar "jatuh" ke ketukan. Attack pendek dipakai supaya nilai
impuls 0 tepat di batas loop — pose akhir loop identik dengan awal (seam 0 mm).

---

## 5. Evaluasi gerak (iteratif)

`src/motionQA.js` mengukur 8 aspek dari lintasan sendi dunia:

| aspek | definisi | target |
|---|---|---|
| activity | rata-rata kecepatan sendi kunci | > 0.4 m/s |
| amplitude | rentang vertikal tangan + kepala | tangan > 1 m |
| smoothness | rata-rata jerk (turunan percepatan) | rendah |
| footwork | kecepatan horizontal **pergelangan** saat telapak menapak | < 0.1 m/s |
| balance | jarak horizontal panggul ke titik tengah kaki | < 0.15 m |
| loop | selisih pose frame awal vs akhir satu loop | < 0.1 m |
| rhythm | fraksi ketukan yang punya puncak percepatan (aksen) | > 0.5 |
| dynamics | pantulan panggul **di dalam satu ketukan** + kontras puncak/rata-rata | pantul > 6 cm, kontras > 2 |

Catatan metodologi (penting): slip diukur di **pergelangan**, bukan ujung jari,
supaya gerakan berputar-di-tempat (pivot) tidak salah dihitung sebagai slaid.
Tinggi lantai diukur dari **sol** (`tinggi bone toe − offset kalibrasi`),
negatif berarti menembus lantai.

Jalankan:

```bash
npm run eval            # tabel metrik + contact sheet di eval/out/
npm run eval -- --json  # keluaran JSON
```

Panel **Motion QA** di kanan layar menjalankan metrik yang sama secara live
(window bergulir ~5 detik) sambil showroom berjalan.

### 5.1 Hasil akhir

```
dance            bpm   energi  slip    lantai   jerk    seam    balance  ritme   skor
neon-groove      110   0.66    0.03    -0.000   34      0.006   0.05     1.00    94
wave-rider        96   0.55    0.01    -0.002   26      0.000   0.03     1.00    95
break-circuit    128   0.83    0.04     0.001   43      0.000   0.07     0.97    94
robot-pop        120   0.48    0.01     0.003   33      0.000   0.05     1.00    95
```

Dinamika (diukur `eval/dynamics-audit.mjs`, satu instance GLTF bersih per dance):

```
dance           kecepatan  p95   kontras  pantul/ketuk  ayun kepala
neon-groove      0.70 m/s  1.73    2.47      9.9 cm        52 cm
wave-rider       0.58 m/s  1.27    2.18      7.5 cm        36 cm
break-circuit    0.88 m/s  3.77    4.31      6.7 cm        56 cm
robot-pop        0.53 m/s  1.59    3.02      6.0 cm        42 cm
```

Contact sheet (8 pose per dance, tampak depan & samping) ada di
`eval/out/*.png`.

### 5.2 Jejak iterasi

| iterasi | perubahan | slip rata-rata | skor rata-rata |
|---|---|---|---|
| 0 | layer aditif tanpa IK | 0.42 m/s | 68 |
| 1 | + IK dua-tulang & target telapak | 0.16 m/s | 75 |
| 2 | + auto-step, unwrap yaw, metrik slip diperbaiki | 0.03 m/s | 87 |
| 3 | + postur dasar, servo IK 2 iterasi, toe-flat, pagar lantai | 0.02 m/s | 88 |
| 4 | + lapisan impuls per ketukan, kunci fase clip dasar, snapshot QA | 0.02 m/s | 94 |

Iterasi 4 berangkat dari keluhan "kurang dinamis". Audit menunjukkan bukan
amplitudo yang kurang, tapi **aksen**: panggul hanya memantul 1–3 cm tiap
ketukan dan `p95/mean` 6× (gerak menumpuk di sedikit frame, sisanya diam).
Setelah impuls per ketukan + `snap` yang lebih pendek: pantul 6–10 cm, kontras
2.2–4.3×. Sekaligus dua bug pengukuran ketahuan — lihat [`eval/report.md`](eval/report.md).

Detail lengkap tiap iterasi (termasuk bug yang ditemukan) ada di
[`eval/report.md`](eval/report.md).

---

## 6. Keterbatasan

- **Tidak ada IK lengan / tangan.** Lengan memakai rotasi aditif; kontak tangan
  ke lantai (breakdance sesungguhnya, freeze bertumpu tangan) belum disupport.
- Gerakan hanya **di tempat**; belum ada root motion berjalan keliling panggung.
- Metrik QA berbasis bone, bukan vertex mesh, jadi sol/telapak diaproksimasi
  dengan offset kalibrasi.
- Rhythm `wave-rider` rendah (0.17) secara desain: dance ini legato, aksennya
  halus. Skor diperbaiki dengan osilator pulse per ketukan, tapi tetap paling
  lembut di antara empat dance.

## 7. Ide lanjutan

- IK lengan untuk freeze/handstand, deteksi kontak tangan otomatis.
- Input audio (mic/beat detection) untuk menggerakkan tempo secara live.
- Ekspor koreografi ke `.glb` (bake pose ke AnimationClip) agar bisa dipakai di
  engine lain.
- Retarget ke model humanoid lain (Mixamo/VRM) dengan pemetaan bone otomatis.

---

## 8. Kredit & lisensi

Model 3D: **“Humanoid Basic Animation Set”** oleh **Snorki**
([Sketchfab](https://sketchfab.com/3d-models/humanoid-basic-animation-set-a2fc1a30e8064fdcb0384fdd7716e869)),
lisensi **[CC-BY-4.0](http://creativecommons.org/licenses/by/4.0/)**.
Atribusi ditampilkan juga di pojok kanan bawah showroom sesuai syarat lisensi.

Kode showroom, engine koreografi, dan metrik QA: dibuat untuk sesi ini
(three.js r180 + Vite).
