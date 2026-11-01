# Kalibrasi aturan collapse (hasil simulasi)

Dihitung analitik lalu dirata-ratakan atas 20000 tarikan `sigma_hat`; tanpa dependensi.
Baseline 5 seed M0 · 10 arm · 6 generasi · konfirmasi >= 2/3 seed.

| Aturan | sigma | FP per-look | **FWER studi** | Power @0,05 | @0,10 | @0,20 | MDE80 (bpb) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `R1_v0.2_3sigma_lantai_0.10` | 0.02 | 0.0000% | **0.00%** | 0% | 99% | 100% | 0.084 |
| `R1_v0.2_3sigma_lantai_0.10` | 0.04 | 0.0146% | **0.87%** | 26% | 72% | 99% | 0.114 |
| `R1_v0.2_3sigma_lantai_0.10` | 0.06 | 0.1825% | **8.47%** | 26% | 49% | 88% | 0.171 |
| `R2_predint_lantai_0.10` | 0.02 | 0.0000% | **0.00%** | 0% | 93% | 100% | 0.087 |
| `R2_predint_lantai_0.10` | 0.04 | 0.0081% | **0.48%** | 15% | 49% | 93% | 0.156 |
| `R2_predint_lantai_0.10` | 0.06 | 0.0902% | **4.21%** | 13% | 29% | 69% | 0.233 |
| `R3_3sigma_tanpa_lantai` | 0.02 | 0.5525% | **11.41%** | 72% | 99% | 100% | 0.057 |
| `R3_3sigma_tanpa_lantai` | 0.04 | 0.5525% | **11.41%** | 36% | 72% | 99% | 0.114 |
| `R3_3sigma_tanpa_lantai` | 0.06 | 0.5525% | **11.41%** | 26% | 49% | 88% | 0.171 |
| `R4_3sigma_lantai_0.05` | 0.02 | 0.0146% | **0.87%** | 72% | 99% | 100% | 0.057 |
| `R4_3sigma_lantai_0.05` | 0.04 | 0.3647% | **11.23%** | 36% | 72% | 99% | 0.114 |
| `R4_3sigma_lantai_0.05` | 0.06 | 0.5073% | **11.41%** | 26% | 49% | 88% | 0.171 |
| `R5_3sigma_lantai_0.10_persist` | 0.02 | 0.0000% | **0.00%** | 0% | 80% | 100% | 0.100 |
| `R5_3sigma_lantai_0.10_persist` | 0.04 | 0.0000% | **0.00%** | 0% | 42% | 96% | 0.149 |
| `R5_3sigma_lantai_0.10_persist` | 0.06 | 0.0006% | **0.04%** | 2% | 19% | 71% | 0.224 |
| `R6_predint_tanpa_lantai_persist` | 0.02 | 0.0095% | **0.40%** | 26% | 84% | 100% | 0.095 |
| `R6_predint_tanpa_lantai_persist` | 0.04 | 0.0095% | **0.40%** | 5% | 26% | 84% | 0.189 |
| `R6_predint_tanpa_lantai_persist` | 0.06 | 0.0095% | **0.40%** | 2% | 10% | 48% | 0.284 |
| `R7_bonferroni_predint` | 0.02 | 0.1220% | **4.96%** | 49% | 93% | 100% | 0.078 |
| `R7_bonferroni_predint` | 0.04 | 0.2406% | **5.55%** | 20% | 49% | 93% | 0.156 |
| `R7_bonferroni_predint` | 0.06 | 0.2513% | **5.55%** | 13% | 29% | 69% | 0.233 |

## Cara membaca

- **FWER studi** = peluang studi melaporkan setidaknya satu *collapse* padahal tidak ada degradasi nyata. Ini angka yang harus ditekan; FP per-look menyesatkan karena ada 180 uji.
- **MDE80** = degradasi terkecil (pada generasi 6) yang bisa dideteksi dengan power 80%. Jika MDE80 lebih besar dari efek yang kita pedulikan, eksperimen ini tidak akan bisa menjawab pertanyaannya.
- `sigma` tidak diketahui sebelum M0 dilatih. Baris terburuk (0,06) adalah skenario yang harus diasumsikan sampai pengukuran nyata membuktikan sebaliknya.

## Temuan utama (ditulis setelah melihat hasil, bukan sebelum)

1. **`3*sigma_hat` bukan aturan 0,3%.** Pada R3 (tanpa lantai), FWER studi = 11,41% dan **tidak berubah** terhadap sigma, karena masalahnya struktural: `sigma_hat` hanya punya 4 derajat bebas, uji memakai `mu_hat` yang berderau, dan ada 180 uji tanpa koreksi multiplisitas. Aturan v0.2 bisa 'menemukan' collapse pada kira-kira satu dari sembilan studi yang sebenarnya kosong.
2. **Lantai 0,10 bpb menekan FWER dengan cara membutakan deteksi.** R1: FWER 0–8,5%, tetapi MDE80 0,084–0,171 bpb. Pada sigma = 0,06 dibutuhkan degradasi >= 17% relatif sebelum terdeteksi.
3. **Koreksi Bonferroni + selang prediksi (R7) memegang FWER di target 5%** (terukur 4,96–5,55%) dengan MDE80 0,078–0,233 bpb. Ini satu-satunya aturan di tabel yang galatnya terkendali **dan** daya deteksinya tidak kolaps.
4. **Persistensi dua generasi terlalu mahal.** R5 dan R6 menekan FWER hampir ke nol, tetapi power pada efek 0,05 bpb hanya 0–26%; efek yang tumbuh perlahan justru terlewat. Persistensi cocok untuk alarm dini, bukan untuk keputusan utama.
5. **Implikasi epistemik yang harus masuk laporan:** dengan 3 seed dan baseline 5 seed, studi ini tidak mampu mendeteksi degradasi di bawah ~5–8% relatif. Karena itu 'tidak terdeteksi' **tidak boleh** dilaporkan sebagai 'tidak terjadi'.

## Rekomendasi teknis

- Aturan primer: **R7** (Bonferroni atas 180 uji x selang prediksi t_{0,99,4} dengan faktor sqrt(1+1/5), lantai 0,03 bpb).
- Laporkan **R1** dan **R3** sebagai analisis sensitivitas, agar pembaca bisa melihat kesimpulan tidak bergantung pada satu pilihan ambang.
- Tulis **MDE80** di pra-registrasi, sebelum run, dan bandingkan dengan ukuran efek yang dihipotesiskan.
- Bila sigma M0 terukur > 0,06, tambah seed baseline (murah) sampai MDE80 turun ke bawah efek terkecil yang kita pedulikan. Baseline lebih murah daripada salah menyimpulkan.
