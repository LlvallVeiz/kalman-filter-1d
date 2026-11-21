# kalman-filter-1d

Filter Kalman satu dimensi dalam Rust murni (std-only, tanpa
dependensi eksternal): filter skalar dan constant-velocity, pembanding
α–β, smoother Rauch–Tung–Striebel, estimasi noise adaptif, dan uji
konsistensi NEES — semuanya bisa dicek hitung tangan pada contoh di
`examples/hand_check.md`.

Sebuah sensor melaporkan besaran `z` yang merupakan keadaan sebenarnya
`x` ditambah noise. Filter Kalman (Kalman, 1960) adalah estimator
rekursif dengan mean-square error minimum untuk masalah ini bila
dinamikanya linear dan semua noise Gaussian: ia menyimpan estimasi
keadaan beserta kovarians errornya, lalu memperbarui keduanya setiap
ada pengukuran baru. Proyek ini mengimplementasikan filter itu untuk
sinyal 1D — suhu ruangan, ketinggian barometrik, level, posisi
sepanjang satu sumbu — lengkap dengan alat untuk membuktikan bahwa
filter tersebut bekerja (RMSE vs pengukuran mentah, uji NEES
chi-square) dan untuk membandingkannya dengan alternatif yang lebih
sederhana (filter α–β).

## Fitur

- **Filter skalar** (random walk): prediksi + update, gain Kalman,
  kovarians; rumus steady-state tertutup `K∞` (diuji vs 500 iterasi).
- **Filter constant-velocity (CV)**: keadaan posisi + kecepatan,
  model discrete white-noise acceleration; aritmetika matriks 2×2
  ditulis tangan (tanpa crate aljabar linear) — plus varian
  `with_discrete_q`.
- **Filter α–β** sebagai pembanding (Benedict & Bordner, 1962),
  dengan gain dari tracking index Kalata (1984).
- **Smoother Rauch–Tung–Striebel** (RTS) untuk skalar dan CV:
  memakai seluruh batch (termasuk pengukuran "masa depan") sehingga
  selalu lebih akurat dari filter di titik interior.
- **Estimasi noise adaptif**: `R` dari covariance matching
  inovasi (Mehra, 1970) dan `Q` dari NIS, dengan peredaman agar
  tidak divergen (lihat `docs/RESEARCH.md` §6 — versi pertama
  sempat divergen dan didokumentasikan).
- **CLI**: baca CSV seri waktu (`t,z[,truth]`), jalankan filter,
  tulis CSV hasil + ringkasan statistik (RMSE, NEES, NIS);
  mode batch untuk banyak file sekaligus.
- **Generator data sintetik**: step, sinus, ramp, sensor suhu
  realistis (ayunan diurnal + drift), dan random-walk ketinggian —
  semua dengan ground truth dan seed yang reproducible.
- **Validasi**: RMSE filter vs pengukuran mentah; uji konsistensi
  NEES chi-square (kuantil chi-square dihitung eksak via fungsi
  gamma tak lengkap — tanpa tabel, tanpa crate statistik).

## Instalasi

Butuh toolchain Rust (cargo ≥ 1.70; diuji dengan rustc 1.99.1).
Tidak ada crate eksternal — proyek ini build sepenuhnya offline.

```bash
cargo build --release
# binary: target/release/kalman-filter-1d
```

## Penggunaan

```text
kalman-filter-1d <subcommand> [options]

    filter   <input.csv>      Jalankan filter; tulis <stem>.filtered.csv
                               + <stem>.summary.txt
        --filter scalar|cv    Jenis filter (default: scalar)
        --q Q                 Varians noise proses / PSD (default: 0,01)
        --r R                 Varians noise pengukuran (default: 0,1)
        --p0 P0               Varians error awal (default: 1,0)
        --x0 X0               Keadaan awal (default: pengukuran pertama)
        --output FILE         Jalur CSV keluaran
        --smooth              Jalankan juga smoother RTS
        --adaptive [LAMBDA]   Retune Q/R dari inovasi (default lambda 0,98)

    generate <scenario>       Tulis seri sintetik dengan ground truth.
        Skenario: step, sine, ramp, temperature, altitude
        --n N --dt DT --noise S --seed S --p1 X --p2 Y --output FILE

    compare  <input.csv>      Scalar KF vs CV KF vs alpha-beta.
        --q Q --r R --p0 P0 --alpha A --beta B --lambda L

    batch    <input-dir>      Filter semua *.csv dalam satu direktori.
        --output-dir DIR --filter scalar|cv --q Q --r R --p0 P0

    handcheck                 Cetak contoh 3 langkah yang bisa dicek tangan.
```

Format CSV masukan (satu header, satu sampel per baris):

```text
t,z,truth
0.0,1.02,1.0
1.0,0.98,1.0
```

Kolom `truth` opsional — bila ada, CLI melaporkan RMSE dan NEES;
bila tidak ada, hanya seri terfilter yang ditulis. Waktu harus
tidak menurun; `dt` untuk filter CV diambil dari selisih waktu,
jadi sampling tak beraturan didukung.

## Contoh — hasil asli dari run beneran

Semua file di `examples/` dibuat ulang oleh
`bash examples/run_examples.sh` (seed 42 di semua skenario).
Angka di bawah disalin langsung dari output perintah itu.

### 1. Sinyal step — filter skalar

```bash
kalman-filter-1d generate step --seed 42 --output examples/step.csv
kalman-filter-1d filter examples/step.csv --q 0.1 --r 0.25 --smooth
```

```text
kalman-filter-1d filter (scalar, q=0.1, r=0.25, p0=1) — step.csv
samples: 200
RMSE raw measurements : 0.496934
RMSE filtered         : 0.337091
improvement factor    : 1.474x
NEES mean             : 0.9796 (95% interval [0.8136, 1.2053]) → CONSISTENT
mean NIS              : 1.0486
wrote step.filtered.csv
wrote step.summary.txt
wrote step.smoothed.csv
RMSE smoothed: 0.296310
```

Isi `step.filtered.csv` (5 baris pertama; `x0` = pengukuran pertama,
jadi inovasi baris pertama 0):

```text
t,z,truth,x,p,gain,innovation
0,0.2073598752157653,0,0.2073598752,0.2037037037,0.8148148148,0.0000000000
1,0.32634061107597145,0,0.2726202119,0.1371237458,0.5484949833,0.1189807359
2,-0.44594310681387817,0,-0.0771644726,0.1216958462,0.4867833848,-0.7185633188
3,0.6634167814070533,0,0.2709068304,0.1174993632,0.4699974528,0.7405812540
4,0.8647965439687012,0,0.5472080292,0.1163099783,0.4652399133,0.5938897133
```

### 2. Ringkasan semua skenario

| Skenario | Filter | RMSE mentah | RMSE filter | Membaik | RMSE smoother | NEES |
|---|---|---|---|---|---|---|
| step (200) | skalar Q=0,1 R=0,25 | 0,4969 | 0,3371 | 1,47× | 0,2963 | 0,98 konsisten |
| sine (200) | skalar Q=0,01 R=0,09 | 0,2982 | 0,1652 | 1,81× | 0,0971 | 1,07 konsisten |
| ramp (200) | CV Q=0,001 R=0,25 | 0,4969 | 0,2583 | 1,92× | 0,1500 | 0,85 konsisten |
| suhu (288) | skalar Q=0,005 R=0,04 | 0,1934 | 0,1006 | 1,92× | 0,0626 | 0,85 konsisten |
| altitude (1000) | CV Q=0,0025 R=1,0 | 1,0043 | 0,5351 | 1,88× | 0,2956 | 1,05 konsisten |

Smoother selalu menang dari filter (0,34→0,30; 0,17→0,10;
0,26→0,15; 0,10→0,06; 0,54→0,30) — harga yang dibayar: ia butuh
seluruh batch dulu, jadi tidak bisa online.

### 3. Perbandingan tiga estimator

```bash
kalman-filter-1d compare examples/step.csv --q 0.1 --r 0.25
kalman-filter-1d compare examples/ramp.csv --q 0.01 --r 0.25
```

```text
compare — step.csv (200 samples, q=0.1, r=0.25, p0=1)
  raw measurements : RMSE 0.496934
  scalar Kalman    : RMSE 0.337091 (1.47x better than raw)
  CV Kalman        : RMSE 0.393930 (1.26x better than raw)
  alpha-beta (a=0.913, b=0.996) : RMSE 0.463210 (1.07x better than raw)
best: scalar Kalman (RMSE 0.337091)

compare — ramp.csv (200 samples, q=0.01, r=0.25, p0=1)
  raw measurements : RMSE 0.496934
  scalar Kalman    : RMSE 0.499785 (0.99x better than raw)
  CV Kalman        : RMSE 0.308756 (1.61x better than raw)
  alpha-beta (a=0.913, b=0.996) : RMSE 0.460898 (1.08x better than raw)
best: CV Kalman (RMSE 0.308756)
```

Pelajaran dari tabel ini: tidak ada filter yang selalu menang.
Pada step, filter skalar menang (sinyal diam di dua level); pada
ramp, filter CV menang karena keadaan kecepatannya membawa tren —
filter skalar malah *tidak lebih baik dari data mentah* (0,99×)
dan NEES-nya 5,50 (inkonsisten): ia tertinggal konstan di belakang
sinyal yang bergerak. File `ramp_scalar.filtered.csv` menyimpan
kasus gagal itu sebagai contoh.

### 4. Mode adaptif — tuning Q/R dari data

```bash
kalman-filter-1d filter examples/temp_sensor.csv --q 0.01 --r 0.04 --adaptive
```

```text
samples: 288
RMSE raw measurements : 0.193367
RMSE filtered         : 0.113257
improvement factor    : 1.707x
NEES mean             : 1.0655 (95% interval [0.8433, 1.1698]) → CONSISTENT
mean NIS              : 1.0503
adaptive: final R estimate 0.021973, mean NIS 0.9541
```

Titik awal yang sama *tanpa* adaptif memberi RMSE 0,1046 dengan
NEES 0,70 (inkonsisten — kovarians terlalu pesimistis). Adaptif
menukar sedikit RMSE (0,105 → 0,113) dengan kovarians yang jujur
(NEES 0,70 → 1,07). Estimasi akhir `R̂ = 0,022` berada pada orde
yang sama dengan nilai benar `R = 0,04`.

### 5. Mode batch

```bash
kalman-filter-1d batch examples --output-dir batch_output --q 0.1 --r 0.25
```

```text
file, samples, rmse_raw, rmse_filtered, improvement, nees_mean, status
altitude.csv, 1000, 1.0043, 0.8467, 1.19x, 6.188, ok
ramp.csv, 200, 0.4969, 0.3107, 1.60x, 0.832, ok
sine.csv, 200, 0.2982, 0.1727, 1.73x, 0.257, ok
step.csv, 200, 0.4969, 0.3371, 1.47x, 0.980, ok
temp_sensor.csv, 288, 0.1934, 0.1118, 1.73x, 0.108, ok
batch: 5/5 files filtered → batch_output
```

Satu pasangan Q/R untuk semua file: semua membaik, tetapi NEES
altitude (6,19) dan sine (0,26) menunjukkan satu tuning tidak
cocok untuk semua sinyal — itulah gunanya mode adaptif dan tuning
per skenario di atas. File yang rusak (CSV tidak valid) dilaporkan
`FAILED` dan dilewati tanpa menggagalkan batch.

### 6. Contoh yang bisa dicek hitung tangan

`kalman-filter-1d handcheck` mencetak tiga langkah pertama filter
skalar (`x0=0, P0=1, Q=0,01, R=0,1`, `z = [1,0, 1,1, 0,9]`)
dengan aritmetika lengkap — `x` akhir 0,9658, `P` akhir 0,0376 —
dan perhitungan yang sama oleh filter itu sendiri, sehingga tidak
mungkin angkanya melenceng dari implementasi. Detail lengkap +
langkah CV dan α–β ada di `examples/hand_check.md`.

## Cara kerja (ringkas)

Filter skalar (random walk), per langkah:

```text
prediksi:  x⁻ = x,  P⁻ = P + Q
gain:      K  = P⁻ / (P⁻ + R)
update:    x  = x⁻ + K·(z − x⁻),   P = (1 − K)·P⁻
```

Filter CV memakai model yang sama dalam bentuk matriks
(`F = [[1, dt], [0, 1]]`, `H = [1, 0]`,
`Q = q·[[dt⁴/4, dt³/2], [dt³/2, dt²]]`). Smoother RTS
menjalankan rekursi mundur dari gain
`G = P·Fᵀ/P⁻_berikutnya`. Teori lengkap, termasuk mengapa NEES
filter 2-keadaan harus ≈ 2 (bukan 1) dan mengapa uji NEES pada run
pendek kurang bertenaga, ada di `docs/RESEARCH.md`.

## Struktur proyek

```text
kalman-filter-1d/
├── Cargo.toml            # std-only, tanpa dependensi eksternal
├── README.md
├── LICENSE               # MIT
├── docs/
│   └── RESEARCH.md       # teori + daftar referensi (judul + URL/DOI)
├── src/
│   ├── lib.rs            # dokumentasi crate + contoh cepat
│   ├── main.rs           # entry point binary
│   ├── cli.rs            # parsing argumen + subcommand
│   ├── scalar.rs         # filter Kalman skalar (random walk)
│   ├── cv.rs             # filter constant-velocity (posisi + kecepatan)
│   ├── alpha_beta.rs     # filter α–β + tracking index Kalata
│   ├── smoother.rs       # smoother Rauch–Tung–Striebel
│   ├── noise.rs          # estimasi adaptif Q/R dari inovasi
│   ├── metrics.rs        # RMSE, NEES, kuantil chi-square eksak
│   ├── rng.rs            # SplitMix64 + Box–Muller (seeded)
│   ├── series.rs         # model seri waktu + parsing CSV
│   ├── synth.rs          # generator skenario sintetik
│   ├── output.rs         # CSV keluaran + ringkasan
│   ├── batch.rs          # mode batch
│   └── errors.rs         # tipe error
├── tests/                # 9 file integration test (CLI end-to-end dll.)
└── examples/
    ├── run_examples.sh   # regenerasi semua contoh dari nol
    ├── hand_check.md     # perhitungan tangan + verifikasi live
    ├── hand_check.txt    # output `handcheck`
    ├── step.csv / sine.csv / ramp.csv / temp_sensor.csv / altitude.csv
    ├── *.filtered.csv / *.smoothed.csv / *.summary.txt
    ├── compare_step.txt / compare_ramp.txt
    ├── adaptive_temp.txt
    └── batch_summary.txt + batch_output/
```

## Tes

```bash
cargo test            # 114 tes: 79 unit + 34 integration + 1 doc test
cargo test --release  # sama, pada build optimasi
cargo clippy --all-targets   # 0 warning
```

Tes mencakup: tiga langkah pertama filter skalar dan CV yang
dicocokkan dengan perhitungan tangan independen (Python);
steady-state `K∞` tertutup vs 500 iterasi; konsistensi NEES pada
run 1000–2000 sampel untuk filter yang *matched* (dan
inkonsistensi yang diharapkan untuk filter yang salah tuning);
smoother yang selalu mengalahkan filter; generator yang
deterministik per seed; CLI end-to-end (generate → filter →
compare → batch) lewat binary asli.

## Keputusan desain

- **Std-only, tanpa dependensi.** Build offline penuh; PRNG sendiri
  (SplitMix64 + Box–Muller) dan kuantil chi-square sendiri (gamma
  tak lengkap Numerical Recipes) supaya tidak ada yang perlu
  diunduh. Konsekuensinya: parsing argumen ditulis tangan dan
  matriks 2×2 dikalikan manual — keduanya diuji menyeluruh.
- **Semua besaran antara tersimpan.** Setiap langkah mengembalikan
  struct (`ScalarStep`, `CvStep`) berisi prediksi, gain, inovasi,
  dan varians — bukan hanya estimasi akhir — supaya smoother,
  estimator adaptif, dan NEES bisa memakai angka yang sama
  persis dengan yang dilihat pengguna di CSV.
- **Kegagalan dilaporkan, bukan disembunyikan.** NEES yang
  inkonsisten dicetak apa adanya; file batch yang rusak
  dilaporkan `FAILED`; `compare` menolak input tanpa ground
  truth; README ini memuat kasus filter skalar yang kalah dari
  data mentah pada ramp.
- **Angka contoh dari run beneran.** `examples/run_examples.sh`
  meregenerasi semua file contoh; README menyalin output
  perintah itu, bukan angka karangan.

## Keterbatasan

- **Hanya model linear-Gaussian 1D.** Tidak ada Extended/Unscented
  Kalman filter; sinyal non-linear (mis. ayunan besar, saturasi
  sensor) butuh model lain. Tidak ada penanganan outlier
  (satu pengukuran liar bisa menarik estimasi).
- **Filter skalar tidak bisa mengikuti tren.** Pada ramp/sinyal
  bergerak ia tertinggal konstan — pakai filter CV atau naikkan Q.
- **Noise harus white dan stasioner** agar optimalitas dan uji
  NEES berlaku; noise berkorelasi atau drift yang cepat butuh
  model keadaan yang lebih kaya.
- **Uji NEES pada run pendek kurang bertenaga.** Nilai NEES
  berurutan saling berkorelasi (filter punya memori), sehingga
  interval chi-square terlalu sempit untuk ~200 sampel: pada
  filter altitude yang *matched*, hanya 56% dari 50 seed yang
  lolos interval 95% pada n=200 (rata-rata mean NEES 1,01 —
  benar). Gunakan run panjang (n=1000 pada contoh altitude)
  untuk putusan NEES.
- **Tuning Q/R tetap perlu.** Nilai default (`Q=0,01, R=0,1`)
  hanyalah titik awal; mode `--adaptive` membantu tetapi bukan
  pengganti pemahaman akan sensornya.
- **CSV sederhana.** Parser tidak mendukung nilai yang dikutip
  (`"..."`); kolom harus persis `t,z[,truth]`.

## Pengembangan lebih lanjut

- Extended Kalman filter untuk model non-linear (mis. sensor
  jarak non-linear); deteksi outlier berbasis NIS
  (tolak pengukuran bila NIS di atas ambang chi-square);
- estimasi `Q`/`R` offline berbasis maximum-likelihood
  (EM) sebagai alternatif covariance matching;
- dukungan `dt` per-sampel pada smoother CV bila ada celah waktu
  besar; output JSON selain CSV.

## Referensi

Daftar lengkap dengan judul + URL/DOI ada di
[`docs/RESEARCH.md`](docs/RESEARCH.md): Kalman (1960), Welch &
Bishop (2006), Rauch–Tung–Striebel (1965), Mehra (1970),
Benedict & Bordner (1962), Kalata (1984), Bar-Shalom, Li &
Kirubarajan (2001), Gelb (1974), Simon (2006), Box & Muller
(1958), Wilson & Hilferty (1931), Numerical Recipes (2007).

## Lisensi

MIT — lihat [LICENSE](LICENSE).
