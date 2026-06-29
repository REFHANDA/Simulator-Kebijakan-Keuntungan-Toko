# Simulator Kebijakan Keuntungan Toko — Integrasi Akhir (M14–M16)

Dashboard simulasi kebijakan berbasis Machine Learning yang menggabungkan
analisis **What-If**, **MLOps (persistensi & monitoring drift)**, serta
**Explainable AI (SHAP)** dan **Sistem Pendukung Keputusan (SAW)** dalam satu
aplikasi Streamlit.

## ✨ Fitur

| Modul | Sumber Materi | Deskripsi |
|---|---|---|
| Simulator What-If | Minggu 14 | Slider intervensi (Iklan & Diskon) + analisis Delta vs Baseline |
| Persistensi Model | Minggu 15 | Model & scaler disimpan/dimuat via `joblib`, tidak training ulang di app |
| Deteksi Data Drift | Minggu 15 | Peringatan otomatis jika input di luar rentang data latih (>2 std) |
| Etika Data | Minggu 15 | Anonimisasi PII (`clean_sensitive_data`) — didemokan langsung di expander Etika |
| Explainability (XAI) | Minggu 16 | Kontribusi fitur terhadap prediksi via SHAP |
| Rekomendasi (SPK/SAW) | Minggu 16 | Bobot kriteria **interaktif** (Keuntungan/Biaya/Risiko) + ranking alternatif |
| Konsistensi ML↔SPK | Minggu 16 | Peringatan otomatis jika bobot SPK mengabaikan sinyal SHAP yang kuat |
| Komunikasi Ketidakpastian | Minggu 16 | RMSE training ditampilkan sebagai rentang ± pada setiap prediksi |
| Robustness / Stress-Test | Minggu 16 | Mode input manual ekstrem + validasi (clipping & peringatan ekstrapolasi) |
| Justifikasi Black-Box | Minggu 16 | Breakdown skor ternormalisasi per kriteria untuk menjawab "kenapa pilih yang mahal?" |

## 📁 Struktur Folder

```
PROYEK-SIMULASI/
├── app.py              # Dashboard Streamlit (fase deployment / inference)
├── train_model.py      # Skrip training & persistensi model (fase development)
├── requirements.txt    # Daftar library yang dibutuhkan (versi dipatok exact)
├── README.md           # Dokumen ini
└── models/              # Dibuat otomatis oleh train_model.py
    ├── model_keuntungan_v1.joblib
    ├── scaler_keuntungan_v1.joblib
    └── metrics_keuntungan_v1.joblib   # RMSE & metadata untuk komunikasi ketidakpastian
```

## 🚀 Cara Menjalankan

### 1. Instalasi dependensi

```bash
pip install -r requirements.txt
```

### 2. Latih & simpan model (jalankan sekali)

```bash
python train_model.py
```

Skrip ini akan membuat folder `models/` beserta file `model_keuntungan_v1.joblib`
dan `scaler_keuntungan_v1.joblib`. **Wajib dijalankan sebelum membuka dashboard**,
karena `app.py` hanya memuat model — tidak pernah melatih ulang (prinsip
pemisahan development vs deployment, Minggu 15).

### 3. Jalankan dashboard

```bash
streamlit run app.py
```

Browser akan otomatis terbuka menampilkan dashboard simulator.

## 🔁 Melatih Ulang Model (Retraining)

Jika data historis berubah (mis. ada data penjualan baru, atau terjadi
*Data Drift* signifikan), edit `X_train`/`y_train` di `train_model.py`, lalu
jalankan ulang:

```bash
python train_model.py
```

File `.joblib` lama akan tertimpa dengan versi baru. Untuk *model versioning*
yang lebih rapi, ganti nama file (mis. `model_keuntungan_v2.joblib`) dan
perbarui path di `app.py`.

## ☁️ Deploy ke Streamlit Cloud

1. Pastikan folder `models/` (berisi `.joblib`) ikut diunggah ke repositori GitHub —
   **jangan** mengandalkan training otomatis di cloud.
2. Buka [share.streamlit.io](https://share.streamlit.io) → **Create app**.
3. Pilih repository, branch, dan isi *Main file path* dengan `app.py`.
4. Klik **Deploy!**

## ⚠️ Batasan & Etika

- Model dilatih pada data sintetis terbatas (Iklan 5–25 Jt, Diskon 5–25%).
  Hasil prediksi di luar rentang ini ditandai sebagai **drift/ekstrapolasi**
  dan kurang dapat dipercaya.
- Tidak ada kredensial atau API key yang di-hardcode di dalam kode sumber.
- Fungsi anonimisasi data (`clean_sensitive_data`) disiapkan untuk mencegah
  kebocoran data pribadi (PII) apabila data operasional ditambahkan di masa depan.

## 👤 Author

**Refhanda Setyadi Eko Wicasono** — NPM: 2313020058 — Kelas: 3B
