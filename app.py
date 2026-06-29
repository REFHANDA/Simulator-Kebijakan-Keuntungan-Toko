"""
app.py
======
Aplikasi Dashboard Streamlit (FASE DEPLOYMENT / Inference).

Integrasi Akhir Minggu 14-16:
- M14: Simulator What-If interaktif (slider, baseline, delta analysis)
- M15: Persistensi model (joblib.load), Monitoring Drift, Etika Data
- M16: Transparansi (SHAP/XAI) & Rekomendasi (MCDM/SAW)

PENTING: Skrip ini TIDAK melatih model. Jalankan train_model.py terlebih
dahulu agar file models/model_keuntungan_v1.joblib dan
models/scaler_keuntungan_v1.joblib tersedia.

Cara menjalankan:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit as st

# =========================================================================
# KEKHASAN: KONFIGURASI HALAMAN & THEME KUSTOM
# =========================================================================
st.set_page_config(
    page_title="Simulator Kebijakan Toko - Final Integrasi",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #4A90E2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    .footer-custom {
        text-align: center;
        padding: 20px;
        font-size: 12px;
        color: #888888;
        border-top: 1px solid #e0e0e0;
        margin-top: 50px;
    }
    .badge-custom {
        background-color: #4A90E2;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# =========================================================================
# M15 - PERSISTENSI MODEL: HANYA LOAD, TIDAK TRAINING DI APLIKASI
# =========================================================================
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model_keuntungan_v1.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_keuntungan_v1.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics_keuntungan_v1.joblib")

# Data historis hanya disimpan di sini sebagai REFERENSI untuk
# drift-check & SHAP background (bukan untuk training ulang).
X_TRAIN_REF = np.array([[5, 10], [10, 20], [15, 5], [20, 25], [25, 15]], dtype=float)


@st.cache_resource
def load_model_dan_scaler():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        st.error(
            f"❌ File model tidak ditemukan di `{MODEL_PATH}` / `{SCALER_PATH}`.\n\n"
            "Jalankan terlebih dahulu: `python train_model.py` sebelum membuka dashboard ini."
        )
        st.stop()
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    # M16 Soal 8: muat metrik error (RMSE) untuk komunikasi ketidakpastian.
    # Fallback aman jika file metrik lama belum di-generate ulang.
    if os.path.exists(METRICS_PATH):
        metrics = joblib.load(METRICS_PATH)
    else:
        metrics = {"rmse": None}
    return model, scaler, metrics


@st.cache_resource
def buat_explainer(_model, _scaler, _X_train):
    background = _scaler.transform(_X_train)
    return shap.LinearExplainer(_model, background)


# =========================================================================
# M16 - VALIDASI INPUT / ROBUSTNESS (Soal Umpan Balik #5)
# Menangani input ekstrem dari mode stress-test manual, agar sistem tidak
# crash dan tetap memberi peringatan yang jujur ke pengguna.
# =========================================================================
def validasi_input(iklan, diskon):
    pesan = []
    iklan_valid, diskon_valid = iklan, diskon

    if iklan < 0:
        iklan_valid = 0.0
        pesan.append("Anggaran Iklan negatif tidak masuk akal secara bisnis -> dipaksa menjadi 0.")
    if diskon < 0:
        diskon_valid = 0.0
        pesan.append("Diskon negatif tidak masuk akal secara bisnis -> dipaksa menjadi 0.")
    if diskon_valid > 100:
        diskon_valid = 100.0
        pesan.append("Diskon tidak bisa lebih dari 100% -> dipotong (clipped) menjadi 100.")
    if iklan_valid > 1000 or diskon_valid > 1000:
        pesan.append(
            "Nilai input sangat jauh di luar jangkauan model (model hanya belajar dari rentang "
            "5–25). Prediksi pada titik ini TIDAK dapat dipercaya — ini murni ekstrapolasi linear, "
            "bukan estimasi yang divalidasi data."
        )
    return iklan_valid, diskon_valid, pesan


# =========================================================================
# M16 - KONSISTENSI LOGIKA ML vs SPK (Soal Umpan Balik #2)
# Mengecek apakah bobot kriteria "Keuntungan" (representasi sinyal ML/SHAP)
# di SAW masih proporsional dengan kekuatan pengaruh model. Jika bobot
# Keuntungan dibuat sangat kecil padahal model punya sinyal kuat, beri
# peringatan -- ini analog dengan kasus "SHAP bilang Suhu penting, tapi
# bobot SPK untuk Suhu hanya 0.01".
# =========================================================================
def cek_konsistensi_ml_spk(bobot_keuntungan, total_kontribusi_shap_abs, ambang_bobot=0.15):
    if bobot_keuntungan < ambang_bobot and total_kontribusi_shap_abs > 5:
        return (
            f"⚠️ **Inkonsistensi Terdeteksi:** Model ML (SHAP) menunjukkan variabel input memiliki "
            f"kontribusi nyata (≈{total_kontribusi_shap_abs:.1f} Jt terhadap pergeseran prediksi), "
            f"namun bobot kriteria **Keuntungan** pada SPK hanya **{bobot_keuntungan:.2f}** "
            f"(di bawah ambang {ambang_bobot}). Akibatnya, rekomendasi akhir SPK akan didominasi "
            f"kriteria statis (Biaya/Risiko) dan **mengabaikan insight prediksi ML** — sama seperti "
            f"kasus 'Suhu penting menurut SHAP tapi bobotnya 0.01 di SPK'. Pertimbangkan menaikkan "
            f"bobot Keuntungan agar keputusan tetap selaras dengan hasil model."
        )
    return None


def get_baseline(model, scaler):
    baseline_input = np.array([[10.0, 10.0]])
    baseline_scaled = scaler.transform(baseline_input)
    return model.predict(baseline_scaled)[0]


def run_simulation(model, scaler, baseline_pred, new_iklan, new_diskon):
    intervention_input = np.array([[new_iklan, new_diskon]], dtype=float)
    intervention_scaled = scaler.transform(intervention_input)
    prediction = model.predict(intervention_scaled)[0]
    delta_y = prediction - baseline_pred
    return prediction, delta_y, intervention_input, intervention_scaled


# =========================================================================
# M15 - MONITORING: DETEKSI DATA DRIFT SEDERHANA
# =========================================================================
def check_data_drift(new_input, X_train, threshold_std=2.0):
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    z_scores = np.abs((new_input[0] - train_mean) / train_std)
    nama_fitur = ["Anggaran Iklan", "Besaran Diskon"]
    drift_terdeteksi = [(nama_fitur[i], z) for i, z in enumerate(z_scores) if z > threshold_std]
    return drift_terdeteksi, z_scores


# =========================================================================
# M15 - ETIKA PRAKTIS: ANONYMIZATION (disiapkan untuk data riwayat operator)
# =========================================================================
def clean_sensitive_data(df_input, cols_to_remove=("Nama_Operator", "NIK_Petugas")):
    return df_input.drop(columns=[c for c in cols_to_remove if c in df_input.columns])


# =========================================================================
# M16 - INTEGRASI MCDM (SAW)
# =========================================================================
def jalankan_saw(matriks, tipe_kriteria, bobot):
    norm = np.zeros_like(matriks, dtype=float)
    for j in range(matriks.shape[1]):
        col = matriks[:, j]
        if tipe_kriteria[j] == "benefit":
            norm[:, j] = col / col.max() if col.max() != 0 else 0
        else:
            norm[:, j] = (col.min() / col) if np.all(col != 0) else 0
    skor_akhir = norm @ bobot
    return skor_akhir, norm


def bangun_alternatif_saw(model, scaler, iklan_user, diskon_user, hasil_pred, bobot):
    alt_input = {
        "Konservatif": (5.0, 5.0),
        "Skenario Anda": (iklan_user, diskon_user),
        "Agresif": (35.0, 35.0),
    }

    nama_alt, keuntungan_list, biaya_list, risiko_list = [], [], [], []
    for nama, (iklan, diskon) in alt_input.items():
        if nama == "Skenario Anda":
            pred = hasil_pred
        else:
            scaled = scaler.transform(np.array([[iklan, diskon]]))
            pred = model.predict(scaled)[0]
        biaya = iklan + diskon
        risiko = 0.3 * diskon + 0.1 * iklan

        nama_alt.append(nama)
        keuntungan_list.append(pred)
        biaya_list.append(biaya)
        risiko_list.append(risiko)

    matriks = np.column_stack([keuntungan_list, biaya_list, risiko_list])
    tipe_kriteria = ["benefit", "cost", "cost"]

    skor, norm = jalankan_saw(matriks, tipe_kriteria, bobot)

    df_saw = pd.DataFrame({
        "Alternatif": nama_alt,
        "Keuntungan (Juta)": keuntungan_list,
        "Biaya (Proksi)": biaya_list,
        "Risiko (Proksi)": risiko_list,
        "Skor SAW": skor,
    }).sort_values("Skor SAW", ascending=False).reset_index(drop=True)
    df_saw["Ranking"] = df_saw.index + 1
    return df_saw, norm, nama_alt


# =========================================================================
# LOAD MODEL (sekali, dari disk) & SIAPKAN STATE
# =========================================================================
model, scaler, metrics = load_model_dan_scaler()
baseline_pred = get_baseline(model, scaler)
explainer = buat_explainer(model, scaler, X_TRAIN_REF)
rmse = metrics.get("rmse")

# --- UI HEADER ---
st.markdown('<h1 class="main-title">Simulator Kebijakan Keuntungan — Integrasi Akhir (M14–M16)</h1>',
            unsafe_allow_html=True)
st.markdown(
    "Simulator ini menggabungkan **Mesin Prediksi (ML)**, **Monitoring Drift**, **Penjelasan SHAP (XAI)**, "
    "dan **Rekomendasi MCDM (SAW)** dalam satu dashboard `What-If` yang utuh."
)
st.markdown(
    '<span class="badge-custom">Model: Linear Regression v1</span> '
    '<span class="badge-custom">Persistensi: joblib</span> '
    '<span class="badge-custom">XAI: SHAP</span> '
    '<span class="badge-custom">SPK: SAW</span>',
    unsafe_allow_html=True,
)
st.caption(
    "🔗 **Alur Data Terintegrasi:** Slider/Input UI → `run_simulation()` (Model ML) → "
    "`bangun_alternatif_saw()` (Matriks Keputusan) → `jalankan_saw()` (Skor) → Tabel Ranking SPK."
)

# --- SIDEBAR: Variabel Kontrol (Tuas Kebijakan) ---
st.sidebar.markdown("### 🛠️ Tuas Kebijakan *(Intervensi)*")
st.sidebar.write("Geser nilai di bawah untuk memanipulasi strategi bisnis:")

iklan_slider = st.sidebar.slider("Anggaran Iklan (Juta IDR)", 0, 50, 10, help="Tentukan alokasi dana pemasaran")
diskon_slider = st.sidebar.slider("Besaran Diskon (%)", 0, 50, 10, help="Potongan harga produk")

st.sidebar.markdown("---")
st.sidebar.caption(
    "📌 Model dilatih pada rentang Iklan 5–25 Jt & Diskon 5–25%. "
    "Nilai di luar rentang ini berisiko ekstrapolasi (lihat peringatan Drift)."
)

# --- M16 Soal 5: Mode Stress-Test (input manual tanpa batas slider) ---
st.sidebar.markdown("---")
mode_stress = st.sidebar.checkbox(
    "🧪 Mode Stress-Test (input manual ekstrem)",
    help="Uji Robustness sistem dengan nilai di luar batas slider, mis. -50 atau 5000."
)
pesan_validasi = []
if mode_stress:
    iklan_manual = st.sidebar.number_input("Iklan Manual (Juta) - bisa ekstrem", value=float(iklan_slider))
    diskon_manual = st.sidebar.number_input("Diskon Manual (%) - bisa ekstrem", value=float(diskon_slider))
    iklan_final, diskon_final, pesan_validasi = validasi_input(iklan_manual, diskon_manual)
else:
    iklan_final, diskon_final = float(iklan_slider), float(diskon_slider)

# --- ENGINE: Jalankan Simulasi ---
hasil_pred, delta, intervensi_raw, intervensi_scaled = run_simulation(
    model, scaler, baseline_pred, iklan_final, diskon_final
)

if pesan_validasi:
    for p in pesan_validasi:
        st.error(f"🛑 **Validasi Input:** {p}")

# --- M15: Cek Drift sebelum menampilkan hasil ---
drift_terdeteksi, z_scores = check_data_drift(intervensi_raw, X_TRAIN_REF, threshold_std=2.0)
if drift_terdeteksi:
    daftar = ", ".join([f"{nama} (z={z:.2f})" for nama, z in drift_terdeteksi])
    st.warning(
        f"⚠️ **Peringatan Drift / Ekstrapolasi:** Input Anda berada jauh di luar pola data historis pada "
        f"variabel: **{daftar}**. Hasil prediksi di bawah ini kemungkinan kurang akurat karena model "
        f"belum pernah belajar dari kondisi seekstrem ini."
    )

# --- UI MAIN CONTENT: Tab agar terorganisir (Prediksi, XAI, SPK) ---
tab1, tab2, tab3 = st.tabs(["📊 Hasil Prediksi", "🔍 Penjelasan (XAI)", "🧭 Rekomendasi (SPK)"])

with tab1:
    col_kiri, col_kanan = st.columns([1, 1], gap="large")

    with col_kiri:
        st.subheader("📊 Metrik Hasil Simulasi")
        m1, m2 = st.columns(2)
        if rmse is not None:
            m1.metric(label="Estimasi Keuntungan", value=f"Rp {hasil_pred:.2f} Jt",
                       help=f"Rentang ketidakpastian model (RMSE): ± Rp {rmse:.2f} Juta")
        else:
            m1.metric(label="Estimasi Keuntungan", value=f"Rp {hasil_pred:.2f} Jt")
        m2.metric(label="Perubahan (Delta)", value=f"Rp {delta:.2f} Jt", delta=f"{delta:.2f} Jt")

        if rmse is not None:
            st.caption(
                f"🎯 **Rentang Estimasi (±1 RMSE):** Rp {hasil_pred - rmse:.2f} Jt "
                f"hingga Rp {hasil_pred + rmse:.2f} Jt. Angka tunggal di atas adalah titik tengah, "
                f"bukan nilai pasti — selalu pertimbangkan rentang ini saat melapor ke atasan."
            )

        st.markdown("### 💡 Analisis Strategi")
        if delta > 15:
            st.balloons()
            st.success(
                f"🔥 **Strategi Jenius!** Skenario ini menghasilkan lonjakan keuntungan signifikan sebesar "
                f"**Rp {delta:.2f} Juta** di atas baseline."
            )
        elif delta > 0:
            st.success(
                f"✅ **Bagus!** Skenario ini menunjukkan tren positif dengan kenaikan sebesar "
                f"**Rp {delta:.2f} Juta** dibandingkan kondisi baseline."
            )
        elif delta < 0:
            st.warning(
                f"⚠️ **Peringatan Korosif!** Kebijakan ini berisiko menurunkan keuntungan sebesar "
                f"**Rp {abs(delta):.2f} Juta**. Pertimbangkan kembali anggarannya."
            )
        else:
            st.info("⚖️ **Stagnan.** Skenario ini menghasilkan keuntungan yang sama persis dengan baseline.")

    with col_kanan:
        st.subheader("📉 Visualisasi Komparatif")
        data_plot = pd.DataFrame({
            'Skenario': ['Kondisi Awal (Baseline)', 'Simulasi Anda'],
            'Keuntungan (Juta)': [baseline_pred, hasil_pred]
        })
        st.bar_chart(data=data_plot, x='Skenario', y='Keuntungan (Juta)', color="#4A90E2")

    st.markdown("---")
    st.info(
        f"📌 **Info Baseline:** Jika Anggaran Iklan = Rp 10 Juta dan Diskon = 10%, "
        f"maka keuntungan standar toko Anda adalah **Rp {baseline_pred:.2f} Juta**."
    )

with tab2:
    st.subheader("🔍 Mengapa Hasilnya Demikian? (SHAP Explainability)")
    st.write(
        "Grafik di bawah menunjukkan seberapa besar kontribusi setiap variabel terhadap pergeseran "
        "prediksi dari nilai dasar (base value) menuju estimasi keuntungan skenario Anda saat ini."
    )

    shap_values = explainer(intervensi_scaled)

    fig, ax = plt.subplots(figsize=(7, 3))
    fitur_names = ["Anggaran Iklan", "Besaran Diskon"]
    kontribusi = shap_values.values[0]
    warna = ["#4A90E2" if v >= 0 else "#FF4B4B" for v in kontribusi]
    ax.barh(fitur_names, kontribusi, color=warna)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Kontribusi terhadap Prediksi (Juta IDR)")
    ax.set_title("Kontribusi Fitur (SHAP values)")
    st.pyplot(fig)

    st.caption(
        f"Base value (rata-rata prediksi model) ≈ Rp {shap_values.base_values[0]:.2f} Jt. "
        f"Total kontribusi fitur menggeser hasil menjadi ≈ Rp {hasil_pred:.2f} Jt."
    )

    if abs(kontribusi[0]) > abs(kontribusi[1]):
        st.success("➡️ Pada skenario ini, **Anggaran Iklan** adalah tuas kebijakan paling sensitif.")
    elif abs(kontribusi[1]) > abs(kontribusi[0]):
        st.success("➡️ Pada skenario ini, **Besaran Diskon** adalah tuas kebijakan paling sensitif.")
    else:
        st.info("➡️ Kedua variabel memberikan kontribusi yang setara pada skenario ini.")

    total_kontribusi_shap_abs = float(np.abs(kontribusi).sum())

with tab3:
    st.subheader("🧭 Rekomendasi Akhir (Multi-Criteria Decision Making — SAW)")
    st.write(
        "Simulator tidak berhenti pada angka prediksi: hasil ML digabungkan dengan kriteria Biaya dan Risiko "
        "untuk memberikan rekomendasi tindakan yang konkret antara tiga alternatif kebijakan."
    )

    # --- M16 Soal 9: Bobot dibuat interaktif agar terlihat sinergi/dominasi ML vs Pakar (SPK) ---
    st.markdown("##### ⚖️ Atur Bobot Kriteria (Analisis Sinergi ML vs Pakar)")
    bc1, bc2, bc3 = st.columns(3)
    w_keuntungan = bc1.slider("Bobot Keuntungan (sinyal ML)", 0.0, 1.0, 0.5, 0.05)
    w_biaya = bc2.slider("Bobot Biaya (statis)", 0.0, 1.0, 0.3, 0.05)
    w_risiko = bc3.slider("Bobot Risiko (statis)", 0.0, 1.0, 0.2, 0.05)
    total_bobot = w_keuntungan + w_biaya + w_risiko
    if total_bobot == 0:
        bobot_norm = np.array([0.34, 0.33, 0.33])
    else:
        bobot_norm = np.array([w_keuntungan, w_biaya, w_risiko]) / total_bobot
    st.caption(
        f"Bobot ternormalisasi: Keuntungan={bobot_norm[0]:.2f} · Biaya={bobot_norm[1]:.2f} · "
        f"Risiko={bobot_norm[2]:.2f}"
    )

    # M16 Soal 2: Cek konsistensi bobot ML vs SPK
    peringatan_konsistensi = cek_konsistensi_ml_spk(bobot_norm[0], total_kontribusi_shap_abs)
    if peringatan_konsistensi:
        st.warning(peringatan_konsistensi)

    df_saw, norm_matrix, nama_alt_list = bangun_alternatif_saw(
        model, scaler, iklan_final, diskon_final, hasil_pred, bobot_norm
    )
    st.dataframe(
        df_saw.style.format({
            "Keuntungan (Juta)": "{:.2f}",
            "Biaya (Proksi)": "{:.2f}",
            "Risiko (Proksi)": "{:.2f}",
            "Skor SAW": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    alternatif_terbaik = df_saw.iloc[0]["Alternatif"]
    st.success(
        f"🏆 **Rekomendasi SPK:** Berdasarkan bobot kriteria saat ini (Keuntungan {bobot_norm[0]:.0%}, "
        f"Biaya {bobot_norm[1]:.0%}, Risiko {bobot_norm[2]:.0%}), alternatif **{alternatif_terbaik}** "
        f"memiliki skor SAW tertinggi dan paling disarankan."
    )

    if alternatif_terbaik != "Skenario Anda":
        st.caption(
            "💡 Skenario yang sedang Anda coba di slider belum menjadi pilihan optimal menurut SPK. "
            "Coba sesuaikan slider mendekati profil alternatif yang direkomendasikan."
        )

    # --- M16 Soal 4: Justifikasi "kenapa pilih yang lebih mahal?" lewat breakdown skor ternormalisasi ---
    with st.expander("🗣️ Jawaban untuk Penguji: \"Kenapa sistem memilih alternatif ini meski lebih mahal?\""):
        df_breakdown = pd.DataFrame(
            norm_matrix,
            columns=["Skor Norm. Keuntungan", "Skor Norm. Biaya", "Skor Norm. Risiko"],
            index=nama_alt_list,
        )
        st.dataframe(df_breakdown.style.format("{:.2f}"), use_container_width=True)
        termahal = df_saw.loc[df_saw["Biaya (Proksi)"].idxmax(), "Alternatif"]
        if alternatif_terbaik == termahal:
            st.write(
                f"**{alternatif_terbaik}** memang memiliki Biaya tertinggi, namun skor ternormalisasi pada "
                f"kolom *Keuntungan* jauh lebih unggul dibanding alternatif lain. Karena bobot Keuntungan "
                f"({bobot_norm[0]:.0%}) cukup besar, keunggulan ini **menutupi (compensate)** kekurangan "
                f"di kriteria Biaya — inilah prinsip *compensatory* pada metode SAW."
            )
        else:
            st.write(
                f"Pada bobot saat ini, alternatif termahal (**{termahal}**) justru **tidak** terpilih — "
                f"artinya kriteria Biaya/Risiko cukup berat untuk mengalahkan keunggulan Keuntungannya. "
                f"Coba naikkan slider *Bobot Keuntungan* di atas untuk melihat kapan alternatif ini mulai unggul."
            )

st.markdown("---")

with st.expander("ℹ️ Catatan Etika, Keterbatasan, dan Replayability Sistem (Wajib M15–M16)"):
    st.markdown(f"""
    - **Persistensi Model:** Model & scaler dimuat dari `{MODEL_PATH}` dan `{SCALER_PATH}` menggunakan
      `joblib`, hasil dari `train_model.py` — aplikasi ini **tidak pernah** melatih ulang model.
    - **Batas Validitas Model:** Model dilatih pada data Iklan 5–25 Jt dan Diskon 5–25%. Di luar rentang
      ini, sistem akan memunculkan peringatan **Data Drift** karena risiko ekstrapolasi.
    - **Tingkat Ketidakpastian:** Prediksi bersifat estimasi linear sederhana (RMSE ≈
      {f'{rmse:.2f} Jt' if rmse is not None else 'belum dihitung'}), bukan nilai pasti — selalu
      pertimbangkan rentang error saat mengambil keputusan bisnis.
    - **Keamanan:** Tidak ada kredensial/API key yang ditulis langsung di kode (hardcoding) pada aplikasi ini.
    """)

    st.markdown("##### 🔒 Demo Anonimisasi Data (M16 Soal Umpan Balik #7)")
    st.write(
        "Andaikan dashboard ini diperluas untuk mencatat siapa yang menjalankan setiap skenario "
        "(misal untuk audit internal), data mentah berikut **mengandung PII** (Nama & NIK Operator) "
        "yang melanggar etika jika ditampilkan di dashboard publik:"
    )
    raw_log = pd.DataFrame({
        "Nama_Operator": ["Budi Santoso"],
        "NIK_Petugas": ["3210xxxxxxxx0001"],
        "Anggaran Iklan": [iklan_final],
        "Diskon": [diskon_final],
        "Prediksi Keuntungan": [round(float(hasil_pred), 2)],
    })
    st.caption("Data mentah (sebelum anonimisasi) — TIDAK boleh ditampilkan publik:")
    st.dataframe(raw_log, use_container_width=True, hide_index=True)

    log_bersih = clean_sensitive_data(raw_log)
    st.caption("Setelah `clean_sensitive_data()` dipanggil — aman ditampilkan publik:")
    st.dataframe(log_bersih, use_container_width=True, hide_index=True)
    st.success(
        "✅ Kolom `Nama_Operator` dan `NIK_Petugas` otomatis dibuang sebelum data ini boleh "
        "ditampilkan di dashboard atau diunggah ke repositori publik."
    )

st.markdown(
    """
    <div class="footer-custom">
        <p>Didevelop secara eksklusif oleh <b>Refhanda Setyadi Eko Wicasono</b> | NPM: 2313020058 | Kelas: 3B</p>
        <p>Integrasi Akhir: Minggu 14 (Simulator What-If) · Minggu 15 (MLOps & Persistensi) · Minggu 16 (XAI & SPK)</p>
    </div>
    """,
    unsafe_allow_html=True
)
