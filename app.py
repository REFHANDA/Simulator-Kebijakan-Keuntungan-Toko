from pathlib import Path
import logging
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Simulator Kebijakan Toko",
    page_icon="📊",
    layout="wide"
)

# ==========================================================
# CONSTANT
# ==========================================================

CONFIG = {
    "MODEL_DIR": Path("models"),
    "MODEL_NAME": "model_keuntungan_v1.joblib",
    "SCALER_NAME": "scaler_keuntungan_v1.joblib",
    "METRIC_NAME": "metrics_keuntungan_v1.joblib",
    "BASELINE_IKLAN": 10.0,
    "BASELINE_DISKON": 10.0,
    "DRIFT_Z": 2.0,
    "MAX_DISKON": 100,
    "MAX_EXTREME": 1000,
    "RISIKO_DISKON": 0.30,
    "RISIKO_IKLAN": 0.10,
}

MODEL_PATH = CONFIG["MODEL_DIR"] / CONFIG["MODEL_NAME"]
SCALER_PATH = CONFIG["MODEL_DIR"] / CONFIG["SCALER_NAME"]
METRIC_PATH = CONFIG["MODEL_DIR"] / CONFIG["METRIC_NAME"]
METRICS_PATH = METRIC_PATH  # alias untuk Replayability section

# ==========================================================
# TRAINING REFERENCE
# ==========================================================

X_TRAIN_REF = np.array(
    [
        [5, 10],
        [10, 20],
        [15, 5],
        [20, 25],
        [25, 15],
    ],
    dtype=float,
)

FEATURE_NAMES = [
    "Anggaran Iklan",
    "Besaran Diskon",
]

TRAIN_MEAN = X_TRAIN_REF.mean(axis=0)
TRAIN_STD = X_TRAIN_REF.std(axis=0)

# ==========================================================
# CUSTOM CSS
# ==========================================================

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
    color: #888;
    border-top: 1px solid #DDD;
    margin-top: 50px;
}
.badge-custom {
    background: #4A90E2;
    color: white;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD MODEL
# ==========================================================

def load_artifacts():
    if not MODEL_PATH.exists():
        st.error(
            "Model belum tersedia.\n\n"
            "Jalankan train_model.py terlebih dahulu."
        )
        st.stop()

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    metrics = joblib.load(METRIC_PATH) if METRIC_PATH.exists() else {}

    logging.info("Model berhasil dimuat.")
    return model, scaler, metrics


# ==========================================================
# SHAP EXPLAINER
# ==========================================================

def load_explainer(
    _model, 
    _scaler
):
    background = _scaler.transform(X_TRAIN_REF)
    return shap.LinearExplainer(
        _model, 
        background
    )


# ==========================================================
# PREDICTION HELPER
# ==========================================================

def predict(model, scaler, X: np.ndarray):
    """
    Prediction helper.
    Menghindari penulisan scaler.transform()
    dan model.predict() berulang.
    """
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    return pred, X_scaled


# ==========================================================
# BASELINE
# ==========================================================

def get_baseline(model, scaler):
    baseline = np.array([[
        CONFIG["BASELINE_IKLAN"],
        CONFIG["BASELINE_DISKON"]
    ]])
    pred, _ = predict(model, scaler, baseline)
    return float(pred[0])


# ==========================================================
# INPUT VALIDATION
# ==========================================================

def validasi_input(iklan: float, diskon: float) -> tuple[float, float, list[str]]:
    """Validasi input pengguna."""
    warning = []
    iklan = float(iklan)
    diskon = float(diskon)

    if iklan < 0:
        iklan = 0
        warning.append("Anggaran iklan tidak boleh negatif. Nilai diubah menjadi 0.")

    if diskon < 0:
        diskon = 0
        warning.append("Diskon tidak boleh negatif. Nilai diubah menjadi 0.")

    if diskon > CONFIG["MAX_DISKON"]:
        diskon = CONFIG["MAX_DISKON"]
        warning.append("Diskon melebihi 100%. Nilai dipotong menjadi 100%.")

    if iklan > CONFIG["MAX_EXTREME"] or diskon > CONFIG["MAX_EXTREME"]:
        warning.append(
            "Input sangat jauh dari data pelatihan. "
            "Prediksi merupakan ekstrapolasi sehingga akurasi tidak dapat dijamin."
        )

    return iklan, diskon, warning


# ==========================================================
# SIMULATION
# ==========================================================

def run_simulation(model, scaler, baseline_pred: float, iklan: float, diskon: float):
    """Menjalankan simulasi satu skenario."""
    input_data = np.array([[iklan, diskon]], dtype=float)
    pred, scaled = predict(model, scaler, input_data)
    prediction = float(pred[0])
    delta = prediction - baseline_pred
    return prediction, delta, input_data, scaled


# ==========================================================
# DRIFT DETECTION
# ==========================================================

def check_data_drift(new_input: np.ndarray, threshold: float = CONFIG["DRIFT_Z"]):
    """Monitoring sederhana menggunakan Z-Score."""
    z_score = np.abs((new_input[0] - TRAIN_MEAN) / TRAIN_STD)
    drift = [
        (FEATURE_NAMES[idx], float(nilai))
        for idx, nilai in enumerate(z_score)
        if nilai > threshold
    ]
    return drift, z_score


# ==========================================================
# PRIVACY
# ==========================================================

def clean_sensitive_data(dataframe: pd.DataFrame, columns=("Nama_Operator", "NIK_Petugas")):
    """Menghapus informasi sensitif."""
    hasil = dataframe.copy()
    for col in columns:
        if col in hasil.columns:
            hasil.drop(columns=col, inplace=True)
    return hasil


# ==========================================================
# SHAP CONSISTENCY
# ==========================================================

def cek_konsistensi_ml_spk(bobot_keuntungan: float, total_shap: float, threshold: float = 0.15):
    """Memastikan hasil SHAP konsisten dengan bobot SAW."""
    if bobot_keuntungan < threshold and total_shap > 5:
        return (
            "⚠️ Bobot Keuntungan pada SAW terlalu kecil dibandingkan kontribusi "
            "yang ditunjukkan oleh SHAP. Rekomendasi SPK berpotensi mengabaikan hasil model ML."
        )
    return None


# ==========================================================
# SAW NORMALIZATION
# ==========================================================

def jalankan_saw(matriks: np.ndarray, tipe_kriteria: list[str], bobot: np.ndarray):
    """Simple Additive Weighting (SAW)."""
    matriks = matriks.astype(float)
    normalisasi = np.zeros_like(matriks, dtype=float)

    for col in range(matriks.shape[1]):
        nilai = matriks[:, col]
        if tipe_kriteria[col] == "benefit":
            maksimum = nilai.max()
            if maksimum != 0:
                normalisasi[:, col] = nilai / maksimum
        else:
            minimum = nilai.min()
            if minimum != 0:
                normalisasi[:, col] = minimum / nilai

    skor = normalisasi @ bobot
    return skor, normalisasi


# ==========================================================
# BUILD SAW ALTERNATIVES
# ==========================================================

def bangun_alternatif_saw(
    model, scaler, iklan_user: float, diskon_user: float,
    hasil_prediksi: float, bobot: np.ndarray
):
    """Membangun alternatif keputusan menggunakan Batch Prediction."""
    nama_alternatif = ["Konservatif", "Skenario Anda", "Agresif"]

    input_model = np.array(
        [[5.0, 5.0], [iklan_user, diskon_user], [35.0, 35.0]],
        dtype=float,
    )

    prediksi, _ = predict(model, scaler, input_model)
    prediksi = prediksi.astype(float)
    prediksi[1] = hasil_prediksi

    biaya = input_model[:, 0] + input_model[:, 1]
    risiko = (
        CONFIG["RISIKO_DISKON"] * input_model[:, 1]
        + CONFIG["RISIKO_IKLAN"] * input_model[:, 0]
    )

    matriks = np.column_stack((prediksi, biaya, risiko))
    tipe = ["benefit", "cost", "cost"]
    skor, normalisasi = jalankan_saw(matriks, tipe, bobot)

    df = pd.DataFrame({
        "Alternatif": nama_alternatif,
        "Keuntungan (Juta)": prediksi,
        "Biaya (Proksi)": biaya,
        "Risiko (Proksi)": risiko,
        "Skor SAW": skor,
    })

    df = df.sort_values("Skor SAW", ascending=False).reset_index(drop=True)
    df["Ranking"] = np.arange(len(df)) + 1

    return df, normalisasi, nama_alternatif


# ==========================================================
# LOAD ARTIFACTS
# ==========================================================

model, scaler, metrics = load_artifacts()
baseline_pred = get_baseline(model, scaler)
explainer = load_explainer(model, scaler)

rmse = None
if "metrics" in metrics:
    rmse = metrics["metrics"].get("RMSE")
elif "rmse" in metrics:
    rmse = metrics["rmse"]

logging.info("Deployment siap digunakan.")

# ==========================================================
# HEADER
# ==========================================================

st.markdown('<h1 class="main-title">Simulator Kebijakan Keuntungan</h1>', unsafe_allow_html=True)

st.write("""
Dashboard ini mengintegrasikan:
- Machine Learning
- What-If Simulation
- SHAP Explainability
- Data Drift Monitoring
- Multi Criteria Decision Making (SAW)

ke dalam satu sistem pendukung keputusan.
""")

col1, col2, col3, col4 = st.columns(4)
col1.success("Linear Regression")
col2.info("Joblib")
col3.warning("SHAP")
col4.error("SAW")

st.divider()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("Parameter Simulasi")

iklan_slider = st.sidebar.slider("Anggaran Iklan (Juta)", min_value=0, max_value=50, value=10)
diskon_slider = st.sidebar.slider("Diskon (%)", min_value=0, max_value=50, value=10)

st.sidebar.caption("Model dilatih pada rentang Iklan 5-25 Juta dan Diskon 5-25%.")
st.sidebar.divider()

mode_stress = st.sidebar.checkbox("Stress Test", value=False)

if mode_stress:
    iklan_manual = st.sidebar.number_input("Iklan Manual", value=float(iklan_slider))
    diskon_manual = st.sidebar.number_input("Diskon Manual", value=float(diskon_slider))
    iklan, diskon, warning_input = validasi_input(iklan_manual, diskon_manual)
else:
    iklan = float(iklan_slider)
    diskon = float(diskon_slider)
    warning_input = []

# ==========================================================
# ENGINE
# ==========================================================

hasil_prediksi, delta_prediksi, input_raw, input_scaled = run_simulation(
    model, scaler, baseline_pred, iklan, diskon
)

# ==========================================================
# WARNING INPUT
# ==========================================================

for warning in warning_input:
    st.warning(warning)

# ==========================================================
# DRIFT MONITOR
# ==========================================================

drift, z_score = check_data_drift(input_raw)

if drift:
    nama = [f"{fitur} (z={nilai:.2f})" for fitur, nilai in drift]
    st.warning("⚠️ Data Drift Terdeteksi\n\n" + ", ".join(nama))

# ==========================================================
# SHAP
# ==========================================================

shap_values = explainer(input_scaled)
kontribusi = shap_values.values[0]
base_value = float(shap_values.base_values[0])
total_shap = float(np.abs(kontribusi).sum())

# ==========================================================
# SAW DEFAULT
# ==========================================================

DEFAULT_BOBOT = np.array([0.50, 0.30, 0.20])

df_default, norm_default, nama_alt_default = bangun_alternatif_saw(
    model, scaler, iklan, diskon, hasil_prediksi, DEFAULT_BOBOT
)

# ==========================================================
# TABS
# ==========================================================

tab1, tab2, tab3 = st.tabs(["📊 Hasil Prediksi", "🔍 Explainability", "🧭 Rekomendasi"])

# ==========================================================
# TAB 1 — HASIL PREDIKSI
# ==========================================================

with tab1:
    kiri, kanan = st.columns(2, gap="large")

    with kiri:
        st.subheader("Hasil Simulasi")
        metric1, metric2 = st.columns(2)

        if rmse is not None:
            metric1.metric(
                "Estimasi Keuntungan",
                f"Rp {hasil_prediksi:.2f} Jt",
                help=f"RMSE ± {rmse:.2f}"
            )
        else:
            metric1.metric("Estimasi Keuntungan", f"Rp {hasil_prediksi:.2f} Jt")

        metric2.metric("Delta", f"{delta_prediksi:.2f} Jt", delta=f"{delta_prediksi:.2f}")

        if rmse is not None:
            st.info(
                f"Rentang prediksi: "
                f"**Rp {hasil_prediksi - rmse:.2f} Jt** hingga **Rp {hasil_prediksi + rmse:.2f} Jt**"
            )

        st.divider()

        if delta_prediksi > 20:
            st.success("🚀 Strategi sangat baik.\n\nPeningkatan keuntungan sangat signifikan.")
        elif delta_prediksi > 0:
            st.success("✅ Strategi memberikan peningkatan keuntungan.")
        elif delta_prediksi == 0:
            st.info("Tidak ada perubahan dibanding baseline.")
        else:
            st.error("Strategi berpotensi menurunkan keuntungan.")

        st.divider()
        st.caption(
            f"Baseline Model\n\n"
            f"Iklan : {CONFIG['BASELINE_IKLAN']:.0f} Juta\n\n"
            f"Diskon : {CONFIG['BASELINE_DISKON']:.0f} %\n\n"
            f"Prediksi : Rp {baseline_pred:.2f} Juta"
        )

    with kanan:
        st.subheader("Perbandingan")

        chart_df = pd.DataFrame({
            "Skenario": ["Baseline", "Simulasi"],
            "Keuntungan": [baseline_pred, hasil_prediksi],
        })
        st.bar_chart(chart_df, x="Skenario", y="Keuntungan")

        st.divider()

        persen = (delta_prediksi / baseline_pred) * 100
        st.metric("Persentase Perubahan", f"{persen:.2f}%")
        st.progress(min(max(hasil_prediksi / 200, 0), 1))
        st.caption("Visualisasi menunjukkan tingkat pencapaian keuntungan terhadap target hipotetik 200 Juta.")

    st.divider()
    st.subheader("Ringkasan Simulasi")

    hasil_ringkas = pd.DataFrame({
        "Parameter": ["Iklan", "Diskon", "Prediksi", "Delta"],
        "Nilai": [
            f"{iklan:.2f}",
            f"{diskon:.2f}",
            f"{hasil_prediksi:.2f}",
            f"{delta_prediksi:.2f}",
        ],
    })
    st.dataframe(hasil_ringkas, use_container_width=True, hide_index=True)

# ==========================================================
# TAB 2 — EXPLAINABILITY
# ==========================================================

with tab2:
    st.subheader("SHAP Explainability")
    st.write(
        "SHAP menunjukkan bagaimana setiap variabel menggeser prediksi model "
        "dari nilai rata-rata (Base Value) menuju hasil akhir."
    )

    fig, ax = plt.subplots(figsize=(8, 3))
    warna = ["#4A90E2" if nilai >= 0 else "#FF4B4B" for nilai in kontribusi]
    ax.barh(FEATURE_NAMES, kontribusi, color=warna)
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_xlabel("Kontribusi (Juta)")
    ax.set_title("Kontribusi Setiap Variabel")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    st.info(
        f"Base Value ≈ Rp {base_value:.2f} Juta\n\n"
        f"Prediksi Akhir ≈ Rp {hasil_prediksi:.2f} Juta"
    )

    df_shap = pd.DataFrame({
        "Variabel": FEATURE_NAMES,
        "Kontribusi": kontribusi,
        "Absolut": np.abs(kontribusi),
    })
    df_shap = df_shap.sort_values("Absolut", ascending=False).reset_index(drop=True)

    st.dataframe(
        df_shap.style.format({"Kontribusi": "{:.2f}", "Absolut": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

    fitur_utama = df_shap.iloc[0]
    arah = "meningkatkan" if fitur_utama["Kontribusi"] > 0 else "menurunkan"
    st.success(
        f"Variabel paling berpengaruh adalah **{fitur_utama['Variabel']}** "
        f"yang **{arah}** prediksi sebesar **{abs(fitur_utama['Kontribusi']):.2f} Juta**"
    )

    st.markdown("### Breakdown")
    for fitur, nilai in zip(FEATURE_NAMES, kontribusi):
        if nilai >= 0:
            st.success(f"{fitur} +{nilai:.2f} Juta")
        else:
            st.error(f"{fitur} {nilai:.2f} Juta")

    st.divider()
    st.metric("Total Kontribusi Absolut", f"{total_shap:.2f}")

    with st.expander("Bagaimana membaca grafik SHAP?"):
        st.write("""
• Nilai positif → menaikkan prediksi.
• Nilai negatif → menurunkan prediksi.
• Semakin jauh dari nol → semakin besar pengaruhnya.
• SHAP membantu menjelaskan mengapa model menghasilkan prediksi tertentu
  sehingga keputusan menjadi lebih transparan.
""")

# ==========================================================
# TAB 3 — REKOMENDASI
# ==========================================================

with tab3:
    st.subheader("Rekomendasi Keputusan (SAW)")
    st.write(
        "Prediksi Machine Learning dikombinasikan dengan metode Simple Additive Weighting "
        "untuk menentukan alternatif terbaik."
    )

    st.markdown("### Bobot Kriteria")
    c1, c2, c3 = st.columns(3)
    w_keuntungan = c1.slider("Keuntungan", 0.0, 1.0, 0.50, 0.05)
    w_biaya = c2.slider("Biaya", 0.0, 1.0, 0.30, 0.05)
    w_risiko = c3.slider("Risiko", 0.0, 1.0, 0.20, 0.05)

    bobot = np.array([w_keuntungan, w_biaya, w_risiko])
    bobot = np.array([0.34, 0.33, 0.33]) if bobot.sum() == 0 else bobot / bobot.sum()

    st.caption(
        f"Keuntungan : {bobot[0]:.2f}\n\n"
        f"Biaya : {bobot[1]:.2f}\n\n"
        f"Risiko : {bobot[2]:.2f}"
    )

    konsistensi_warning = cek_konsistensi_ml_spk(bobot[0], total_shap)
    if konsistensi_warning:
        st.warning(konsistensi_warning)

    df_saw, norm_matrix, nama_alt = bangun_alternatif_saw(
        model, scaler, iklan, diskon, hasil_prediksi, bobot
    )

    st.dataframe(
        df_saw.style.format({
            "Keuntungan (Juta)": "{:.2f}",
            "Biaya (Proksi)": "{:.2f}",
            "Risiko (Proksi)": "{:.2f}",
            "Skor SAW": "{:.3f}",
        }),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### Skor Alternatif")
    chart = pd.DataFrame({"Alternatif": df_saw["Alternatif"], "Skor": df_saw["Skor SAW"]})
    st.bar_chart(chart, x="Alternatif", y="Skor")

    terbaik = df_saw.iloc[0]
    st.success(
        f"Alternatif terbaik adalah **{terbaik['Alternatif']}** "
        f"dengan skor **{terbaik['Skor SAW']:.3f}**"
    )

    if terbaik["Alternatif"] != "Skenario Anda":
        st.info("Skenario yang Anda buat belum menjadi pilihan terbaik menurut SAW.")

    st.markdown("### Breakdown Normalisasi")
    df_norm = pd.DataFrame(
        norm_matrix,
        columns=["Keuntungan", "Biaya", "Risiko"],
        index=nama_alt,
    )
    st.dataframe(df_norm.style.format("{:.2f}"), use_container_width=True)

    with st.expander("Mengapa alternatif ini dipilih?"):
        st.write(f"""
Metode SAW memilih **{terbaik['Alternatif']}** karena mempunyai skor total terbesar
(**{terbaik['Skor SAW']:.3f}**). Nilai tersebut diperoleh dari penjumlahan seluruh
kriteria yang telah dinormalisasi dan dikalikan dengan bobot.

Semakin besar bobot Keuntungan, semakin besar pengaruh prediksi Machine Learning.
Semakin besar bobot Biaya atau Risiko, semakin konservatif rekomendasi yang diberikan.
""")

    st.divider()
    st.metric("Alternatif Terbaik", terbaik["Alternatif"])
    st.metric("Skor SAW", f"{terbaik['Skor SAW']:.3f}")

# ==========================================================
# INFORMASI SISTEM
# ==========================================================

st.divider()

with st.expander("Informasi Sistem", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Machine Learning")
        info_model = pd.DataFrame({
            "Komponen": ["Algoritma", "Versi", "Scaler", "Framework", "Deployment", "Inference"],
            "Nilai": ["Linear Regression", "1.0", "StandardScaler", "Scikit-Learn", "Streamlit", "Realtime"],
        })
        st.dataframe(info_model, hide_index=True, use_container_width=True)
        st.success("Model berhasil dimuat menggunakan Joblib.")

    with col2:
        st.subheader("Monitoring")
        monitor = {
            "RMSE": f"{rmse:.2f}" if rmse else "-",
            "Data Drift": "Ya" if drift else "Tidak",
            "Stress Test": "Aktif" if mode_stress else "Tidak",
            "Alternatif": len(df_saw),
        }
        for key, value in monitor.items():
            st.metric(key, value)

# ==========================================================
# PRIVASI DAN ETIKA AI
# ==========================================================

st.divider()

with st.expander("Privasi dan Etika AI"):
    st.write("Dashboard ini menerapkan prinsip Responsible AI.")

    raw = pd.DataFrame({
        "Nama_Operator": ["Budi"],
        "NIK": ["321xxxxxxxx"],
        "Iklan": [iklan],
        "Diskon": [diskon],
        "Prediksi": [hasil_prediksi],
    })

    st.caption("Data Mentah")
    st.dataframe(raw, hide_index=True, use_container_width=True)

    st.caption("Sesudah Anonimisasi")
    st.dataframe(clean_sensitive_data(raw), hide_index=True, use_container_width=True)
    st.success("Data sensitif berhasil dihapus.")

# ==========================================================
# KETERBATASAN MODEL
# ==========================================================

st.divider()

with st.expander("Keterbatasan Model"):
    st.warning(
        "Model hanya belajar dari rentang:\n\n"
        "Iklan 5–25 Juta\n\n"
        "Diskon 5–25%\n\n"
        "Input di luar rentang merupakan ekstrapolasi."
    )
    if rmse is not None:
        st.info(f"RMSE ± {rmse:.2f} Juta digunakan sebagai estimasi ketidakpastian.")

# ==========================================================
# REPLAYABILITY
# ==========================================================

st.divider()

with st.expander("Replayability"):
    st.code(
        f"Model\n{MODEL_PATH}\n\nScaler\n{SCALER_PATH}\n\nMetrics\n{METRICS_PATH}",
        language="text",
    )
    st.success("Deployment dapat direproduksi dengan menjalankan train_model.py")

# ==========================================================
# SESSION SUMMARY
# ==========================================================

st.divider()
st.subheader("Ringkasan Simulasi")

summary = pd.DataFrame({
    "Parameter": ["Iklan", "Diskon", "Prediksi", "Delta", "Drift", "Alternatif"],
    "Nilai": [
        iklan,
        diskon,
        round(hasil_prediksi, 2),
        round(delta_prediksi, 2),
        len(drift),
        df_saw.iloc[0]["Alternatif"],
    ],
})
st.dataframe(summary, hide_index=True, use_container_width=True)

# ==========================================================
# FOOTER
# ==========================================================

st.divider()
st.caption("""
Developed by Refhanda Setyadi Eko Wicasono
NPM 2313020058 — Pemodelan dan Simulasi — Semester 6
Machine Learning | What If Analysis | MLOps | Explainable AI | Decision Support System
2026
""")
