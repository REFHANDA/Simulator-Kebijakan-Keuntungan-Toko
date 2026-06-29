"""
train_model.py
==============
Skrip FASE PENGEMBANGAN (Development).

Sesuai prinsip MLOps Minggu 15: proses training dipisahkan dari aplikasi
deployment (app.py). Jalankan skrip ini SEKALI untuk melatih model dan
menyimpannya ke disk dalam format .joblib. Setelah itu, app.py hanya
perlu memuat (load) file hasil training ini tanpa melatih ulang.

Cara menjalankan:
    python train_model.py

Output:
    models/model_keuntungan_v1.joblib
    models/scaler_keuntungan_v1.joblib
"""

import os
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model_keuntungan_v1.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_keuntungan_v1.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics_keuntungan_v1.joblib")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Data historis (Baseline)
    # Fitur: [Anggaran Iklan (Juta), Besaran Diskon (%)]
    X_train = np.array([[5, 10], [10, 20], [15, 5], [20, 25], [25, 15]], dtype=float)
    # Target: Keuntungan (Juta)
    y_train = np.array([50, 80, 110, 90, 150], dtype=float)

    # 2. Preprocessing (Scaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 3. Training Model (Mesin Replika / Digital Twin)
    model = LinearRegression().fit(X_train_scaled, y_train)
    print("Model dan Scaler berhasil dilatih pada data historis.")

    # 4. Evaluasi Error Model (untuk komunikasi ketidakpastian di UI - M16 Soal 8)
    y_pred_train = model.predict(X_train_scaled)
    rmse = root_mean_squared_error(y_train, y_pred_train)
    metrics = {
        "rmse": float(rmse),
        "feature_names": ["Anggaran Iklan", "Besaran Diskon"],
        "training_range": {
            "iklan_min": float(X_train[:, 0].min()), "iklan_max": float(X_train[:, 0].max()),
            "diskon_min": float(X_train[:, 1].min()), "diskon_max": float(X_train[:, 1].max()),
        },
    }
    print(f"RMSE pada data latih: {rmse:.2f} Juta (ini merepresentasikan rentang ketidakpastian model)")

    # 5. Persistensi Model (Model Exporting)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(metrics, METRICS_PATH)
    print(f"Tersimpan: {MODEL_PATH}")
    print(f"Tersimpan: {SCALER_PATH}")
    print(f"Tersimpan: {METRICS_PATH}")

    # 6. Validasi cepat (Validation Script): load ulang & coba prediksi
    loaded_model = joblib.load(MODEL_PATH)
    loaded_scaler = joblib.load(SCALER_PATH)

    data_baru = np.array([[10.0, 10.0]])  # Baseline check
    data_baru_scaled = loaded_scaler.transform(data_baru)
    hasil = loaded_model.predict(data_baru_scaled)[0]
    print(f"Validasi baseline (Iklan=10, Diskon=10) -> Prediksi Keuntungan: "
          f"Rp {hasil:.2f} ± {rmse:.2f} Juta")


if __name__ == "__main__":
    main()
