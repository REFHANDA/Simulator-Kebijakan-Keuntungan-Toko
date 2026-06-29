from pathlib import Path
from datetime import datetime
import logging

import joblib
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)

# ==========================================================
# KONFIGURASI
# ==========================================================

CONFIG = {
    "random_seed": 42,
    "model_version": "1.0.0",
    "algorithm": "Linear Regression",
}

np.random.seed(CONFIG["random_seed"])

# ==========================================================
# PATH
# ==========================================================

MODEL_DIR = Path("models")

MODEL_PATH   = MODEL_DIR / "model_keuntungan_v1.joblib"
SCALER_PATH  = MODEL_DIR / "scaler_keuntungan_v1.joblib"
METRICS_PATH = MODEL_DIR / "metrics_keuntungan_v1.joblib"
BUNDLE_PATH  = MODEL_DIR / "model_bundle.joblib"
LOG_PATH     = MODEL_DIR / "training.log"

# ==========================================================
# LOGGING
# ==========================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

logger.info("=" * 60)
logger.info("MEMULAI PROSES TRAINING")
logger.info("=" * 60)

# ==========================================================
# DATA PREPARATION
# ==========================================================

def prepare_data() -> tuple[np.ndarray, np.ndarray]:
    """Menyiapkan data historis untuk proses training."""
    logger.info("Menyiapkan data historis...")

    training_data = {
        "iklan":      [5, 10, 15, 20, 25],
        "diskon":     [10, 20, 5, 25, 15],
        "keuntungan": [50, 80, 110, 90, 150],
    }

    X_train = np.column_stack([
        training_data["iklan"],
        training_data["diskon"],
    ]).astype(float)

    y_train = np.array(training_data["keuntungan"], dtype=float)

    logger.info("Jumlah sampel : %d", len(X_train))
    logger.info("Jumlah fitur  : %d", X_train.shape[1])
    logger.info("Shape X       : %s", X_train.shape)
    logger.info("Shape y       : %s", y_train.shape)

    return X_train, y_train


# ==========================================================
# TRAIN MODEL
# ==========================================================

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[LinearRegression, StandardScaler, np.ndarray]:
    """
    Melatih model Linear Regression beserta scaler.

    Parameters
    ----------
    X_train : np.ndarray
        Data fitur.
    y_train : np.ndarray
        Target.

    Returns
    -------
    model, scaler, X_scaled
    """
    logger.info("Memulai proses training model...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = LinearRegression(fit_intercept=True, copy_X=True, positive=False)
    model.fit(X_scaled, y_train)

    logger.info("Training selesai.")
    logger.info("Koefisien Model : %s", np.round(model.coef_, 4))
    logger.info("Intercept       : %.4f", model.intercept_)

    return model, scaler, X_scaled


# ==========================================================
# EVALUASI MODEL
# ==========================================================

def evaluate_model(
    model: LinearRegression,
    X_scaled: np.ndarray,
    y_train: np.ndarray,
    X_train: np.ndarray,
) -> dict:
    """Menghitung performa model dan membuat metadata."""
    logger.info("Menghitung metrik evaluasi...")

    y_pred = model.predict(X_scaled)

    mae  = mean_absolute_error(y_train, y_pred)
    rmse = root_mean_squared_error(y_train, y_pred)
    r2   = r2_score(y_train, y_pred)

    logger.info("-" * 50)
    logger.info("MAE  : %.3f", mae)
    logger.info("RMSE : %.3f", rmse)
    logger.info("R²   : %.4f", r2)
    logger.info("-" * 50)

    metadata = {
        "version":    CONFIG["model_version"],
        "algorithm":  CONFIG["algorithm"],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "samples":    int(len(X_train)),
        "features":   int(X_train.shape[1]),
        "feature_names": ["Anggaran Iklan", "Besaran Diskon"],
        "training_range": {
            "iklan_min":  float(X_train[:, 0].min()),
            "iklan_max":  float(X_train[:, 0].max()),
            "diskon_min": float(X_train[:, 1].min()),
            "diskon_max": float(X_train[:, 1].max()),
        },
        "metrics": {
            "MAE":  float(mae),
            "RMSE": float(rmse),
            "R2":   float(r2),
        },
        "baseline_prediction": float(np.mean(y_pred)),
    }

    logger.info("Metadata model berhasil dibuat.")
    return metadata


# ==========================================================
# SAVE ARTIFACTS
# ==========================================================

def save_artifacts(
    model: LinearRegression,
    scaler: StandardScaler,
    metadata: dict,
) -> None:
    """Menyimpan seluruh artefak model ke dalam folder models."""
    logger.info("Menyimpan artefak model...")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model,    MODEL_PATH)
    joblib.dump(scaler,   SCALER_PATH)
    joblib.dump(metadata, METRICS_PATH)

    bundle = {"model": model, "scaler": scaler, "metadata": metadata}
    joblib.dump(bundle, BUNDLE_PATH)

    logger.info("Seluruh artefak berhasil disimpan.")

    # Verifikasi file
    artifacts = [MODEL_PATH, SCALER_PATH, METRICS_PATH, BUNDLE_PATH]

    logger.info("-" * 60)
    logger.info("Verifikasi file hasil penyimpanan")

    for file in artifacts:
        if file.exists():
            size_kb = file.stat().st_size / 1024
            logger.info("✓ %s (%.2f KB)", file.name, size_kb)
        else:
            raise FileNotFoundError(f"{file} gagal dibuat.")

    logger.info("-" * 60)


# ==========================================================
# VALIDATION
# ==========================================================

def validate_model() -> None:
    """
    Memastikan model yang telah disimpan dapat dimuat kembali
    dan menghasilkan prediksi.
    """
    logger.info("Melakukan validasi artefak...")

    bundle   = joblib.load(BUNDLE_PATH)
    model    = bundle["model"]
    scaler   = bundle["scaler"]
    metadata = bundle["metadata"]

    sample        = np.array([[10.0, 10.0]])
    sample_scaled = scaler.transform(sample)
    prediction    = model.predict(sample_scaled)[0]

    logger.info("Prediksi validasi : %.2f", prediction)
    logger.info("Versi model       : %s",   metadata["version"])
    logger.info("Algoritma         : %s",   metadata["algorithm"])
    logger.info("Training Time     : %s",   metadata["created_at"])
    logger.info("Validasi artefak berhasil.")


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    """
    Pipeline utama training model.

    Tahapan:
    1. Menyiapkan data
    2. Training model
    3. Evaluasi performa
    4. Menyimpan artefak
    5. Validasi hasil simpan
    """
    logger.info("=" * 60)
    logger.info("PIPELINE TRAINING DIMULAI")
    logger.info("=" * 60)

    try:
        X_train, y_train = prepare_data()

        model, scaler, X_scaled = train_model(X_train, y_train)

        metadata = evaluate_model(
            model=model,
            X_scaled=X_scaled,
            y_train=y_train,
            X_train=X_train,
        )

        save_artifacts(model=model, scaler=scaler, metadata=metadata)

        validate_model()

        logger.info("=" * 60)
        logger.info("TRAINING BERHASIL DISELESAIKAN")
        logger.info("=" * 60)

    except Exception as err:
        logger.exception("Training gagal karena terjadi error:")
        raise err


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()