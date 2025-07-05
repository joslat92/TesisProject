# -*- coding: utf-8 -*-
"""
infer_lstm_cv.py
----------------
Genera predicciones para TODO el histórico con un modelo LSTM ya entrenado
(cross-validation). Graba un CSV de cortesía en outputs/ y el
predictions_all_folds.csv dentro de la carpeta del modelo.

Uso:
    python src/infer_lstm_cv.py <model_dir> <csv_path> <target_column> [<lookback>]

Ejemplo:
    python src/infer_lstm_cv.py models/LSTM_PLAIN_TUNED \
           data/df_final_ready_plus_vix.csv Target_Price 40
"""
import sys, json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def make_windows(arr: np.ndarray, lookback: int) -> np.ndarray:
    """Crea un array 3-D (n_samples, lookback, n_features)."""
    X = [arr[i : i + lookback] for i in range(len(arr) - lookback)]
    return np.asarray(X)


def print_head(df: pd.DataFrame, n=5):
    """Imprime las primeras n filas bonito."""
    with pd.option_context("display.width", None, "display.max_columns", None):
        print(df.head(n))


# --------------------------------------------------------------------------- #
# 1) Argumentos CLI
# --------------------------------------------------------------------------- #
if len(sys.argv) < 4:
    print(
        "Uso: python infer_lstm_cv.py <model_dir> <csv_path> "
        "<target_column> [<lookback>]"
    )
    sys.exit(1)

MODEL_DIR = Path(sys.argv[1])
CSV_PATH = Path(sys.argv[2])
TARGET_COL = sys.argv[3]
LOOKBACK_CLI = int(sys.argv[4]) if len(sys.argv) >= 5 else None

# --------------------------------------------------------------------------- #
# 2) Metadatos del modelo
# --------------------------------------------------------------------------- #
meta_path = MODEL_DIR / "model_metadata.json"
if meta_path.exists():
    # utf-8-sig ignora el BOM si lo hubiera
    with open(meta_path, encoding="utf-8-sig") as fp:
        meta = json.load(fp)

    FEATURES = meta["features_used"]
    TARGET = meta["target"]
    LOOKBACK = meta["lookback"]
    SCALER_PATH = MODEL_DIR / meta["scaler_path"]
    MODEL_PATH = MODEL_DIR / meta["model_path"]
else:
    # Fallback (mínimo) a los argumentos
    if LOOKBACK_CLI is None:
        print("❌ Falta el lookback y no existe model_metadata.json")
        sys.exit(1)
    FEATURES = [TARGET_COL]          # sólo la target
    TARGET = TARGET_COL
    LOOKBACK = LOOKBACK_CLI
    SCALER_PATH = MODEL_DIR / "scaler.save"
    MODEL_PATH = MODEL_DIR / "model.keras"

print(f"📁 Modelo:          {MODEL_PATH}")
print(f"📈 Columnas usadas: {FEATURES}")
print(f"🔁 Lookback:        {LOOKBACK}")

# --------------------------------------------------------------------------- #
# 3) Carga de datos CSV
# --------------------------------------------------------------------------- #
df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
df = df.loc[:, ~df.columns.duplicated()]  # elimina duplicadas

missing = set(FEATURES) - set(df.columns)
if missing:
    raise ValueError(f"❌ Faltan columnas en el CSV: {missing}")

# --------------------------------------------------------------------------- #
# 4) Cargar scaler y transformar
# --------------------------------------------------------------------------- #
scaler: MinMaxScaler = joblib.load(SCALER_PATH)
if scaler.n_features_in_ != len(FEATURES):
    raise ValueError(
        f"❌ El scaler esperaba {scaler.n_features_in_} columnas y "
        f"recibió {len(FEATURES)}: {FEATURES}"
    )

X_scaled = scaler.transform(df[FEATURES].values)

# --------------------------------------------------------------------------- #
# 5) Ventanas y predicción
# --------------------------------------------------------------------------- #
X = make_windows(X_scaled, LOOKBACK)
print(f"🧩 Shape final de entrada: {X.shape}")

model = load_model(MODEL_PATH)
y_pred_scaled = model.predict(X, verbose=0).flatten()

# --------------------------------------------------------------------------- #
# 6) Des-escalado (solo target) para y_pred e y_true
# --------------------------------------------------------------------------- #
# Creamos un scaler “parcial” solo para la columna target
col_idx = FEATURES.index(TARGET)
target_scaler = MinMaxScaler()
target_scaler.min_, target_scaler.scale_ = (
    scaler.min_[col_idx],
    scaler.scale_[col_idx],
)

y_pred = (y_pred_scaled - target_scaler.min_) / target_scaler.scale_
y_true = df[TARGET].values[LOOKBACK:]  # recorta los primeros LOOKBACK

# --------------------------------------------------------------------------- #
# 7) Guardar resultados
# --------------------------------------------------------------------------- #
out_df = pd.DataFrame(
    {
        "Date": df["Date"].values[LOOKBACK:],
        "y_true": y_true,
        "y_pred": y_pred,
    }
)
# a) dentro de la carpeta del modelo
out_df.to_csv(MODEL_DIR / "predictions_all_folds.csv", index=False)
print(f"📦  Predicciones guardadas en {MODEL_DIR/'predictions_all_folds.csv'}")

# b) copia de cortesía en outputs/
Path("outputs").mkdir(exist_ok=True)
out_df.to_csv("outputs/predicciones_lstm.csv", index=False)
print("✅ Predicciones generadas y CSV de cortesía en outputs/predicciones_lstm.csv")
print("🔍 Primeras 5 filas:")
print_head(out_df)

# --------------------------------------------------------------------------- #
# Fin
# --------------------------------------------------------------------------- #
