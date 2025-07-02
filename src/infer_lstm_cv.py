# -*- coding: utf-8 -*-
"""
Genera predicciones para un modelo LSTM entrenado con *cross-validation*.

Uso
----
    python src/infer_lstm_cv.py <model_dir> <csv_path> <target_column> [<lookback>]

Ejemplo
-------
    python src/infer_lstm_cv.py models/LSTM_PLAIN_TUNED \
                                data/df_final_ready_plus_vix.csv \
                                Target_Price 40
"""
# ---------------------------------------------------------------------------

import sys
import os
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# 1. Funciones auxiliares
# ---------------------------------------------------------------------------

def make_windows(arr: np.ndarray, window: int) -> np.ndarray:
    """Devuelve un array 3-D (n, window, n_features) con todas las ventanas
    posibles de tamaño ``window``."""
    X = [arr[i : i + window] for i in range(len(arr) - window)]
    return np.asarray(X)

# ---------------------------------------------------------------------------
# 2. Argumentos desde consola
# ---------------------------------------------------------------------------

if len(sys.argv) < 4:
    print("Uso: python infer_lstm_cv.py <model_dir> <csv_path> <target_column> [<lookback>]")
    sys.exit(1)

MODEL_DIR   = Path(sys.argv[1])
CSV_PATH    = Path(sys.argv[2])
TARGET_COL  = sys.argv[3]
LOOKBACK_OV = int(sys.argv[4]) if len(sys.argv) >= 5 else None  # opcional

# ---------------------------------------------------------------------------
# 3. Leer metadatos del modelo
# ---------------------------------------------------------------------------

meta_path = MODEL_DIR / "model_metadata.json"
if not meta_path.exists():
    raise FileNotFoundError(f"❌ No se encontró {meta_path}")

with meta_path.open(encoding="utf-8") as fp:
    meta = json.load(fp)

FEATURES     = meta.get("features_used") or [TARGET_COL]
LOOKBACK     = LOOKBACK_OV or meta.get("lookback", 40)
SCALER_FILE  = MODEL_DIR / meta.get("scaler_path",  "scaler.save")
MODEL_FILE   = MODEL_DIR / meta.get("model_path",   "model.keras")

print(f"📁 Modelo:          {MODEL_FILE}")
print(f"📈 Columnas usadas: {FEATURES}")
print(f"🔁 Lookback:        {LOOKBACK}")

# ---------------------------------------------------------------------------
# 4. Cargar datos y scaler
# ---------------------------------------------------------------------------

df = pd.read_csv(CSV_PATH)
df = df.loc[:, ~df.columns.duplicated()]      # elimina duplicados de cabecera

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"❌ Faltan columnas en el CSV: {missing}")

scaler: MinMaxScaler = joblib.load(SCALER_FILE)
if scaler.n_features_in_ != len(FEATURES):
    raise ValueError(
        f"❌ Scaler esperaba {scaler.n_features_in_} features, "
        f"pero se le pasaron {len(FEATURES)}"
    )

# Normaliza únicamente las columnas necesarias
X_scaled = scaler.transform(df[FEATURES].values)

# ---------------------------------------------------------------------------
# 5. Ventanas y predicción
# ---------------------------------------------------------------------------

X = make_windows(X_scaled, LOOKBACK)
print(f"🧩 Shape final de entrada: {X.shape}")

model = load_model(MODEL_FILE)
y_pred_scaled = model.predict(X, verbose=0)

# ---------------------------------------------------------------------------
# 6. Des-escalado de predicciones
# ---------------------------------------------------------------------------

if len(FEATURES) == 1:
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
else:
    dummy = np.zeros((len(y_pred_scaled), scaler.n_features_in_))
    idx_target = FEATURES.index(TARGET_COL)
    dummy[:, idx_target] = y_pred_scaled.flatten()
    y_pred = scaler.inverse_transform(dummy)[:, idx_target]

# Serie real sin escalar
y_true = df[TARGET_COL].values[LOOKBACK:]

# ---------------------------------------------------------------------------
# 7. Guardar resultados
# ---------------------------------------------------------------------------

date_col = "Date" if "Date" in df.columns else None
out_df = pd.DataFrame({
    "Date": df[date_col].iloc[LOOKBACK:] if date_col else np.arange(len(y_true)),
    "y_true": y_true,
    "y_pred": y_pred
})

out_df.to_csv(MODEL_DIR / "predictions_all_folds.csv", index=False)
print(f"📦  Predicciones guardadas en {MODEL_DIR/'predictions_all_folds.csv'}")

# Además, copia rápida de cortesía a outputs/
out_df.to_csv("outputs/predicciones_lstm.csv", index=False)
print("✅ Predicciones generadas y CSV de cortesía en outputs/predicciones_lstm.csv")

# Muestra un vistazo
print("🔍 Primeras 5 filas:")
print(out_df.head())
