import sys
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# FUNCIONES AUXILIARES
# -----------------------------

def make_windows(data, window_size):
    """Crea ventanas de tamaño fijo para series de tiempo"""
    X = []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
    return np.array(X)

# -----------------------------
# PARÁMETROS DESDE CONSOLA
# -----------------------------

if len(sys.argv) < 4:
    print("Uso: python infer_lstm_cv.py <model_path> <csv_path> <target_column> [<lookback>]")
    sys.exit(1)

model_path = sys.argv[1]
csv_path = sys.argv[2]
target_column = sys.argv[3]

# -----------------------------
# CARGAR METADATOS
# -----------------------------

metadata_path = os.path.join(model_path, "model_metadata.json")

if not os.path.exists(metadata_path):
    raise FileNotFoundError(f"❌ No se encontró el archivo de metadatos en: {metadata_path}")

with open(metadata_path, "r") as f:
    metadata = json.load(f)

FEATURES = metadata["features_used"]
LOOKBACK = metadata["lookback"]
scaler_path = os.path.join(model_path, metadata["scaler_path"])
model_file = os.path.join(model_path, metadata["model_path"])

print(f"📁 Modelo: {model_file}")
print(f"📈 Columnas usadas: {FEATURES}")
print(f"🔁 Lookback: {LOOKBACK}")

# -----------------------------
# CARGAR DATOS
# -----------------------------

df = pd.read_csv(csv_path)

if not all(col in df.columns for col in FEATURES):
    raise ValueError(f"❌ Las columnas {FEATURES} no están completas en el CSV")

df = df.loc[:, ~df.columns.duplicated()]  # por si acaso

# -----------------------------
# CARGAR SCALER
# -----------------------------

scaler = joblib.load(scaler_path)

if scaler.n_features_in_ != len(FEATURES):
    raise ValueError(f"❌ ERROR: El scaler fue entrenado con {scaler.n_features_in_} columnas, pero recibiste {len(FEATURES)}: {FEATURES}")

X_input = scaler.transform(df[FEATURES].values)

# -----------------------------
# CREAR VENTANAS
# -----------------------------

X = make_windows(X_input, LOOKBACK)
print(f"🧩 Shape final de entrada: {X.shape}")

# -----------------------------
# CARGAR MODELO Y PREDECIR
# -----------------------------

model = load_model(model_file)
predictions = model.predict(X)

# -----------------------------
# RESULTADOS
# -----------------------------

print(f"✅ Predicciones generadas: {predictions.shape}")
print("🔍 Primeras 5 predicciones:")
print(predictions[:5].flatten())
