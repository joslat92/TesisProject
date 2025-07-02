import json
import os

# Ruta al modelo
model_path = "models/LSTM_PLAIN_TUNED"

# Diccionario de metadatos
metadata = {
    "features_used": ["Target_Price"],        # <- Muy importante: columnas que usaste para entrenar
    "lookback": 40,                           # <- Número de pasos atrás en la ventana temporal
    "target_column": "Target_Price",          # <- Variable objetivo
    "scaler_path": "scaler.save",             # <- Nombre del archivo del scaler
    "model_path": "model.keras"               # <- Nombre del archivo del modelo
}

# Guardar como JSON
metadata_path = os.path.join(model_path, "model_metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"✅ Metadata guardada en: {metadata_path}")
