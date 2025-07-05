# -*- coding: utf-8 -*-
"""
Crea un modelo híbrido.

Toma las predicciones del mejor modelo LSTM y modela los residuos
con un modelo AR(p) en un esquema de validación walk-forward.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from tqdm import tqdm

# --- Configuración ---
# Carga las predicciones del mejor modelo actual
BEST_MODEL_PREDS = Path("models/LSTM_VIX_TUNED/predictions_all_folds.csv")
OUT_DIR = Path("models/LSTM_HYBRID")
OUT_DIR.mkdir(exist_ok=True, parents=True)

MIN_TRAIN_SAMPLES = 100  # Mínimo historial de errores para entrenar el primer AR
MAX_AR_ORDER = 15      # Máximo orden 'p' a probar para el modelo AR

# --- Carga y Preparación de Datos ---
print(f"Cargando predicciones base de {BEST_MODEL_PREDS}...")
df = pd.read_csv(BEST_MODEL_PREDS, parse_dates=["Date"])

# Calcula los residuos del modelo base
df['resid'] = df['y_true'] - df['y_pred']
residuals = df['resid']

hybrid_predictions = []
print(f"Iniciando entrenamiento walk-forward del modelo AR sobre {len(df) - MIN_TRAIN_SAMPLES} pasos...")

# --- Bucle Walk-Forward ---
# tqdm es para tener una barra de progreso, ya que esto puede tardar un poco
for i in tqdm(range(MIN_TRAIN_SAMPLES, len(df))):
    # Historial de errores hasta el día anterior
    train_residuals = residuals.iloc[:i]

    # Seleccionar el mejor orden 'p' para el modelo AR basado en AIC
    selector = ar_select_order(train_residuals, maxlag=MAX_AR_ORDER, glob=True)
    best_p = selector.ar_lags[-1] if selector.ar_lags else 0

    # Entrenar el modelo AR con el mejor orden 'p'
    if best_p > 0:
        model = AutoReg(train_residuals, lags=best_p).fit()
        # Predecir el error del siguiente día
        pred_residual = model.predict(start=len(train_residuals), end=len(train_residuals)).iloc[0]
    else:
        # Si no se selecciona ningún lag, la mejor predicción del error es 0
        pred_residual = 0.0

    # La predicción híbrida es la predicción original del LSTM más la predicción del error
    final_pred = df['y_pred'].iloc[i] + pred_residual
    hybrid_predictions.append(final_pred)

# --- Guardado de Resultados ---
# Crea un DataFrame con los resultados del modelo híbrido
results_df = pd.DataFrame({
    'Date': df['Date'].iloc[MIN_TRAIN_SAMPLES:],
    'y_true': df['y_true'].iloc[MIN_TRAIN_SAMPLES:],
    'y_pred': hybrid_predictions
})

# Guardar el CSV global de predicciones
output_csv_path = OUT_DIR / "predictions_all_folds.csv"
results_df.to_csv(output_csv_path, index=False)
print(f"✅ Predicciones del modelo híbrido guardadas en {output_csv_path}")

# Calcular y guardar métricas
rmse = np.sqrt(np.mean((results_df['y_true'] - results_df['y_pred'])**2))
metrics = {"RMSE": rmse}
output_metrics_path = OUT_DIR / "metrics.json"
with open(output_metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"✅ Métricas guardadas en {output_metrics_path} | RMSE = {rmse:.6f}")
