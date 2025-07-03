# -*- coding: utf-8 -*-
"""
Recorre todos los modelos en models/, lee preds.csv,
compara contra el valor real y calcula MAE, RMSE, MASE.
Resultados → outputs/metrics_summary.csv
"""
import os, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = "data/df_final_ready_plus_vix.csv"   # contiene la serie Target_Price
TARGET_COL = "Target_Price"
MODELS_DIR = Path("models")
OUT_PATH   = Path("outputs/metrics_summary.csv")
OUT_PATH.parent.mkdir(exist_ok=True)

# --- carga serie real ---
df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).set_index("Date")
y_true_full = df[TARGET_COL]

records = []
for model_dir in MODELS_DIR.iterdir():
    if not model_dir.is_dir():
        continue
    preds_file = model_dir / "preds.csv"
    if not preds_file.exists():
        print(f"⚠️  {preds_file} no encontrado, salto…")
        continue

    preds = pd.read_csv(preds_file, parse_dates=["Date"]).set_index("Date")["y_pred"]
    # Alinea para evitar días faltantes
    common_idx = y_true_full.index.intersection(preds.index)
    y_true = y_true_full.loc[common_idx]
    y_pred = preds.loc[common_idx]

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mase = mae / mean_absolute_error(y_true, y_true.shift(1).dropna())  # escala naïve

    # lee RMSE guardado por el modelo si existe
    metrics_json = model_dir / "metrics.json"
    saved_rmse = json.load(open(metrics_json))["RMSE"] if metrics_json.exists() else None

    records.append({
        "model": model_dir.name,
        "MAE": mae,
        "RMSE": rmse,
        "RMSE_saved": saved_rmse,
        "MASE": mase,
        "n_obs": len(common_idx)
    })

summary = pd.DataFrame(records).sort_values("RMSE")
summary.to_csv(OUT_PATH, index=False)
print("✅  Métricas guardadas en", OUT_PATH)
print(summary)
