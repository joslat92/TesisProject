# -*- coding: utf-8 -*-
"""
Construye la Tabla 7.1:
  • Recorre todos los   models/*/predictions_all_folds.csv
  • Detecta las columnas de verdad / predicción (y_true, actual-forecast, etc.)
  • Calcula MAE y RMSE sobre las 5 536 observaciones de CV
  • Guarda la tabla ordenada por RMSE  →  tables/table_7_1_metrics.csv
         y un JSON individual por modelo  →  tables/<model>.json
"""
import glob, os, json, pathlib, sys
import pandas as pd, numpy as np

pathlib.Path("tables").mkdir(exist_ok=True)

CANDIDATES = [("y_true","y_pred"), ("actual","forecast"),
              ("y","yhat"), ("obs","pred")]

records, skipped = [], []

for csv in glob.glob("models/*/predictions_all_folds.csv"):
    model = os.path.basename(os.path.dirname(csv))
    df    = pd.read_csv(csv)

    col_y = col_p = None
    for a,b in CANDIDATES:
        if a in df.columns and b in df.columns:
            col_y, col_p = a,b; break
    if col_y is None:
        skipped.append(model); continue

    diff  = df[col_y] - df[col_p]
    mae   = np.mean(np.abs(diff))
    rmse  = np.sqrt(np.mean(diff**2))
    rec   = {"model": model,
             "MAE": round(mae,4),
             "RMSE": round(rmse,4)}
    records.append(rec)
    with open(f"tables/{model}.json","w") as fp:
        json.dump(rec, fp, indent=2)

summary = (pd.DataFrame(records)
             .sort_values("RMSE")
             .reset_index(drop=True))
summary.to_csv("tables/table_7_1_metrics.csv", index=False)

print("\nTabla 7.1 – Métricas globales de CV (ordenadas por RMSE)")
print(summary.to_string(index=False))
if skipped:
    print("\n⚠  Saltados por encabezados no estándar:",", ".join(skipped))
