#!/usr/bin/env python
"""
Unifica las predicciones walk‑forward de Hybrid y No‑VIX
y crea appendices/data/robust_WF.csv (327 filas).
"""
import pandas as pd
from pathlib import Path

# ---------- Rutas ----------

HYB_PATH = "models/LSTM_HYBRID/preds.csv"
NVX_PATH = "models/LSTM_NO_VIX_TUNED/preds.csv"

OUT_PATH = Path("appendices/data/robust_WF.csv")
START, END = "2024-01-01", "2025-05-31"   # fechas del período

# ---------- Cargar ----------
hyb = pd.read_csv(HYB_PATH, parse_dates=["Date"])
nvx = pd.read_csv(NVX_PATH, parse_dates=["Date"])

# Normalizar nombre de columna predicción
for cand in ["y_pred", "y_y_pred", "yhat"]:
    if cand in hyb.columns: hyb.rename(columns={cand: "y_pred_hybrid"}, inplace=True)
    if cand in nvx.columns: nvx.rename(columns={cand: "y_pred_no_vix"}, inplace=True)

# Filtrar rango de fechas
hyb = hyb[(hyb["Date"] >= START) & (hyb["Date"] <= END)]
nvx = nvx[(nvx["Date"] >= START) & (nvx["Date"] <= END)]

# ---------- Unir ----------
merged = (
    hyb[["Date", "y_true", "y_pred_hybrid"]]
    .merge(nvx[["Date", "y_pred_no_vix"]], on="Date", how="inner")
    .sort_values("Date")
)

assert len(merged) == 327, f"Esperaba 327 filas y obtuve {len(merged)}"

# ---------- Guardar ----------
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUT_PATH, index=False)
print(f"Archivo creado: {OUT_PATH}  ({len(merged)} filas)")
