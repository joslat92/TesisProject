"""
Toma un modelo LSTM existente, calcula sus residuos, entrena un AR(p)
corto y guarda ambos caminos.
"""
import argparse, joblib, json, numpy as np, pandas as pd
from pathlib import Path
import statsmodels.api as sm

parser = argparse.ArgumentParser()
parser.add_argument("base_model_dir", help="models/LSTM_PLAIN_TUNED")
parser.add_argument("--output_dir", default="models/LSTM_AR_HYB")
args = parser.parse_args()

BASE_DIR = Path(args.base_model_dir)
OUT_DIR  = Path(args.output_dir); OUT_DIR.mkdir(exist_ok=True)

# --- 1. Cargar predicciones existentes ---
preds_path = BASE_DIR / "predictions_all_folds.csv"
df = pd.read_csv(preds_path, parse_dates=["Date"]).set_index("Date")
resid = df["y_true"] - df["y_pred"]

# --- 2. Seleccionar orden AR según AIC (p=1..3) ---
best_aic, best_p = 1e9, None
for p in range(1, 4):
    model = sm.tsa.ARIMA(resid, order=(p,0,0)).fit()
    if model.aic < best_aic:
        best_aic, best_p, best_model = model.aic, p, model

print(f"🔧 Mejor AR({best_p})  AIC={best_aic:.2f}")
joblib.dump(best_model, OUT_DIR / "ar_model.pkl")

# --- 3. Predicción in-sample (para guardar csv) ---
resid_pred = best_model.predict()
df["y_pred_hybrid"] = df["y_pred"] + resid_pred

df_hyb = df.reset_index()[["Date","y_true","y_pred_hybrid"]]
df_hyb.columns = ["Date","y_true","y_pred"]
df_hyb.to_csv(OUT_DIR / "predictions_all_folds.csv", index=False)

# --- 4. Métricas rápidas ---
rmse = np.sqrt(((df_hyb.y_true - df_hyb.y_pred)**2).mean())
json.dump({"RMSE": float(rmse), "AR_order": best_p},
          open(OUT_DIR/"metrics.json","w"), indent=2)

json.dump({"base_model": str(BASE_DIR),
           "AR_order": best_p},
          open(OUT_DIR/"run_config.json","w"), indent=2)

print("✅  Híbrido guardado en", OUT_DIR, " RMSE=", rmse)
