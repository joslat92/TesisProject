"""
Train ARIMA baseline in a rolling-window backtest
-------------------------------------------------
Usage (from project root):
    python src/train_arima.py --train_len 1500 --test_len 20 --stride 20 \
                              --target diff --out_dir models/ARIMA
"""
from pmdarima.model_selection import train_test_split
from pmdarima.arima import ARIMA
import joblib
import argparse, os, json
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from utils_backtest import rolling_splits   # (lo creaste en fase 3)

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--csv",       default="data/df_final_ready_plus_vix.csv")
parser.add_argument("--train_len", type=int, default=1500)
parser.add_argument("--test_len",  type=int, default=20)
parser.add_argument("--stride",    type=int, default=20)
parser.add_argument("--target",    choices=["diff", "logret"], default="diff")
parser.add_argument("--out_dir",   default="models/ARIMA")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ---------- Load & prep ----------
df = pd.read_csv(args.csv, parse_dates=["Date"]).set_index("Date")
price = df["Target_Price"]

if args.target == "diff":
    y = price.diff().dropna()
else:  # log-ret
    y = np.log(price).diff().dropna()

# Align features to target index (no exógenas en este baseline)
y = y.to_frame("y")

# ---------- Rolling training ----------
fold_preds = []
for fold, (train_df, test_df) in enumerate(
        rolling_splits(y, args.train_len, args.test_len, args.stride), 1):
    
    y_train, y_test = train_df["y"], test_df["y"]

    model = auto_arima(
        y_train,
        start_p=0, start_q=0, max_p=5, max_q=5,
        d=None,           # determina d automáticamente
        seasonal=False,   # SARIMA=True en el script siguiente
        error_action="ignore", suppress_warnings=True,
        stepwise=True, n_fits=20,   # rápido pero razonable
        trace=False
    )

    # Guardar resumen del modelo (opcional)
    summary_path = os.path.join(args.out_dir, f"fold_{fold:02d}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(model.summary().as_text())

    # Predicciones
    preds = model.predict(n_periods=len(y_test))
    fold_df = pd.DataFrame({
        "Date": y_test.index,
        "y_true": y_test.values,
        "y_pred": preds,
        "fold": fold,
        "model": "ARIMA"
    })
    fold_preds.append(fold_df)

    # Persistir modelo (pkl) si quieres re-usar
    model_path = os.path.join(args.out_dir, f"fold_{fold:02d}.pkl")
    joblib.dump(model, model_path)

# ---------- Aggregate & export ----------
preds_all = pd.concat(fold_preds, ignore_index=True)
preds_all.to_csv(os.path.join(args.out_dir, "predictions_all_folds.csv"), index=False)

# Extra: guardar parámetros de ejecución para reproducibilidad
with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
    json.dump(vars(args), f, indent=2)

print("✅ ARIMA rolling-backtest completado y resultados guardados")
