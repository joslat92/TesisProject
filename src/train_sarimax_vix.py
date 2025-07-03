# -*- coding: utf-8 -*-
"""
Entrena SARIMAX con VIX como variable exógena en validación walk-forward
Salida:
    models/SARIMAX_VIX/
        ├── fold_##.pkl
        ├── fold_##_summary.txt
        ├── predictions_all_folds.csv
        ├── metrics.json
        └── run_config.json
"""
import warnings, json, joblib, pandas as pd
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np

# ---------------- CONFIG -----------------
DATA_PATH      = "data/df_final_ready_plus_vix.csv"
TARGET_COL     = "Target_Price"
EXOG_COLS      = ["VIX_Close"]          #  único exógeno
TRAIN_RATIO    = 0.80                   # resto se usa en walk-forward
FREQ           = 5                      # tamaño fold ~ 1 semana
ORDERS         = (1,1,1)                # p,d,q    igual que SARIMA
SEAS_ORDERS    = (0,1,1,52)             # P,D,Q,s  anualidad semanal
OUT_DIR        = Path("models/SARIMAX_VIX")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- LOAD DATA --------------
df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).set_index("Date")
y_full   = df[TARGET_COL]
X_full   = df[EXOG_COLS]

split = int(len(df) * TRAIN_RATIO)
y_train, y_test = y_full.iloc[:split], y_full.iloc[split:]
X_train, X_test = X_full.iloc[:split], X_full.iloc[split:]

pred_records = []
rmse_folds   = []

start, end = 0, len(y_test) // FREQ
for fold in range(1, end + 1):
    # rango de entrenamiento hasta el momento t
    y_tr  = y_full.iloc[:split + fold * FREQ]
    X_tr  = X_full.iloc[:split + fold * FREQ]
    y_val = y_full.iloc[split + (fold-1)*FREQ : split + fold*FREQ]
    X_val = X_full.iloc[split + (fold-1)*FREQ : split + fold*FREQ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(y_tr, exog=X_tr, order=ORDERS,
                        seasonal_order=SEAS_ORDERS,
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit(disp=False)

    preds = model.predict(start=y_val.index[0],
                          end=y_val.index[-1],
                          exog=X_val)

    rmse = mean_squared_error(y_val, preds, squared=False)
    rmse_folds.append(rmse)

    # --- persist fold ---
    joblib.dump(model, OUT_DIR / f"fold_{fold:02d}.pkl")
    with open(OUT_DIR / f"fold_{fold:02d}_summary.txt", "w") as f:
        f.write(model.summary().as_text())

    pred_records.append(
        pd.DataFrame({
            "Date": y_val.index,
            "y_true": y_val.values,
            "y_pred": preds.values
        })
    )
    print(f"Fold {fold:02d}  RMSE = {rmse:.5f}")

# ---------------- SAVE AGGREGATE ---------
all_preds = pd.concat(pred_records, ignore_index=True)
all_preds.to_csv(OUT_DIR / "predictions_all_folds.csv", index=False)

metrics = {"RMSE_mean": float(np.mean(rmse_folds)),
           "RMSE_std":  float(np.std(rmse_folds))}
json.dump(metrics, open(OUT_DIR / "metrics.json", "w"), indent=2)

json.dump({"order": ORDERS,
           "seasonal_order": SEAS_ORDERS,
           "exog": EXOG_COLS},
          open(OUT_DIR / "run_config.json", "w"), indent=2)

print("  Listo – artefactos en", OUT_DIR)
