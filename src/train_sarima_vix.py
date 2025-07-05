# -*- coding: utf-8 -*-
"""
SARIMAX con VIX como exógeno – CV de 53 pliegues igual que SARIMA base
Guarda modelo, summary, predicciones y metrics.json
"""
import os, json, joblib
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from utils5_estable import expanding_cv_splits, save_fold_preds  # ya definido

DATA   = "data/df_final_ready_plus_vix.csv"
TARGET = "Target_Price"
EXOG   = "VIX_Close"
OUT    = Path("models/SARIMA_VIX")
OUT.mkdir(exist_ok=True, parents=True)

df = pd.read_csv(DATA, parse_dates=["Date"]).set_index("Date")
y  = df[TARGET]
x  = df[[EXOG]]

# mismos (p,d,q)(P,D,Q,s) descubiertos para SARIMA base
order, seasonal_order = (2,1,2), (1,0,1,12)

all_preds = []
for i, (train_idx, test_idx) in enumerate(expanding_cv_splits(y, n_folds=53)):
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    x_tr, x_te = x.iloc[train_idx], x.iloc[test_idx]

    model = sm.tsa.SARIMAX(y_tr, exog=x_tr, order=order,
                           seasonal_order=seasonal_order,
                           enforce_stationarity=False,
                           enforce_invertibility=False).fit(disp=False)

    pred = model.get_forecast(len(test_idx), exog=x_te).predicted_mean

    # --- persistencia fold ---
    model.save(OUT / f"fold_{i+1:02d}.pkl")
    with open(OUT / f"fold_{i+1:02d}_summary.txt", "w") as fp:
        fp.write(model.summary().as_text())

    all_preds.append(save_fold_preds(y_te, pred, i+1, OUT))

# -------- csv global ----------
pd.concat(all_preds).to_csv(OUT / "predictions_all_folds.csv", index=False)
json.dump({"RMSE": float(np.sqrt(((all_preds[-1]['y_true']-all_preds[-1]['y_pred'])**2).mean()))},
          open(OUT / "metrics.json", "w"), indent=2)
print(" SARIMA + VIX entrenado y guardado")
