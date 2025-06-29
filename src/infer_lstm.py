# -*- coding: utf-8 -*-
"""
Genera un CSV de predicciones para un modelo LSTM entrenado
Uso:
    python src/infer_lstm.py models/LSTM_Plain  data/df_final_ready_plus_vix.csv  Target_Price 40
    python src/infer_lstm.py models/LSTM_VIX    data/df_final_ready_plus_vix.csv  "Target_Price,VIX_Close" 40
"""
import sys, os, json, joblib
import numpy as np, pandas as pd
from tensorflow.keras.models import load_model

# --------------------- CLI ----------------------------
model_dir, data_path, cols_arg, lookback = sys.argv[1:]
cols      = cols_arg.split(",")
lookback  = int(lookback)

out_csv   = os.path.join(model_dir, "predictions_all_folds.csv")
run_cfg   = os.path.join(model_dir, "run_config")

# --------------------- Datos --------------------------
df = pd.read_csv(data_path, parse_dates=["Date"]).set_index("Date")[cols]
scaler = joblib.load(os.path.join(model_dir, "scaler.save"))
df_scaled = pd.DataFrame(
    scaler.transform(df.values), index=df.index, columns=cols
)

# -> usamos el mismo split que en el entrenamiento (último 20 %)
split = int(len(df_scaled) * 0.8)
test_df = df_scaled.iloc[split-lookback:]

def make_windows(arr, look):
    X, ix = [], []
    for i in range(len(arr) - look):
        X.append(arr[i:i+look])
        ix.append(i+look)
    return np.array(X), ix

X_test, idx_pos = make_windows(test_df.values, lookback)
model  = load_model(os.path.join(model_dir, "model.keras"))
pred_y = model.predict(X_test, verbose=0).flatten()

# Des-escalado (solo primera columna del scaler)
dummy = np.zeros((len(pred_y), len(cols)))
dummy[:,0] = pred_y
pred_inv  = scaler.inverse_transform(dummy)[:,0]

truth_scaled = test_df.values[lookback:, 0]
dummy[:,0]   = truth_scaled
truth_inv = scaler.inverse_transform(dummy)[:,0]

dates = test_df.index[lookback:]

pd.DataFrame({
    "Date":  dates,
    "Truth": truth_inv,
    "Pred":  pred_inv
}).to_csv(out_csv, index=False)

with open(run_cfg, "w") as fp:
    json.dump({"lookback": lookback, "cols": cols}, fp, indent=2)

print(f"  Predicciones guardadas en {out_csv}")
