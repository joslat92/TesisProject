# -*- coding: utf-8 -*-
# walk_forward.py – robustez single-split
import sys, json, joblib, pandas as pd, numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from utils5_estable import make_windows     # devuelve SOLO X

if len(sys.argv) != 4:
    print("Uso: python src/walk_forward.py <model_dir> <run_cfg.json> <csv_path>")
    sys.exit(1)

MODEL_DIR = Path(sys.argv[1])
CFG_PATH  = Path(sys.argv[2])
CSV_PATH  = Path(sys.argv[3])

cfg    = json.load(open(CFG_PATH, encoding="utf-8"))
target = cfg["target"]                       # 'Target_Price'
feats  = [c for c in cfg["features"] if c != target]   # ['VIX_Close']
look   = cfg["lookback"]

# -------- datos --------
df = (pd.read_csv(CSV_PATH, parse_dates=["Date"])
        .set_index("Date")
        .loc[cfg["train_start"]:cfg["test_end"], feats + [target]])

test_df = df.loc[cfg["test_start"]:cfg["test_end"]]

# -------- scaler --------
scaler = joblib.load(MODEL_DIR / "scaler.save")        # 2 features esperadas

X_test  = scaler.transform(test_df[feats + [target]])
X_test  = make_windows(X_test, look)                   # shape (n, look, 2)

# -------- modelo --------
model = load_model(MODEL_DIR / "model.keras")
y_pred_scaled = model.predict(X_test, verbose=0).flatten()

# -------- des-escalar SOLO la 1ª columna --------
last_feats = X_test[:, -1, :].copy()   # (n, 2)
last_feats[:, 0] = y_pred_scaled       # sustituye precio escalado
y_pred = scaler.inverse_transform(last_feats)[:, 0]

out = pd.DataFrame({
    "Date"  : test_df.index[look:],           # se pierden los primeros look días
    "y_true": test_df[target].values[look:],
    "y_pred": y_pred
})
out.to_csv("outputs/robust_WF.csv", index=False)
print("✅  Guardado outputs/robust_WF.csv  ->  shape:", out.shape)
