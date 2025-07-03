# -*- coding: utf-8 -*-
"""
Auto-tuning de un LSTM con 2 entradas: precio objetivo + VIX
Guarda:
    • model.keras
    • scaler.save
    • metrics.json      (RMSE validación)
    • best_params.json  (hiperparámetros)
    • outputs/tuning_log_vix.csv   (historial completo)
"""

import os, json, itertools, csv
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ----------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------
DATA_PATH   = "data/df_final_ready_plus_vix.csv"
TARGET_COL  = "Target_Price"
COLS        = ["Target_Price", "VIX_Close"]     # <-- 2 features
LOOKBACK    = 40
TEST_RATIO  = 0.20
VAL_RATIO   = 0.10
EARLY_STOP  = 10

OUT_DIR = Path("models/LSTM_VIX_TUNED")
LOG_PATH = Path("outputs/tuning_log_vix.csv")

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------
df = (
    pd.read_csv(DATA_PATH, parse_dates=["Date"])
      .set_index("Date")[COLS]
      .astype(float)
      .dropna()
)

scaler = MinMaxScaler().fit(df.values)
df_scaled = pd.DataFrame(
    scaler.transform(df.values),
    index=df.index,
    columns=COLS
)

def make_windows(arr: np.ndarray, lookback: int):
    """Devuelve X (n-lookback, lookback, n_feat) y y (n-lookback,)"""
    X, y = [], []
    for i in range(len(arr) - lookback):
        X.append(arr[i:i+lookback])
        y.append(arr[i+lookback, 0])       # la 1ª columna es el target
    return np.array(X), np.array(y)

X_all, y_all = make_windows(df_scaled.values, LOOKBACK)

# Train / Test split
split = int(len(X_all) * (1 - TEST_RATIO))
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]

# ----------------------------------------------------------------------
# Grid de hiperparámetros
# ----------------------------------------------------------------------
param_grid = {
    "units"        : [32, 64],
    "n_layers"     : [1, 2],
    "dropout"      : [0.0, 0.2],
    "learning_rate": [1e-3, 5e-4],
    "epochs"       : [80],
    "batch_size"   : [32],
}
grid_iter   = list(itertools.product(*param_grid.values()))
param_names = list(param_grid.keys())

# Crear header de log
with open(LOG_PATH, "w", newline="") as f:
    csv.writer(f).writerow(
        list(param_grid.keys()) + ["val_RMSE", "test_RMSE"]
    )

best_rmse   = np.inf
best_model  = None
best_params = None

# ----------------------------------------------------------------------
# Auxiliares
# ----------------------------------------------------------------------
def build_model(units: int, n_layers: int, dropout: float, lr: float):
    model = Sequential()
    for i in range(n_layers):
        return_seq = i < n_layers - 1
        model.add(
            LSTM(
                units,
                return_sequences=return_seq,
                input_shape=(LOOKBACK, len(COLS)) if i == 0 else None
            )
        )
        if dropout > 0:
            model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer=Adam(lr))
    return model

rmse = lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))

# ----------------------------------------------------------------------
# Bucle de tuning
# ----------------------------------------------------------------------
for combo in grid_iter:
    params = dict(zip(param_names, combo))
    print("--->", params)

    model = build_model(
        units=params["units"],
        n_layers=params["n_layers"],
        dropout=params["dropout"],
        lr=params["learning_rate"]
    )

    # Split interno de validación
    val_split = int(len(X_train) * (1 - VAL_RATIO))
    X_tr, X_val = X_train[:val_split], X_train[val_split:]
    y_tr, y_val = y_train[:val_split], y_train[val_split:]

    early = EarlyStopping(
        monitor="val_loss", patience=EARLY_STOP,
        restore_best_weights=True, verbose=0
    )

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        verbose=0,
        callbacks=[early]
    )

    val_rmse  = rmse(y_val,  model.predict(X_val,  verbose=0))
    test_rmse = rmse(y_test, model.predict(X_test, verbose=0))

    print(f"    Val RMSE = {val_rmse:.6f} | Test RMSE = {test_rmse:.6f}")

    # log
    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            params["units"], params["n_layers"], params["dropout"],
            params["learning_rate"], params["epochs"], params["batch_size"],
            val_rmse, test_rmse
        ])

    # mejor modelo
    if val_rmse < best_rmse:
        best_rmse, best_model, best_params = val_rmse, model, params
        print("    🆕  Nuevo mejor modelo")

# ----------------------------------------------------------------------
# Guardado final
# ----------------------------------------------------------------------
if best_model is None:
    raise RuntimeError("No se entrenó ningún modelo :/")

# modelo y scaler
best_model.save(OUT_DIR / "model.keras")
import joblib; joblib.dump(scaler, OUT_DIR / "scaler.save")

# métricas + params
with open(OUT_DIR / "metrics.json", "w") as fp:
    json.dump({"RMSE_val": float(best_rmse)}, fp, indent=2)
with open(OUT_DIR / "best_params.json", "w") as fp:
    json.dump(best_params, fp, indent=2)

# metadata para inferencia posterior
with open(OUT_DIR / "model_metadata.json", "w") as fp:
    json.dump({
        "features_used": COLS,
        "target": TARGET_COL,
        "lookback": LOOKBACK,
        "scaler_path": "scaler.save",
        "model_path": "model.keras"
    }, fp, indent=2)

print("\n🚀  FIN TUNING")
print(f"   • Mejor Val RMSE = {best_rmse:.6f}")
print(f"   • Hiperparámetros = {best_params}")
print(f"   • Artefactos en {OUT_DIR}")
print(f"   • Log completo en {LOG_PATH}")
