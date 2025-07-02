# -*- coding: utf-8 -*-

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
# Configuración general
# ----------------------------------------------------------------------
DATA_PATH   = "data/df_final_ready_plus_vix.csv"
TARGET_COL  = "Target_Price"
COLS        = ["Target_Price", "VIX_Close"] 
LOOKBACK    = 40
TEST_RATIO  = 0.20
VAL_RATIO   = 0.10               # % de TRAIN que va a validación
EARLY_STOP  = 10                 # paciencia en epochs
OUT_DIR     = Path("models/LSTM_PLAIN_TUNED")
LOG_PATH    = Path("outputs/tuning_log.csv")

OUT_DIR     = Path("models/LSTM_VIX_TUNED")      
LOG_PATH    = Path("outputs/tuning_log_vix.csv") 

# ----------------------------------------------------------------------
# Carga y preparación de datos
# ----------------------------------------------------------------------
df_series = (
    pd.read_csv(DATA_PATH, parse_dates=["Date"])
      .set_index("Date")[[TARGET_COL]]
      .astype(float)
)

scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(df_series.values)

def make_windows(arr: np.ndarray, lookback: int):
    """Convierte una serie (n,1) en pares X,y con look-back."""
    X, y = [], []
    for i in range(len(arr) - lookback):
        X.append(arr[i : i + lookback])
        y.append(arr[i + lookback, 0])
    return np.array(X), np.array(y)

X, y = make_windows(y_scaled, LOOKBACK)

# Split train / test
split = int(len(X) * (1 - TEST_RATIO))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ----------------------------------------------------------------------
# Grid de hiperparámetros
# ----------------------------------------------------------------------
param_grid = {
    "units"        : [32, 64],
    "n_layers"     : [1, 2],
    "dropout"      : [0.0, 0.2],
    "learning_rate": [1e-3, 5e-4],
    "epochs"       : [80],          # se puede ampliar
    "batch_size"   : [32]
}

grid_iter = list(itertools.product(*param_grid.values()))
param_names = list(param_grid.keys())

# Registro de resultados
with open(LOG_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["units", "n_layers", "dropout", "lr", "epochs", "batch",
         "val_RMSE", "test_RMSE"]
    )

best_rmse = np.inf
best_model = None
best_params = None

# ----------------------------------------------------------------------
# Funciones auxiliares
# ----------------------------------------------------------------------
def build_model(units: int, n_layers: int, dropout: float, lr: float):
    model = Sequential()
    for i in range(n_layers):
        return_sequences = i < n_layers - 1
        model.add(
            LSTM(
                units,
                return_sequences=return_sequences,
                input_shape=(LOOKBACK, 1) if i == 0 else None
            )
        )
        if dropout > 0:
            model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer=Adam(lr))
    return model


def calc_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ----------------------------------------------------------------------
# Bucle de experimentos
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

    # Split train -> train/val
    val_split_idx = int(len(X_train) * (1 - VAL_RATIO))
    X_tr, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
    y_tr, y_val = y_train[:val_split_idx], y_train[val_split_idx:]

    early = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOP,
        restore_best_weights=True,
        verbose=0
    )

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        verbose=0,
        callbacks=[early]
    )

    # --------- métrica validación / test ----------
    val_pred  = model.predict(X_val,  verbose=0)
    test_pred = model.predict(X_test, verbose=0)

    val_rmse  = calc_rmse(y_val,  val_pred)
    test_rmse = calc_rmse(y_test, test_pred)

    print(f"    Val RMSE = {val_rmse:.6f} | Test RMSE = {test_rmse:.6f}")

    # log
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            params["units"], params["n_layers"], params["dropout"],
            params["learning_rate"], params["epochs"], params["batch_size"],
            val_rmse, test_rmse
        ])

    # guarda el mejor
    if val_rmse < best_rmse:
        best_rmse = val_rmse
        best_params = params
        best_model = model
        print("    🆕  Nuevo mejor modelo")

# ----------------------------------------------------------------------
# Persistencia del mejor modelo
# ----------------------------------------------------------------------
if best_model is None:
    raise RuntimeError("No se entrenó ningún modelo :/")

best_model.save(OUT_DIR / "model.keras")
import joblib
joblib.dump(scaler, OUT_DIR / "scaler.save")

with open(OUT_DIR / "metrics.json", "w") as fp:
    json.dump({"RMSE_val": float(best_rmse)}, fp, indent=2)

with open(OUT_DIR / "best_params.json", "w") as fp:
    json.dump(best_params, fp, indent=2)

print("\n🚀  FIN TUNING")
print(f"   • Mejor Val RMSE = {best_rmse:.6f}")
print(f"   • Hiperparámetros = {best_params}")
print(f"   • Artefactos en {OUT_DIR}")
print(f"   • Log completo en {LOG_PATH}")
