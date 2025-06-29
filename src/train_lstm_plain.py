# -*- coding: utf-8 -*-
"""
Entrena un LSTM univariado sobre Target_Price
Guardará el modelo y las métricas en ./models/
"""
import os, joblib, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# --------------------  CONFIG  -------------------------------------------
DATA_PATH   = "data/df_final_ready.csv"
COLS        = ["Target_Price"]      # solo precio
LOOKBACK    = 40
TEST_RATIO  = 0.20
BATCH_SIZE  = 32
EPOCHS      = 100
PATIENCE    = 10
OUT_DIR     = "models/LSTM_Plain"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------- 1. Dataset y escalado ---------------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).set_index("Date")[COLS]
scaler = MinMaxScaler().fit(df.values)
df_scaled = pd.DataFrame(scaler.transform(df.values), index=df.index, columns=COLS)

# split
split = int(len(df_scaled) * (1 - TEST_RATIO))
train_df, test_df = df_scaled.iloc[:split], df_scaled.iloc[split-LOOKBACK:]

def make_windows(series: np.ndarray, lookback: int = 40):
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i:i+lookback])
        y.append(series[i+lookback])
    return np.array(X), np.array(y)

X_train, y_train = make_windows(train_df.values, LOOKBACK)
X_test,  y_test  = make_windows(test_df.values,  LOOKBACK)

# Reshape para LSTM: (samples, timesteps, features)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# ----------------- 2. Modelo --------------------------------------------
model = Sequential([
    LSTM(64, input_shape=(LOOKBACK, len(COLS))),
    Dense(1)
])
model.compile(loss="mse", optimizer="adam")

cb = EarlyStopping(monitor="val_loss",
                   patience=PATIENCE,
                   restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[cb],
    verbose=2
)

# ----------------- 3. Evaluación y guardado ------------------------------
mse = model.evaluate(X_test, y_test, verbose=0)
rmse = np.sqrt(mse)
print(f"✅  Test RMSE: {rmse:.6f}")

model.save(os.path.join(OUT_DIR, "model.keras"))
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.save"))

with open(os.path.join(OUT_DIR, "metrics.json"), "w") as fp:
    json.dump({"RMSE": float(rmse)}, fp, indent=2)

print("📦  Modelo y scaler guardados en", OUT_DIR)
