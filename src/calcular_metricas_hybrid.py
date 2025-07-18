import pandas as pd, numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

TEST_START = "2024-01-01"
TEST_END   = "2025-05-31"
PRED_PATH  = "models/LSTM_NO_VIX_TUNED/preds.csv"   # ← ruta al CSV

# 1. Cargar
preds = pd.read_csv(PRED_PATH, parse_dates=["Date"])      # Date, y_true, y_pred

# 2. Filtrar al mismo rango de fechas
mask  = (preds["Date"] >= TEST_START) & (preds["Date"] <= TEST_END)
test  = preds.loc[mask].copy()

# 3. Métricas
rmse = np.sqrt(mean_squared_error(test["y_true"], test["y_pred"]))
mae  = mean_absolute_error(test["y_true"], test["y_pred"])
true_ret = test["y_true"].diff()
pred_ret = test["y_pred"] - test["y_true"].shift(1)
da = (np.sign(true_ret[1:]) == np.sign(pred_ret[1:])).mean()

print(f"n_obs = {len(test)}")
print(f"RMSE  = {rmse:.2f}")
print(f"MAE   = {mae :.2f}")
print(f"DA    = {da  :.2%}")
