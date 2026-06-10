import pandas as pd
import os, sys, yaml
sys.path.append(os.getcwd())
from src.core.dm import diebold_mariano_test

with open("config.yaml", "r") as f: config = yaml.safe_load(f)
preds_dir = config['paths']['preds_oos_dir']
df_arima = pd.read_csv(os.path.join(preds_dir, "preds_ARIMA.csv"))
df_lstm = pd.read_csv(os.path.join(preds_dir, "preds_lstm_sent_vix.csv"))

dm_stat, p_val = diebold_mariano_test(df_arima['Target_Price'], df_arima['y_pred'], df_lstm['y_pred'], h=1)
print(f"\n--- COMPARACIÓN FINAL ---")
print(f"DM Stat (ARIMA vs LSTM_SENT): {dm_stat:.4f}")
print(f"P-Value: {p_val:.4f}")
if p_val < 0.05: print("RESULTADO: Hay una diferencia estadísticamente significativa.")
else: print("RESULTADO: Empate técnico estadístico (No se puede rechazar H0).")
