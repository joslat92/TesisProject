# src/plot_acf_pacf.py
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pathlib import Path

DATA_PATH = "data/df_final_ready.csv"     # ← ajusta a tu ruta
SERIES_COL = "Target_Price"
OUT_DIR = Path("appendices/figs")          # carpeta de destino
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Leer y ordenar datos
df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).sort_values("Date")
y = df[SERIES_COL].astype(float)

# 2. ACF
plt.figure(figsize=(8,4))
plot_acf(y, lags=40, zero=False)
plt.title("ACF – Precio NASDAQ‑100 (nivel)")
plt.tight_layout()
plt.savefig(OUT_DIR/"acf_raw.png", dpi=300)
plt.close()

# 3. PACF
plt.figure(figsize=(8,4))
plot_pacf(y, lags=40, zero=False, method="ywm")
plt.title("PACF – Precio NASDAQ‑100 (nivel)")
plt.tight_layout()
plt.savefig(OUT_DIR/"pacf_raw.png", dpi=300)
plt.close()

print("Gráficos guardados en:", OUT_DIR)
