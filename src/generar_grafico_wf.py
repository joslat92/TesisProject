import pandas as pd
import matplotlib.pyplot as plt
import pathlib

print("--- Iniciando generación de gráfico walk-forward ---")

csv_path = pathlib.Path("outputs/robust_WF.csv")

if not csv_path.exists():
    print(f" Error: El archivo '{csv_path}' no fue encontrado. Asegúrate de que la prueba de robustez se ejecutó correctamente.")
else:
    try:
        df = pd.read_csv(csv_path, parse_dates=["Date"])

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(df["Date"], df["y_true"], label="Real", linewidth=2, color="black")
        ax.plot(df["Date"], df["y_pred"], label="Predicho (LSTM sin VIX)", alpha=0.7, linestyle='--')

        ax.set_title("Prueba de Robustez Walk-Forward (2024-2025) – LSTM sin VIX")
        ax.set_ylabel("Precio NASDAQ-100 (USD)")
        ax.set_xlabel("Fecha")
        ax.legend()
        ax.grid(alpha=0.3)

        out_path = pathlib.Path("figs/wf_no_vix.png")
        out_path.parent.mkdir(exist_ok=True, parents=True)

        fig.tight_layout()
        fig.savefig(out_path, dpi=300)

        print(f" Figura guardada exitosamente en: {out_path}")
    except Exception as e:
        print(f" Ocurrió un error al generar el gráfico: {e}")
