# src/visualize_results.py
# ------------------------------------------------------------
"""
Genera los plots FINALES para la tesis.
- Carga la verdad desde el dataset maestro para garantizar la escala correcta.
- Es flexible con los nombres de las columnas de los archivos de predicciones.
- Filtra los gráficos para mostrar comparaciones justas y claras.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- CONFIGURACIÓN ---
MASTER_DATA_PATH = Path("data/df_final_ready_plus_vix.csv")

def load_all_predictions(models_dir: Path) -> pd.DataFrame:
    """Carga y consolida predicciones, usando el dataset maestro como fuente de verdad."""
    if not MASTER_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset maestro no encontrado en: {MASTER_DATA_PATH}")

    # 1. Cargar la serie 'Real' (precio) desde la ÚNICA fuente de verdad.
    truth_df = pd.read_csv(MASTER_DATA_PATH, usecols=["Date", "Target_Price"], parse_dates=["Date"])
    truth_df = truth_df.rename(columns={"Target_Price": "y_true_price"})
    truth_df = truth_df.set_index("Date")
    
    # 2. Cargar y normalizar las predicciones de cada modelo.
    all_rows = []
    print("--- Cargando y consolidando todas las predicciones ---")
    for csv_path in models_dir.glob("*/predictions_all_folds.csv"):
        model_name = csv_path.parent.name
        try:
            df_pred = pd.read_csv(csv_path, parse_dates=["Date"])
            # --- Lógica flexible para nombres de columnas ---
            cols = {c.lower(): c for c in df_pred.columns}
            y_pred_col = cols.get("y_pred") or cols.get("pred")
            
            if not y_pred_col:
                print(f"⚠️  Ignorado '{model_name}': no se encontró la columna de predicción ('y_pred' o 'Pred').")
                continue

            df_pred = df_pred.rename(columns={y_pred_col: "y_pred"})
            df_pred["model"] = model_name
            all_rows.append(df_pred[["Date", "y_pred", "model"]])
            print(f"✓ Cargado: {model_name}")

        except Exception as e:
            print(f"⚠️  Error al procesar '{csv_path}': {e}")
            continue
    
    if not all_rows:
        raise RuntimeError("No se encontraron predicciones válidas.")

    # 3. Unir todo en un DataFrame final.
    full_pred_df = pd.concat(all_rows, ignore_index=True)
    final_df = pd.merge(full_pred_df, truth_df.reset_index(), on="Date", how="inner")
    
    # Renombrar 'y_true_price' a 'y_true' para consistencia.
    final_df = final_df.rename(columns={"y_true_price": "y_true"})
    
    return final_df

def plot_time_series_and_boxplot(df: pd.DataFrame, window: int, out_dir: Path):
    """Genera ambos gráficos, asegurando que solo se comparan modelos en la misma escala."""
    
    # Heurística para identificar modelos que predicen precio vs. retornos
    main_scale_mean = df["y_true"].mean()
    price_models = [name for name, grp in df.groupby("model") if grp["y_pred"].mean() > main_scale_mean * 0.5]
    
    print(f"\nModelos identificados en escala de PRECIO: {price_models}")
    df_price = df[df['model'].isin(price_models)].copy()

    # --- Gráfico de Series de Tiempo ---
    last_dates = df_price["Date"].sort_values(ascending=False).unique()[:window]
    df_win = df_price[df_price["Date"].isin(last_dates)]

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    truth = df_win.groupby("Date")["y_true"].first().sort_index()
    plt.plot(truth.index, truth.values, label="Valor Real", linewidth=2.5, color='black', zorder=10)
    
    for model_name, grp in df_win.groupby("model"):
        plt.plot(grp["Date"], grp["y_pred"], label=model_name, linewidth=1.0, alpha=0.9)

    plt.title(f"Predicciones de Precio – Últimos {window} Días", fontsize=16)
    plt.xlabel("Fecha", fontsize=12); plt.ylabel("Precio NASDAQ-100", fontsize=12)
    plt.legend(loc="upper left"); plt.tight_layout()
    plt.savefig(out_dir / "time_series_final.png", dpi=300)
    plt.close()
    print(f"✅ Gráfico de series de tiempo final guardado en '{out_dir / 'time_series_final.png'}'")

    # --- Gráfico de Cajas de Error ---
    df_price["abs_err"] = np.abs(df_price["y_true"] - df_price["y_pred"])
    median_errors = df_price.groupby("model")["abs_err"].median().sort_values()
    sorted_labels = median_errors.index
    sorted_data = [df_price[df_price["model"] == model]["abs_err"].dropna().values for model in sorted_labels]

    plt.figure(figsize=(10, 6))
    plt.boxplot(sorted_data, tick_labels=sorted_labels, showmeans=True)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    
    plt.ylabel("|Error Absoluto|", fontsize=12)
    plt.title("Distribución del Error Absoluto (Modelos de Precio)", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_dir / "error_boxplot_final.png", dpi=300)
    plt.close()
    print(f"✅ Gráfico de cajas de error final guardado en '{out_dir / 'error_boxplot_final.png'}'")

def main():
    parser = argparse.ArgumentParser(description="Genera gráficos de resultados para la tesis.")
    parser.add_argument("--models_dir", default="models", type=str)
    parser.add_argument("--window", default=250, type=int)
    parser.add_argument("--out", dest="out_dir", default="figs", type=str)
    args = parser.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    
    try:
        df = load_all_predictions(Path(args.models_dir))
        plot_time_series_and_boxplot(df, args.window, out_dir)
        print("\n🎉 ¡Gráficos finales generados con éxito!")
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    main()