# src/visualize_results.py
# ------------------------------------------------------------
"""
Genera plots de resultados a partir de los CSV de predicciones
que cada modelo guarda en <models/*/predictions_all_folds.csv>.

Uso:
    python src/visualize_results.py --window 250 --out figs/
"""
import argparse, itertools
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_all_predictions(models_dir: Path) -> pd.DataFrame:
    """Une todos los CSV de predicciones en un único DataFrame
       Devuelve columnas: Date, y_true, y_pred, model"""
    rows = []
    for csv in models_dir.glob("*/predictions_all_folds.csv"):
        df = pd.read_csv(csv, parse_dates=["Date"])
        # normaliza nombres de columnas
        cols = {c.lower(): c for c in df.columns}
        y_true  = cols.get("y_true")  or cols.get("truth")
        y_pred  = cols.get("y_pred")  or cols.get("pred")
        if y_true is None or y_pred is None:
            print(f"⚠️  {csv} ignorado: columnas esperadas y_true / y_pred")
            continue
        df = df.rename(columns={y_true: "y_true", y_pred: "y_pred"})
        df["model"] = csv.parent.name
        rows.append(df[["Date", "y_true", "y_pred", "model"]])
    if not rows:
        raise RuntimeError("No se encontraron predicciones válidas")
    return pd.concat(rows, ignore_index=True)

def plot_time_series(df: pd.DataFrame, window: int, out_dir: Path):
    last_dates = df["Date"].sort_values().unique()[-window:]
    df_win = df[df["Date"].isin(last_dates)]

    plt.figure()
    # curva real
    truth = (df_win.groupby("Date")["y_true"].first()
             .sort_index())
    plt.plot(truth.index, truth.values, label="Real")
    # curvas modelo
    for m, grp in df_win.groupby("model"):
        plt.plot(grp["Date"], grp["y_pred"], label=m, linewidth=0.8)
    plt.title(f"Predicciones – últimos {window} días")
    plt.xlabel("Fecha"); plt.ylabel("Precio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "time_series.png", dpi=250)
    plt.close()

def plot_error_boxplot(df: pd.DataFrame, out_dir: Path):
    df["abs_err"] = np.abs(df["y_true"] - df["y_pred"])
    data = [g["abs_err"].values for _, g in df.groupby("model")]
    labels = df["model"].unique()

    plt.figure()
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("|Error|")
    plt.title("Distribución del error absoluto por modelo")
    plt.tight_layout()
    plt.savefig(out_dir / "error_boxplot.png", dpi=250)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", default="models", type=str)
    parser.add_argument("--window", default=250, type=int,
                        help="nº de días recientes a graficar")
    parser.add_argument("--out", "--out_dir", dest="out_dir",
                        default="figs", type=str)
    args = parser.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    df = load_all_predictions(Path(args.models_dir))
    plot_time_series(df, args.window, out_dir)
    plot_error_boxplot(df, out_dir)
    print("  Gráficos guardados en", out_dir)

if __name__ == "__main__":
    main()
