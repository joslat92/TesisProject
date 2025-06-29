# dm_test.py ──────────────────────────────────────────────────────────────
"""
Diebold-Mariano bilateral para comparar errores de predicción entre modelos.

Cada CSV debe contener las columnas (en cualquier orden / alias):
    Date | y_true | y_pred
Alias aceptados:
    Date : date, ds
    y_true: y_true, Truth, target, actual, Target_Price
    y_pred: y_pred, Pred, yhat, prediction
"""

import argparse, itertools
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# ------------------------------------------------------------------------
def dm_statistic(e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 2
                 ) -> Tuple[float, float]:
    """Diebold-Mariano (bilateral, H0: misma capacidad predictiva)."""
    d = (np.abs(e1) if power == 1 else e1**2) - (np.abs(e2) if power == 1 else e2**2)
    mean_d = np.mean(d)
    n = len(d)
    # varianza Newey–West con ventana h-1
    gamma = [np.sum((d[:n-k] - mean_d) * (d[k:] - mean_d)) / n for k in range(h)]
    var_d = gamma[0] + 2 * np.sum(gamma[1:])
    dm = mean_d / np.sqrt(var_d / n)
    pval = 2 * (1 - stats.norm.cdf(abs(dm)))
    return dm, pval

# ------------------------------------------------------------------------
ALIASES: Dict[str, set] = {
    "Date":   {"Date", "date", "ds"},
    "y_true": {"y_true", "Truth", "target", "actual", "Target_Price"},
    "y_pred": {"y_pred", "Pred", "yhat", "prediction"},
}

def load_predictions(csv_file: Path) -> pd.DataFrame:
    """Lee CSV y lo devuelve con columnas estándar."""
    txt = csv_file.read_text(encoding="utf-8", errors="ignore")[:1000]
    sep = ";" if ";" in txt and "," not in txt.split("\n")[0] else ","
    df = pd.read_csv(csv_file, sep=sep)

    # Renombrar usando alias
    colmap = {}
    for std, opts in ALIASES.items():
        for c in df.columns:
            if c in opts:
                colmap[c] = std
                break
    df = df.rename(columns=colmap)

    required = {"Date", "y_true", "y_pred"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_file} debe contener columnas {required}")

    return df[["Date", "y_true", "y_pred"]].set_index("Date").sort_index()

# ------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="models",
                    help="carpeta donde buscar subcarpetas de modelo (por defecto 'models')")
    ap.add_argument("--h", type=int, default=1, help="horizonte de predicción (steps-ahead)")
    ap.add_argument("--power", type=int, choices=[1,2], default=2,
                    help="1 = ABS, 2 = RMSE (default)")
    args = ap.parse_args()

    model_dirs = [d for d in Path(args.dir).iterdir() if d.is_dir()]
    csv_paths = {d.name: next(d.glob("predictions_all_folds.csv"), None)
                 for d in model_dirs}
    csv_paths = {k: v for k, v in csv_paths.items() if v is not None}

    if len(csv_paths) < 2:
        print("  Se necesitan al menos dos CSV de predicciones para comparar.")
        return

    # Cargar todas las series de errores
    errors: Dict[str, pd.Series] = {}
    for name, path in csv_paths.items():
        df = load_predictions(path)
        errors[name] = df["y_true"] - df["y_pred"]

    # Comparar todos los pares
    results = []
    for (m1, e1), (m2, e2) in itertools.combinations(errors.items(), 2):
        common = e1.index.intersection(e2.index)
        if len(common) < 30:            # muy pocas observaciones en común
            continue
        dm, pval = dm_statistic(e1.loc[common].values,
                                e2.loc[common].values,
                                h=args.h, power=args.power)
        results.append((m1, m2, dm, pval, len(common)))

    if not results:
        print("⚠️  No se encontraron pares con fechas en común suficientes.")
        return

    out = pd.DataFrame(results, columns=["model_1","model_2","DM_stat","p_value","n_obs"])
    out = out.sort_values("p_value")
    out.to_csv("outputs/dm_results.csv", index=False)
    print("✅  Resultados guardados en outputs/dm_results.csv")
    print(out.to_string(index=False))

# ------------------------------------------------------------------------
if __name__ == "__main__":
    main()
