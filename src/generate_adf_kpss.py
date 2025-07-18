#!/usr/bin/env python
"""
Genera adf_kpss_full.csv con resultados detallados de las pruebas de estacionariedad.
Uso:
    python src/generate_adf_kpss.py --data data/df_final_ready.csv --col Target_Price \
        --out appendices/data/adf_kpss_full.csv --maxlag 10
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

def run_adf(x, lag=None, regression="c"):
    """ADF con lag fijo (int) o autolag ('AIC'). Maneja 5 o 6 outputs."""
    if lag is None:                       # autolag por AIC
        result = adfuller(x, autolag="AIC", regression=regression)
        lag_lbl = "auto"
    else:                                 # lag fijo
        result = adfuller(x, maxlag=lag, autolag=None, regression=regression)
        lag_lbl = str(lag)

    # Desempaquetado robusto (5 o 6 elementos)
    stat, pval, usedlag, nobs, crit = result[:5]
    icbest = result[5] if len(result) == 6 else float("nan")

    conc = "Reject_unit_root" if pval < 0.05 else "Fail_to_reject_unit_root"
    return {
        "test": "ADF",
        "lag": lag_lbl,
        "stat": stat,
        "pvalue": pval,
        "nobs": nobs,
        "crit_1": crit["1%"],
        "crit_5": crit["5%"],
        "crit_10": crit["10%"],
        "conclusion": conc,
    }


def run_kpss(x, lag=None, regression="c"):
    """KPSS con nlags fijo o auto (Bartlett kernel)."""
    if lag is None:
        stat, pval, lags, crit = kpss(x, nlags="auto", regression=regression)
        lag_lbl = "auto"
    else:
        stat, pval, lags, crit = kpss(x, nlags=lag, regression=regression)
        lag_lbl = str(lag)
    conc = "Reject_stationarity" if pval < 0.05 else "Do_not_reject_stationarity"
    return {
        "test": "KPSS",
        "lag": lag_lbl,
        "stat": stat,
        "pvalue": pval,
        "nobs": len(x),
        "crit_1": crit["1%"],
        "crit_5": crit["5%"],
        "crit_10": crit["10%"],
        "conclusion": conc,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Ruta al CSV con la serie.")
    ap.add_argument("--col", required=True, help="Nombre de la columna numérica a analizar.")
    ap.add_argument("--out", required=True, help="Ruta de salida del CSV completo.")
    ap.add_argument("--maxlag", type=int, default=5, help="Máximo lag explícito a evaluar.")
    ap.add_argument("--trend", default="c", choices=["c","ct"], help="Regresión: c=const, ct=const+trend.")
    args = ap.parse_args()

    df = pd.read_csv(args.data, parse_dates=["Date"])
    df = df.sort_values("Date")
    x = df[args.col].values.astype(float)

    # Serie en diferencia
    dx = np.diff(x)

    rows = []
    # ADF nivel: auto + 0..maxlag
    rows.append({"series": args.col, "transform": "level", **run_adf(x, lag=None, regression=args.trend)})
    for lag in range(0, args.maxlag+1):
        rows.append({"series": args.col, "transform": "level", **run_adf(x, lag=lag, regression=args.trend)})

    # KPSS nivel: auto + 0..maxlag
    rows.append({"series": args.col, "transform": "level", **run_kpss(x, lag=None, regression=args.trend)})
    for lag in range(0, args.maxlag+1):
        rows.append({"series": args.col, "transform": "level", **run_kpss(x, lag=lag, regression=args.trend)})

    # ADF diff1
    rows.append({"series": args.col, "transform": "diff1", **run_adf(dx, lag=None, regression=args.trend)})
    for lag in range(0, args.maxlag+1):
        rows.append({"series": args.col, "transform": "diff1", **run_adf(dx, lag=lag, regression=args.trend)})

    # KPSS diff1
    rows.append({"series": args.col, "transform": "diff1", **run_kpss(dx, lag=None, regression=args.trend)})
    for lag in range(0, args.maxlag+1):
        rows.append({"series": args.col, "transform": "diff1", **run_kpss(dx, lag=lag, regression=args.trend)})

    out_df = pd.DataFrame(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(out_df)} filas)")

    # Resumen corto (lags 0,1,2) útil para Tabla A-1
    short = out_df[out_df["lag"].isin(["0","1","2"])].copy()
    short_path = out_path.with_name(out_path.stem + "_summary.csv")
    short.to_csv(short_path, index=False)
    print(f"Resumen corto: {short_path}")

if __name__ == "__main__":
    main()
