# src/merge_vix.py
import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/df_final_ready.csv")
    ap.add_argument("--vix", default="data/vix.csv")
    ap.add_argument("--out", default="data/df_final_ready_plus_vix.csv")
    ap.add_argument("--prices-date-col", dest="prices_date_col", default="Date")
    ap.add_argument("--vix-date-col", dest="vix_date_col", default="Date")
    ap.add_argument("--vix-close-col", dest="vix_close_col", default="Close")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Precios base ---
    prices = pd.read_csv(args.prices)
    if args.prices_date_col not in prices.columns:
        raise ValueError(
            f"Columna de fecha '{args.prices_date_col}' no encontrada en {args.prices}. "
            f"Columnas disponibles: {list(prices.columns)}"
        )
    prices[args.prices_date_col] = pd.to_datetime(
        prices[args.prices_date_col], errors="coerce"
    )
    prices = (
        prices.dropna(subset=[args.prices_date_col])
        .sort_values(args.prices_date_col)
        .set_index(args.prices_date_col)
    )

    # --- VIX ---
    vix = pd.read_csv(args.vix)
    for col in (args.vix_date_col, args.vix_close_col):
        if col not in vix.columns:
            raise ValueError(
                f"Columna '{col}' no encontrada en {args.vix}. "
                f"Columnas disponibles: {list(vix.columns)}"
            )

    vix = vix[[args.vix_date_col, args.vix_close_col]]
    vix.columns = pd.Index([args.vix_date_col, "VIX_Close"])
    vix[args.vix_date_col] = pd.to_datetime(vix[args.vix_date_col], errors="coerce")
    vix = (
        vix.dropna(subset=[args.vix_date_col])
        .sort_values(args.vix_date_col)
        .set_index(args.vix_date_col)
    )

    # --- Join por fecha, rellenando festivos con último valor válido ---
    merged = prices.join(vix, how="left").ffill()

    # Guardar con columna Date explícita
    merged.reset_index(names="Date").to_csv(out_path, index=False)
    print(f"OK -> {out_path} ({len(merged)} filas)")


if __name__ == "__main__":
    main()
