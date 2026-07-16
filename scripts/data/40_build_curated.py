"""Construye el dataset canonico una vez sellada la seleccion de GDELT."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import sha256_file, utc_now, write_json
from gdelt_utils import aggregate_to_market_dates, longest_missing_run


def main() -> None:
    config_path = ROOT / "config_data.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    scope = config["gdelt"]["selected_scope"]
    if scope not in config["gdelt"]["allowed_scopes"]:
        raise ValueError(
            "config_data.yaml debe declarar gdelt.selected_scope despues de la "
            "auditoria de cobertura y relevancia"
        )

    ndx_path = ROOT / "data" / "source" / "nasdaq100_fred.csv"
    vix_path = ROOT / "data" / "source" / "vix_cboe.csv"
    gdelt_path = ROOT / "data" / "source" / "gdelt_daily_tone_candidates.csv"
    for path in [ndx_path, vix_path, gdelt_path]:
        if not path.exists():
            raise FileNotFoundError(f"Falta la fuente normalizada: {path}")

    ndx = pd.read_csv(ndx_path, parse_dates=["Date"])
    vix = pd.read_csv(vix_path, parse_dates=["Date"])
    gdelt = pd.read_csv(gdelt_path, parse_dates=["Date"])
    eligible_dates = pd.Series(sorted(set(ndx["Date"]) & set(vix["Date"])))
    eligible_dates = eligible_dates[
        eligible_dates.between(config["period"]["start"], config["period"]["end"])
    ].reset_index(drop=True)
    sentiment = aggregate_to_market_dates(gdelt, eligible_dates, scope)

    curated = (
        ndx.loc[ndx["Date"].isin(eligible_dates)]
        .merge(vix[["Date", "VIX_Close"]], on="Date", how="left", validate="one_to_one")
        .merge(sentiment, on="Date", how="left", validate="one_to_one")
        .sort_values("Date")
        .reset_index(drop=True)
        .rename(columns={"NDX_Close": "Target_Price"})
    )
    present = curated["Sentiment_GDELT"].notna()
    quality = {
        "created_at_utc": utc_now(),
        "selected_scope": scope,
        "rows_before_sentiment_exclusion": int(len(curated)),
        "date_start": curated["Date"].min().date().isoformat(),
        "date_end": curated["Date"].max().date().isoformat(),
        "target_dates_excluded_by_market_intersection": sorted(
            d.date().isoformat() for d in set(ndx["Date"]) - set(vix["Date"])
            if pd.Timestamp(config["period"]["start"]) <= d <= pd.Timestamp(config["period"]["end"])
        ),
        "missing_counts_before_derived_columns": {
            column: int(curated[column].isna().sum())
            for column in ["Target_Price", "VIX_Close", "Sentiment_GDELT"]
        },
        "sentiment_longest_missing_run": longest_missing_run(present),
        "sentiment_missing_dates_first_20": curated.loc[~present, "Date"]
            .dt.date.astype(str).head(20).tolist(),
        "duplicate_dates": int(curated["Date"].duplicated().sum()),
    }
    quality_path = ROOT / "data" / "quality" / "curated_data_quality.json"
    write_json(quality_path, quality)
    if quality["duplicate_dates"]:
        raise ValueError("El dataset curado contiene fechas duplicadas")
    required_market = ["Target_Price", "VIX_Close"]
    if curated[required_market].isna().any().any():
        raise ValueError(
            f"Las fuentes de mercado tienen faltantes; revise {quality_path.relative_to(ROOT)}"
        )
    curated = curated.loc[present].copy().reset_index(drop=True)
    quality["rows_after_sentiment_exclusion"] = int(len(curated))
    write_json(quality_path, quality)
    if curated["Sentiment_GDELT"].isna().any():
        raise ValueError("La exclusion explicita no elimino todos los tonos faltantes")
    if (curated[["Target_Price", "VIX_Close"]] <= 0).any().any():
        raise ValueError("Precio objetivo o VIX no positivo")
    if not curated["Sentiment_GDELT"].between(-100, 100).all():
        raise ValueError("Sentiment_GDELT esta fuera del rango teorico de V2Tone")

    curated["logP"] = np.log(curated["Target_Price"])
    curated["ret_log"] = curated["logP"].diff()
    output_columns = [
        "Date", "Target_Price", "Sentiment_GDELT", "VIX_Close", "logP", "ret_log"
    ]
    output = ROOT / "data" / "curated" / "model_input_ndx.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    curated[output_columns].to_csv(output, index=False, date_format="%Y-%m-%d")
    manifest = {
        "manifest_version": 1,
        "created_at_utc": utc_now(),
        "config_file": "config_data.yaml",
        "config_sha256": sha256_file(config_path),
        "inputs": {
            "nasdaq100_sha256": sha256_file(ndx_path),
            "vix_sha256": sha256_file(vix_path),
            "gdelt_candidates_sha256": sha256_file(gdelt_path),
        },
        "output_file": "data/curated/model_input_ndx.csv",
        "output_sha256": sha256_file(output),
        "rows": int(len(curated)),
        "date_start": quality["date_start"],
        "date_end": quality["date_end"],
        "decisions": config,
    }
    manifest_path = ROOT / "data" / "manifests" / "curated_dataset.json"
    write_json(manifest_path, manifest)
    print(f"OK: {output.relative_to(ROOT)} ({len(curated)} filas)")
    print(f"Manifiesto: {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
