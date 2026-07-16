"""Perfila y congela la evidencia del dataset integrado heredado."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import max_trailing_run, sha256_file, utc_now, write_json


LEGACY_PATH = ROOT / "data" / "raw" / "data.csv"
MANIFEST_PATH = ROOT / "data" / "manifests" / "legacy_data_manifest.json"
REQUIRED = [
    "Date", "Target_Price", "Sentiment_GDELT", "VIX_Close", "logP", "ret_log"
]


def main() -> None:
    df = pd.read_csv(LEGACY_PATH, parse_dates=["Date"])
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas heredadas: {missing}")
    if df.empty:
        raise ValueError("El dataset heredado esta vacio")

    tail_rows, tail_value = max_trailing_run(df["Sentiment_GDELT"])
    tail_start = df.iloc[-tail_rows]["Date"] if tail_rows else None
    numeric = {}
    for column in ["Target_Price", "Sentiment_GDELT", "VIX_Close"]:
        series = pd.to_numeric(df[column], errors="coerce")
        numeric[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "unique": int(series.nunique(dropna=True)),
            "nulls": int(series.isna().sum()),
        }

    payload = {
        "manifest_version": 1,
        "created_at_utc": utc_now(),
        "classification": "integrated_derived_dataset_with_unknown_acquisition",
        "source_file": str(LEGACY_PATH.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256_file(LEGACY_PATH),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "date_start": df["Date"].min().date().isoformat(),
        "date_end": df["Date"].max().date().isoformat(),
        "duplicate_dates": int(df["Date"].duplicated().sum()),
        "null_counts": {key: int(value) for key, value in df.isna().sum().items()},
        "numeric_profile": numeric,
        "sentiment_constant_tail": {
            "rows": int(tail_rows),
            "value": float(tail_value),
            "start": tail_start.date().isoformat() if tail_start is not None else None,
            "end": df["Date"].max().date().isoformat(),
        },
        "known_limitations": [
            "No acquisition scripts or source query are present in the active repository.",
            "The file already contains merged sources and derived log/return columns.",
            "Target_Price identity is not documented in the active repository.",
            "Sentiment_GDELT is constant over the trailing period reported above.",
        ],
    }
    write_json(MANIFEST_PATH, payload)
    print(f"OK: {MANIFEST_PATH.relative_to(ROOT)}")
    print(f"sha256={payload['sha256']} rows={payload['rows']}")
    print(
        "sentiment_tail="
        f"{tail_rows} rows ({payload['sentiment_constant_tail']['start']} -> "
        f"{payload['sentiment_constant_tail']['end']})"
    )


if __name__ == "__main__":
    main()
