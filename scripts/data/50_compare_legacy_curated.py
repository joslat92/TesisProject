"""Compara el dataset heredado y el reconstruido sin evaluar modelos."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import max_trailing_run, sha256_file, utc_now, write_json


def _distribution(series: pd.Series) -> dict:
    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "median": float(series.median()),
        "max": float(series.max()),
    }


def main() -> None:
    legacy_path = ROOT / "data" / "raw" / "data.csv"
    curated_path = ROOT / "data" / "curated" / "model_input_ndx.csv"
    if not curated_path.exists():
        raise FileNotFoundError("Ejecute primero scripts/data/40_build_curated.py")

    legacy = pd.read_csv(legacy_path, parse_dates=["Date"]).sort_values("Date")
    curated = pd.read_csv(curated_path, parse_dates=["Date"]).sort_values("Date")
    legacy["target_return"] = np.log(legacy["Target_Price"]).diff()
    curated["target_return"] = np.log(curated["Target_Price"]).diff()
    merged = legacy.merge(curated, on="Date", how="inner", suffixes=("_legacy", "_curated"))

    target_ratio = merged["Target_Price_legacy"] / merged["Target_Price_curated"]
    sentiment_difference = (
        merged["Sentiment_GDELT_legacy"] - merged["Sentiment_GDELT_curated"]
    )
    vix_difference = merged["VIX_Close_legacy"] - merged["VIX_Close_curated"]
    trailing_rows, trailing_value = max_trailing_run(legacy["Sentiment_GDELT"])
    legacy_tail_start = legacy.iloc[-trailing_rows]["Date"]
    rebuilt_on_legacy_tail = curated.loc[curated["Date"] >= legacy_tail_start]

    report = {
        "created_at_utc": utc_now(),
        "inputs": {
            "legacy_sha256": sha256_file(legacy_path),
            "curated_sha256": sha256_file(curated_path),
        },
        "date_alignment": {
            "legacy_rows": int(len(legacy)),
            "curated_rows": int(len(curated)),
            "matched_rows": int(len(merged)),
            "legacy_only_dates": sorted(
                d.date().isoformat() for d in set(legacy["Date"]) - set(curated["Date"])
            ),
            "curated_only_dates": sorted(
                d.date().isoformat() for d in set(curated["Date"]) - set(legacy["Date"])
            ),
        },
        "target": {
            "legacy_inference": "QQQ adjusted close",
            "curated_definition": "NASDAQ-100 official daily close",
            "level_correlation": float(
                merged["Target_Price_legacy"].corr(merged["Target_Price_curated"])
            ),
            "log_return_correlation": float(
                merged["target_return_legacy"].corr(merged["target_return_curated"])
            ),
            "median_level_ratio_legacy_to_curated": float(target_ratio.median()),
            "level_ratio_cv": float(target_ratio.std() / target_ratio.mean()),
        },
        "vix": {
            "legacy_inference": "daily open mislabeled as close",
            "curated_definition": "official Cboe daily close",
            "correlation": float(
                merged["VIX_Close_legacy"].corr(merged["VIX_Close_curated"])
            ),
            "mean_absolute_difference": float(vix_difference.abs().mean()),
            "maximum_absolute_difference": float(vix_difference.abs().max()),
        },
        "sentiment": {
            "legacy_definition": "unknown historical extraction and aggregation",
            "curated_definition": "nasdaq_market; weighted GDELT mean tone",
            "correlation": float(
                merged["Sentiment_GDELT_legacy"].corr(
                    merged["Sentiment_GDELT_curated"]
                )
            ),
            "mean_absolute_difference": float(sentiment_difference.abs().mean()),
            "legacy_distribution": _distribution(merged["Sentiment_GDELT_legacy"]),
            "curated_distribution": _distribution(merged["Sentiment_GDELT_curated"]),
            "legacy_constant_tail": {
                "rows": int(trailing_rows),
                "start": legacy_tail_start.date().isoformat(),
                "value": float(trailing_value),
            },
            "curated_on_legacy_tail": {
                "rows": int(len(rebuilt_on_legacy_tail)),
                "unique_values": int(rebuilt_on_legacy_tail["Sentiment_GDELT"].nunique()),
                "std": float(rebuilt_on_legacy_tail["Sentiment_GDELT"].std()),
            },
        },
        "interpretation": (
            "The rebuilt dataset changes all three substantive inputs. Model results "
            "from the inherited dataset cannot be transferred to the rebuilt dataset "
            "without rerunning the complete sealed pipeline."
        ),
    }
    output = ROOT / "data" / "quality" / "legacy_vs_curated.json"
    write_json(output, report)
    print(f"OK: {output.relative_to(ROOT)}")
    print(
        f"Target return corr={report['target']['log_return_correlation']:.6f} | "
        f"VIX corr={report['vix']['correlation']:.6f} | "
        f"Sentiment corr={report['sentiment']['correlation']:.6f}"
    )


if __name__ == "__main__":
    main()
