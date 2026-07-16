"""Contrasta Target_Price heredado con NDX y QQQ (solo identificacion forense)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timezone
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import download, sha256_file, utc_now, write_json


def _yahoo_url(start: pd.Timestamp, end: pd.Timestamp) -> str:
    period1 = int(start.tz_localize(timezone.utc).timestamp())
    period2 = int((end + pd.Timedelta(days=1)).tz_localize(timezone.utc).timestamp())
    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    return "https://query1.finance.yahoo.com/v8/finance/chart/QQQ?" + urlencode(params)


def _series_stats(frame: pd.DataFrame, candidate: str) -> dict:
    valid = frame[["Target_Price", candidate]].dropna()
    legacy_ret = np.log(valid["Target_Price"]).diff()
    candidate_ret = np.log(valid[candidate]).diff()
    ratio = valid["Target_Price"] / valid[candidate]
    return {
        "matched_rows": int(len(valid)),
        "level_correlation": float(valid["Target_Price"].corr(valid[candidate])),
        "log_return_correlation": float(legacy_ret.corr(candidate_ret)),
        "median_level_ratio": float(ratio.median()),
        "level_ratio_cv": float(ratio.std() / ratio.mean()),
        "mean_absolute_percentage_gap": float(
            ((valid["Target_Price"] / valid[candidate] - 1).abs()).mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-qqq", action="store_true")
    args = parser.parse_args()

    legacy = pd.read_csv(ROOT / "data" / "raw" / "data.csv", parse_dates=["Date"])
    ndx_path = ROOT / "data" / "source" / "nasdaq100_fred.csv"
    if not ndx_path.exists():
        raise FileNotFoundError("Ejecute primero scripts/data/10_fetch_market_data.py")
    ndx = pd.read_csv(ndx_path, parse_dates=["Date"])
    merged = legacy[["Date", "Target_Price"]].merge(ndx, on="Date", how="left")

    report = {
        "created_at_utc": utc_now(),
        "legacy_sha256": sha256_file(ROOT / "data" / "raw" / "data.csv"),
        "comparisons": {"NDX_Close": _series_stats(merged, "NDX_Close")},
        "historical_repository_evidence": {
            "observation": (
                "Earlier Git history contains the same target at higher precision "
                "together with an equity Volume column."
            ),
            "interpretation": (
                "This supports an ETF-price origin and is inconsistent with an "
                "untradeable index-level series, but it is not standalone proof of QQQ."
            ),
        },
    }

    if not args.skip_qqq:
        source_dir = ROOT / "data" / "source"
        yahoo_raw = source_dir / "qqq_yahoo_forensic.json"
        yahoo_meta = download(
            _yahoo_url(legacy["Date"].min(), legacy["Date"].max()), yahoo_raw
        )
        payload = json.loads(yahoo_raw.read_text(encoding="utf-8"))
        result = payload["chart"]["result"][0]
        quote = result["indicators"]["quote"][0]
        adjusted = result["indicators"]["adjclose"][0]["adjclose"]
        qqq = pd.DataFrame({
            "Date": pd.to_datetime(result["timestamp"], unit="s", utc=True)
                .tz_convert(None).normalize(),
            "QQQ_Close": quote["close"],
            "QQQ_AdjClose": adjusted,
            "QQQ_Volume": quote["volume"],
        }).dropna(subset=["Date"])
        qqq_path = source_dir / "qqq_yahoo_forensic.csv"
        qqq.to_csv(qqq_path, index=False, date_format="%Y-%m-%d")
        merged = merged.merge(qqq, on="Date", how="left")
        report["comparisons"]["QQQ_Close"] = _series_stats(merged, "QQQ_Close")
        report["comparisons"]["QQQ_AdjClose"] = _series_stats(
            merged, "QQQ_AdjClose"
        )
        report["qqq_forensic_source"] = {
            "role": "forensic_only_not_canonical",
            "download": yahoo_meta,
            "normalized_sha256": sha256_file(qqq_path),
        }
        qqq_adjusted = report["comparisons"]["QQQ_AdjClose"]
        if (
            qqq_adjusted["log_return_correlation"] > 0.999999
            and qqq_adjusted["level_ratio_cv"] < 0.00001
        ):
            report["legacy_target_inference"] = {
                "confidence": "high",
                "instrument": "Invesco QQQ ETF",
                "field": "adjusted close",
                "basis": (
                    "The inherited target and current QQQ adjusted close have "
                    "virtually identical log returns and a nearly constant level "
                    "ratio. The constant ratio is consistent with a later adjustment "
                    "vintage and does not affect log returns. Earlier Git history "
                    "also associates the target with equity trading volume."
                ),
                "canonical_reconstruction_decision": (
                    "Use the official NASDAQ-100 close as the thesis target and keep "
                    "QQQ adjusted close only as a documented sensitivity analysis."
                ),
            }

    output = ROOT / "data" / "quality" / "legacy_target_forensics.json"
    write_json(output, report)
    print(f"OK: {output.relative_to(ROOT)}")
    for name, stats in report["comparisons"].items():
        print(
            f"{name}: return_corr={stats['log_return_correlation']:.8f} "
            f"ratio_cv={stats['level_ratio_cv']:.6f}"
        )


if __name__ == "__main__":
    main()
