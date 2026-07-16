"""Descarga y normaliza NASDAQ-100 (FRED/Nasdaq) y VIX (Cboe)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import download, sha256_file, utc_now, write_json


FRED_TEMPLATE = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv?"
    "id=NASDAQ100&cosd={start}&coed={end}"
)
CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"


def _validate(frame: pd.DataFrame, value_columns: list[str], label: str) -> None:
    if frame.empty:
        raise ValueError(f"{label}: no hay observaciones")
    if frame["Date"].isna().any():
        raise ValueError(f"{label}: contiene fechas invalidas")
    if frame["Date"].duplicated().any():
        raise ValueError(f"{label}: contiene fechas duplicadas")
    if not frame["Date"].is_monotonic_increasing:
        raise ValueError(f"{label}: las fechas no estan ordenadas")
    for value_column in value_columns:
        if frame[value_column].isna().any() or (frame[value_column] <= 0).any():
            raise ValueError(f"{label}: {value_column} contiene valores invalidos")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2015-02-17")
    parser.add_argument("--end", default="2025-04-22")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if start >= end:
        raise ValueError("--start debe ser anterior a --end")

    source_dir = ROOT / "data" / "source"
    fred_raw = source_dir / "fred_nasdaq100_download.csv"
    cboe_raw = source_dir / "cboe_vix_download.csv"
    fred_meta = download(FRED_TEMPLATE.format(start=args.start, end=args.end), fred_raw)
    cboe_meta = download(CBOE_URL, cboe_raw)

    ndx_raw = pd.read_csv(fred_raw)
    date_column = "observation_date" if "observation_date" in ndx_raw else "DATE"
    ndx = ndx_raw.rename(columns={date_column: "Date", "NASDAQ100": "NDX_Close"})
    ndx = ndx[["Date", "NDX_Close"]]
    ndx["Date"] = pd.to_datetime(ndx["Date"], errors="coerce")
    ndx["NDX_Close"] = pd.to_numeric(ndx["NDX_Close"], errors="coerce")
    ndx = ndx[(ndx["Date"] >= start) & (ndx["Date"] <= end)].dropna()
    ndx = ndx.sort_values("Date").reset_index(drop=True)
    _validate(ndx, ["NDX_Close"], "NASDAQ-100")

    vix_raw = pd.read_csv(cboe_raw)
    vix_raw.columns = [column.strip().upper() for column in vix_raw.columns]
    vix = vix_raw.rename(
        columns={"DATE": "Date", "OPEN": "VIX_Open", "CLOSE": "VIX_Close"}
    )
    vix = vix[["Date", "VIX_Open", "VIX_Close"]]
    vix["Date"] = pd.to_datetime(vix["Date"], errors="coerce")
    vix["VIX_Open"] = pd.to_numeric(vix["VIX_Open"], errors="coerce")
    vix["VIX_Close"] = pd.to_numeric(vix["VIX_Close"], errors="coerce")
    vix = vix[(vix["Date"] >= start) & (vix["Date"] <= end)].dropna()
    vix = vix.sort_values("Date").reset_index(drop=True)
    _validate(vix, ["VIX_Open", "VIX_Close"], "VIX")

    ndx_path = source_dir / "nasdaq100_fred.csv"
    vix_path = source_dir / "vix_cboe.csv"
    ndx.to_csv(ndx_path, index=False, date_format="%Y-%m-%d")
    vix.to_csv(vix_path, index=False, date_format="%Y-%m-%d")

    manifest = {
        "manifest_version": 1,
        "created_at_utc": utc_now(),
        "requested_range": {"start": args.start, "end": args.end},
        "redistribution_note": (
            "Source files are excluded from Git pending a licensing review; "
            "the URLs, retrieval metadata and checksums are versioned."
        ),
        "sources": {
            "nasdaq100": {
                "provider": "FRED, Federal Reserve Bank of St. Louis",
                "upstream_source": "Nasdaq, Inc.",
                "series": "NASDAQ100",
                "fields": ["daily open", "daily close"],
                "download": fred_meta,
                "normalized_file": "data/source/nasdaq100_fred.csv",
                "normalized_sha256": sha256_file(ndx_path),
                "rows": int(len(ndx)),
                "date_start": ndx["Date"].min().date().isoformat(),
                "date_end": ndx["Date"].max().date().isoformat(),
            },
            "vix": {
                "provider": "Cboe Global Markets",
                "series": "VIX Index",
                "field": "daily close",
                "download": cboe_meta,
                "normalized_file": "data/source/vix_cboe.csv",
                "normalized_sha256": sha256_file(vix_path),
                "rows": int(len(vix)),
                "date_start": vix["Date"].min().date().isoformat(),
                "date_end": vix["Date"].max().date().isoformat(),
            },
        },
    }
    manifest_path = ROOT / "data" / "manifests" / "market_sources.json"
    write_json(manifest_path, manifest)
    print(f"OK: NDX={len(ndx)} filas | VIX={len(vix)} filas")
    print(f"Manifiesto: {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
