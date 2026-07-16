"""Identifica el campo VIX usado por el dataset heredado y su origen historico."""

from __future__ import annotations

import subprocess
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import sha256_file, utc_now, write_json


HISTORICAL_REF = "5674181:data/vix.csv"
MERGE_SCRIPT_REF = "159ff23:src/merge_vix.py"


def _comparison(frame: pd.DataFrame, candidate: str) -> dict:
    valid = frame[["Legacy_VIX_Close", candidate]].dropna()
    difference = (valid["Legacy_VIX_Close"] - valid[candidate]).abs()
    return {
        "matched_rows": int(len(valid)),
        "correlation": float(valid["Legacy_VIX_Close"].corr(valid[candidate])),
        "mean_absolute_difference": float(difference.mean()),
        "maximum_absolute_difference": float(difference.max()),
        "rows_within_0_011": int((difference <= 0.011).sum()),
    }


def _pair_comparison(frame: pd.DataFrame, left: str, right: str) -> dict:
    valid = frame[[left, right]].dropna()
    difference = (valid[left] - valid[right]).abs()
    return {
        "matched_rows": int(len(valid)),
        "correlation": float(valid[left].corr(valid[right])),
        "mean_absolute_difference": float(difference.mean()),
        "maximum_absolute_difference": float(difference.max()),
        "rows_within_0_011": int((difference <= 0.011).sum()),
    }


def _git_text(ref: str) -> str:
    result = subprocess.run(
        ["git", "show", ref],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8-sig",
    )
    return result.stdout


def main() -> None:
    legacy_path = ROOT / "data" / "raw" / "data.csv"
    cboe_path = ROOT / "data" / "source" / "vix_cboe.csv"
    if not cboe_path.exists():
        raise FileNotFoundError("Ejecute primero scripts/data/10_fetch_market_data.py")

    legacy = pd.read_csv(legacy_path, parse_dates=["Date"])
    cboe = pd.read_csv(cboe_path, parse_dates=["Date"])
    inherited = legacy[["Date", "VIX_Close"]].rename(
        columns={"VIX_Close": "Legacy_VIX_Close"}
    )
    merged = inherited.merge(cboe, on="Date", how="left")

    historical = pd.read_csv(
        StringIO(_git_text(HISTORICAL_REF)),
        header=None,
        names=[
            "Date",
            "historical_field_1",
            "historical_field_2",
            "historical_field_3",
            "historical_field_4",
            "historical_field_5",
        ],
        parse_dates=["Date"],
    )
    historical_on_legacy_dates = legacy[["Date"]].merge(historical, on="Date", how="left")
    historical_on_legacy_dates["historical_field_4_ffill"] = (
        historical_on_legacy_dates["historical_field_4"].ffill()
    )
    historical_with_cboe = historical.merge(cboe, on="Date", how="inner")
    merged = merged.merge(
        historical_on_legacy_dates[["Date", "historical_field_4_ffill"]],
        on="Date",
        how="left",
    )

    open_difference = (merged["Legacy_VIX_Close"] - merged["VIX_Open"]).abs()
    exceptional = merged.loc[
        open_difference > 0.011,
        ["Date", "Legacy_VIX_Close", "VIX_Open", "VIX_Close"],
    ].copy()
    exceptional = exceptional.rename(
        columns={
            "Legacy_VIX_Close": "legacy_value",
            "VIX_Close": "official_close",
        }
    )
    exceptional["Date"] = exceptional["Date"].dt.date.astype(str)

    history_difference = (
        merged["Legacy_VIX_Close"] - merged["historical_field_4_ffill"]
    ).abs()
    merge_script = _git_text(MERGE_SCRIPT_REF)
    report = {
        "created_at_utc": utc_now(),
        "legacy_sha256": sha256_file(legacy_path),
        "official_cboe_sha256": sha256_file(cboe_path),
        "comparisons_with_current_cboe": {
            "VIX_Open": _comparison(merged, "VIX_Open"),
            "VIX_Close": _comparison(merged, "VIX_Close"),
        },
        "current_cboe_open_exceptions_over_0_011": exceptional.to_dict("records"),
        "historical_repository_evidence": {
            "raw_csv_ref": HISTORICAL_REF,
            "merge_script_ref": MERGE_SCRIPT_REF,
            "merge_script_renames_headerless_fields_as": [
                "Date", "Open", "High", "Low", "Close", "Volume"
            ],
            "merge_script_selects": "the fourth numeric field under the name Close",
            "field_mapping_against_current_cboe": {
                "historical_field_1_to_VIX_Close": _pair_comparison(
                    historical_with_cboe, "historical_field_1", "VIX_Close"
                ),
                "historical_field_4_to_VIX_Open": _pair_comparison(
                    historical_with_cboe, "historical_field_4", "VIX_Open"
                ),
            },
            "fourth_numeric_field_matches_inherited_values": {
                "rows": int(history_difference.notna().sum()),
                "rows_within_0_00001": int((history_difference <= 0.00001).sum()),
                "maximum_absolute_difference": float(history_difference.max()),
                "used_forward_fill": ".ffill()" in merge_script,
                "last_forward_filled_date": "2025-04-22",
                "last_source_date": "2025-04-21",
            },
        },
        "legacy_vix_inference": {
            "confidence": "high",
            "actual_field": "daily open",
            "inherited_label": "VIX_Close",
            "basis": (
                "The inherited series reproduces the fourth numeric field of the "
                "historical headerless CSV, including its final forward fill. That "
                "field agrees with the official Cboe open in 2542 of 2561 rows within "
                "0.011, while the first numeric field agrees with the official close. "
                "The historical merge script assigned the field names in the opposite "
                "order and therefore mislabeled the selected open as Close."
            ),
            "canonical_reconstruction_decision": (
                "Use the official Cboe daily close, retain the open for audit, and do "
                "not forward-fill missing observations silently."
            ),
        },
    }
    output = ROOT / "data" / "quality" / "legacy_vix_forensics.json"
    write_json(output, report)
    print(f"OK: {output.relative_to(ROOT)}")
    print(
        "Legacy vs Cboe: "
        f"open MAE={report['comparisons_with_current_cboe']['VIX_Open']['mean_absolute_difference']:.6f}, "
        f"close MAE={report['comparisons_with_current_cboe']['VIX_Close']['mean_absolute_difference']:.6f}"
    )
    print(
        "Legacy vs historical fourth field: "
        f"{int((history_difference <= 0.00001).sum())}/{history_difference.notna().sum()}"
    )


if __name__ == "__main__":
    main()
