"""Funciones deterministas para alinear GDELT con sesiones de mercado."""

from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_to_market_dates(
    candidates: pd.DataFrame,
    market_dates: pd.Series | pd.DatetimeIndex,
    scope: str,
) -> pd.DataFrame:
    selected = candidates.loc[candidates["scope"] == scope].copy()
    if selected.empty:
        raise ValueError(f"GDELT no contiene filas para el scope {scope}")

    selected["Date"] = pd.to_datetime(selected["Date"], errors="raise").dt.normalize()
    dates = pd.DatetimeIndex(pd.to_datetime(market_dates)).sort_values().unique()
    positions = dates.searchsorted(selected["Date"], side="left")
    inside = positions < len(dates)
    selected = selected.loc[inside].copy()
    selected["Market_Date"] = dates[positions[inside]]
    selected["weighted_tone"] = selected["mean_tone"] * selected["n_articles"]

    grouped = selected.groupby("Market_Date", as_index=False).agg(
        weighted_tone=("weighted_tone", "sum"),
        GDELT_Articles=("n_articles", "sum"),
        GDELT_Calendar_Days=("Date", "nunique"),
        GDELT_Source_Day_Count=("n_sources", "sum"),
    )
    grouped["Sentiment_GDELT"] = (
        grouped["weighted_tone"] / grouped["GDELT_Articles"]
    )
    return grouped.rename(columns={"Market_Date": "Date"}).drop(
        columns="weighted_tone"
    )


def longest_missing_run(present: pd.Series) -> int:
    longest = 0
    current = 0
    for value in present.astype(bool):
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest
