from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "data"))

from gdelt_utils import aggregate_to_market_dates, longest_missing_run


def test_calendar_news_maps_to_next_market_date_with_weighted_tone():
    candidates = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08"]
            ),
            "scope": ["exact_ndx"] * 4,
            "n_articles": [2, 2, 1, 3],
            "n_sources": [2, 1, 1, 2],
            "mean_tone": [2.0, -2.0, 4.0, 1.0],
        }
    )
    market_dates = pd.to_datetime(["2024-01-05", "2024-01-08"])

    result = aggregate_to_market_dates(candidates, market_dates, "exact_ndx")

    assert result["Date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-05",
        "2024-01-08",
    ]
    assert result["GDELT_Articles"].tolist() == [2, 6]
    assert result["GDELT_Calendar_Days"].tolist() == [1, 3]
    assert result.loc[0, "Sentiment_GDELT"] == pytest.approx(2.0)
    assert result.loc[1, "Sentiment_GDELT"] == pytest.approx(0.5)


def test_news_after_last_market_date_is_not_backfilled():
    candidates = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-08", "2024-01-09"]),
            "scope": ["broad_nasdaq", "broad_nasdaq"],
            "n_articles": [1, 100],
            "n_sources": [1, 10],
            "mean_tone": [3.0, -50.0],
        }
    )

    result = aggregate_to_market_dates(
        candidates, pd.to_datetime(["2024-01-08"]), "broad_nasdaq"
    )

    assert len(result) == 1
    assert result.loc[0, "Sentiment_GDELT"] == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("present", "expected"),
    [
        ([True, True], 0),
        ([False, False, True, False], 2),
        ([True, False, False, False], 3),
    ],
)
def test_longest_missing_run(present, expected):
    assert longest_missing_run(pd.Series(present)) == expected
