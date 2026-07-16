"""Reproducible ADF/KPSS diagnostics used by the thesis appendix."""

from __future__ import annotations

import warnings

import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss


def stationarity_table(series: pd.Series, series_name: str) -> pd.DataFrame:
    """Run the thesis' fixed-lag ADF/KPSS specification on level and diff1."""
    level = pd.to_numeric(series, errors="raise").dropna()
    if len(level) < 10:
        raise ValueError("Stationarity diagnostics require at least 10 observations.")

    transforms = {
        "level": level,
        "diff1": level.diff().dropna(),
    }
    rows: list[dict[str, object]] = []

    for transform, values in transforms.items():
        for test in ("ADF", "KPSS"):
            for lag in range(3):
                if test == "ADF":
                    stat, pvalue = adfuller(
                        values,
                        maxlag=lag,
                        regression="c",
                        autolag=None,
                    )[:2]
                    conclusion = (
                        "Reject_unit_root"
                        if pvalue < 0.05
                        else "Fail_to_reject_unit_root"
                    )
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", InterpolationWarning)
                        stat, pvalue = kpss(
                            values,
                            regression="c",
                            nlags=lag,
                        )[:2]
                    conclusion = (
                        "Reject_stationarity"
                        if pvalue < 0.05
                        else "Do_not_reject_stationarity"
                    )

                rows.append(
                    {
                        "series": series_name,
                        "transform": transform,
                        "test": test,
                        "lag": lag,
                        "stat": float(stat),
                        "pvalue": float(pvalue),
                        "conclusion": conclusion,
                    }
                )

    return pd.DataFrame(rows)
