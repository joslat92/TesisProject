import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

from src.core.stationarity import stationarity_table


def test_stationarity_table_uses_fixed_lags_and_expected_specification():
    rng = np.random.default_rng(2025)
    series = pd.Series(10_000 + np.cumsum(rng.normal(size=250)))

    result = stationarity_table(series, "Target_Price")

    assert list(result.columns) == [
        "series",
        "transform",
        "test",
        "lag",
        "stat",
        "pvalue",
        "conclusion",
    ]
    assert len(result) == 12
    assert set(result["transform"]) == {"level", "diff1"}
    assert set(result["test"]) == {"ADF", "KPSS"}
    assert set(result["lag"]) == {0, 1, 2}

    adf_row = result.query("transform == 'level' and test == 'ADF' and lag == 2").iloc[0]
    expected_adf = adfuller(series, maxlag=2, regression="c", autolag=None)
    assert adf_row["stat"] == expected_adf[0]
    assert adf_row["pvalue"] == expected_adf[1]

    kpss_row = result.query("transform == 'diff1' and test == 'KPSS' and lag == 1").iloc[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        expected_kpss = kpss(series.diff().dropna(), regression="c", nlags=1)
    assert kpss_row["stat"] == expected_kpss[0]
    assert kpss_row["pvalue"] == expected_kpss[1]
