"""
Chequeos estadisticos puntuales para la auditoria RC2.1.

Usa pandas/numpy/scipy y CSVs comprometidos. Reimplementa DM-HLN localmente
para verificar las celdas sensibles sin llamar a src/core/dm.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]


def dm_hln(y_true, y_bench, y_chall, h):
    y_true = np.asarray(y_true, dtype=float)
    e_bench = (y_true - np.asarray(y_bench, dtype=float)) ** 2
    e_chall = (y_true - np.asarray(y_chall, dtype=float)) ** 2
    d = e_bench - e_chall
    n = len(d)
    gamma_0 = np.var(d)
    gamma_sum = 0.0
    for lag in range(1, h):
        gamma_sum += (1.0 - lag / h) * np.cov(d[lag:], d[:-lag])[0][1]
    var_d = (gamma_0 + 2 * gamma_sum) / n
    if var_d <= 1e-16:
        return 0.0, 1.0, 0.0, 1.0, n, var_d
    dm = float(np.mean(d) / np.sqrt(var_d))
    p = float(2 * (1 - stats.norm.cdf(abs(dm))))
    k = (n + 1 - 2 * h + h * (h - 1) / n) / n
    dm_hln_stat = float(dm * np.sqrt(k))
    p_hln = float(2 * (1 - stats.t.cdf(abs(dm_hln_stat), df=n - 1)))
    return dm, p, dm_hln_stat, p_hln, n, var_d


def load_pred(period, h, model):
    return pd.read_csv(
        ROOT / "outputs" / "preds" / period / f"preds_T{h}_{model}.csv",
        parse_dates=["Date"],
    )


def verify_dm_cell(period, h, bench, chall):
    b = load_pred(period, h, bench)[["Date", "y_pred_ret"]]
    c = load_pred(period, h, chall)[["Date", "y_true_ret", "y_pred_ret"]]
    m = pd.merge(c, b, on="Date", suffixes=("", "_bench"))
    dm, p, dm_hln_stat, p_hln, n, var_d = dm_hln(
        m["y_true_ret"], m["y_pred_ret_bench"], m["y_pred_ret"], h
    )
    print(
        f"DM {period} h={h} challenger={chall} bench={bench}: "
        f"n={n} DM={dm:.4f} p={p:.4f} DM_HLN={dm_hln_stat:.4f} "
        f"p_HLN={p_hln:.4f} var={var_d:.6e}"
    )


def seasonal_gate():
    raw = pd.read_csv(ROOT / "data" / "data.csv", parse_dates=["Date"])
    raw = raw.sort_values("Date").reset_index(drop=True)
    raw["logP"] = np.log(raw["Target_Price"])
    raw["ret_1d"] = raw["logP"].diff()
    for col in ["Sentiment_GDELT", "VIX_Close"]:
        for lag in [1, 2, 5, 10, 15]:
            raw[f"{col}_lag{lag}"] = raw[col].shift(lag)
    for lag in [1, 2, 5]:
        raw[f"ret_lag{lag}"] = raw["ret_1d"].shift(lag)
    raw["Target_Ret_h1"] = raw["logP"].shift(-1) - raw["logP"]
    feature_cols = (
        ["ret_1d"]
        + [f"{col}_lag{lag}" for col in ["Sentiment_GDELT", "VIX_Close"] for lag in [1, 2, 5, 10, 15]]
        + [f"ret_lag{lag}" for lag in [1, 2, 5]]
    )
    df = raw.dropna(subset=["Target_Ret_h1"] + feature_cols).set_index("Date")
    y = df.loc[df.index <= pd.Timestamp("2023-12-29"), "ret_1d"]
    groups = [y[y.index.dayofweek == d].values for d in range(5)]
    stat, p_value = stats.kruskal(*groups)
    print(f"seasonality_gate T1 IS: KW={stat:.4f} p={p_value:.4f}")


def arimax_sarimax_identity(period):
    for h in [1, 5, 10, 20]:
        a = load_pred(period, h, "ARIMAX")
        s = load_pred(period, h, "SARIMAX")
        m = pd.merge(
            a[["Date", "y_pred_ret"]],
            s[["Date", "y_pred_ret"]],
            on="Date",
            suffixes=("_arimax", "_sarimax"),
        )
        diff = (m["y_pred_ret_arimax"] - m["y_pred_ret_sarimax"]).abs().max()
        print(f"{period} h={h} max|ARIMAX-SARIMAX y_pred_ret|={diff:.3e}")


def multiseed():
    df = pd.read_csv(ROOT / "reports" / "data" / "multiseed_dm_T20_LSTM_FULL.csv")
    seed_p = df[df["Seed"] != "ENSEMBLE"]["p_value"].astype(float)
    ens = df[df["Seed"] == "ENSEMBLE"].iloc[0]
    print(
        f"multiseed median_p={seed_p.median():.4f} "
        f"p_lt_0.05={(seed_p < 0.05).sum()}/10 "
        f"ensemble_rmse={float(ens['RMSE']):.5f} ensemble_p={float(ens['p_value']):.4f}"
    )


def pt_degenerate_rows():
    df = pd.read_csv(ROOT / "reports" / "data" / "tbl_PT_OOS.csv")
    deg = df[df["PT_Stat"].isna()][["Horizon", "Model", "HitRate", "HitRate_H0"]]
    print("PT degenerate rows:")
    print(deg.to_string(index=False))


def mz_hac_sensitivity():
    rows = []
    for h in [1, 5, 10, 20]:
        for model in ["RW", "ARIMA", "ARIMAX", "SARIMAX", "LSTM", "LSTM_SENT", "LSTM_FULL"]:
            df = load_pred("OOS", h, model)
            X = sm.add_constant(df["y_pred_level"])
            for lag in [1, max(h - 1, 0)]:
                res = sm.OLS(df["y_true_level"], X).fit(
                    cov_type="HAC", cov_kwds={"maxlags": lag}
                )
                beta = res.params.iloc[1]
                se = res.bse.iloc[1]
                p_beta1 = 2 * (1 - stats.t.cdf(abs((beta - 1.0) / se), df=res.df_resid))
                rows.append({"h": h, "model": model, "lag": lag, "p_beta1": p_beta1})
    out = pd.DataFrame(rows)
    wide = out.pivot_table(index=["h", "model"], columns="lag", values="p_beta1").reset_index()
    print("MZ HAC sensitivity examples (p_beta1):")
    for h in [5, 10, 20]:
        lag_h = h - 1
        sub = wide[wide["h"].eq(h)].copy()
        sub["abs_delta"] = (sub[1] - sub[lag_h]).abs()
        top = sub.sort_values("abs_delta", ascending=False).head(3)
        for _, r in top.iterrows():
            print(
                f"h={h} model={r['model']} p_L1={r[1]:.4g} "
                f"p_L{lag_h}={r[lag_h]:.4g} delta={r['abs_delta']:.4g}"
            )


def main():
    seasonal_gate()
    verify_dm_cell("OOS", 5, "RW", "LSTM_SENT")
    verify_dm_cell("OOS_2025", 5, "RW", "LSTM")
    arimax_sarimax_identity("OOS")
    arimax_sarimax_identity("OOS_2025")
    multiseed()
    pt_degenerate_rows()
    mz_hac_sensitivity()


if __name__ == "__main__":
    main()
