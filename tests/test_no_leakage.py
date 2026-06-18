"""
Gate anti-fuga del pipeline (contrato §8: y_hat_t(h) solo usa información <= t).

Principio: si alteramos TODOS los datos posteriores a t y la predicción
para t cambia, el modelo está leyendo el futuro.
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def _load_stage(filename, module_name):
    """Carga un stage por ruta (los nombres 11_*.py no son importables)."""
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "src" / "stages" / filename
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _load_returns(h=20):
    df = pd.read_parquet(ROOT / "data" / "processed" / f"features_T{h}.parquet")
    return df.set_index("Date")["ret_1d"]

def test_arima_prediction_ignores_future():
    """Para t en OOS, corromper los datos > t no debe mover y_hat_t(h)."""
    os.chdir(ROOT)  # los stages leen config.yaml relativo a la raíz
    stage = _load_stage("11_train_arima.py", "stage_11_train_arima")

    t = pd.Timestamp("2024-06-14")
    rng = np.random.default_rng(123)

    for h in [1, 5, 10, 20]:
        y = _load_returns(h)
        assert y.index.max() > t + pd.Timedelta(days=60), "se necesita futuro que corromper"

        base = stage.predict_oos_iterated(y, pd.DatetimeIndex([t]), h, (1, 0, 1))

        y_corrupt = y.copy()
        future = y_corrupt.index > t
        y_corrupt.loc[future] = rng.normal(0.0, 0.05, int(future.sum()))
        corrupt = stage.predict_oos_iterated(
            y_corrupt, pd.DatetimeIndex([t]), h, (1, 0, 1)
        )

        assert np.array_equal(base, corrupt), (
            f"FUGA en h={h}: y_hat_t cambió al alterar datos posteriores a t "
            f"({base} vs {corrupt})"
        )

def test_sarimax_prediction_ignores_future():
    """
    Para t en OOS, corromper retornos Y exógenas posteriores a t no debe
    mover y_hat_t(h) de ARIMAX/SARIMAX (las exógenas futuras del forecast
    deben venir congeladas en su valor en t, nunca de los datos reales).
    """
    os.chdir(ROOT)
    stage = _load_stage("13_train_sarimax.py", "stage_13_train_sarimax")

    import yaml
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    lags_cfg = cfg["features"]["exog"]

    raw = pd.read_csv(ROOT / cfg["data"]["raw_source"], parse_dates=["Date"]).set_index("Date")
    exog_raw = raw[list(lags_cfg.keys())]

    t = pd.Timestamp("2024-06-14")
    rng = np.random.default_rng(321)

    for h in [1, 20]:
        y = _load_returns(h)
        dates = pd.DatetimeIndex([t])

        base = stage.predict_oos_iterated_exog(
            y, exog_raw, dates, h, (1, 0, 1), (0, 0, 0, 0), lags_cfg
        )

        # 1) Corromper SOLO exógenas posteriores a t
        exog_corrupt = exog_raw.copy()
        future_x = exog_corrupt.index > t
        exog_corrupt.loc[future_x] = rng.normal(50, 20, (int(future_x.sum()), exog_raw.shape[1]))
        only_exog = stage.predict_oos_iterated_exog(
            y, exog_corrupt, dates, h, (1, 0, 1), (0, 0, 0, 0), lags_cfg
        )
        assert np.array_equal(base, only_exog), (
            f"FUGA DE EXOGENAS en h={h}: y_hat_t cambió al alterar VIX/sentimiento > t"
        )

        # 2) Corromper también los retornos posteriores a t
        y_corrupt = y.copy()
        future_y = y_corrupt.index > t
        y_corrupt.loc[future_y] = rng.normal(0.0, 0.05, int(future_y.sum()))
        both = stage.predict_oos_iterated_exog(
            y_corrupt, exog_corrupt, dates, h, (1, 0, 1), (0, 0, 0, 0), lags_cfg
        )
        assert np.array_equal(base, both), (
            f"FUGA en h={h}: y_hat_t cambió al alterar retornos y exógenas > t"
        )

def test_lstm_prediction_ignores_future():
    """
    Anti-fuga para la LSTM (cierra el hueco de cobertura señalado por las
    auditorías): para una fecha t del OOS, corromper features Y target de
    TODAS las filas posteriores a t no debe mover la predicción de t.

    Se reentrena en modo reducido (epochs=1) con la misma semilla en ambas
    corridas. El scaler se ajusta solo con el train (≤ train_end), el
    entrenamiento usa solo muestras del IS, y la ventana de entrada de t
    termina en t; nada de eso depende de filas > t, así que ŷ_t debe ser
    bit-idéntico. Se usa la variante LSTM_FULL (ret_1d + Sent_lag1 + VIX_lag1)
    para ejercitar también los canales exógenos.
    """
    os.chdir(ROOT)
    import yaml
    stage = _load_stage("12_train_lstm.py", "stage_12_train_lstm")

    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    params = dict(cfg["models"]["params"]["lstm"])
    params["epochs"] = 1  # modo reducido: rápido y determinista
    feature_cols = params["variants"]["LSTM_FULL"]
    seed = cfg["project"]["seed"]
    price_col = cfg["data"]["target_col"]
    train_end = pd.Timestamp(cfg["data"]["splits"]["train_end"])
    oos_start = pd.Timestamp(cfg["data"]["splits"]["oos_start"])
    oos_end = pd.Timestamp(cfg["data"]["splits"]["oos_end"])
    h = 5

    df = pd.read_parquet(ROOT / "data" / "processed" / f"features_T{h}.parquet")

    # t a ~1 mes dentro del OOS, con futuro de sobra para corromper
    oos_dates = df.loc[(df["Date"] >= oos_start) & (df["Date"] <= oos_end), "Date"]
    t = oos_dates.iloc[20]
    assert (df["Date"] > t).sum() > h + 5, "se necesita futuro que corromper"

    base = stage.fit_predict(df, feature_cols, params, seed, h, price_col,
                             train_end, oos_start, oos_end)
    base_t = base.loc[base["Date"] == t, "y_pred_ret"].values

    df_corrupt = df.copy()
    fut = df_corrupt["Date"] > t
    rng = np.random.default_rng(999)
    cols_to_corrupt = list(feature_cols) + [f"Target_Ret_h{h}",
                                            f"Target_Price_h{h}", price_col]
    for c in cols_to_corrupt:
        df_corrupt.loc[fut, c] = rng.normal(0.0, 1.0, int(fut.sum()))

    corrupt = stage.fit_predict(df_corrupt, feature_cols, params, seed, h,
                                price_col, train_end, oos_start, oos_end)
    corrupt_t = corrupt.loc[corrupt["Date"] == t, "y_pred_ret"].values

    assert np.array_equal(base_t, corrupt_t), (
        f"FUGA LSTM: ŷ_t({t.date()}) cambió al corromper el futuro "
        f"({base_t} vs {corrupt_t})"
    )

def test_rw_preds_are_zero_return():
    """RW del contrato: y_hat_t(h) = 0 en retornos, nivel = P_t."""
    for h in [1, 5, 10, 20]:
        path = ROOT / "outputs" / "preds" / "OOS" / f"preds_T{h}_RW.csv"
        if not path.exists():
            import pytest
            pytest.skip("aún no se generaron las predicciones RW")
        preds = pd.read_csv(path)
        assert (preds["y_pred_ret"] == 0).all(), f"RW h={h} tiene retornos != 0"

if __name__ == "__main__":
    test_arima_prediction_ignores_future()
    test_sarimax_prediction_ignores_future()
    test_lstm_prediction_ignores_future()
    test_rw_preds_are_zero_return()
    print("OK: sin fuga detectada (ARIMA + ARIMAX/SARIMAX + LSTM + RW)")
