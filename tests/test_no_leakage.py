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
    test_rw_preds_are_zero_return()
    print("OK: sin fuga detectada (ARIMA iterado + RW)")
