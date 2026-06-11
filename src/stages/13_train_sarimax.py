import pandas as pd
import numpy as np
import yaml
import os
import sys
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.core.contract import ContractValidator

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def seasonality_gate(y_ret_is, m=5, alpha=0.10):
    """
    Gating de estacionalidad del contrato: incluir términos estacionales (m=5)
    SOLO si hay evidencia con p < 0.10 en el IS.

    Test: Kruskal-Wallis de efecto día-de-semana sobre los retornos diarios
    del IS (5 grupos lunes..viernes). Es el test estándar para estacionalidad
    semanal en datos de trading diarios y entrega un único p-value.
    """
    groups = [y_ret_is[y_ret_is.index.dayofweek == d].values for d in range(5)]
    groups = [g for g in groups if len(g) > 0]
    stat, p_value = stats.kruskal(*groups)
    return p_value < alpha, p_value

def build_exog_lags(exog_raw, lags_cfg):
    """Matriz de exógenas rezagadas: col '{name}_lag{k}' = serie cruda shift(k)."""
    out = {}
    for col, lags in lags_cfg.items():
        for k in lags:
            out[f"{col}_lag{k}"] = exog_raw[col].shift(k)
    return pd.DataFrame(out, index=exog_raw.index)

def predict_oos_iterated_exog(y_ret, exog_raw, oos_dates, h, order,
                              seasonal_order, lags_cfg, refit="M", fit_upto=None):
    """
    Pronóstico iterado anti-fuga con exógenas para y_hat_t(h) = sum(ret_{t+1..t+h}).

    Para cada t de OOS el modelo solo recibe retornos y exógenas CRUDAS <= t.
    Las exógenas entran rezagadas (lags de config). Para los pasos futuros
    t+1..t+h del forecast, cada exógena cruda se CONGELA en su último valor
    conocido en t (random-walk para la exógena): los lag-k con t+j-k <= t usan
    el valor real ya observado, y solo los que caerían después de t usan el
    valor congelado. Elección documentada: es la convención conservadora que
    no requiere un modelo auxiliar para las exógenas y garantiza información <= t.

    Esquema expanding con re-fit mensual + apply() diario (misma justificación
    que en 11_train_arima: re-fit diario ~30x más caro, parámetros estables).

    fit_upto (walk-forward): si se indica, los parámetros se estiman una sola
    vez con datos hasta esa fecha (día previo al bloque; contrato: HPs solo
    con IS o bloques previos) y cada t del bloque solo re-filtra.

    exog_raw debe venir en el índice CRUDO completo (anterior al recorte por
    lags) para que los primeros rezagos del historial existan.
    """
    max_h_idx = y_ret.index
    preds = []
    res = None
    fitted_month = None
    for t in oos_dates:
        y_t = y_ret.loc[:t]

        pos_t = max_h_idx.get_loc(t)
        future_idx = max_h_idx[pos_t + 1: pos_t + 1 + h]

        hist = exog_raw.loc[:t]
        frozen = pd.DataFrame(
            np.tile(hist.iloc[-1].values, (h, 1)),
            columns=hist.columns, index=future_idx
        )
        X = build_exog_lags(pd.concat([hist, frozen]), lags_cfg)
        X_hist = X.loc[y_t.index]
        X_fut = X.loc[future_idx]

        if fit_upto is not None:
            if res is None:
                y_fit = y_ret.loc[:fit_upto]
                res = ARIMA(y_fit, exog=X.loc[y_fit.index], order=order,
                            seasonal_order=seasonal_order).fit()
            res = res.apply(y_t, exog=X_hist)
        else:
            month = (t.year, t.month) if refit == "M" else None
            if res is None or month != fitted_month:
                res = ARIMA(y_t, exog=X_hist, order=order,
                            seasonal_order=seasonal_order).fit()
                fitted_month = month
            else:
                res = res.apply(y_t, exog=X_hist)
        preds.append(float(res.forecast(steps=h, exog=X_fut).sum()))
    return np.array(preds)

def run_sarimax():
    cfg = load_config()
    print(">>> [Fase 2] Entrenando ARIMAX/SARIMAX (iterado, anti-fuga) bajo Contrato...")

    horizons = cfg['features']['horizons']
    input_pattern = cfg['paths']['features_parquet_pattern']
    output_dir = cfg['paths']['preds_oos_dir']

    train_end = pd.Timestamp(cfg['data']['splits']['train_end'])
    oos_start = pd.Timestamp(cfg['data']['splits']['oos_start'])
    oos_end = pd.Timestamp(cfg['data']['splits']['oos_end'])

    order = tuple(cfg['models']['params']['arima']['order'])
    seasonal_cfg = tuple(cfg['models']['params']['sarimax']['seasonal_order'])
    lags_cfg = cfg['features']['exog']

    # Exógenas crudas desde el raw (los parquets solo traen los lags)
    raw = pd.read_csv(cfg['data']['raw_source'], parse_dates=['Date']).set_index('Date')
    exog_raw = raw[list(lags_cfg.keys())]

    for h in horizons:
        print(f"   Procesando h={h}...")
        df = pd.read_parquet(input_pattern.format(h=h)).set_index('Date')
        y_ret = df['ret_1d']

        df_oos = df[(df.index >= oos_start) & (df.index <= oos_end)]
        oos_dates = df_oos.index

        # Gating de estacionalidad sobre el IS (mismo y_ret para todo h;
        # se evalúa y registra por horizonte según contrato)
        use_seasonal, p_seas = seasonality_gate(
            y_ret[y_ret.index <= train_end], m=seasonal_cfg[3]
        )
        print(f"      Gate estacional m={seasonal_cfg[3]}: p={p_seas:.4f} -> "
              f"{'CON' if use_seasonal else 'SIN'} términos estacionales")

        for model_name, seasonal_order in [
            ("ARIMAX", (0, 0, 0, 0)),
            ("SARIMAX", seasonal_cfg if use_seasonal else (0, 0, 0, 0)),
        ]:
            y_pred_ret = predict_oos_iterated_exog(
                y_ret, exog_raw, oos_dates, h, order, seasonal_order, lags_cfg
            )

            current_price_col = cfg['data']['target_col']
            y_pred_level = df_oos[current_price_col].values * np.exp(y_pred_ret)

            df_out = pd.DataFrame()
            df_out['Date'] = oos_dates
            df_out['h'] = h
            df_out['model'] = model_name
            df_out['y_true_ret'] = df_oos[f'Target_Ret_h{h}'].values
            df_out['y_true_level'] = df_oos[f'Target_Price_h{h}'].values
            df_out['y_pred_ret'] = y_pred_ret
            df_out['y_pred_level'] = y_pred_level

            filename = cfg['contract']['naming']['oos'].format(h=h, model=model_name)
            df_out.to_csv(os.path.join(output_dir, filename), index=False)
            print(f"      -> Generado {model_name} h={h}")

    print("   Validando artefactos ARIMAX/SARIMAX...")
    ContractValidator().validate_all_outputs()

if __name__ == "__main__":
    run_sarimax()
