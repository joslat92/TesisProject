import pandas as pd
import numpy as np
import yaml
import os
import sys
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.core.contract import ContractValidator

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def predict_oos_iterated(y_ret, oos_dates, h, order, refit="M"):
    """
    Pronóstico iterado anti-fuga de ŷ_t(h) = sum(ret_{t+1..t+h}).

    Para cada fecha t de OOS, el modelo solo ve retornos diarios hasta t
    (incluido: ret_t es observable al cierre de t). Se pronostican h pasos
    con forecast(h) y se suman para obtener el retorno acumulado.

    Esquema expanding: los parámetros se re-estiman al inicio de cada mes
    (re-fit diario cuesta ~30x más y los parámetros ARMA de retornos diarios
    son estables intra-mes); entre re-fits, res.apply() re-filtra con los
    parámetros fijos usando estrictamente la muestra hasta t.
    """
    preds = []
    res = None
    fitted_month = None
    for t in oos_dates:
        y_t = y_ret.loc[:t]
        month = (t.year, t.month) if refit == "M" else None
        if res is None or month != fitted_month:
            res = ARIMA(y_t, order=order).fit()
            fitted_month = month
        else:
            res = res.apply(y_t)
        preds.append(float(res.forecast(steps=h).sum()))
    return np.array(preds)

def run_arima():
    cfg = load_config()
    print(">>> [Fase 2] Entrenando ARIMA (iterado, anti-fuga) bajo Contrato...")

    horizons = cfg['features']['horizons']
    model_name = "ARIMA"

    input_pattern = cfg['paths']['features_parquet_pattern']
    output_dir = cfg['paths']['preds_oos_dir']

    oos_start = pd.Timestamp(cfg['data']['splits']['oos_start'])
    oos_end = pd.Timestamp(cfg['data']['splits']['oos_end'])

    p, d, q = cfg['models']['params']['arima']['order']

    for h in horizons:
        print(f"   Procesando h={h}...")
        df = pd.read_parquet(input_pattern.format(h=h))
        df = df.set_index('Date')

        # Serie de retornos diarios: única entrada del modelo.
        y_ret = df['ret_1d']

        df_oos = df[(df.index >= oos_start) & (df.index <= oos_end)]
        oos_dates = df_oos.index

        y_pred_ret = predict_oos_iterated(y_ret, oos_dates, h, (p, d, q))

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

    print("   Validando artefactos ARIMA...")
    ContractValidator().validate_all_outputs()

if __name__ == "__main__":
    run_arima()
