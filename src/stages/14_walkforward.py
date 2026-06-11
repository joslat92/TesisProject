"""
Stage 14 — Walk-forward 2024 (12 bloques mensuales) para los 7 modelos
efectivos: RW, ARIMA, ARIMAX, SARIMAX, LSTM, LSTM_SENT, LSTM_FULL.

Contrato (contrato.docx §8):
- Bloque b = mes b de 2024. Parámetros/scalers/HPs SOLO con IS + bloques
  previos (fit_upto / train_end = último día hábil ANTERIOR al bloque).
- Dentro del bloque, los clásicos re-filtran diariamente con información <= t
  (sin re-estimar parámetros); LSTM se re-entrena por bloque (scaler por
  bloque, purga de frontera y early stopping con embargo, ver stage 12).
- Salida: preds_T{h}_{model}_block{b}.csv con columna block.
"""
import pandas as pd
import numpy as np
import yaml
import os
import sys
import importlib.util
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(ROOT)
from src.core.contract import ContractValidator

def _load_stage(filename, module_name):
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(ROOT, "src", "stages", filename)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

stage_arima = _load_stage("11_train_arima.py", "stage_11_train_arima")
stage_sarimax = _load_stage("13_train_sarimax.py", "stage_13_train_sarimax")
stage_lstm = _load_stage("12_train_lstm.py", "stage_12_train_lstm")

def load_config():
    with open(os.path.join(ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_block(df_out, model_name, h, b, cfg):
    df_out = df_out.copy()
    df_out['block'] = b
    filename = cfg['contract']['naming']['wf'].format(h=h, model=model_name, b=b)
    df_out.to_csv(os.path.join(cfg['paths']['preds_wf_dir'], filename), index=False)

def contract_frame(df_oos_block, h, model_name, y_pred_ret, cfg):
    out = pd.DataFrame()
    out['Date'] = df_oos_block.index
    out['h'] = h
    out['model'] = model_name
    out['y_true_ret'] = df_oos_block[f'Target_Ret_h{h}'].values
    out['y_true_level'] = df_oos_block[f'Target_Price_h{h}'].values
    out['y_pred_ret'] = y_pred_ret
    out['y_pred_level'] = (df_oos_block[cfg['data']['target_col']].values
                           * np.exp(y_pred_ret))
    return out

def run_walkforward():
    cfg = load_config()
    print(">>> [Fase 2b] Walk-forward 2024 (12 bloques) bajo Contrato...")

    horizons = cfg['features']['horizons']
    input_pattern = cfg['paths']['features_parquet_pattern']
    os.makedirs(cfg['paths']['preds_wf_dir'], exist_ok=True)

    wf_year = cfg['data']['splits']['wf_year']
    n_blocks = cfg['data']['splits']['wf_blocks']
    train_end_is = pd.Timestamp(cfg['data']['splits']['train_end'])

    order = tuple(cfg['models']['params']['arima']['order'])
    seasonal_cfg = tuple(cfg['models']['params']['sarimax']['seasonal_order'])
    lags_cfg = cfg['features']['exog']
    lstm_params = cfg['models']['params']['lstm']
    seed = cfg['project']['seed']

    raw = pd.read_csv(cfg['data']['raw_source'], parse_dates=['Date']).set_index('Date')
    exog_raw = raw[list(lags_cfg.keys())]

    month_starts = pd.date_range(start=f'{wf_year}-01-01', periods=n_blocks, freq='MS')

    for h in horizons:
        print(f"   Horizonte h={h}...")
        df = pd.read_parquet(input_pattern.format(h=h))
        df_idx = df.set_index('Date')
        y_ret = df_idx['ret_1d']

        # Gate estacional: decidido SOLO con IS (mismo criterio que stage 13),
        # válido para todos los bloques (HPs nunca usan el bloque corriente).
        use_seasonal, p_seas = stage_sarimax.seasonality_gate(
            y_ret[y_ret.index <= train_end_is], m=seasonal_cfg[3]
        )
        seasonal_order = seasonal_cfg if use_seasonal else (0, 0, 0, 0)
        print(f"      Gate estacional m={seasonal_cfg[3]}: p={p_seas:.4f} -> "
              f"{'CON' if use_seasonal else 'SIN'} términos estacionales")

        for b, month_start in enumerate(month_starts, start=1):
            month_end = month_start + pd.offsets.MonthEnd(1)
            block_mask = (df_idx.index >= month_start) & (df_idx.index <= month_end)
            df_block = df_idx[block_mask]
            if df_block.empty:
                print(f"      [WARN] Bloque {b} vacío. Saltando.")
                continue
            block_dates = df_block.index

            # Último día hábil ANTERIOR al bloque: límite de estimación
            fit_upto = y_ret.index[y_ret.index < month_start].max()

            # --- RW ---
            zeros = np.zeros(len(df_block))
            save_block(contract_frame(df_block, h, "RW", zeros, cfg), "RW", h, b, cfg)

            # --- ARIMA ---
            preds = stage_arima.predict_oos_iterated(
                y_ret, block_dates, h, order, fit_upto=fit_upto
            )
            save_block(contract_frame(df_block, h, "ARIMA", preds, cfg), "ARIMA", h, b, cfg)

            # --- ARIMAX / SARIMAX ---
            for model_name, s_order in [("ARIMAX", (0, 0, 0, 0)),
                                        ("SARIMAX", seasonal_order)]:
                preds = stage_sarimax.predict_oos_iterated_exog(
                    y_ret, exog_raw, block_dates, h, order, s_order,
                    lags_cfg, fit_upto=fit_upto
                )
                save_block(contract_frame(df_block, h, model_name, preds, cfg),
                           model_name, h, b, cfg)

            # --- Variantes LSTM (re-entrenadas por bloque) ---
            for model_name, feature_cols in lstm_params['variants'].items():
                df_out = stage_lstm.fit_predict(
                    df, feature_cols, lstm_params, seed, h,
                    cfg['data']['target_col'],
                    train_end=fit_upto,
                    pred_start=month_start, pred_end=month_end
                )
                df_out.insert(2, 'model', model_name)
                save_block(df_out, model_name, h, b, cfg)

            print(f"      Bloque {b:02d} ({month_start:%Y-%m}) listo "
                  f"({len(df_block)} fechas x 7 modelos)")

    print("   Validando artefactos WF...")
    ContractValidator().validate_all_outputs()

if __name__ == "__main__":
    run_walkforward()
