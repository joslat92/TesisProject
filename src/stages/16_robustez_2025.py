"""
Stage 16 — Bloque de robustez 2025 (decisión D3 del Registro de Decisiones).

OOS extendido: train hasta 2024-12-31, OOS 2025-01-02 a 2025-04-22 (incluye
el shock de volatilidad de abril 2025). Config propio (config_robustez2025.yaml);
el estudio principal (OOS=2024) no se toca. Sin walk-forward ni multi-semilla:
clásicos + 3 variantes LSTM con seed canónica 42.

Gate anti-fuga en dos capas antes de generar nada:
1. Suite pytest canónica (tests/).
2. Chequeo inline específico de 2025: en t=2025-02-14 se corrompen retornos y
   exógenas posteriores a t y se exige y_hat_t(h=20) idéntico para ARIMA y
   ARIMAX/SARIMAX.

Nota de datos: para h>1 el último origen evaluable es h días hábiles antes
del fin del raw (h=20 → 2025-03-24), porque el target no existe después.
Para que el forecast de los últimos orígenes tenga sus h pasos de calendario,
la serie de retornos de los clásicos se toma del RAW (hasta 2025-04-22),
recortada al primer día del parquet (lags completos).

Uso: python src/stages/16_robustez_2025.py [--tables-only]
--tables-only: recalcula métricas/mensual/DM desde las predicciones ya
generadas en outputs/preds/OOS_2025 (sin re-entrenar ni re-pronosticar).
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
from src.core.dm import diebold_mariano

CFG_PATH = "config_robustez2025.yaml"

def _load_stage(filename, module_name):
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(ROOT, "src", "stages", filename)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def contract_frame(df_block, h, model_name, y_pred_ret, cfg):
    out = pd.DataFrame()
    out['Date'] = df_block.index
    out['h'] = h
    out['model'] = model_name
    out['y_true_ret'] = df_block[f'Target_Ret_h{h}'].values
    out['y_true_level'] = df_block[f'Target_Price_h{h}'].values
    out['y_pred_ret'] = y_pred_ret
    out['y_pred_level'] = (df_block[cfg['data']['target_col']].values
                           * np.exp(y_pred_ret))
    return out

def inline_leakage_gate_2025(stage_arima, stage_sarimax, y_ret, exog_raw,
                             order, lags_cfg):
    """Corrupción del futuro en t de 2025: y_hat no debe moverse (h=20)."""
    t = pd.Timestamp("2025-02-14")
    rng = np.random.default_rng(777)
    dates = pd.DatetimeIndex([t])

    base_a = stage_arima.predict_oos_iterated(y_ret, dates, 20, order)
    base_x = stage_sarimax.predict_oos_iterated_exog(
        y_ret, exog_raw, dates, 20, order, (0, 0, 0, 0), lags_cfg)

    y_c = y_ret.copy()
    fut_y = y_c.index > t
    y_c.loc[fut_y] = rng.normal(0.0, 0.05, int(fut_y.sum()))
    x_c = exog_raw.copy()
    fut_x = x_c.index > t
    x_c.loc[fut_x] = rng.normal(50, 20, (int(fut_x.sum()), exog_raw.shape[1]))

    corr_a = stage_arima.predict_oos_iterated(y_c, dates, 20, order)
    corr_x = stage_sarimax.predict_oos_iterated_exog(
        y_c, x_c, dates, 20, order, (0, 0, 0, 0), lags_cfg)

    if not (np.array_equal(base_a, corr_a) and np.array_equal(base_x, corr_x)):
        raise RuntimeError("[Leakage GATE 2025] y_hat cambió al corromper el "
                           "futuro de 2025. Pipeline detenido.")
    print("--- GATE ANTI-FUGA INLINE 2025 (t=2025-02-14, h=20): OK ---")

def generate_predictions(cfg, val):
    """Genera las predicciones 2025 de los 7 modelos (con gates)."""
    stage_arima = _load_stage("11_train_arima.py", "stage_11_train_arima")
    stage_sarimax = _load_stage("13_train_sarimax.py", "stage_13_train_sarimax")
    stage_lstm = _load_stage("12_train_lstm.py", "stage_12_train_lstm")

    # GATE capa 1: suite anti-fuga + cordura canónica
    val.run_leakage_gate()

    horizons = cfg['features']['horizons']
    order = tuple(cfg['models']['params']['arima']['order'])
    seasonal_cfg = tuple(cfg['models']['params']['sarimax']['seasonal_order'])
    lags_cfg = cfg['features']['exog']
    lstm_params = cfg['models']['params']['lstm']
    seed = cfg['project']['seed']

    train_end = pd.Timestamp(cfg['data']['splits']['train_end'])
    oos_start = pd.Timestamp(cfg['data']['splits']['oos_start'])
    oos_end = pd.Timestamp(cfg['data']['splits']['oos_end'])

    out_dir = os.path.join(ROOT, cfg['paths']['preds_oos_dir'])
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(ROOT, cfg['paths']['preds_wf_dir']), exist_ok=True)

    raw = pd.read_csv(os.path.join(ROOT, cfg['data']['raw_source']),
                      parse_dates=['Date']).set_index('Date').sort_index()
    raw['ret_1d'] = np.log(raw[cfg['data']['target_col']]).diff()
    exog_raw = raw[list(lags_cfg.keys())]

    df_h20 = pd.read_parquet(cfg['paths']['features_parquet_pattern'].format(h=20))
    first_feat_date = df_h20['Date'].min()
    y_ret = raw['ret_1d'].loc[first_feat_date:]

    # GATE capa 2: corrupción del futuro dentro de 2025
    inline_leakage_gate_2025(stage_arima, stage_sarimax, y_ret, exog_raw,
                             order, lags_cfg)

    # Gate estacional con IS extendido (<= 2024-12-31)
    use_seasonal, p_seas = stage_sarimax.seasonality_gate(
        y_ret[y_ret.index <= train_end], m=seasonal_cfg[3])
    seasonal_order = seasonal_cfg if use_seasonal else (0, 0, 0, 0)
    print(f"   Gate estacional m={seasonal_cfg[3]} (IS<=2024): p={p_seas:.4f} -> "
          f"{'CON' if use_seasonal else 'SIN'} términos estacionales")

    for h in horizons:
        print(f"   Horizonte h={h}...")
        df = pd.read_parquet(cfg['paths']['features_parquet_pattern'].format(h=h))
        df_idx = df.set_index('Date')
        df_block = df_idx[(df_idx.index >= oos_start) & (df_idx.index <= oos_end)]
        oos_dates = df_block.index
        print(f"      OOS efectivo: {oos_dates.min():%Y-%m-%d} a "
              f"{oos_dates.max():%Y-%m-%d} ({len(oos_dates)} obs)")

        # RW
        frame = contract_frame(df_block, h, "RW", np.zeros(len(df_block)), cfg)
        frame.to_csv(os.path.join(
            out_dir, cfg['contract']['naming']['oos'].format(h=h, model="RW")),
            index=False)

        # ARIMA (iterado, re-fit mensual)
        preds = stage_arima.predict_oos_iterated(y_ret, oos_dates, h, order)
        contract_frame(df_block, h, "ARIMA", preds, cfg).to_csv(os.path.join(
            out_dir, cfg['contract']['naming']['oos'].format(h=h, model="ARIMA")),
            index=False)

        # ARIMAX / SARIMAX (exógenas congeladas en t)
        for model_name, s_order in [("ARIMAX", (0, 0, 0, 0)),
                                    ("SARIMAX", seasonal_order)]:
            preds = stage_sarimax.predict_oos_iterated_exog(
                y_ret, exog_raw, oos_dates, h, order, s_order, lags_cfg)
            contract_frame(df_block, h, model_name, preds, cfg).to_csv(
                os.path.join(out_dir, cfg['contract']['naming']['oos'].format(
                    h=h, model=model_name)), index=False)

        # Variantes LSTM (seed 42, train <= 2024-12-31)
        for model_name, feature_cols in lstm_params['variants'].items():
            df_out = stage_lstm.fit_predict(
                df, feature_cols, lstm_params, seed, h,
                cfg['data']['target_col'], train_end, oos_start, oos_end)
            df_out.insert(2, 'model', model_name)
            df_out.to_csv(os.path.join(
                out_dir, cfg['contract']['naming']['oos'].format(
                    h=h, model=model_name)), index=False)
        print(f"      7 modelos listos", flush=True)

    # Validación de contrato fail-fast sobre el set 2025
    val.validate_all_outputs(fail_fast=True)

def run_robustez(tables_only=False):
    with open(os.path.join(ROOT, CFG_PATH), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(">>> [Robustez 2025] OOS ene-abr 2025 bajo Contrato (D3)...")
    val = ContractValidator(os.path.join(ROOT, CFG_PATH))

    if tables_only:
        print("   (--tables-only: usando predicciones existentes de OOS_2025)")
    else:
        generate_predictions(cfg, val)

    horizons = cfg['features']['horizons']
    out_dir = os.path.join(ROOT, cfg['paths']['preds_oos_dir'])

    # --- Métricas, mensual y DM ---
    metrics_rows, monthly_rows, dm_rows = [], [], []
    models = cfg['models']['active_models']
    for h in horizons:
        loaded = {}
        for model in models:
            path = os.path.join(out_dir, cfg['contract']['naming']['oos'].format(
                h=h, model=model))
            loaded[model] = pd.read_csv(path, parse_dates=['Date'])

        for model, dfp in loaded.items():
            err = dfp['y_true_ret'] - dfp['y_pred_ret']
            metrics_rows.append({
                'Horizon': h, 'Model': model, 'n_obs': len(dfp),
                'RMSE': round(np.sqrt((err**2).mean()), 5),
                'MAE': round(err.abs().mean(), 5),
                'MDA': round((np.sign(dfp['y_true_ret'])
                              == np.sign(dfp['y_pred_ret'])).mean(), 4),
            })
            for month, dfm in dfp.groupby(dfp['Date'].dt.to_period('M')):
                errm = dfm['y_true_ret'] - dfm['y_pred_ret']
                monthly_rows.append({
                    'Horizon': h, 'Model': model, 'Month': str(month),
                    'n_obs': len(dfm),
                    'RMSE': round(np.sqrt((errm**2).mean()), 5),
                })

        for bench in ["RW", "SARIMAX"]:
            df_b = loaded[bench]
            for model, dfp in loaded.items():
                if model == bench:
                    continue
                common = pd.merge(dfp[['Date', 'y_true_ret', 'y_pred_ret']],
                                  df_b[['Date', 'y_pred_ret']],
                                  on='Date', suffixes=('', '_base'))
                if len(common) <= 10:
                    continue
                dm_stat, p_val, dm_hln, p_hln = diebold_mariano(
                    common['y_true_ret'].values,
                    common['y_pred_ret_base'].values,
                    common['y_pred_ret'].values, h=h)
                dm_rows.append({
                    'Horizon': h, 'Challenger': model, 'Benchmark': bench,
                    'DM_Stat': round(dm_stat, 4), 'p_value': round(p_val, 4),
                    'DM_HLN': round(dm_hln, 4), 'p_HLN': round(p_hln, 4),
                    'Significant': 'YES' if p_hln < 0.05 else 'NO'})

    df_met = pd.DataFrame(metrics_rows)
    df_met.to_csv(os.path.join(ROOT, cfg['paths']['metrics_oos']), index=False)
    df_month = pd.DataFrame(monthly_rows)
    df_month.to_csv(os.path.join(ROOT, "reports", "data",
                                 "metrics_OOS_2025_monthly.csv"), index=False)
    df_dm = pd.DataFrame(dm_rows)
    df_dm.to_csv(os.path.join(ROOT, cfg['paths']['dm_results']), index=False)

    print("\n--- METRICAS OOS 2025 (RMSE) ---")
    print(df_met.pivot(index='Horizon', columns='Model', values='RMSE'))
    print("\n--- RMSE MENSUAL h=20 ---")
    print(df_month[df_month['Horizon'] == 20].pivot(
        index='Model', columns='Month', values='RMSE'))
    print("\n--- DM 2025 vs RW (p_HLN) ---")
    sub = df_dm[df_dm['Benchmark'] == 'RW']
    print(sub.pivot(index='Challenger', columns='Horizon', values='p_HLN'))
    print("\n>>> [Robustez 2025] Completado.")

if __name__ == "__main__":
    run_robustez(tables_only="--tables-only" in sys.argv)
