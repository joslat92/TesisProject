"""
Stage 15 — Fase 3: análisis multi-semilla de las variantes LSTM (solo OOS).

Corre LSTM / LSTM_SENT / LSTM_FULL x 4 horizontes x 10 semillas
(42, 123, 7, 2024, 31, 99, 555, 1000, 8, 77) = 120 entrenamientos sobre el
OOS 2024, con la misma rutina anti-fuga del stage 12 (fit_predict: scaler
train-only, purga de frontera, early stopping con embargo).

DECISIÓN DOCUMENTADA — el walk-forward queda FUERA del multi-semilla: WF
re-entrena por bloque (x12), así que cubrirlo costaría 1.440 entrenamientos
(12x el OOS) para responder la misma pregunta (sensibilidad a la
inicialización), que el OOS multi-semilla ya contesta. El WF canónico (seed
42) queda como evidencia de estabilidad temporal; este stage aporta la
evidencia de estabilidad a la inicialización.

Salidas:
- outputs/preds/MULTISEED/preds_T{h}_{model}_seed{s}.csv (120 archivos)
- reports/data/multiseed_metrics_by_seed.csv  (métrica por semilla)
- reports/data/multiseed_summary.csv          (media/std/min/max por variante x h)
- reports/data/multiseed_dm_T20_LSTM_FULL.csv (DM vs RW por semilla + ensemble)
- reports/figs/Fig_multiseed_boxplot.png      (RMSE por variante x h, línea RW)

Criterio de lectura (bitácora): si la mediana de p-valores DM (T=20,
LSTM_FULL vs RW) < 0.05 y el RMSE medio queda bajo RW, el hallazgo es
robusto; si solo algunas semillas lo logran, se reporta como sensible a la
inicialización.
"""
import pandas as pd
import numpy as np
import yaml
import os
import sys
import importlib.util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(ROOT)

SEEDS = [42, 123, 7, 2024, 31, 99, 555, 1000, 8, 77]

def _load_stage(filename, module_name):
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(ROOT, "src", "stages", filename)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_config():
    with open(os.path.join(ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def rw_rmse_from_predictions(cfg, horizons, root=ROOT):
    """Compute the RW reference without depending on the later evaluation stage."""
    preds_dir = os.path.join(root, cfg['paths']['preds_oos_dir'])
    rmse = {}
    for h in horizons:
        filename = cfg['contract']['naming']['oos'].format(h=h, model='RW')
        path = os.path.join(preds_dir, filename)
        df = pd.read_csv(path)
        missing = {'y_true_ret', 'y_pred_ret'} - set(df.columns)
        if missing:
            raise ValueError(f"{filename} no contiene columnas requeridas: {sorted(missing)}")
        error = df['y_true_ret'] - df['y_pred_ret']
        rmse[h] = float(np.sqrt((error ** 2).mean()))
    return pd.Series(rmse, name='RMSE')

def run_trainings(cfg, stage_lstm):
    """120 entrenamientos; persiste preds por semilla y métricas por semilla."""
    horizons = cfg['features']['horizons']
    params = cfg['models']['params']['lstm']
    variants = params['variants']
    input_pattern = cfg['paths']['features_parquet_pattern']

    train_end = pd.Timestamp(cfg['data']['splits']['train_end'])
    oos_start = pd.Timestamp(cfg['data']['splits']['oos_start'])
    oos_end = pd.Timestamp(cfg['data']['splits']['oos_end'])

    out_dir = os.path.join(ROOT, "outputs", "preds", "MULTISEED")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for model_name, feature_cols in variants.items():
        for h in horizons:
            df = pd.read_parquet(input_pattern.format(h=h))
            for seed in SEEDS:
                df_out = stage_lstm.fit_predict(
                    df, feature_cols, params, seed, h,
                    cfg['data']['target_col'], train_end, oos_start, oos_end
                )
                df_out.insert(2, 'model', model_name)
                df_out['seed'] = seed
                fname = f"preds_T{h}_{model_name}_seed{seed}.csv"
                df_out.to_csv(os.path.join(out_dir, fname), index=False)

                err = df_out['y_true_ret'] - df_out['y_pred_ret']
                rows.append({
                    'Variant': model_name, 'Horizon': h, 'Seed': seed,
                    'RMSE': np.sqrt((err ** 2).mean()),
                    'MAE': err.abs().mean(),
                    'MDA': (np.sign(df_out['y_true_ret'])
                            == np.sign(df_out['y_pred_ret'])).mean(),
                })
                print(f"   {model_name} h={h} seed={seed} listo", flush=True)

    df_seeds = pd.DataFrame(rows)
    df_seeds[['RMSE', 'MAE', 'MDA']] = df_seeds[['RMSE', 'MAE', 'MDA']].round(5)
    df_seeds.to_csv(os.path.join(ROOT, "reports", "data",
                                 "multiseed_metrics_by_seed.csv"), index=False)

def run_analysis(cfg, stage_eval):
    """Resumen, DM por semilla + ensemble y figura, desde los artefactos."""
    horizons = cfg['features']['horizons']
    out_dir = os.path.join(ROOT, "outputs", "preds", "MULTISEED")
    df_seeds = pd.read_csv(os.path.join(ROOT, "reports", "data",
                                        "multiseed_metrics_by_seed.csv"))

    agg = df_seeds.groupby(['Variant', 'Horizon'])[['RMSE', 'MAE', 'MDA']].agg(
        ['mean', 'std', 'min', 'max'])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.round(5).reset_index()
    agg.to_csv(os.path.join(ROOT, "reports", "data", "multiseed_summary.csv"),
               index=False)
    print("\n--- RESUMEN MULTI-SEMILLA (RMSE mean +- std) ---")
    print(agg[['Variant', 'Horizon', 'RMSE_mean', 'RMSE_std', 'RMSE_min', 'RMSE_max']].to_string(index=False))

    # --- DM vs RW para LSTM_FULL T=20: por semilla + ensemble ---
    rw_path = os.path.join(ROOT, cfg['paths']['preds_oos_dir'],
                           cfg['contract']['naming']['oos'].format(h=20, model='RW'))
    df_rw = pd.read_csv(rw_path, parse_dates=['Date'])
    dm_rows = []
    ens = None
    for seed in SEEDS:
        dfp = pd.read_csv(os.path.join(out_dir, f"preds_T20_LSTM_FULL_seed{seed}.csv"),
                          parse_dates=['Date'])
        common = pd.merge(dfp, df_rw[['Date', 'y_pred_ret']],
                          on='Date', suffixes=('', '_rw'))
        dm_stat, p_val = stage_eval.diebold_mariano_test(
            common['y_true_ret'].values, common['y_pred_ret_rw'].values,
            common['y_pred_ret'].values, h=20)
        err = common['y_true_ret'] - common['y_pred_ret']
        dm_rows.append({'Seed': str(seed), 'RMSE': round(np.sqrt((err**2).mean()), 5),
                        'DM_Stat': round(dm_stat, 4), 'p_value': round(p_val, 4)})
        ens = common[['Date', 'y_true_ret']].copy() if ens is None else ens
        ens[f'pred_{seed}'] = common['y_pred_ret'].values

    pred_cols = [c for c in ens.columns if c.startswith('pred_')]
    ens['y_pred_ens'] = ens[pred_cols].mean(axis=1)
    common = pd.merge(ens, df_rw[['Date', 'y_pred_ret']], on='Date')
    dm_stat, p_val = stage_eval.diebold_mariano_test(
        common['y_true_ret'].values, common['y_pred_ret'].values,
        common['y_pred_ens'].values, h=20)
    err = common['y_true_ret'] - common['y_pred_ens']
    dm_rows.append({'Seed': 'ENSEMBLE', 'RMSE': round(np.sqrt((err**2).mean()), 5),
                    'DM_Stat': round(dm_stat, 4), 'p_value': round(p_val, 4)})

    df_dm = pd.DataFrame(dm_rows)
    df_dm.to_csv(os.path.join(ROOT, "reports", "data",
                              "multiseed_dm_T20_LSTM_FULL.csv"), index=False)
    print("\n--- DM vs RW (LSTM_FULL, T=20) ---")
    print(df_dm.to_string(index=False))
    p_seeds = df_dm[df_dm['Seed'] != 'ENSEMBLE']['p_value']
    print(f"\nMediana p-valor (10 semillas): {p_seeds.median():.4f} | "
          f"semillas con p<0.05: {(p_seeds < 0.05).sum()}/10")

    # --- Figura: boxplot RMSE por variante x horizonte, con línea RW ---
    horizons = cfg['features']['horizons']
    rw_rmse = rw_rmse_from_predictions(cfg, horizons)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=False)
    for ax, h in zip(axes, horizons):
        sub = df_seeds[df_seeds['Horizon'] == h]
        sns.boxplot(data=sub, x='Variant', y='RMSE', hue='Variant', ax=ax,
                    palette={'LSTM': '#d62728', 'LSTM_SENT': '#ff7f0e',
                             'LSTM_FULL': '#2ca02c'}, legend=False)
        sns.stripplot(data=sub, x='Variant', y='RMSE', ax=ax, color='black',
                      size=3, alpha=0.6)
        ax.axhline(rw_rmse.loc[h], color='gray', linestyle='--', linewidth=1.4,
                   label=f'RW ({rw_rmse.loc[h]:.4f})')
        ax.set_title(f'h={h}')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=20)
        ax.legend(fontsize=8)
    fig.suptitle('RMSE OOS por semilla (10 seeds) — variantes LSTM vs RW', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(ROOT, "reports", "figs", "Fig_multiseed_boxplot.png"),
                dpi=300)
    plt.close(fig)
    print("\n-> Fig_multiseed_boxplot.png y tablas multiseed_* generadas")

def run_multiseed(analysis_only=False):
    cfg = load_config()
    stage_eval = _load_stage("20_evaluate_stats.py", "stage_20_evaluate_stats")
    if not analysis_only:
        stage_lstm = _load_stage("12_train_lstm.py", "stage_12_train_lstm")
        print(f">>> [Fase 3] Multi-semilla LSTM: {len(SEEDS)} semillas x 3 variantes x 4 horizontes...")
        run_trainings(cfg, stage_lstm)
    run_analysis(cfg, stage_eval)

if __name__ == "__main__":
    run_multiseed(analysis_only="--analysis-only" in sys.argv)
