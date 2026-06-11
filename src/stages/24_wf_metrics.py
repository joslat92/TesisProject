"""
Stage 24 — Métricas del walk-forward (contrato §8):
- metrics_WF_blocks.csv: RMSE/MAE/MDA por (Horizon, Model, Block).
- metrics_WF.csv: promedio across-blocks por (Horizon, Model) [WF_mean].
Métricas sobre retornos, mismas definiciones que 20_evaluate_stats.
"""
import pandas as pd
import numpy as np
import yaml
import os
import glob
import re

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_wf_metrics():
    cfg = load_config()
    print(">>> [Fase 3] Métricas walk-forward por bloque...")

    wf_dir = cfg['paths']['preds_wf_dir']
    files = glob.glob(os.path.join(wf_dir, "preds_T*_block*.csv"))
    if not files:
        print("[ERROR] No hay predicciones WF. Ejecuta primero 14_walkforward.")
        return

    rows = []
    pattern = re.compile(r"preds_T(\d+)_(.+)_block(\d+)\.csv$")
    for f in files:
        m = pattern.search(os.path.basename(f))
        if not m:
            continue
        h, model, b = int(m.group(1)), m.group(2), int(m.group(3))
        df = pd.read_csv(f)
        err = df['y_true_ret'] - df['y_pred_ret']
        rows.append({
            'Horizon': h,
            'Model': model,
            'Block': b,
            'n_obs': len(df),
            'RMSE': np.sqrt((err ** 2).mean()),
            'MAE': err.abs().mean(),
            'MDA': (np.sign(df['y_true_ret']) == np.sign(df['y_pred_ret'])).mean(),
        })

    df_blocks = pd.DataFrame(rows).sort_values(['Horizon', 'Model', 'Block'])
    df_blocks[['RMSE', 'MAE', 'MDA']] = df_blocks[['RMSE', 'MAE', 'MDA']].round(5)

    blocks_path = os.path.join("reports", "data", "metrics_WF_blocks.csv")
    df_blocks.to_csv(blocks_path, index=False)

    df_mean = (df_blocks.groupby(['Horizon', 'Model'])[['RMSE', 'MAE', 'MDA']]
               .mean().round(5).reset_index())
    df_mean.to_csv(cfg['paths']['metrics_wf'], index=False)

    print(f"   -> {blocks_path} ({len(df_blocks)} filas)")
    print(f"   -> {cfg['paths']['metrics_wf']} (WF_mean, {len(df_mean)} filas)")
    print(df_mean.pivot(index='Horizon', columns='Model', values='RMSE'))

if __name__ == "__main__":
    run_wf_metrics()
