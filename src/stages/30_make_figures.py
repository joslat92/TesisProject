"""
Stage 30 — Figuras canónicas del Cap. 7 (nombres de la guía de replicación):
- Fig_T{h}_bars_vs_RW.png          barras dRMSE% vs RW (OOS)
- Fig_T{h}_volcano_vs_RW.png       volcano: dRMSE% vs -log10 p(DM vs RW)
- Fig_T{h}_calibracion_scatter.png scatter de calibración MZ por modelo
- Fig_T{h}_WF_heatmap_vs_RW.png    heatmap bloques WF x modelo (dRMSE% vs RW)
- Fig_T{h}_cumloss_vs_RW.png       pérdida cuadrática acumulada vs RW (OOS)
- Fig_T{h}_dumbbell_dRMSE.png      dumbbell dRMSE% OOS vs WF_mean
- Fig_bump_ranking.png             bump de ranking RMSE OOS multi-horizonte
Una figura por archivo, DPI uniforme. Si falta una tabla fuente, se avisa y
se salta la figura (caída graciosa).
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os

sns.set_theme(style="whitegrid")
DPI = 300

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _load_csv(path):
    if not os.path.exists(path):
        print(f"   [WARN] Falta {path}; figuras dependientes se saltan.")
        return None
    return pd.read_csv(path)

def model_palette(models):
    base = {'RW': '#7f7f7f', 'ARIMA': '#1f77b4', 'ARIMAX': '#17becf',
            'SARIMAX': '#9467bd', 'LSTM': '#d62728', 'LSTM_SENT': '#ff7f0e',
            'LSTM_FULL': '#2ca02c'}
    return {m: base.get(m, '#8c564b') for m in models}

def delta_rmse_vs_rw(df_met):
    """dRMSE% = (RMSE_model / RMSE_RW - 1) * 100 por (Horizon, Model)."""
    rw = df_met[df_met['Model'] == 'RW'].set_index('Horizon')['RMSE']
    out = df_met[df_met['Model'] != 'RW'].copy()
    out['dRMSE_pct'] = out.apply(
        lambda r: (r['RMSE'] / rw.loc[r['Horizon']] - 1) * 100, axis=1)
    return out

def fig_bars_vs_rw(df_met, h, figs_dir, palette):
    d = delta_rmse_vs_rw(df_met)
    d = d[d['Horizon'] == h].sort_values('dRMSE_pct')
    if d.empty:
        return
    plt.figure(figsize=(9, 5))
    colors = [palette[m] for m in d['Model']]
    plt.barh(d['Model'], d['dRMSE_pct'], color=colors)
    plt.axvline(0, color='black', linewidth=1)
    plt.xlabel('dRMSE vs RW (%)  [negativo = mejor que RW]')
    plt.title(f'dRMSE vs RW — OOS 2024 (h={h})')
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, f"Fig_T{h}_bars_vs_RW.png"), dpi=DPI)
    plt.close()

def fig_volcano(df_met, df_dm, h, figs_dir, palette):
    if df_dm is None:
        return
    d = delta_rmse_vs_rw(df_met)
    d = d[d['Horizon'] == h]
    dm = df_dm[(df_dm['Horizon'] == h) & (df_dm['Benchmark'] == 'RW')]
    merged = pd.merge(d, dm, left_on=['Horizon', 'Model'],
                      right_on=['Horizon', 'Challenger'])
    if merged.empty:
        return
    merged['neglog_p'] = -np.log10(merged['p_value'].clip(lower=1e-6))
    plt.figure(figsize=(8, 6))
    for _, r in merged.iterrows():
        plt.scatter(r['dRMSE_pct'], r['neglog_p'], s=90,
                    color=palette[r['Model']], zorder=3)
        plt.annotate(r['Model'], (r['dRMSE_pct'], r['neglog_p']),
                     textcoords="offset points", xytext=(6, 5), fontsize=9)
    plt.axhline(-np.log10(0.05), color='red', linestyle='--', linewidth=0.9,
                label='p=0.05')
    plt.axhline(-np.log10(0.10), color='orange', linestyle=':', linewidth=0.9,
                label='p=0.10')
    plt.axvline(0, color='black', linewidth=1)
    plt.xlabel('dRMSE vs RW (%)  [negativo = mejor]')
    plt.ylabel('-log10 p (DM vs RW, HAC L=h-1)')
    plt.title(f'Volcano DM vs RW — OOS 2024 (h={h})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, f"Fig_T{h}_volcano_vs_RW.png"), dpi=DPI)
    plt.close()

def fig_calibracion_scatter(cfg, h, figs_dir, models, df_mz):
    preds_dir = cfg['paths']['preds_oos_dir']
    models_plot = [m for m in models if m != 'RW']
    n = len(models_plot)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    plotted = False
    for ax, model in zip(axes, models_plot):
        fpath = os.path.join(preds_dir, cfg['contract']['naming']['oos'].format(h=h, model=model))
        if not os.path.exists(fpath):
            ax.axis('off')
            continue
        df = pd.read_csv(fpath)
        ax.scatter(df['y_true_level'], df['y_pred_level'], alpha=0.45, s=14,
                   color='purple')
        low = min(df['y_true_level'].min(), df['y_pred_level'].min())
        high = max(df['y_true_level'].max(), df['y_pred_level'].max())
        margin = (high - low) * 0.05
        ax.plot([low, high], [low, high], color='gray', linestyle='--', linewidth=0.9)
        ax.set_xlim(low - margin, high + margin)
        ax.set_ylim(low - margin, high + margin)
        title = model
        if df_mz is not None:
            row = df_mz[(df_mz['Horizon'] == h) & (df_mz['Model'] == model)]
            if not row.empty:
                title += f"  (alpha={row['Alpha'].iloc[0]:.3f}, beta={row['Beta'].iloc[0]:.2f})"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('nivel real'); ax.set_ylabel('nivel pronosticado')
        plotted = True
    for ax in axes[n:]:
        ax.axis('off')
    if not plotted:
        plt.close(fig)
        return
    fig.suptitle(f'Calibración (MZ) — OOS 2024 (h={h})', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(figs_dir, f"Fig_T{h}_calibracion_scatter.png"), dpi=DPI)
    plt.close(fig)

def fig_wf_heatmap(df_wf_blocks, h, figs_dir):
    if df_wf_blocks is None:
        return
    d = df_wf_blocks[df_wf_blocks['Horizon'] == h]
    if d.empty:
        return
    rw = d[d['Model'] == 'RW'].set_index('Block')['RMSE']
    d = d[d['Model'] != 'RW'].copy()
    d['dRMSE_pct'] = d.apply(lambda r: (r['RMSE'] / rw.loc[r['Block']] - 1) * 100, axis=1)
    mat = d.pivot(index='Model', columns='Block', values='dRMSE_pct')
    plt.figure(figsize=(11, 4.5))
    vmax = np.nanmax(np.abs(mat.values))
    sns.heatmap(mat, cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax,
                annot=True, fmt='.0f', annot_kws={'size': 7},
                cbar_kws={'label': 'dRMSE vs RW (%)'})
    plt.title(f'Walk-forward 2024 por bloque — dRMSE vs RW (%) (h={h})')
    plt.xlabel('Bloque (mes 2024)')
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, f"Fig_T{h}_WF_heatmap_vs_RW.png"), dpi=DPI)
    plt.close()

def fig_cumloss(cfg, h, figs_dir, models, palette):
    preds_dir = cfg['paths']['preds_oos_dir']
    rw_path = os.path.join(preds_dir, cfg['contract']['naming']['oos'].format(h=h, model='RW'))
    if not os.path.exists(rw_path):
        return
    df_rw = pd.read_csv(rw_path, parse_dates=['Date'])
    e2_rw = (df_rw['y_true_ret'] - df_rw['y_pred_ret']) ** 2
    plt.figure(figsize=(11, 5.5))
    for model in models:
        if model == 'RW':
            continue
        fpath = os.path.join(preds_dir, cfg['contract']['naming']['oos'].format(h=h, model=model))
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath, parse_dates=['Date'])
        e2 = (df['y_true_ret'] - df['y_pred_ret']) ** 2
        cum = (e2.values - e2_rw.values).cumsum()
        plt.plot(df['Date'], cum, label=model, color=palette[model], linewidth=1.3)
    plt.axhline(0, color='black', linewidth=1)
    plt.ylabel('Suma acumulada de (e2_modelo - e2_RW)')
    plt.xlabel('Fecha (OOS 2024)')
    plt.title(f'Pérdida acumulada vs RW (h={h})  [debajo de 0 = mejor que RW]')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, f"Fig_T{h}_cumloss_vs_RW.png"), dpi=DPI)
    plt.close()

def fig_dumbbell(df_met, df_wf_mean, h, figs_dir, palette):
    if df_wf_mean is None:
        return
    d_oos = delta_rmse_vs_rw(df_met)
    d_oos = d_oos[d_oos['Horizon'] == h][['Model', 'dRMSE_pct']].rename(
        columns={'dRMSE_pct': 'OOS'})
    d_wf = delta_rmse_vs_rw(df_wf_mean)
    d_wf = d_wf[d_wf['Horizon'] == h][['Model', 'dRMSE_pct']].rename(
        columns={'dRMSE_pct': 'WF_mean'})
    merged = pd.merge(d_oos, d_wf, on='Model').sort_values('OOS')
    if merged.empty:
        return
    plt.figure(figsize=(9, 5))
    for i, r in enumerate(merged.itertuples()):
        plt.plot([r.OOS, r.WF_mean], [i, i], color='gray', linewidth=1.4, zorder=1)
        plt.scatter(r.OOS, i, color=palette[r.Model], s=80, zorder=3, marker='o')
        plt.scatter(r.WF_mean, i, color=palette[r.Model], s=80, zorder=3, marker='s')
    plt.yticks(range(len(merged)), merged['Model'])
    plt.axvline(0, color='black', linewidth=1)
    plt.xlabel('dRMSE vs RW (%)   o = OOS,  s = WF_mean')
    plt.title(f'dRMSE vs RW: OOS vs walk-forward (h={h})')
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, f"Fig_T{h}_dumbbell_dRMSE.png"), dpi=DPI)
    plt.close()

def fig_bump_ranking(df_met, figs_dir, palette, horizons):
    d = df_met.copy()
    d['rank'] = d.groupby('Horizon')['RMSE'].rank(method='min')
    plt.figure(figsize=(9, 5.5))
    for model in d['Model'].unique():
        sub = d[d['Model'] == model].sort_values('Horizon')
        plt.plot(sub['Horizon'], sub['rank'], marker='o', label=model,
                 color=palette[model], linewidth=1.6)
    plt.gca().invert_yaxis()
    plt.xticks(horizons)
    plt.yticks(range(1, d['Model'].nunique() + 1))
    plt.xlabel('Horizonte h')
    plt.ylabel('Ranking RMSE OOS (1 = mejor)')
    plt.title('Bump de ranking por horizonte — OOS 2024')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "Fig_bump_ranking.png"), dpi=DPI)
    plt.close()

def run_figures():
    cfg = load_config()
    print(">>> [Fase 4] Generando figuras canónicas...")

    figs_dir = cfg['paths']['figures_dir']
    os.makedirs(figs_dir, exist_ok=True)

    horizons = cfg['features']['horizons']
    models = cfg['models']['active_models']
    palette = model_palette(models)

    df_met = _load_csv(cfg['paths']['metrics_oos'])
    if df_met is None:
        print("[ERROR] Sin metrics_OOS.csv no hay figuras. Corre 20_evaluate_stats.")
        return
    df_dm = _load_csv(os.path.join("reports", "data", "tbl_DM_OOS.csv"))
    df_mz = _load_csv(cfg['paths']['mz_results'])
    df_wf_blocks = _load_csv(os.path.join("reports", "data", "metrics_WF_blocks.csv"))
    df_wf_mean = _load_csv(cfg['paths']['metrics_wf'])

    for h in horizons:
        fig_bars_vs_rw(df_met, h, figs_dir, palette)
        fig_volcano(df_met, df_dm, h, figs_dir, palette)
        fig_calibracion_scatter(cfg, h, figs_dir, models, df_mz)
        fig_wf_heatmap(df_wf_blocks, h, figs_dir)
        fig_cumloss(cfg, h, figs_dir, models, palette)
        fig_dumbbell(df_met, df_wf_mean, h, figs_dir, palette)
        print(f"   -> Figuras h={h} listas")

    fig_bump_ranking(df_met, figs_dir, palette, horizons)
    print(f"   -> Fig_bump_ranking lista")
    print(f">>> [Fase 4] Figuras en {figs_dir}")

if __name__ == "__main__":
    run_figures()
