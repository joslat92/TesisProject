import pandas as pd
import numpy as np
import yaml
import os
import sys
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.stats.multitest import multipletests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.core.dm import diebold_mariano
from src.core.metrics import pesaran_timmermann

# Asegurar encoding UTF-8 siempre
def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def calculate_mda(y_true, y_pred):
    """Mean Directional Accuracy: % de acierto en el signo."""
    return np.mean(np.sign(y_true) == np.sign(y_pred))

def diebold_mariano_test(y_true, y_pred_base, y_pred_chal, h):
    """
    Wrapper de compatibilidad (lo usa 15_multiseed_lstm): devuelve el par
    HLN (estadístico corregido, p-valor t(n−1)) de src/core/dm.py — la misma
    convención que publicó RC1.
    """
    _, _, dm_hln, p_hln = diebold_mariano(y_true, y_pred_base, y_pred_chal, h=h)
    return dm_hln, p_hln

def mincer_zarnowitz_test(y_true, y_pred, h=1):
    """
    Regresión MZ: y_true = alpha + beta * y_pred + error.
    Devuelve alpha, beta, R2, pruebas marginales y el Wald HAC conjunto
    H0: (alpha, beta) = (0, 1).

    Errores HAC con maxlags = h-1 (coherente con el kernel del DM): los
    retornos acumulados solapados inducen autocorrelación MA(h-1) en el
    residuo MZ, así que el lag de truncamiento debe escalar con el
    horizonte. Para h=1 (sin solape) maxlags=0 ⇒ HAC se reduce a robustez
    de heterocedasticidad. Antes se usaba maxlags=1 fijo, lo que subestimaba
    la incertidumbre en horizontes largos (decisión D7, 2026-06-16).
    """
    y_true = pd.Series(y_true, dtype=float).reset_index(drop=True)
    y_pred = pd.Series(y_pred, dtype=float).reset_index(drop=True)
    if len(y_true) != len(y_pred) or len(y_true) < 3:
        raise ValueError("MZ requiere vectores de igual longitud y al menos 3 observaciones.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("MZ no admite NaN o infinitos.")
    if y_pred.nunique() < 2:
        raise ValueError("MZ requiere variación en el pronóstico.")

    X = pd.DataFrame({'const': 1.0, 'forecast': y_pred})
    model = sm.OLS(y_true, X)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': max(h - 1, 0)})

    alpha = results.params.iloc[0]
    beta = results.params.iloc[1]
    r2 = results.rsquared
    p_alpha = results.pvalues.iloc[0]
    p_beta0 = results.pvalues.iloc[1]
    p_beta1 = float(results.t_test('forecast = 1').pvalue)
    p_joint = float(results.wald_test('const = 0, forecast = 1', scalar=True).pvalue)

    return alpha, beta, r2, p_alpha, p_beta0, p_beta1, p_joint

def run_evaluation():
    cfg = load_config()
    print(">>> [Fase 3] Calculando Estadísticas Consolidada (Métricas + DM + MZ)...")

    # GATE del pipeline (contrato §11): estructura + contrato + anti-fuga.
    # Cualquier violación detiene la evaluación.
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.core.contract import ContractValidator
    ContractValidator().run_full_gate()

    horizons = cfg['features']['horizons']
    models = cfg['models']['active_models']

    preds_dir = cfg['paths']['preds_oos_dir']

    metrics_data = []
    dm_data = []
    mz_data = []
    pt_data = []

    # Contrato: DM contra RW y contra SARIMAX
    benchmarks = ["RW", "SARIMAX"]

    for h in horizons:
        # Cargar predicciones de los benchmarks para este horizonte
        df_bench = {}
        for bench in benchmarks:
            bfile = os.path.join(preds_dir, cfg['contract']['naming']['oos'].format(h=h, model=bench))
            if os.path.exists(bfile):
                df_bench[bench] = pd.read_csv(bfile)

        for model in models:
            filename = cfg['contract']['naming']['oos'].format(h=h, model=model)
            filepath = os.path.join(preds_dir, filename)

            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Falta predicción requerida: {filepath}")

            df = pd.read_csv(filepath)

            # --- 1. MÉTRICAS (Sobre Retornos) ---
            rmse = np.sqrt(mean_squared_error(df['y_true_ret'], df['y_pred_ret']))
            mae = mean_absolute_error(df['y_true_ret'], df['y_pred_ret'])
            mda = calculate_mda(df['y_true_ret'], df['y_pred_ret'])

            metrics_data.append({
                'Horizon': h,
                'Model': model,
                'RMSE': round(rmse, 5),
                'MAE': round(mae, 5),
                'MDA': round(mda, 4)
            })

            # --- 2. DIEBOLD-MARIANO (vs RW y vs SARIMAX) ---
            # DM_Stat/p_value: DM clásico (normal). DM_HLN/p_HLN: corrección
            # de muestra pequeña Harvey-Leybourne-Newbold (factor √k + t(n−1)).
            # El veredicto (Significant) se toma del p_HLN, el conservador.
            for bench, df_b in df_bench.items():
                if model == bench:
                    continue
                common = pd.merge(df[['Date', 'y_true_ret', 'y_pred_ret']],
                                  df_b[['Date', 'y_pred_ret']],
                                  on='Date', suffixes=('', '_base'))

                if len(common) > 10:
                    dm_stat, p_val, dm_hln, p_hln = diebold_mariano(
                        common['y_true_ret'].values,
                        common['y_pred_ret_base'].values,
                        common['y_pred_ret'].values,
                        h=h
                    )

                    dm_data.append({
                        'Horizon': h,
                        'Challenger': model,
                        'Benchmark': bench,
                        'DM_Stat': round(dm_stat, 4),
                        'p_value': round(p_val, 4),
                        'DM_HLN': round(dm_hln, 4),
                        'p_HLN': round(p_hln, 4),
                        'Significant': 'YES' if p_hln < 0.05 else 'NO'
                    })

            # --- 2b. PESARAN-TIMMERMANN (direccional; indicativo en h>1) ---
            if model != 'RW':
                pt_stat, pt_p, p_hat, p_star = pesaran_timmermann(
                    df['y_true_ret'].values, df['y_pred_ret'].values)
                pt_data.append({
                    'Horizon': h,
                    'Model': model,
                    'PT_Stat': round(pt_stat, 4) if np.isfinite(pt_stat) else np.nan,
                    'p_value': round(pt_p, 4) if np.isfinite(pt_p) else np.nan,
                    'HitRate': round(p_hat, 4),
                    'HitRate_H0': round(p_star, 4),
                })

            # --- 3. MINCER-ZARNOWITZ (Sobre Niveles) ---
            alpha, beta, r2, p_a, p_b0, p_b1, p_joint = mincer_zarnowitz_test(
                df['y_true_level'], df['y_pred_level'], h=h)
            mz_data.append({
                'Horizon': h,
                'Model': model,
                'Alpha': round(alpha, 4),
                'Beta': round(beta, 4),
                'R2': round(r2, 4),
                'p_Alpha': round(p_a, 4),   # H0: alpha=0
                'p_Beta1': round(p_b1, 4),  # H0: beta=1
                'p_Joint': round(p_joint, 4),  # H0: (alpha,beta)=(0,1)
            })

    # Guardar Reportes
    output_dir = "reports/data"
    os.makedirs(output_dir, exist_ok=True)

    # Metrics
    if metrics_data:
        df_met = pd.DataFrame(metrics_data)
        df_met.to_csv(cfg['paths']['metrics_oos'], index=False)
        print("\n--- RESUMEN MÉTRICAS (OOS) ---")
        print(df_met.pivot(index='Horizon', columns='Model', values=['RMSE', 'MDA']))

    # DM: tabla completa (tbl_DM_OOS) + resumen compacto (dm_summary, p_HLN)
    if dm_data:
        df_dm = pd.DataFrame(dm_data)
        df_dm.to_csv(os.path.join(output_dir, "tbl_DM_OOS.csv"), index=False)

        # Familia primaria preespecificada: cinco modelos únicos vs RW por
        # cuatro horizontes. SARIMAX se excluye porque es idéntico a ARIMAX.
        primary_models = ['ARIMA', 'ARIMAX', 'LSTM', 'LSTM_SENT', 'LSTM_FULL']
        df_primary = df_dm[
            (df_dm['Benchmark'] == 'RW') &
            (df_dm['Challenger'].isin(primary_models))
        ].copy()
        df_primary['p_Holm'] = multipletests(df_primary['p_HLN'], method='holm')[1]
        df_primary['p_BH'] = multipletests(df_primary['p_HLN'], method='fdr_bh')[1]
        df_primary['Sig_Holm_5pct'] = np.where(df_primary['p_Holm'] < 0.05, 'YES', 'NO')
        df_primary['Sig_BH_5pct'] = np.where(df_primary['p_BH'] < 0.05, 'YES', 'NO')
        df_primary[['p_Holm', 'p_BH']] = df_primary[['p_Holm', 'p_BH']].round(4)
        df_primary.to_csv(os.path.join(output_dir, "tbl_DM_primary_adjusted.csv"), index=False)

        # Contrastes que responden directamente al aporte incremental de las
        # exógenas, con corrección conjunta sobre las 16 ablaciones.
        ablation_pairs = [
            ('ARIMA', 'ARIMAX'),
            ('LSTM', 'LSTM_SENT'),
            ('LSTM', 'LSTM_FULL'),
            ('LSTM_SENT', 'LSTM_FULL'),
        ]
        ablation_rows = []
        for h in horizons:
            for benchmark, challenger in ablation_pairs:
                base = pd.read_csv(os.path.join(
                    preds_dir, cfg['contract']['naming']['oos'].format(h=h, model=benchmark)))
                chal = pd.read_csv(os.path.join(
                    preds_dir, cfg['contract']['naming']['oos'].format(h=h, model=challenger)))
                common = pd.merge(
                    chal[['Date', 'y_true_ret', 'y_pred_ret']],
                    base[['Date', 'y_pred_ret']],
                    on='Date', suffixes=('_chall', '_base'))
                _, _, dm_hln, p_hln = diebold_mariano(
                    common['y_true_ret'], common['y_pred_ret_base'],
                    common['y_pred_ret_chall'], h=h)
                ablation_rows.append({
                    'Horizon': h,
                    'Benchmark': benchmark,
                    'Challenger': challenger,
                    'DM_HLN': dm_hln,
                    'p_HLN': p_hln,
                })
        df_ablation = pd.DataFrame(ablation_rows)
        df_ablation['p_Holm'] = multipletests(df_ablation['p_HLN'], method='holm')[1]
        df_ablation['p_BH'] = multipletests(df_ablation['p_HLN'], method='fdr_bh')[1]
        for col in ['DM_HLN', 'p_HLN', 'p_Holm', 'p_BH']:
            df_ablation[col] = df_ablation[col].round(4)
        df_ablation.to_csv(os.path.join(output_dir, "tbl_DM_ablations.csv"), index=False)
        dm_compact = df_dm.pivot_table(index=['Horizon', 'Challenger'],
                                       columns='Benchmark', values='p_HLN').reset_index()
        dm_compact.columns = ['Horizon', 'Challenger'] + [f"pHLN_vs_{c}" for c in dm_compact.columns[2:]]
        dm_compact.to_csv(cfg['paths']['dm_results'], index=False)
        print("\n--- RESUMEN DIEBOLD-MARIANO (p-values HLN) ---")
        print(dm_compact.to_string(index=False))

    # PT: acierto direccional (indicativo en h>1 por solape de ventanas)
    if pt_data:
        df_pt = pd.DataFrame(pt_data)
        df_pt.to_csv(os.path.join(output_dir, "tbl_PT_OOS.csv"), index=False)
        print("\n--- PESARAN-TIMMERMANN (direccional; h>1 indicativo) ---")
        print(df_pt.to_string(index=False))

    # MZ: core + resumen
    if mz_data:
        df_mz = pd.DataFrame(mz_data)
        df_mz.to_csv(os.path.join(output_dir, "tbl_MZ_core.csv"), index=False)
        df_mz.to_csv(cfg['paths']['mz_results'], index=False)

    print(f"\n>>> [Fase 3] Completada. Reportes guardados en {output_dir}")

if __name__ == "__main__":
    run_evaluation()
