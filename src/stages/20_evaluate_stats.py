import pandas as pd
import numpy as np
import yaml
import os
import sys
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Asegurar encoding UTF-8 siempre
def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def calculate_mda(y_true, y_pred):
    """Mean Directional Accuracy: % de acierto en el signo."""
    return np.mean(np.sign(y_true) == np.sign(y_pred))

def diebold_mariano_test(y_true, y_pred_base, y_pred_chal, h):
    """
    Test DM con corrección HAC (Harvey, Leybourne, Newbold).
    H0: Ambos modelos tienen el mismo error.
    """
    e1 = (y_true - y_pred_base)**2
    e2 = (y_true - y_pred_chal)**2
    d = e1 - e2
    
    T = len(d)
    mean_d = np.mean(d)
    
    # Autocovarianza para HAC (Newey-West con lag = h-1)
    # Usamos statsmodels para consistencia
    # O cálculo manual robusto:
    gamma_0 = np.var(d)
    gamma_sum = 0
    for lag in range(1, h):
        gamma_sum += np.cov(d[lag:], d[:-lag])[0][1]
        
    var_d = (gamma_0 + 2 * gamma_sum) / T
    
    if var_d <= 1e-16: return 0.0, 1.0 # Varianza cero -> p-value 1
    
    dm_stat = mean_d / np.sqrt(var_d)
    
    # Corrección HLN para muestras finitas
    k = (T + 1 - 2*h + h*(h-1)/T) / T
    dm_stat_adj = dm_stat * np.sqrt(k)
    
    # p-value (dos colas)
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat_adj), df=T-1))
    
    return dm_stat_adj, p_value

def mincer_zarnowitz_test(y_true, y_pred):
    """
    Regresión MZ: y_true = alpha + beta * y_pred + error.
    Devuelve alpha, beta, R2, p(alpha=0), p(beta=0) y p(beta=1) — el test
    de insesgadez relevante es la pareja H0: alpha=0 y H0: beta=1.
    """
    X = sm.add_constant(y_pred)
    model = sm.OLS(y_true, X)
    try:
        results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 1})

        alpha = results.params.iloc[0]
        beta = results.params.iloc[1] if len(results.params) > 1 else np.nan
        r2 = results.rsquared
        p_alpha = results.pvalues.iloc[0]
        p_beta0 = results.pvalues.iloc[1] if len(results.pvalues) > 1 else np.nan
        # H0: beta = 1 (t robusto HAC)
        se_beta = results.bse.iloc[1] if len(results.bse) > 1 else np.nan
        if np.isfinite(se_beta) and se_beta > 0:
            t_b1 = (beta - 1.0) / se_beta
            p_beta1 = 2 * (1 - stats.t.cdf(np.abs(t_b1), df=results.df_resid))
        else:
            p_beta1 = np.nan
    except Exception:
        return (np.nan,) * 6

    return alpha, beta, r2, p_alpha, p_beta0, p_beta1

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
                continue

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
            for bench, df_b in df_bench.items():
                if model == bench:
                    continue
                common = pd.merge(df[['Date', 'y_true_ret', 'y_pred_ret']],
                                  df_b[['Date', 'y_pred_ret']],
                                  on='Date', suffixes=('', '_base'))

                if len(common) > 10:
                    dm_stat, p_val = diebold_mariano_test(
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
                        'Significant': 'YES' if p_val < 0.05 else 'NO'
                    })

            # --- 3. MINCER-ZARNOWITZ (Sobre Niveles) ---
            alpha, beta, r2, p_a, p_b0, p_b1 = mincer_zarnowitz_test(
                df['y_true_level'], df['y_pred_level'])
            mz_data.append({
                'Horizon': h,
                'Model': model,
                'Alpha': round(alpha, 4),
                'Beta': round(beta, 4),
                'R2': round(r2, 4),
                'p_Alpha': round(p_a, 4),   # H0: alpha=0
                'p_Beta1': round(p_b1, 4),  # H0: beta=1
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

    # DM: tabla completa (tbl_DM_OOS) + resumen compacto (dm_summary)
    if dm_data:
        df_dm = pd.DataFrame(dm_data)
        df_dm.to_csv(os.path.join(output_dir, "tbl_DM_OOS.csv"), index=False)
        dm_compact = df_dm.pivot_table(index=['Horizon', 'Challenger'],
                                       columns='Benchmark', values='p_value').reset_index()
        dm_compact.columns = ['Horizon', 'Challenger'] + [f"p_vs_{c}" for c in dm_compact.columns[2:]]
        dm_compact.to_csv(cfg['paths']['dm_results'], index=False)
        print("\n--- RESUMEN DIEBOLD-MARIANO (p-values) ---")
        print(dm_compact.to_string(index=False))

    # MZ: core + resumen
    if mz_data:
        df_mz = pd.DataFrame(mz_data)
        df_mz.to_csv(os.path.join(output_dir, "tbl_MZ_core.csv"), index=False)
        df_mz.to_csv(cfg['paths']['mz_results'], index=False)

    print(f"\n>>> [Fase 3] Completada. Reportes guardados en {output_dir}")

if __name__ == "__main__":
    run_evaluation()
