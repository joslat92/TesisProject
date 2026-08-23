"""
Stage 31 — Regímenes de volatilidad (contrato §9.4).

Genera reports/data/regimes_summary.csv con columnas:
    model, h, regime ∈ {Q1,Q2,Q3,Q4}, n_obs, rmse_ret, mda

Estratifica el OOS 2024 por CUARTIL de volatilidad, definida como el VIX_Close
en la fecha de origen t (observable al cierre de t; es una partición de
EVALUACIÓN, no entra en ningún modelo ⇒ sin fuga). Los umbrales de cuartil se
calculan sobre las fechas de origen del OOS para cada horizonte, y se aplican
idénticos a todos los modelos. Q1 = volatilidad más baja (calma), Q4 = más alta
(estrés). rmse_ret sobre retornos; mda por coincidencia de signo (convención
del proyecto: sign(0) del RW no puntúa ⇒ MDA(RW)=0).

La definicion del regimen (proxy = VIX contemporaneo en t, cuartiles de la
muestra OOS) se usa solo como analisis descriptivo de robustez. No selecciona
modelos, no interviene en el entrenamiento y no sustenta por si sola inferencia
causal ni confirmatoria.
"""
import pandas as pd
import numpy as np
import yaml
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

def load_config():
    with open(os.path.join(ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_regimes():
    cfg = load_config()
    print(">>> [Fase 4] Regímenes de volatilidad (contrato §9.4)...")

    horizons = cfg['features']['horizons']
    models = cfg['models']['active_models']
    preds_dir = os.path.join(ROOT, cfg['paths']['preds_oos_dir'])

    # VIX por fecha desde el crudo (proxy de volatilidad en el origen t)
    raw = pd.read_csv(os.path.join(ROOT, cfg['data']['raw_source']),
                      parse_dates=['Date'])[['Date', 'VIX_Close']]

    rows = []
    for h in horizons:
        # Conjunto canónico de fechas de origen (compartido por todos los
        # modelos): se toma del archivo RW del horizonte.
        rw_path = os.path.join(preds_dir, cfg['contract']['naming']['oos'].format(h=h, model='RW'))
        if not os.path.exists(rw_path):
            print(f"   [WARN] Falta {rw_path}; salto h={h}.")
            continue
        dates_h = pd.read_csv(rw_path, parse_dates=['Date'])[['Date']]
        vix_h = dates_h.merge(raw, on='Date', how='left')
        # Cuartiles de VIX → Q1 (calma) .. Q4 (estrés)
        vix_h['regime'] = pd.qcut(vix_h['VIX_Close'], 4,
                                  labels=['Q1', 'Q2', 'Q3', 'Q4'])
        regime_map = dict(zip(vix_h['Date'], vix_h['regime']))

        for model in models:
            mpath = os.path.join(preds_dir, cfg['contract']['naming']['oos'].format(h=h, model=model))
            if not os.path.exists(mpath):
                continue
            df = pd.read_csv(mpath, parse_dates=['Date'])
            df['regime'] = df['Date'].map(regime_map)
            for regime in ['Q1', 'Q2', 'Q3', 'Q4']:
                sub = df[df['regime'] == regime]
                if sub.empty:
                    continue
                err = sub['y_true_ret'] - sub['y_pred_ret']
                rows.append({
                    'model': model,
                    'h': h,
                    'regime': regime,
                    'n_obs': len(sub),
                    'rmse_ret': round(np.sqrt((err ** 2).mean()), 6),
                    'mda': round((np.sign(sub['y_true_ret'])
                                  == np.sign(sub['y_pred_ret'])).mean(), 4),
                })

    out = pd.DataFrame(rows)
    out_path = os.path.join(ROOT, "reports", "data", "regimes_summary.csv")
    out.to_csv(out_path, index=False)
    print(f"   -> {out_path} ({len(out)} filas)")
    if not out.empty:
        print("\n--- rmse_ret por régimen (h=20) ---")
        piv = out[out['h'] == 20].pivot(index='model', columns='regime', values='rmse_ret')
        print(piv.to_string())
    print(">>> [Fase 4] Regímenes completado.")

if __name__ == "__main__":
    run_regimes()
