# TesisProject — NASDAQ-100 Forecasting (ARIMA/SARIMA/LSTM)

## 1) Instalación (3 pasos)
1) `git clone https://github.com/joslat92/TesisProject && cd TesisProject`
2) `python -m venv .venv && .\.venv\Scripts\activate`
3) `pip install -r requirements.txt`

## 2) Estructura
- `data/df_final_ready_plus_vix.csv`
- `src/` (merge_vix, train_*, infer_*, dm_test, evaluate_metrics, plots)
- `models/` (pesos en LFS o excluidos)
- `outputs/` (`all_metrics.csv`, `dm_results.csv`)
- `reports/figures/` (png)

## 3) Reproducir todo
`make all`

## 4) Targets útiles
- `make prep` — mergea VIX y prepara dataset
- `make train` — ARIMA / SARIMA / LSTM (plain y VIX)
- `make infer` — CV para LSTM
- `make dm` — Diebold–Mariano
- `make eval` — métricas unificadas
- `make plots` — figuras finales
- `make precommit` — formatea/lint

## 5) FAQs
- **sklearn: `squared`** → usa `scikit-learn>=1.2`
- **Fechas desalineadas** → ver `dm_test.py`
- **Repo pesado** → Git LFS (`*.keras`, `*.pkl`, `*.h5`)
- **Falta `df_final_ready_plus_vix.csv`** → `make prep` (`merge_vix.py`)
