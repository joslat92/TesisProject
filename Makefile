PY?=python
SRC=src
DATA_PLUS_VIX=data/df_final_ready_plus_vix.csv
OUT=outputs
FIG=reports/figures
SEED?=42

.PHONY: all prep data train infer dm eval plots tune precommit clean
all: prep train infer dm eval plots

prep:
@echo ">> Preparando datos y VIX"
-$(PY) $(SRC)/merge_vix.py

data:
@test -f $(DATA_PLUS_VIX) || (echo "Falta $(DATA_PLUS_VIX). Ejecuta: make prep"; exit 1)

train: data
@echo ">> Entrenando modelos estadísticos y LSTM"
-$(PY) $(SRC)/train_arima.py --seed $(SEED)
-$(PY) $(SRC)/train_sarima.py --seed $(SEED)
-$(PY) $(SRC)/train_sarima_vix.py --seed $(SEED)
-$(PY) $(SRC)/train_lstm_plain.py --seed $(SEED)
-$(PY) $(SRC)/train_lstm_vix.py --seed $(SEED)

infer:
@echo ">> Inferencia CV (LSTM)"
-$(PY) $(SRC)/infer_lstm_cv.py

dm:
@echo ">> Prueba DM"
$(PY) $(SRC)/dm_test.py --out $(OUT)/dm_results.csv

eval:
@echo ">> Consolidando métricas"
-$(PY) $(SRC)/evaluate_metrics.py --out $(OUT)/all_metrics.csv || $(PY) $(SRC)/evaluate_metrics.py

plots:
@echo ">> Generando figuras"
$(PY) -c "import pathlib; pathlib.Path('reports/figures').mkdir(parents=True, exist_ok=True)"
-$(PY) $(SRC)/plot_acf_pacf.py --outdir $(FIG) || true
-$(PY) $(SRC)/generar_grafico_wf.py --outdir $(FIG) || true
-$(PY) $(SRC)/visualize_results.py --outdir $(FIG) || true

tune:
-$(PY) $(SRC)/tune_lstm.py --seed $(SEED)
-$(PY) $(SRC)/tune_lstm_vix.py --seed $(SEED)

precommit:
pre-commit install
pre-commit run --all-files

clean:
rm -rf .ruff_cache .mypy_cache __pycache__ $(OUT)/*.tmp
