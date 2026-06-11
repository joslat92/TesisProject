# CLAUDE.md — Contexto del Proyecto de Tesis

## Qué es este proyecto

Tesis de maestría (MMACC, Universidad del Rosario) de José Luis Agudelo: comparación de
modelos clásicos de series de tiempo (RW, ARIMA, ARIMAX, SARIMA, SARIMAX) contra LSTM
(plain, +Sentiment, +Sentiment+VIX) para pronosticar retornos acumulados del NASDAQ-100
en horizontes T ∈ {1, 5, 10, 20}, evaluando el aporte de exógenas (VIX y sentimiento
GDELT). Protocolo: OOS 2024 + walk-forward de 12 bloques mensuales, métricas RMSE/MAE/MDA,
tests Diebold–Mariano (HAC, L=h−1, vs RW y vs SARIMAX) y Mincer–Zarnowitz en niveles.

El proyecto estuvo pausado ~1 año por motivos médicos del autor. Se retomó en junio 2026.

## Historia y estado de las versiones (IMPORTANTE)

1. **Versión T+1** (Documento Tesis 4.docx, ago-2025): estudio cerrado y coherente.
   T+1 en nivel, 6 modelos + RW1. Conclusión: empate técnico, SARIMAX_BOTH mejor RMSE OOS.
2. **Versión multi-horizonte** (doc "Numeros normales"): expansión con T∈{1,5,10,20} sobre
   retornos. Tablas completas de 8 modelos (incluye SARIMAX y LSTM con exógenas), pero
   resumen/abstract desactualizados y discusión a medio redactar. **Es la versión que se
   va a terminar** (pendiente de confirmar con el director).
3. **Historial de git (commits hasta e7ac2d3)**: la base de código ANTERIOR a la
   reestructuración. NO BORRAR NI REESCRIBIR ese historial. ⚠️ Verificado 2026-06-09:
   sus resultados (tables/table_7_1_metrics.csv, dm_results.csv) son de una TERCERA
   campaña — backtest rolling walk-forward de 53 folds en NIVELES de precio (RMSE 4–11),
   todo T+1/bloques de 20 pasos. NO contiene el código multi-horizonte (T=5/10/20 sobre
   retornos) que generó las tablas del doc "Numeros normales"; ese código hay que
   buscarlo fuera del repo o reimplementarlo. Sí contiene SARIMA/SARIMAX, LSTM con
   exógenas, hybrid y ablation (entrenar_sensibilidad.py), y su ARIMA multi-paso es
   correcto (forecast genuino, sin fuga) — usarlo como referencia para el fix del bug.
   Disponible en worktree `../TesisProject-legacy`.
4. **Árbol de trabajo actual (29-12-2025)**: reestructuración limpia y modular del
   pipeline, SIN commitear. Corrida preliminar solo con RW/ARIMA/LSTM. ⚠️ Sus resultados
   son SOSPECHOSOS (ver "Bug conocido") y sus figuras no son las canónicas de la tesis.

## Bug conocido — CORREGIDO 2026-06-10

La fuga estaba en `11_train_arima.py`: modelaba la serie solapada `Target_Ret_h{h}`
con predicción one-step-ahead (`apply().predict()`), de modo que y_{t−1}(h) —que
contiene precios hasta t+h−1— entraba como regresor. Fix: enfoque iterado sobre
retornos diarios (`predict_oos_iterated`: fit con datos ≤ t, `forecast(h)` y suma),
re-fit mensual + apply diario, orden [1,0,1] sobre ret_1d. Además: purga de frontera
train/OOS en `12_train_lstm.py` (targets de train que invadían el OOS).
Métricas post-fix DENTRO de la cordura de la tesis (T=1 RMSE≈0.0113 todos;
T=20: RW 0.0440, ARIMA 0.0403/MDA 0.742, LSTM 0.0391; DM ARIMA vs RW no
significativo). Gate permanente: `tests/test_no_leakage.py` (corrupción del futuro
⇒ ŷ_t(h) no cambia; correr con `python tests/test_no_leakage.py`, pytest no está
en el venv). ⚠️ Las figuras de `reports/figs` aún son de la corrida buggy del
29-12-2025; regenerar antes de usarlas.

## Especificación canónica (fuente de la verdad)

- `contrato.docx` — contrato de diseño: esquema de predicciones (Date, h, model,
  y_true_ret, y_pred_ret, y_true_level, y_pred_level [+block en WF]), naming
  (preds_T{h}_{model}.csv / preds_T{h}_{model}_block{b}.csv), splits (IS→2023-12-29,
  OOS 2024, WF 12 bloques), escalado train-only sin fuga, DM HAC L=h−1, MZ, regímenes.
- `Contrato_Control_actualizado.xlsx` — libro de control (SSOT) con errores conocidos.
- `pasos replicacion 220925 2230.xlsx` — guía de replicación etapa por etapa.
- HPs LSTM canónicos (los del documento de tesis, salvo que el director indique otro):
  window=40, hidden=64, dropout=0.2, Adam lr=0.001, epochs=30, batch=64, seed=42,
  early-stopping con val 20%.

## Reglas de trabajo

- NUNCA reescribir el historial de git ni hacer force-push. La rama main debe seguir
  apuntando a la versión pre-reestructuración hasta que se valide la nueva.
- Trabajar la reestructuración en la rama `reestructura-dic2025`.
- Commit + push al final de CADA sesión de trabajo.
- Todo cambio al pipeline debe respetar el contrato (contrato.docx §8 y §11).
- Sin fuga de información: scalers y selección de hiperparámetros solo con IS o bloque
  WF previo; exógenas siempre rezagadas.
- Sesiones de ~2 horas diarias. Mantener bitácora en `logs/bitacora.md` (qué se hizo,
  qué sigue, decisiones tomadas).
- El plan completo de 12 semanas está en `Plan_Retoma_Tesis.md` (fases 0–5).

## Prioridades inmediatas (en orden)

1. Crear `.gitignore` (.venv/, venv/, __pycache__/, *.pyc) y commitear el árbol actual
   en la rama `reestructura-dic2025`; push al remoto.
2. Explorar el historial de git: localizar el código que generó las corridas SARIMAX y
   LSTM con exógenas de la tesis (buscar en commits previos a e7ac2d3). Extraerlo a una
   carpeta de referencia (`legacy/`) usando git worktree o checkout selectivo, sin tocar main.
3. Auditar y corregir el bug de fuga en ARIMA multi-paso.
4. Implementar el validador de contrato (contrato.docx §11) como gate del pipeline.
5. Completar stages SARIMAX y variantes LSTM con exógenas en la estructura nueva.
6. Re-correr OOS + WF completo y validar contra las tablas de la tesis.

## Datos

- `data/data.csv` (= data/raw/data.csv): 2,561 obs diarias, 2015-02-17 a 2025-04-22.
  Columnas: Date, Target_Price, Sentiment_GDELT, VIX_Close, logP, ret_log.
- El estudio principal usa OOS=2024. Los datos de ene–abr 2025 (incluyen el shock de
  volatilidad de abril) quedan reservados para un bloque opcional de robustez.
