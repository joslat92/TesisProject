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
5. **Episodio RC2.1 (jun-2026, rama reestructura-dic2025)**: la retoma produjo RC1
   (WF 12×7 + gate fail-fast); el bloque de robustez 2025 destapó el bug #2
   (y_true desplazado en salidas LSTM) ⇒ regeneración total como **RC2.1** (rotulada
   "RC2" en las bitácoras de jun-2026; se adopta **RC2.1** como etiqueta única,
   alineada con config.yaml version 2.1.0), DM con
   kernel de Bartlett (el rectangular daba varianza negativa en 2025), y sellado
   2026-06-12: y_true de los 56 archivos verificado contra el CSV crudo
   (tests/test_ytrue_sanity.py en la suite del gate), reproducibilidad de punta
   a punta con run_all.py (--quick verificado en clon limpio) y requirements
   congelado. **RC2.1 es el set de números del documento final** (pendiente
   ratificación del director; ver Registro_Decisiones_1.md y bitácora).

## Bugs conocidos — ambos CORREGIDOS

1. **Fuga ARIMA multi-paso (corregido 2026-06-10).** `11_train_arima.py` modelaba
   la serie solapada `Target_Ret_h{h}` one-step-ahead ⇒ y_{t−1}(h), que contiene
   precios hasta t+h−1, entraba como regresor. Fix: enfoque iterado sobre retornos
   diarios (`predict_oos_iterated`), orden [1,0,1] sobre ret_1d, re-fit mensual.
   Gate permanente: `tests/test_no_leakage.py` (pytest instalado; corre solo y
   dentro de `run_full_gate()`).
2. **y_true desalineado en salidas LSTM (corregido 2026-06-11).** `fit_predict`
   usaba el índice reseteado del frame post-secuencias como etiqueta de fila del
   parquet ⇒ y_true_ret/y_true_level/y_pred_level desplazados seq_len=40 filas en
   TODOS los artefactos LSTM (OOS 2024, WF, multiseed). El ENTRENAMIENTO y
   y_pred_ret siempre fueron correctos; solo la verdad reportada estaba corrida.
   En 2024 se camufló (vol estable ⇒ RMSE plausible); el bloque 2025 lo destapó
   (LSTM "ganando" 50–60% a RW en el crash, imposible). Fix: `orig_rows =
   idx + seq_len` + assert de fechas; guard permanente en ContractValidator
   (`_check_truth_consistency`: y_true idéntico entre modelos por horizonte,
   fail-fast). ⚠️ TODA métrica LSTM anterior al 2026-06-11 (RC1, multiseed, D5
   del Registro de Decisiones) quedó invalidada; el set vigente es **RC2.1**
   (bitácora 2026-06-11). Cordura post-fix OOS 2024: T=1 todos ≈0.0113; T=20
   RW .0440, ARIMA .0403 (MDA .742), LSTM .0414, LSTM_FULL .0423. DM-HLN con
   kernel de Bartlett (2026-06-12, el rectangular daba varianza negativa en
   2025): única celda significativa al 5% es LSTM_SENT T=5 vs RW (p=0.048,
   marginal/no-hallazgo); en T=20 mín p≈0.060. RC2.1 sellado 2026-06-12:
   y_true verificado contra fuente primaria (tests/test_ytrue_sanity.py,
   en la suite del gate).

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

1. ✔ (2026-06-09) .gitignore + commit del árbol en `reestructura-dic2025` + push.
2. ✔ (2026-06-09) Historial explorado; worktree `../TesisProject-legacy` en e7ac2d3.
   Hallazgo: el código multi-horizonte de la tesis NO está en el repo (ver Historia #3).
3. ✔ (2026-06-10) Bug de fuga ARIMA corregido (ver "Bug conocido — CORREGIDO").
4. ✔ (2026-06-10) Gate fail-fast: contract.py levanta RuntimeError ante violaciones
   y `run_full_gate()` (estructura + contrato + pytest anti-fuga) corre al inicio de
   20_evaluate_stats. pytest instalado. MDA(RW)=0 documentado como convención
   (sign(0) no puntúa); pendiente decidir presentación en el texto.
5. ✔ (2026-06-10) Stages ARIMAX/SARIMAX (13_train_sarimax.py, iterado, exógenas
   congeladas en t, gate estacional KW p<0.10 → sin m=5) y variantes LSTM_SENT/
   LSTM_FULL con early stopping + embargo. OOS dentro del corredor de cordura;
   T=20 queda ~7-11% sobre las referencias de la tesis (esperable: especificación
   exógena/tuning distintos) — contrastar cuando aparezca el código original.
6. ✔ (2026-06-10) ★RC1 de resultados★: WF 2024 completo (12 bloques × 7 modelos,
   336 archivos, re-fit/scalers por bloque), DM vs RW y vs SARIMAX, MZ con p(β=1),
   tablas consolidadas en reports/data y 25 figuras canónicas en reports/figs
   (las buggy del 29-12 en _archive_corrida_buggy_20251229/). Ver bitácora.
7. ✔ (2026-06-11) Cierre Fase 3: DM clásico+HLN en src/core/dm.py (tbl_DM_OOS
   con ambas columnas), test Pesaran–Timmermann (tbl_PT_OOS, indicativo en h>1),
   bloque de robustez 2025 (D3: config_robustez2025.yaml, 16_robustez_2025.py,
   metrics_OOS_2025 + tbl_DM_2025 + mensual). El bloque 2025 destapó el bug #2
   (ver arriba); todo regenerado como ★RC2.1★ (reemplaza a RC1; mismas
   predicciones clásicas, métricas LSTM corregidas, ningún DM significativo).
8. ✔ (2026-06-12) Sellado RC2.1: y_true verificado contra fuente primaria
   (test_ytrue_sanity.py, en gate), anomalía DM=0.0 resuelta (kernel Bartlett
   en dm.py; tablas DM regeneradas), run_all.py + requirements congelado +
   prueba de clon limpio (--quick), docs/REPRODUCIR.md.
9. Validar RC2.1 con el director: D5 actualizada en el Registro (mediana p≈0.19,
   1/10 semillas <0.05, ensemble n.s.; única celda 5% es LSTM_SENT T=5 vs RW
   p=0.048, no-hallazgo por comparaciones múltiples) y decidir narrativa final.
10. Seguir buscando fuera del repo el código multi-horizonte original (tablas
    del doc "Numeros normales"). ⚠️ La estructura docs/ con material RC2.1 que el
    autor mencionó (2026-06-12) no llegó al árbol de trabajo; solo existe
    docs/REPRODUCIR.md (creado por la tarea de reproducibilidad).

## Datos

- `data/data.csv` (= data/raw/data.csv): 2,561 obs diarias, 2015-02-17 a 2025-04-22.
  Columnas: Date, Target_Price, Sentiment_GDELT, VIX_Close, logP, ret_log.
- El estudio principal usa OOS=2024. Los datos de ene–abr 2025 (incluyen el shock de
  volatilidad de abril) quedan reservados para un bloque opcional de robustez.
