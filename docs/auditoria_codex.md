# Auditoria tecnica independiente RC2.1

Fecha de auditoria: 2026-06-16  
Rama auditada: `reestructura-dic2025` (`origin/reestructura-dic2025`)  
Rama de entrega: `auditoria-codex-rc2-1`

## Veredicto general

RC2.1 corrige de forma verificable los dos bugs criticos de evaluacion reportados y reproduce las predicciones/tablas con reentrenamiento completo; el principal problema metodologico que encontre es que Mincer-Zarnowitz usa HAC con `maxlags=1` fijo para todos los horizontes.

## Hallazgos

### IMPORTANTE - Mincer-Zarnowitz usa HAC con `maxlags=1` fijo, no dependiente del horizonte

Evidencia:

- `src/stages/20_evaluate_stats.py:32-58` implementa `mincer_zarnowitz_test`.
- `src/stages/20_evaluate_stats.py:41` fija `model.fit(cov_type='HAC', cov_kwds={'maxlags': 1})` para todos los horizontes.
- El proyecto usa retornos acumulados solapados para `h > 1`; por analogia con el DM (`L=h-1`) y con la propia advertencia de solape en PT, el error MZ en niveles tambien queda serialmente correlacionado por construccion.
- Recalculo independiente en `scripts/audit_stats_check.py`:
  - `h=5 LSTM_FULL`: `p_Beta1` pasa de `0.00359` con L=1 a `0.01796` con L=4.
  - `h=10 LSTM_FULL`: `p_Beta1` pasa de `5.161e-05` con L=1 a `0.01666` con L=9.
  - `h=20 RW`: `p_Beta1` pasa de `1.749e-05` con L=1 a `0.04620` con L=19.
  - `h=20 LSTM_FULL`: `p_Beta1` pasa de `9.089e-06` con L=1 a `0.04161` con L=19.

Impacto:

- No vi que cambie la conclusion cualitativa principal de MZ, porque los ejemplos siguen bajo 5%.
- Pero las tablas comprometidas `reports/data/tbl_MZ_core.csv` subestiman la incertidumbre para horizontes largos. Recomiendo regenerar MZ con `maxlags=h-1` o documentar explicitamente por que MZ usa L=1 mientras DM usa L=h-1.

### OBSERVACION - Reproduccion completa con reentrenamiento verificada; no todos los artefactos son bit-a-bit

Evidencia:

- Cree entorno limpio con Python 3.11.9:
  - `python -m venv .codex_audit_venv`
  - `.codex_audit_venv\Scripts\python.exe -m pip install -r requirements.txt`
- Instalacion exitosa, pero `pip` aviso que `numpy==2.4.0` esta yanked en PyPI por `Backward compatibility bug`.
- Primero ejecute `.codex_audit_venv\Scripts\python.exe run_all.py --quick`: `RUN_ALL_OK (QUICK)` en 8.2 min. Como era esperable, ese modo cambio filas LSTM por `epochs=2` y solo sirve como prueba mecanica.
- Despues ejecute reproduccion completa en worktree limpio `TesisProject-full-repro` desde `origin/reestructura-dic2025`:
  - `.venv_full_repro\Scripts\python.exe run_all.py`
  - Resultado del log `full_repro_stdout.log`: `RUN_ALL_OK (COMPLETO) en 19.5 min`.
  - Tiempos por etapa: `00_prepare` 4s, RW 9s, ARIMA 39s, ARIMAX/SARIMAX 219s, LSTM OOS 51s, walk-forward 505s, multiseed 218s, robustez 2025 104s, evaluacion/gate 12s, figuras 10s.
  - Gate interno: `5 passed in 8.36s`, `GATE ANTI-FUGA: OK`, `stderr` vacio.
- Comparacion posterior con `git diff`:
  - Sin diferencias en `outputs/preds/`.
  - Sin diferencias en `reports/data/metrics_OOS.csv`, `metrics_OOS_2025.csv`, `metrics_WF.csv`, `tbl_DM_OOS.csv`, `tbl_DM_2025.csv`, `tbl_MZ_core.csv`, `tbl_PT_OOS.csv`, `multiseed_dm_T20_LSTM_FULL.csv` ni el resto de CSV de `reports/data`, salvo `metadata_snapshot.csv`.
  - Diferencias no sustantivas:
    - `data/processed/splits.json`: solo `created_at`.
    - `reports/data/metadata_snapshot.csv`: `timestamp` y `config_hash`; `raw_data_hash` y `n_rows_raw=2561` coinciden. El `config_hash` comprometido no corresponde al `config.yaml` actual, aunque eso no afecto las predicciones/tablas regeneradas.
    - `reports/figs/Fig_multiseed_boxplot.png`: difiere como PNG regenerado; los datos subyacentes `multiseed_*` coinciden.

Impacto:

- La reproduccion completa si valida que los modelos reentrenados regeneran exactamente las predicciones y tablas numericas de RC2.1.
- La reproducibilidad no es bit-a-bit para todos los artefactos porque hay timestamps, un `config_hash` inconsistente en metadata y una figura PNG no identica.
- El pin a una version yanked de NumPy sigue siendo un riesgo de reproducibilidad futura.

### MENOR - `config.yaml` contiene un parametro SARIMAX muerto o contradictorio

Evidencia:

- `config.yaml:50-52` declara `sarimax.order: [1, 1, 1]` y `seasonal_order: [0, 1, 0, 5]`.
- `src/stages/13_train_sarimax.py:112` toma `order = tuple(cfg['models']['params']['arima']['order'])`, no `cfg['models']['params']['sarimax']['order']`.
- `src/stages/13_train_sarimax.py:135-140` usa el mismo `order` para ARIMAX y SARIMAX; solo cambia `seasonal_order`.

Impacto:

- No invalida los resultados actuales: con gate estacional apagado, SARIMAX==ARIMAX es coherente.
- Pero el parametro `sarimax.order` puede inducir a error. Recomiendo eliminarlo o hacer que el codigo lo use explicitamente.

### MENOR - El gate anti-fuga no prueba LSTM por corrupcion del futuro

Evidencia:

- `tests/test_no_leakage.py` corrompe futuro para ARIMA y ARIMAX/SARIMAX, y verifica RW.
- No hay test equivalente que reentrene o evalue LSTM tras corromper features/targets posteriores a una fecha `t`.
- El codigo LSTM inspeccionado es consistente con no fuga:
  - `src/stages/12_train_lstm.py:105-109`: `StandardScaler.fit` se hace sobre `df_train`.
  - `src/stages/12_train_lstm.py:115-126`: purga por `target_end_dates <= train_end` y embargo entre train/val.
  - `src/stages/12_train_lstm.py:128-154`: predicciones OOS usan secuencias construidas antes de la fila objetivo y recuperan `y_true` por `orig_rows`.

Impacto:

- No encontre fuga LSTM por inspeccion.
- Pero el gate automatico no cubre esta familia con el mismo tipo de prueba adversarial que los modelos clasicos.

### OBSERVACION - Bug de fuga por targets solapados corregido en ARIMA/ARIMAX/SARIMAX

Evidencia:

- `src/stages/00_prepare.py:16-18` define el target acumulado con `shift(-h)`.
- `src/stages/11_train_arima.py:17-51` ya no modela `Target_Ret_h{h}`; modela `ret_1d`, ajusta con datos `<=t` y usa `forecast(steps=h).sum()`.
- `src/stages/13_train_sarimax.py:40-97` aplica el mismo enfoque iterado con exogenas congeladas en el ultimo valor conocido en `t`.
- `pytest tests/ -q`: `5 passed in 29.68s`.
- `run_all.py --quick`: el gate interno volvio a ejecutar la suite y reporto `5 passed in 21.06s`.

Conclusion:

- No encontre evidencia de la fuga original en el codigo actual.

### OBSERVACION - Bug de `y_true` LSTM desplazado 40 filas corregido y verificado contra fuente primaria

Evidencia:

- `src/stages/12_train_lstm.py:129-138` corrige el mapeo con `orig_rows = dates_all[mask_pred].index + seq_len` y valida fecha/fila con `assert`.
- `src/core/contract.py:121-157` agrega `_check_truth_consistency`, que exige `y_true_ret` identico entre modelos por horizonte en fechas comunes.
- Script independiente escrito para esta auditoria: `scripts/audit_ytrue_manual.py`.
- Comando: `python scripts\audit_ytrue_manual.py`.
- Resultado:
  - `raw_rows=2561`
  - `checked_files=56`
  - `seed=20260616`
  - `tol=1e-10`
  - maxima diferencia en `y_true_ret`: entre `8.326673e-17` y `9.714451e-17`, segun horizonte/periodo.
  - maxima diferencia en `y_true_level`: `0.0`.

Conclusion:

- El desplazamiento de 40 filas no esta presente en los artefactos RC2.1 auditados.

### OBSERVACION - Diebold-Mariano/HLN esta implementado conforme a la especificacion auditada

Evidencia:

- `src/core/dm.py:39-48` usa kernel Bartlett/Newey-West con pesos `w = 1 - lag/h` y truncamiento por `range(1, h)`, es decir `L=h-1`.
- `src/core/dm.py:58-61` implementa HLN con `k = (n + 1 - 2h + h(h-1)/n)/n`, multiplica el estadistico por `sqrt(k)` y usa t de Student con `df=n-1`.
- Recalculo independiente en `scripts/audit_stats_check.py`:
  - OOS 2024, `h=5`, LSTM_SENT vs RW: `n=252`, `DM=2.0142`, `p=0.0440`, `DM_HLN=1.9782`, `p_HLN=0.0490`.
  - Bloque 2025, `h=5`, LSTM vs RW: `n=70`, `DM=-2.2448`, `p=0.0248`, `DM_HLN=-2.1004`, `p_HLN=0.0394`.

Conclusion:

- Las dos celdas significativas reportadas se reproducen.
- El signo de 2025 indica que LSTM pierde contra RW bajo la convencion `d=e2_bench-e2_chall`.

### OBSERVACION - Anomalia DM=0.0 contra SARIMAX es degeneracion genuina por predicciones identicas

Evidencia:

- Gate estacional recomputado: `KW=4.8634`, `p=0.3016`; no activa terminos estacionales.
- `scripts/audit_stats_check.py` encontro:
  - `OOS h=1/5/10/20 max|ARIMAX-SARIMAX y_pred_ret|=0.000e+00`
  - `OOS_2025 h=1/5/10/20 max|ARIMAX-SARIMAX y_pred_ret|=0.000e+00`
- `reports/data/tbl_DM_2025.csv` contiene `ARIMAX,SARIMAX` con `DM_Stat=0.0`, `p_value=1.0`, `DM_HLN=0.0`, `p_HLN=1.0` para todos los horizontes.

Conclusion:

- Es el caso degenerado genuino por SARIMAX==ARIMAX, no un error de merge ni de generacion de tabla.

### OBSERVACION - PT trata degenerados como NaN y los documenta como indicativos

Evidencia:

- `src/core/metrics.py:13-19` advierte que PT asume independencia y que para `h>1` es indicativo por solape.
- `src/core/metrics.py:43-45` devuelve `NaN` cuando la varianza degenera.
- Tabla comprometida `reports/data/tbl_PT_OOS.csv` y recomputo:
  - Filas degeneradas: ARIMA en `h=5/10/20`, LSTM en `h=10/20`.
  - En esos casos `HitRate == HitRate_H0`, por ejemplo `h=20 ARIMA: 0.7421 == 0.7421`.

Conclusion:

- El tratamiento de predicciones de un solo signo es razonable y no vi falsos positivos PT.

### OBSERVACION - Multi-semilla RC2.1 coincide con la afirmacion documental

Evidencia:

- `reports/data/multiseed_dm_T20_LSTM_FULL.csv`.
- Recalculo/resumen en `scripts/audit_stats_check.py`:
  - mediana de p-valores: `0.1941`.
  - semillas con `p<0.05`: `1/10`.
  - ensemble RMSE: `0.04154`.
  - ensemble p-value: `0.1400`.

Conclusion:

- La afirmacion "mediana p≈0.194 y ensemble p≈0.140" esta verificada.

## Afirmaciones no verificadas o contradictorias

- La reproduccion completa fresca de `run_all.py` sin flags quedo verificada: predicciones y tablas numericas coinciden. No queda verificado bit-a-bit el PNG `reports/figs/Fig_multiseed_boxplot.png`, y `metadata_snapshot.csv` no es estable por timestamp/config_hash.
- `Registro_Decisiones.md`, pedido en la orientacion inicial, no existe con ese nombre. El archivo real es `Registro_Decisiones_1.md`.
- `docs/REPRODUCIR.md` afirma prueba de clon limpio con `run_all.py --quick`; mi auditoria reproduce el modo quick y ademas ejecuta `run_all.py` completo en worktree limpio, aunque no en un segundo clon remoto independiente.
- `config.yaml` declara `sarimax.order: [1,1,1]`, pero el codigo usa el orden ARIMA `[1,0,1]` para ARIMAX/SARIMAX. Esta contradiccion no cambia los artefactos actuales, pero si contradice la lectura natural del config.

## Recomendaciones

1. Corregir MZ para aceptar `h` y usar `cov_kwds={'maxlags': h-1}` en horizontes acumulados, o documentar formalmente por que MZ debe usar L=1.
2. Arreglar la metadata de reproducibilidad: `metadata_snapshot.csv` deberia registrar un `config_hash` que corresponda al `config.yaml` vigente o excluir campos temporales de la comparacion.
3. Reemplazar `numpy==2.4.0` por una version no yanked si los resultados no cambian materialmente, o documentar el riesgo de depender de una version retirada.
4. Eliminar o activar `models.params.sarimax.order` para evitar parametros muertos.
5. Agregar un test anti-fuga LSTM por corrupcion de futuro, aunque sea en modo reducido con `epochs=1`, para cubrir la familia neuronal con el mismo criterio adversarial que ARIMA/ARIMAX.
6. Mantener `scripts/audit_ytrue_manual.py` como prueba externa minima de integridad de `y_true`; no depende de modulos del proyecto y captura exactamente el bug de desplazamiento que causo RC2.
