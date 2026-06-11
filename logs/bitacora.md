# Bitácora del proyecto

## 2026-06-10 (sesión 3) — ★ RC1 DE RESULTADOS ★ (prioridades 4 y 6) ✔

**Este punto queda marcado como RC1 (release candidate 1) de resultados: el
conjunto OOS 2024 + walk-forward 12 bloques de los 7 modelos efectivos es el
candidato a entrar en el documento final de la tesis**, generado con pipeline
sin fuga (gate automático), contrato validado fail-fast y figuras canónicas.
Etiqueta de referencia: commit de esta sesión en `reestructura-dic2025`.

### 1. Gate fail-fast del pipeline (prioridad 4)
- `pytest` y `openpyxl` instalados en el venv (requirements.txt actualizado).
- `src/core/contract.py`: `validate_all_outputs(fail_fast=True)` ahora LEVANTA
  RuntimeError con la lista de violaciones (antes solo imprimía); valida
  también unicidad de `block` en WF. Nuevo `run_leakage_gate()` ejecuta
  `pytest tests/test_no_leakage.py` y detiene el pipeline si falla. 
  `run_full_gate()` = estructura + contrato + anti-fuga; es el gate que corre
  20_evaluate_stats ANTES de calcular nada, y el CLI
  `python -m src.core.contract` / `python src/core/contract.py`.
- Gate de esta corrida: 364 archivos validados (28 OOS + 336 WF), anti-fuga
  3/3 PASSED.

### 2. Walk-forward 2024 — 12 bloques × 7 modelos (prioridad 6)
- `14_walkforward.py` reescrito por completo (el anterior era pre-contrato:
  SARIMAX(2,1,3) sobre niveles a dataset inexistente).
- Por bloque b (mes de 2024): parámetros/scalers SOLO con datos < inicio del
  bloque (`fit_upto` nuevo en predict_oos_iterated[_exog]; el LSTM se
  re-entrena por bloque con scaler del bloque, purga y embargo). Dentro del
  bloque los clásicos re-filtran diariamente (apply) sin re-estimar.
- Salida contrato: preds_T{h}_{model}_block{b}.csv (336 archivos).
- Gate estacional (IS): p=0.3016 → sin m=5; SARIMAX ≡ ARIMAX también en WF.

### 3. Tablas consolidadas (reports/data/)
- metrics_OOS.csv (7 modelos × 4 h) — sin cambios vs sesión 2.
- metrics_WF_blocks.csv (336 filas: por bloque) y metrics_WF.csv (WF_mean).
- tbl_DM_OOS.csv (DM completo vs RW Y vs SARIMAX, HAC L=h−1) y dm_summary.csv
  (pivot compacto de p-values). ARIMAX vs SARIMAX da p=1.0 (idénticos, gate
  off — el caso degenerado se maneja con p=1, sin abortar).
- tbl_MZ_core.csv y mz_summary_OOS.csv (MZ en niveles, ahora con p(β=1) HAC).

### 4. Resultados clave RC1
RMSE WF_mean (across 12 bloques):
| T | RW | ARIMA | ARIMAX/SARIMAX | LSTM | LSTM_SENT | LSTM_FULL |
|---|------|-------|------|------|------|------|
| 1 | .01098 | .01104 | .01096 | .01046 | .01050 | .01054 |
| 5 | .02354 | .02279 | .02314 | .02244 | .02263 | .02246 |
| 10 | .03239 | .03048 | .03136 | .03107 | .03215 | .03212 |
| 20 | .04165 | .03674 | .03984 | .03863 | .03777 | .03842 |
- Coherente con OOS: mismas familias, mismos órdenes de magnitud, ARIMA lidera
  T≥10, las LSTM lideran T≤5. El heatmap WF muestra el patrón esperable:
  bloques 8–9 (ago–sep 2024) castigan a las LSTM (+70–90% dRMSE), bloque 4
  las favorece (−70%).
- DM OOS vs RW (p): T=20 → LSTM_SENT 0.030, LSTM_FULL 0.047, LSTM 0.059.
  DM OOS vs SARIMAX: LSTM_SENT significativo en T=5 (p=0.036).
- MZ: p(β=1) ahora reportado por modelo/horizonte en tbl_MZ_core.

### 5. Figuras canónicas (guía de replicación, paso 10) — 25 archivos
- Por h ∈ {1,5,10,20}: Fig_T{h}_bars_vs_RW, Fig_T{h}_volcano_vs_RW,
  Fig_T{h}_calibracion_scatter, Fig_T{h}_WF_heatmap_vs_RW,
  Fig_T{h}_cumloss_vs_RW, Fig_T{h}_dumbbell_dRMSE; global: Fig_bump_ranking.
- Las 7 figuras de la corrida buggy del 29-12-2025 quedaron archivadas en
  `reports/figs/_archive_corrida_buggy_20251229/` (no mezclar con RC1).
- 30_make_figures.py reescrito para generarlas todas desde las tablas
  consolidadas (caída graciosa si falta una fuente).

### Pendientes post-RC1
- Validar RC1 con el director; decidir si las desviaciones vs tablas de la
  tesis (T=20 ~7-11% en exógenos, ver sesión 2) requieren recuperar la
  especificación original o re-narrar con estos números.
- Bloque opcional de robustez ene–abr 2025 (datos reservados).
- 05_features.py y 30_reports.py/23_final_comparison.py siguen siendo código
  muerto/pre-contrato; decidir limpieza.
- MDA(RW)=0 sigue siendo artefacto de sign(0) (RW predice retorno 0):
  documentar la convención en el texto o excluir RW de la columna MDA.

## 2026-06-10 (sesión 2) — SARIMAX/ARIMAX + variantes LSTM (prioridad 5) ✔

### Qué se hizo
1. **`src/stages/13_train_sarimax.py` (nuevo)**: ARIMAX y SARIMAX con el mismo
   enfoque iterado anti-fuga del ARIMA corregido (`predict_oos_iterated_exog`).
   - Exógenas (VIX, Sentiment GDELT) crudas desde `data/raw/data.csv`, entran
     REZAGADAS con los lags del config (`features.exog`: 1,2,5,10,15).
   - Pasos futuros del forecast: cada exógena CONGELADA en su último valor
     conocido en t (convención random-walk, documentada en docstring). Los lag-k
     que caen ≤ t usan el valor real ya observado.
   - Expanding, re-fit mensual + apply diario (igual que stage 11).
   - **Gating estacional del contrato**: m=5 solo con evidencia p<0.10.
     Test Kruskal–Wallis de efecto día-de-semana sobre ret_1d del IS:
     p=0.3016 → SIN términos estacionales en los 4 horizontes. En consecuencia
     SARIMAX ≡ ARIMAX en esta corrida (ambos se emiten por contrato; la
     distinción se activará si el gate cambia con otros datos/splits).
2. **`src/stages/12_train_lstm.py` reescrito**: variantes desde config
   (LSTM=ret_1d; LSTM_SENT=+Sentiment_GDELT_lag1; LSTM_FULL=+VIX_Close_lag1),
   HPs canónicos de la tesis (hidden 64, dropout 0.2, batch 64, lr 0.001,
   epochs 30, seed 42) y **early stopping**: val = último 20% del train
   (cronológico) con **embargo de h muestras** entre train y val para que los
   targets solapados no crucen el corte; restaura mejores pesos; paciencia 5.
   Se mantiene la purga de frontera train/OOS de la sesión 1.
3. **`config.yaml`**: active_models += ARIMAX/LSTM_SENT/LSTM_FULL; HPs LSTM
   canónicos; variantes redefinidas sobre columnas reales del parquet
   (exógenas lag1, regla "siempre rezagadas").
4. **`tests/test_no_leakage.py` extendido**: nuevo test SARIMAX/ARIMAX —
   corromper SOLO las exógenas posteriores a t, y también retornos+exógenas,
   debe dejar ŷ_t(h) idéntico. PASA (h=1 y h=20).
5. Re-corrida OOS completa (7 modelos × 4 horizontes, 28 archivos validados)
   + 20_evaluate_stats.

### Resultados vs criterio de aceptación
- **T=1: CUMPLIDO** — todos en ≈0.011 (0.01089–0.01139).
- **T=20 (referencias tesis: sarimax≈0.0397; lstm_sentvix≈0.0388, DM p≈0.009)**:
  - SARIMAX/ARIMAX: RMSE 0.04249 (~+7% sobre la referencia), MDA 0.659.
  - LSTM_FULL: RMSE 0.04325 (~+11%), DM p=0.0469 (significativo, mismo signo);
    LSTM_SENT: 0.04256, p=0.0298. LSTM plain: 0.04206, p=0.059.
  - Todos dentro del corredor de cordura (mejores que RW 0.04396, sin aplastarlo;
    sin firma de fuga — la fuga infla, no degrada).
  - Desviaciones esperables de una reimplementación: (a) el SARIMAX de la tesis
    probablemente modelaba el horizonte directo con otra especificación de
    exógenas, no el agregado iterado de retornos diarios con exógenas congeladas;
    (b) las variantes LSTM de la tesis estaban TUNED por variante, aquí van con
    HPs canónicos fijos; (c) sensibilidad a semilla única (seed 42).
  - PENDIENTE: contrastar especificación exacta del SARIMAX de la tesis
    (lags exactos, ¿exógena contemporánea rezagada?, ¿estimación directa por
    horizonte?) cuando aparezca el código multi-horizonte original.

### Métricas completas (RMSE OOS)
| T | RW | ARIMA | ARIMAX/SARIMAX | LSTM | LSTM_SENT | LSTM_FULL |
|---|------|-------|----------------|------|-----------|-----------|
| 1 | .01132 | .01139 | .01127 | .01092 | .01112 | .01089 |
| 5 | .02472 | .02422 | .02436 | .02425 | .02408 | .02484 |
| 10 | .03457 | .03337 | .03360 | .03392 | .03439 | .03429 |
| 20 | .04396 | .04032 | .04249 | .04206 | .04256 | .04325 |

### Qué sigue
- Prioridad 4: validador de contrato como gate fail-fast (incluir test anti-fuga
  y revisar MDA(RW)=0 por sign(0)).
- Prioridad 6: walk-forward 12 bloques para los 7 modelos (14_walkforward está
  para RW/ARIMA/LSTM viejos; actualizarlo) y regenerar figuras (las actuales
  siguen siendo de la corrida buggy del 29-12).
- Buscar el código multi-horizonte original para cerrar la comparación SARIMAX.

## 2026-06-10 — Fix de la fuga en ARIMA multi-paso (prioridad 3) ✔

### Qué se hizo
1. **`src/stages/11_train_arima.py` reescrito (enfoque iterado anti-fuga)**:
   - Modela los retornos DIARIOS (`ret_1d`), nunca la serie solapada `Target_Ret_h`.
   - Para cada t del OOS: `forecast(h)` con datos ≤ t y suma de los h retornos
     pronosticados → ŷ_t(h). Función `predict_oos_iterated()` (importable por tests).
   - Esquema expanding con **re-fit mensual** + `res.apply()` diario (re-filtrado con
     parámetros fijos, muestra estricta ≤ t). Decisión documentada: re-fit diario
     cuesta ~30x y los parámetros ARMA de retornos diarios son estables intra-mes.
2. **`config.yaml`**: orden ARIMA pasa de [1,1,1] a **[1,0,1] sobre ret_1d**
   (la serie ya es la primera diferencia de logP; equivale a ARIMA(1,1,1) en logP).
3. **`src/stages/12_train_lstm.py`**: purga anti-fuga en la frontera train/OOS —
   se excluyen las muestras de train cuyo target (t→t+h) se realiza después de
   train_end. Antes, las últimas h muestras de train veían precios de enero-2024.
4. **`src/stages/10_baselines.py`**: revisado, correcto (ŷ_ret=0, nivel=P_t). Sin cambios.
5. **Targets**: revisados en `00_prepare.py` — `Target_Ret_h = logP.shift(-h) − logP`
   y dropna recorta las últimas h filas. Correcto. (`05_features.py` es código muerto:
   escribe `dataset_features.parquet`/`target_R{h}` que nadie consume; los modelos
   usan `features_T{h}.parquet` de 00_prepare. Pendiente decidir si se elimina.)
6. **`tests/test_no_leakage.py` (gate anti-fuga permanente)**: corrompe con ruido
   todos los datos posteriores a t y verifica que ŷ_t(h) no cambia (h=1,5,10,20);
   además verifica RW≡0. PASA. Nota: pytest no está en el venv; el test corre
   standalone con `python tests/test_no_leakage.py`. Integrarlo al gate del
   validador (prioridad 4).
7. **`src/core/contract.py`**: emojis ✅/❌ → [OK]/[ERROR] (UnicodeEncodeError en
   consola Windows cp1252; mismo fix que ya existía en legacy para dm_test).
8. Re-corrida OOS completa (RW, ARIMA, LSTM re-entrenado con purga) + 20_evaluate_stats.

### Resultado — criterio de aceptación CUMPLIDO
RMSE OOS (RW / ARIMA / LSTM) vs cordura de la tesis:
- T=1: 0.01132 / 0.01139 / 0.01117 (esperado ≈0.011) ✓ — ARIMA ligeramente PEOR que RW
- T=5: 0.02472 / 0.02422 / 0.02378 (≈0.025) ✓ — margen ARIMA +2.0%
- T=10: 0.03457 / 0.03337 / 0.03482 (≈0.035) ✓ — margen +3.5%
- T=20: 0.04396 / 0.04032 / 0.03914 (≈0.042) ✓ — margen +8.3%, bajo el umbral 10–15%
- ARIMA T=20: RMSE 0.0403, MDA 0.742 ≈ referencia tesis (arimax 0.039, MDA 0.74).
- DM vs RW: ARIMA nunca significativo (p 0.22–0.48; antes p≈0.000 en T=20).
  LSTM significativo en T=1 (p=0.014, mejor) y T=20 (p=0.042).
- El patrón imposible (T=20 RMSE 0.0167, MDA 0.89, p≈0.000) desapareció.

### Notas / pendientes
- MDA de RW = 0 es artefacto de la métrica (sign(0) no cuenta como acierto
  direccional); revisar definición en 20_evaluate_stats al implementar prioridad 4.
- LSTM sigue siendo variante Plain con HPs del config (hidden=50, batch=32, sin
  early-stopping); alinear con HPs canónicos (hidden=64, batch=64, val 20%) cuando
  se completen las variantes (prioridad 5).
- Las figuras de reports/figs siguen siendo de la corrida buggy del 29-12; regenerar
  con 30_make_figures cuando se cierre el set de modelos.

### Qué sigue
- Prioridad 4: validador de contrato como gate (incluir test anti-fuga, fail-fast).
- Prioridad 5: SARIMAX + variantes LSTM con exógenas en la estructura nueva.
- Localizar fuera del repo el código multi-horizonte original (ver 2026-06-09).

## 2026-06-09 — Retoma: rama, worktree legacy y análisis de procedencia

### Qué se hizo
1. Creado `.gitignore` y commit `3bbc5f0` con todo el árbol reestructurado en la rama
   `reestructura-dic2025`; push a origin. main intacto en `e7ac2d3`.
   - Se eliminó un hook pre-commit huérfano (su `.pre-commit-config.yaml` ya no existe).
2. Worktree de referencia: `git worktree add ../TesisProject-legacy e7ac2d3`.
3. Análisis de procedencia de los resultados en legacy.

### Hallazgo 1 — Los resultados de e7ac2d3 NO corresponden a ninguna de las dos versiones de la tesis
Valores en legacy (`tables/table_7_1_metrics.csv`, `tables/*.json`, `outputs/dm_results.csv`):
RMSE entre 4.14 (LSTM_HYBRID) y 11.44 (SARIMA_VIX), en NIVELES de precio (~520 pts),
provenientes de un backtest rolling walk-forward de 53 folds (train_len=1500, test_len=20,
stride=20; DM con n_obs 53–2521).

Referencias de la tesis:
- Estudio T+1 (Documento Tesis 4): SARIMAX_BOTH 7.294, ARIMA 7.324, LSTM_MV 7.357,
  LSTM_PLAIN 7.331 (OOS 252 obs) → NO coincide (ni valores ni protocolo ni n).
- Estudio multi-horizonte: T=1 RMSE≈0.011 (retornos), T=20 arimax≈0.039/MDA≈0.74 →
  NO coincide (legacy está en niveles, no retornos, y no hay T=5/10/20).

Conclusión: e7ac2d3 es una TERCERA campaña de experimentos (rolling WF en niveles,
2021–2025, incluye modelos LSTM_HYBRID y variantes TUNED que no aparecen con esos
nombres en las tablas citadas). Este commit puede reproducir la "Tabla 7.1" de esa
campaña WF, pero NO las tablas del estudio T+1 ni las del multi-horizonte.
⚠️ El código que generó las tablas multi-horizonte (T∈{1,5,10,20} sobre retornos)
NO está en el historial de este repo — buscar en otra carpeta/equipo/backup.

### Hallazgo 2 — Legacy es todo T+1 / bloques de 20 pasos; no hay multi-horizonte
Búsqueda en `legacy/src/*.py`: el único parámetro de horizonte es `--h` de `dm_test.py`
(default 1). `train_lstm_cv.py` declara explícitamente "horizonte = 1 día, predicción
directa". El ARIMA legacy pronostica bloques de 20 días (`test_len=20`) pero como
trayectoria multi-paso dentro de cada fold, no como targets T+5/T+10/T+20 separados.

### Hallazgo 3 — Causa probable del bug de fuga en el árbol nuevo (confirmada en código)
- **Legacy (sin fuga)**: `train_arima.py` ajusta solo con el train del fold y llama
  `model.predict(n_periods=len(y_test))` → pronóstico multi-paso genuino, solo usa
  información ≤ t (utils_backtest.rolling_splits separa train/test estrictamente).
- **Nuevo (`src/stages/11_train_arima.py`, líneas 53–66)**: ajusta ARIMA sobre la serie
  de retornos acumulados `Target_Ret_h{h}` y luego hace `res.apply(df[target])` +
  `predict()` one-step-ahead sobre TODO el dataset. Para h>1 eso es FUGA: la predicción
  de y_t(h) = retorno t→t+h usa como regresor el valor real de y_{t-1}(h) = retorno
  t−1→t−1+h, que contiene precios hasta t+h−1 (futuro). Como los targets se solapan
  h−1 días, el modelo "ve" casi todo el retorno objetivo. Explica RMSE 0.0167 vs RW
  0.044 y MDA 0.89 en T=20 — y por qué T=1 sí es razonable (sin solape).
- **Corrección pendiente**: para cada t en OOS, pronosticar con información ≤ t
  (p. ej., modelar retornos diarios y agregar h pasos con `forecast(h)`, o re-anclar
  el apply solo hasta t y usar dynamic). Revisar también 10_baselines y 12_train_lstm
  por el mismo patrón de targets solapados.

### Qué sigue
- Corregir 11_train_arima.py (prioridad 3 del CLAUDE.md) y re-correr OOS.
- Localizar fuera del repo el código multi-horizonte que generó las tablas del doc
  "Numeros normales"; si no aparece, habrá que reimplementarlo en la estructura nueva
  (prioridad 5) y validar contra las tablas.
