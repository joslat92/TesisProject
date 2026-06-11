# Bitácora del proyecto

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
