# Bitácora del proyecto

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
