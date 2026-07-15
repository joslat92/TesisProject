# Cierre de auditoría — 2026-06-16

Refinamientos menores tras dos auditorías técnicas independientes (Codex + revisión
previa) y una reproducción completa con reentrenamiento que confirmó determinismo.
Ninguna auditoría halló problemas críticos ni importantes de integridad; este cierre
es **pulido**, no reconstrucción: no se cambió la lógica de modelado.

Trabajo realizado en la rama `cierre-auditoria` (un commit por punto, para revisión
antes de mergear). El set de resultados sellado pasa a llamarse **RC2.1** (etiqueta
única, alineada con `config.yaml` 2.1.0); tag git anotado `RC2.1` sobre el commit
sellado `3cc38fb`.

## Resumen de cambios (un commit por punto)

| # | Commit | Cambio | ¿Cambia números? |
|---|--------|--------|------------------|
| 1 | `a7a6f87` | **MZ → HAC con L=h−1** (D7) | Sí (solo p de MZ; sin cambiar veredictos) |
| 2 | `d51a75b` | **numpy 2.4.0 (yanked) → 2.3.3** | No (clásicos bit-idénticos) |
| 3 | `6a56a55` | **Código muerto** (mz.py, calculate_metrics, sarimax.order) | No |
| 4 | `145df01` | **DM: estimador HAC unificado** (Newey-West de libro) | Sí (<0.0025; sin cambiar veredictos) |
| 5 | `69eb02c` | **Test anti-fuga LSTM** | No |
| 6 | `9a39676` | **Metadata §10 + regímenes §9.4 + reconciliación RC2.1** | No (artefactos nuevos) |

## 1. Mincer–Zarnowitz: HAC con maxlags = h−1 (D7)

Antes la MZ usaba `cov_type='HAC'` con `maxlags=1` fijo. Los retornos acumulados
solapados inducen autocorrelación MA(h−1) en el residuo; el lag debe escalar con h,
igual que el DM. Ahora `maxlags = max(h−1, 0)`.

**p-valor de H0: β=1 — antes → después** (no cambia ningún veredicto):

| h | antes (maxlags=1) | después (maxlags=h−1) | veredicto |
|---|-------------------|------------------------|-----------|
| 1 | ~0.10 – 0.31 | ~0.09 – 0.29 | no se rechaza (igual) |
| 5 | ~0.001 – 0.004 | ~0.008 – 0.018 | se rechaza (igual) |
| 10 | ~0.0000 | ~0.008 – 0.017 | se rechaza (igual) |
| 20 | ~0.0000 | **~0.020 – 0.046** | se rechaza (igual) |

β=1 se rechaza al 5% desde T+5; en T+1 no se rechaza. Ningún rechazo al 5% se
invierte. En esta rama limpia se conserva `tbl_MZ_core.csv` como salida MZ canónica.

## 2. Pin de numpy

`numpy==2.4.0` fue retirada (yanked) de PyPI ⇒ `pip install` rompía en entorno nuevo.
Re-pineado a `numpy==2.3.3` (último parche estable de la serie 2.3.x). Validado en
venv limpio (`scripts/numpy_pin_check.ps1`, log en `logs/numpy_pin_check_2026-06-16.log`):
resuelve con todos los pins, `pytest tests/` 6/6, y las predicciones clásicas
regeneradas con 2.3.3 son **bit a bit idénticas** a RC2.1. No cambia ningún número.

## 3. Código muerto

Verificado con grep que no se importa en el pipeline vigente (solo aparecía en el
snapshot textual histórico retirado de la rama limpia):
- `src/core/mz.py` (β=0 con HC3) — borrado; el MZ real (β=1, HAC) vive en el stage 20.
- `calculate_metrics` de `metrics.py` (MDA por diff de niveles) — borrada; la MDA real
  es por signo del retorno en el stage 20.
- `config.yaml` `sarimax.order: [1,1,1]` — eliminado (los stages usan `arima.order`;
  además [1,1,1] re-diferenciaba una serie ya estacionaria). Nota explicativa en su lugar.

## 4. Estimador HAC del DM (cosmético)

En `dm.py`, `gamma_0` usaba `np.var` (ddof=0, media global) mientras las autocovarianzas
usaban `np.cov` (ddof=1, de-mediando cada subserie). Unificado a Newey-West de libro:
media global y ÷n en todas las autocovarianzas. Kernel de Bartlett y sentinela de
varianza degenerada sin cambios.

**Celdas DM significativas (HLN) — se mantienen; ninguna otra cruza 0.05:**

| Celda | antes | después | veredicto |
|-------|-------|---------|-----------|
| OOS  LSTM_SENT vs RW, T=5 | 0.0490 | **0.0482** | significativa (se mantiene) |
| 2025 LSTM vs RW, T=5 | 0.0394 | **0.0375** | significativa (se mantiene) |
| OOS  LSTM vs RW, T=20 (vecina) | 0.0623 | 0.0599 | no significativa (no cruza) |
| 2025 LSTM vs SARIMAX, T=10 (vecina) | 0.0684 | 0.0641 | no significativa (no cruza) |

Multiseed LSTM_FULL T=20 vs RW: mediana p 0.194→0.190, ensemble 0.140→0.136 (misma
lectura: no significativo). Tablas regeneradas sin reentrenar: `tbl_DM_OOS`,
`dm_summary`, `tbl_DM_2025`, `multiseed_dm_T20_LSTM_FULL`. Métricas/PT/MZ y
predicciones intactas.

## 5. Test anti-fuga para la LSTM

Nuevo `test_lstm_prediction_ignores_future` en `tests/test_no_leakage.py`: corrompe
features y target posteriores a t, reentrena LSTM_FULL (epochs=1, misma semilla) y
verifica que ŷ_t es bit-idéntico. Cierra el único hueco de cobertura señalado por las
auditorías. Suite: 6/6, integrada al gate del pipeline.

## 6. Completitud de contrato y nomenclatura

- **§10 metadata**: `metadata_snapshot.csv` ahora registra timestamp, dataset_sha256,
  git_commit, python_version, pip_freeze_hash (hash de requirements.txt) y config_sha256.
- **§9.4 regímenes**: nuevo `src/stages/31_regimes.py` → `regimes_summary.csv`
  (model, h, regime∈{Q1..Q4}, n_obs, rmse_ret, mda), estratificando el OOS 2024 por
  cuartil de VIX en la fecha de origen. **PROVISIONAL**: la definición del proxy de
  volatilidad (VIX contemporáneo, cuartiles intra-muestra) es una elección
  metodológica pendiente de ratificación del director — el patrón Q1>Q3 en h=20
  sugiere que podría preferirse la volatilidad realizada sobre la ventana de pronóstico.
  Reversible y aislado en su stage.
- **RC2.1**: etiqueta única en CLAUDE.md / REPRODUCIR.md, nota de reconciliación en el
  Registro; bitácoras fechadas conservan "RC2". Tag anotado `RC2.1` sobre `3cc38fb`.

## Verificación final

- `pytest tests/`: 6/6 (incluye el nuevo test LSTM), también bajo venv limpio con numpy 2.3.3.
- `run_all.py --quick` en clon limpio de `cierre-auditoria` (numpy 2.3.3 pineado):
  **RUN_ALL_OK en 5.9 min** — cadena completa 00→31, ambos gates (contrato +
  anti-fuga + cordura y_true) en verde, regimes_summary.csv generado (112 filas).
  Script: `scripts/run_all_quick_check.ps1`; log: `logs/run_all_quick_2026-06-16.log`.
- Determinismo de clásicos confirmado (numpy check, log versionado). La reproducción
  completa CON reentrenamiento (determinismo de LSTM/WF/multiseed) se realizó en la
  fase de sellado previa; si se desea un log versionado fresco de esa corrida completa
  (≈2 h), basta `python run_all.py` sin `--quick`.

## Pendiente para el director
- Ratificar D7 (MZ L=h−1), la etiqueta RC2.1 y la definición de régimen de §9.4.
- El resto del estado RC2.1 (D1–D6 + adenda D5) sigue como en `Registro_Decisiones_1.md`.
