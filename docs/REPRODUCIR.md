# Reproduccion de punta a punta

Esta guia separa dos procesos: reconstruir los datos desde proveedores
identificados y ejecutar el pipeline de modelado sobre el dataset resultante.

## Requisitos

- Windows 11 o un entorno equivalente con Python 3.12.
- Git.
- Aproximadamente 3 GB de espacio disponible.
- Para reconstruir GDELT: un proyecto de Google Cloud con BigQuery habilitado,
  facturacion activa y credenciales de aplicacion. La consulta puede generar
  costo; el dry-run es obligatorio antes de `--execute`.

## Clon y entornos

```powershell
git clone --branch entrega-final-tutor https://github.com/joslat92/TesisProject.git
cd TesisProject
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -r requirements-data.txt
```

## Reconstruccion de datos

`data/raw/data.csv` es el dataset heredado y solo se usa para auditoria
forense. El pipeline canonico lee `data/curated/model_input_ndx.csv`, que no se
publica en Git y debe reconstruirse.

```powershell
# Evidencia y fuentes de mercado sin BigQuery
.venv\Scripts\python.exe scripts/data/00_snapshot_legacy.py
.venv\Scripts\python.exe scripts/data/10_fetch_market_data.py
.venv\Scripts\python.exe scripts/data/20_forensic_legacy_target.py
.venv\Scripts\python.exe scripts/data/21_forensic_legacy_vix.py

# GDELT: primero estimar. PROYECTO_GCP debe ser un proyecto propio autorizado.
gcloud auth application-default login
gcloud auth application-default set-quota-project PROYECTO_GCP
.venv\Scripts\python.exe scripts/data/30_fetch_gdelt.py --billing-project PROYECTO_GCP --maximum-gib 1250

# Ejecutar solo despues de revisar bytes estimados, tarifa y presupuesto.
.venv\Scripts\python.exe scripts/data/30_fetch_gdelt.py --billing-project PROYECTO_GCP --maximum-gib 1250 --execute

# Seleccion, integracion y comparacion
.venv\Scripts\python.exe scripts/data/35_audit_gdelt_candidates.py
.venv\Scripts\python.exe scripts/data/40_build_curated.py
.venv\Scripts\python.exe scripts/data/50_compare_legacy_curated.py
```

La consulta sellada leyo 1.209,42 GiB y produjo 7.430 filas agregadas. El costo
real depende de precios y cuotas vigentes. `--maximum-gib 1250` es un limite de
facturacion de la tarea, no una prediccion del cobro ni permiso para elevarlo.

El resultado validado tiene 2.558 filas entre 2015-02-19 y 2025-04-22:

```text
SHA-256: abf82d900314ce09cd00113785804e9f6742c38dd0867880ae9620c2a12f92ce
```

Verificacion en PowerShell:

```powershell
(Get-FileHash data/curated/model_input_ndx.csv -Algorithm SHA256).Hash.ToLower()
```

Los manifiestos en `data/manifests/` registran las URLs, la consulta SQL, el job
de BigQuery, los hashes de entrada y las reglas de calendario. La fecha
2017-08-29 se excluye por ausencia de tono; no se aplica `ffill` ni imputacion
neutral.

## Pipeline de modelado

Verificacion rapida aislada:

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe run_all.py --quick
```

Reproduccion completa recomendada:

```powershell
.venv\Scripts\python.exe run_all.py --fresh
```

`--fresh` mueve `data/processed`, `outputs` y `reports` a una carpeta
fechada dentro de `_run_archive/`, recrea los directorios vacios y ejecuta todas
las etapas. No borra los resultados anteriores. No se combina con `--quick`.

La cadena completa incluye preparacion, diagnostico ADF/KPSS y correlogramas,
RW, ARIMA, ARIMAX/SARIMAX, variantes LSTM, walk-forward, multi-semilla,
robustez 2025, evaluacion, figuras y regimenes. Se detiene ante cualquier codigo
de salida distinto de cero. El gate final valida contrato, ausencia de fuga y
`y_true` contra el dataset configurado.

## Salidas

- `outputs/preds/OOS/`: 28 archivos del OOS 2024.
- `outputs/preds/WF/`: 336 archivos walk-forward.
- `outputs/preds/MULTISEED/`: 120 archivos LSTM por semilla.
- `outputs/preds/OOS_2025/`: 28 archivos de robustez.
- `reports/data/`: metricas, DM-HLN, MZ, PT, ADF/KPSS y resumenes.
- `reports/figs/`: 28 figuras selladas, incluidos boxplot y correlogramas.
- `Tesis_Maestro_Final.docx`: tesis final actualizada con los resultados del
  dataset reconstruido.

## Validacion efectuada

El 2026-07-16 se ejecuto el pipeline sobre el dataset reconstruido en un
worktree aislado con Python 3.12.13. La verificacion definitiva con
`run_all.py --fresh` termino con `RUN_ALL_OK` en 38,0 minutos; todas las etapas
produjeron artefactos y el gate consolidado aprobo 17 pruebas. La primera
corrida limpia expuso una
dependencia de orden en el analisis multisemilla; fue corregida y cubierta con
una prueba de regresion antes de sellar los resultados.

Las 512 predicciones y las 14 tablas comparables fueron identicas byte por byte
en dos corridas independientes. Veinticinco de las 26 figuras tambien fueron
identicas. El boxplot multi-semilla diferia porque el jitter grafico no fijaba
semilla; se corrigio y dos regeneraciones consecutivas produjeron el mismo
SHA-256. Luego se agregaron ADF/KPSS, dos correlogramas y el resumen
multi-semilla por variante. `data/manifests/reproduction_results.json` sella los
hashes finales de 17 tablas y 28 figuras.

Los resultados reconstruidos no coinciden con RC2.1, porque RC2.1 provenia del
dataset heredado. `docs/reconstruccion_datos.md` resume las diferencias y su
interpretacion. La version Word vigente es
`Tesis_Maestro_Final.docx`.

El extracto verificable de la ejecucion sellada se conserva en
`logs/reproduction_final_summary_2026-07-16.log`. Incluye el SHA-256 del log
local completo, el gate de 17 casos de `run_all.py --fresh`, el cierre
`RUN_ALL_OK` y la ejecucion posterior de la suite completa con 19 casos.
