# Reproduccion de punta a punta

Esta guia separa la construccion de los datos desde proveedores identificados y
la ejecucion del pipeline de modelado sobre el dataset resultante.

## Requisitos

- Windows 11 o un entorno equivalente con Python 3.12.
- Git.
- Aproximadamente 3 GB de espacio disponible.
- Para consultar GDELT: un proyecto de Google Cloud con BigQuery habilitado,
  facturacion activa y credenciales de aplicacion.

La consulta GDELT puede generar costo. El dry-run y la revision del limite de
bytes son obligatorios antes de usar `--execute`.

## Clon y entorno

```powershell
git clone https://github.com/joslat92/TesisProject.git
cd TesisProject
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -r requirements-data.txt
```

## Construccion de datos

El pipeline canonico lee `data/curated/model_input_ndx.csv`. Este archivo no se
publica en Git y debe construirse a partir de FRED/Nasdaq, Cboe y GDELT.

```powershell
# Fuentes de mercado
.venv\Scripts\python.exe scripts/data/10_fetch_market_data.py

# GDELT: primero estimar. PROYECTO_GCP debe ser un proyecto autorizado.
gcloud auth application-default login
gcloud auth application-default set-quota-project PROYECTO_GCP
.venv\Scripts\python.exe scripts/data/30_fetch_gdelt.py `
  --billing-project PROYECTO_GCP --maximum-gib 1250

# Ejecutar solo despues de revisar bytes estimados, tarifa y presupuesto.
.venv\Scripts\python.exe scripts/data/30_fetch_gdelt.py `
  --billing-project PROYECTO_GCP --maximum-gib 1250 --execute

# Seleccion e integracion
.venv\Scripts\python.exe scripts/data/35_audit_gdelt_candidates.py
.venv\Scripts\python.exe scripts/data/40_build_curated.py
```

La consulta sellada leyo 1.209,42 GiB y produjo 7.430 filas agregadas. El costo
real depende de precios y cuotas vigentes. `--maximum-gib 1250` limita los bytes
facturables de la tarea; no es una prediccion del cobro ni una autorizacion para
elevar el presupuesto.

El resultado validado contiene 2.558 filas entre 2015-02-19 y 2025-04-22:

```text
SHA-256: abf82d900314ce09cd00113785804e9f6742c38dd0867880ae9620c2a12f92ce
```

Verificacion en PowerShell:

```powershell
(Get-FileHash data/curated/model_input_ndx.csv -Algorithm SHA256).Hash.ToLower()
```

Los manifiestos en `data/manifests/` registran las URLs, la consulta SQL, el job
de BigQuery, los hashes de entrada y las reglas de calendario. No se aplica
`forward-fill`, interpolacion ni imputacion neutral. La construccion se describe
con mayor detalle en `docs/construccion_datos.md`.

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

`--fresh` verifica primero que exista el dataset canonico y que coincidan sus
2.558 filas y su SHA-256. Luego mueve `data/processed`, `outputs` y `reports` a
una carpeta fechada dentro de `_run_archive/`, recrea los directorios vacios y
ejecuta todas las etapas. No borra los resultados anteriores y no se combina
con `--quick`.

La cadena incluye preparacion, diagnostico ADF/KPSS y correlogramas, RW, ARIMA,
ARIMAX/SARIMAX, variantes LSTM, walk-forward, multi-semilla, robustez 2025,
evaluacion, figuras y regimenes. Se detiene ante cualquier codigo de salida
distinto de cero. El gate final valida el contrato, la ausencia de fuga,
`y_true` contra el dataset configurado, la ventana LSTM que termina en la fecha
de origen y las rutinas de inferencia. Una corrida completa exitosa genera
automaticamente `logs/reproduction_latest_summary.log` y
`data/manifests/reproduction_results.json` (manifiesto v3 con hashes portables).

## Salidas

- `outputs/preds/OOS/`: 28 archivos del OOS 2024.
- `outputs/preds/WF/`: 336 archivos walk-forward.
- `outputs/preds/MULTISEED/`: 120 archivos LSTM por semilla.
- `outputs/preds/OOS_2025/`: 28 archivos de robustez.
- `reports/data/`: 19 tablas de metricas, DM-HLN, correcciones Holm/BH,
  ablaciones, MZ conjunta, PT, ADF/KPSS y resumenes.
- `reports/figs/`: 28 figuras selladas, incluidos boxplot y correlogramas.
- `Tesis Final.docx`: tesis final con los resultados del dataset canonico.

## Validacion efectuada

El 22 de agosto de 2026 se ejecuto el pipeline completo con Python 3.12.13 desde
arboles de artefactos vacios. La corrida mediante `run_all.py --fresh` termino
con `RUN_ALL_OK` en 19,916 minutos y la suite final registro `27 passed`.

La auditoria independiente verifico 512/512 predicciones, reconstruyo cada
`y_true` desde el dataset canonico, recalculo las metricas y comprobo los hashes
portables de 19 tablas y 28 figuras. El manifiesto v3 en
`data/manifests/reproduction_results.json` conserva el commit ejecutado, las
versiones del entorno y la evidencia de la corrida. El extracto verificable se
encuentra en `logs/reproduction_latest_summary.log`.

Para reproducir exactamente la entrega evaluada, usar el tag
`entrega-definitiva-2026-08-22` y construir primero el dataset, que no se
distribuye en Git.
