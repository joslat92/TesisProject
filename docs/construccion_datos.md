# Construccion y trazabilidad de la base de datos

## Fuentes

La base diaria integra tres fuentes identificadas:

1. NASDAQ-100: cierre diario de la serie `NASDAQ100` publicada por FRED, cuya
   fuente original es Nasdaq.
2. VIX: cierre diario del archivo historico publicado por Cboe Global Markets.
3. Noticias: GDELT 2.0 Global Knowledge Graph, consultado mediante el conjunto
   publico `gdelt-bq.gdeltv2.gkg_partitioned` en Google BigQuery.

Las URLs, las fechas de descarga, los parametros, la consulta SQL, el job de
BigQuery y los hashes SHA-256 se conservan en `data/manifests/`.

## Indicador de tono

El sentimiento es la media diaria, ponderada por numero de articulos, del
primer componente de `V2Tone`. El alcance sellado `nasdaq_market` exige una
mencion organizacional a Nasdaq y el tema `ECON_STOCKMARKET`.

El alcance se selecciono por cobertura y relevancia documental, no por
rendimiento predictivo. Por ello, la variable se interpreta como un proxy del
tono de noticias del mercado Nasdaq y del sector tecnologico, no como un corpus
exclusivo de las empresas del NASDAQ-100.

## Calendario e integracion

El calendario canonico es la interseccion de las fechas observadas del
NASDAQ-100 y el VIX. Las noticias publicadas en fines de semana se asignan a la
primera fecha de mercado posterior.

No se aplica `forward-fill`, interpolacion ni imputacion neutral. La fecha
2019-04-19 se excluye por no tener observacion VIX y 2017-08-29 por ausencia de
tono GDELT.

El resultado contiene 2.558 filas entre 2015-02-19 y 2025-04-22:

```text
SHA-256: abf82d900314ce09cd00113785804e9f6742c38dd0867880ae9620c2a12f92ce
```

## Secuencia de construccion

Despues de crear el entorno e instalar `requirements.txt` y
`requirements-data.txt`, se ejecutan las etapas siguientes:

```powershell
.venv\Scripts\python.exe scripts/data/10_fetch_market_data.py

# Dry-run obligatorio antes de autorizar la consulta con costo.
.venv\Scripts\python.exe scripts/data/30_fetch_gdelt.py `
  --billing-project PROYECTO_GCP --maximum-gib 1250

.venv\Scripts\python.exe scripts/data/30_fetch_gdelt.py `
  --billing-project PROYECTO_GCP --maximum-gib 1250 --execute

.venv\Scripts\python.exe scripts/data/35_audit_gdelt_candidates.py
.venv\Scripts\python.exe scripts/data/40_build_curated.py
```

La consulta sellada leyo 1.209,42 GiB y genero 7.430 filas agregadas. El costo
real depende de los precios y cuotas vigentes; `--maximum-gib 1250` limita los
bytes facturables de la tarea y no sustituye la revision previa del presupuesto.

## Verificacion

Antes de ejecutar los modelos se comprueban el numero de filas y el hash:

```powershell
(Get-FileHash data/curated/model_input_ndx.csv -Algorithm SHA256).Hash.ToLower()
```

El pipeline aplica ademas un contrato de datos con cierre inmediato, controles
anti-fuga y verificacion de `y_true` contra el conjunto canonico. Los resultados
sellados se relacionan con los datos de entrada mediante
`data/manifests/reproduction_results.json`.
