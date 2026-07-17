# Capas de datos

La construccion del dataset utiliza estas capas:

- `data/source/`: descargas y normalizaciones de cada proveedor. Esta excluida
  de Git mientras se revisan licencias de redistribucion.
- `data/manifests/`: URLs, fechas de consulta, parametros, conteos y SHA-256.
- `data/quality/`: controles de cobertura y calidad de las fuentes canonicas.
- `data/curated/`: dataset integrado sellado que alimenta el pipeline.
- `data/processed/`: features generadas por el pipeline.

Secuencia:

```powershell
python scripts/data/10_fetch_market_data.py
python scripts/data/30_fetch_gdelt.py --billing-project PROYECTO_GCP
python scripts/data/30_fetch_gdelt.py --billing-project PROYECTO_GCP --maximum-gib 1250 --execute
python scripts/data/35_audit_gdelt_candidates.py
python scripts/data/40_build_curated.py
```

La consulta GDELT hace un `dry-run` por defecto y muestra los bytes estimados.
No se ejecuta ni genera costos sin la bandera explicita `--execute`. La consulta
sellada se estimo en 1.209,42 GiB. El valor `1250` es un limite de facturacion,
no una estimacion ni una autorizacion abierta. El costo depende de la tarifa y
la cuota vigentes del proyecto de Google Cloud.

`config_data.yaml` deja selladas las decisiones de fuente, campo, calendario,
agregacion y faltantes. El constructor se detiene mientras no se haya elegido el
scope GDELT y falla ante cualquier faltante: no aplica `ffill` ni imputa ceros.

`data/source/` y `data/curated/` se generan localmente y permanecen fuera de Git.
Sus manifiestos, hashes, consultas y reportes de calidad si se versionan.
