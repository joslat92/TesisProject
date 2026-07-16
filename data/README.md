# Capas de datos

El archivo historico `data/raw/data.csv` se conserva sin cambios para comparar
los resultados anteriores. No debe considerarse una fuente primaria: ya integra
precios, sentimiento, VIX y variables derivadas.

La reconstruccion utiliza estas capas:

- `data/source/`: descargas y normalizaciones de cada proveedor. Esta excluida de
  Git mientras se revisan licencias de redistribucion.
- `data/manifests/`: URLs, fechas de consulta, parametros, conteos y SHA-256.
- `data/quality/`: perfiles y comparaciones entre fuentes.
- `data/curated/`: dataset integrado que alimentara el pipeline una vez sellado.
- `data/processed/`: features generadas por el pipeline existente.

Orden previsto:

```powershell
python scripts/data/00_snapshot_legacy.py
python scripts/data/10_fetch_market_data.py
python scripts/data/20_forensic_legacy_target.py
python scripts/data/21_forensic_legacy_vix.py
python scripts/data/30_fetch_gdelt.py --billing-project PROYECTO_GCP
python scripts/data/30_fetch_gdelt.py --billing-project PROYECTO_GCP --execute
python scripts/data/35_audit_gdelt_candidates.py
# Tras revisar cobertura y URLs, registrar gdelt.selected_scope en config_data.yaml
python scripts/data/40_build_curated.py
python scripts/data/50_compare_legacy_curated.py
```

La consulta GDELT hace un `dry-run` por defecto y muestra los bytes estimados.
No se ejecuta ni genera costos sin la bandera explicita `--execute`.
La ejecucion real tiene ademas un tope predeterminado de 850 GiB mediante
`maximum_bytes_billed`; puede reducirse con `--maximum-gib`, pero nunca debe
aumentarse sin revisar primero la cuota disponible del proyecto.

`config_data.yaml` deja selladas las decisiones de fuente, campo, calendario,
agregacion y faltantes. El constructor se detiene mientras el scope GDELT no se
haya elegido y tambien falla ante cualquier faltante: no aplica `ffill` ni
imputa ceros de forma silenciosa.

Tanto `data/source/` como `data/curated/` se regeneran localmente y permanecen
fuera de Git hasta cerrar la revision de licencias de redistribucion. Sus
manifiestos, hashes, consultas y reportes de calidad si se versionan.
