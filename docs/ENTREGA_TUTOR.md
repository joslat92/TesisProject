# Entrega final para tutor

Este repositorio contiene la tesis vigente y la evidencia necesaria para
revisar o reproducir sus resultados.

## Contenido principal

- `Tesis Final.docx`: documento academico final.
- `src/`, `run_all.py` y `config*.yaml`: pipeline ejecutable y configuracion.
- `tests/`: 19 pruebas del contrato experimental y la construccion de datos.
- `scripts/data/`: adquisicion documentada de NASDAQ-100, VIX y GDELT.
- `data/manifests/` y `data/quality/`: procedencia, hashes y controles.
- `outputs/preds/`, `reports/data/` y `reports/figs/`: evidencia sellada usada
  en resultados, tablas y figuras.
- `Registro_Decisiones_1.md`: decisiones metodologicas y alcance.

## Fuentes y alcance

La base diaria integra el cierre del NASDAQ-100 publicado por FRED con fuente
original Nasdaq, el cierre diario del VIX publicado por Cboe y un indicador de
tono derivado de GDELT 2.0 GKG. El alcance `nasdaq_market` se eligio por
cobertura y relevancia documental, no por rendimiento predictivo, y se
interpreta como un proxy de noticias Nasdaq/tecnologia.

El dataset canonico no se redistribuye en Git. Su construccion depende de la
disponibilidad de las fuentes y, para GDELT, de un proyecto de BigQuery con
facturacion activa. Antes de ejecutar una consulta con costo se debe revisar el
dry-run, la cuota y el presupuesto. Consulte `docs/REPRODUCIR.md`.
