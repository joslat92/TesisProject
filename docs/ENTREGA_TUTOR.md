# Entrega final para tutor

Esta rama contiene la tesis vigente y la evidencia necesaria para revisar o
reproducir sus resultados. No se hizo merge sobre la rama de reconstruccion.

## Contenido principal

- `Tesis_Maestro_Final.docx`: documento academico final.
- `src/`, `run_all.py` y `config*.yaml`: pipeline ejecutable y configuracion.
- `tests/`: 19 pruebas del contrato experimental y la reconstruccion.
- `scripts/data/`: reconstruccion documentada de NASDAQ-100, VIX y GDELT.
- `data/manifests/` y `data/quality/`: procedencia, hashes y controles.
- `outputs/preds/`, `reports/data/` y `reports/figs/`: evidencia sellada usada
  en resultados, tablas y figuras.
- `Registro_Decisiones_1.md`: decisiones metodologicas y alcance.

## Archivos retirados de esta rama

Los siguientes antecedentes siguen disponibles en
`reconstruccion-datos-origen`, pero se excluyen de la entrega porque fueron
reemplazados o eran material interno de trabajo:

- maestro Word anterior y generadores de integracion;
- instrucciones de contexto para otros asistentes;
- informes intermedios de cierre, correccion y limpieza;
- scripts de auditorias puntuales ya cubiertas por la suite actual;
- bitacoras y logs rapidos anteriores al cierre sellado.

## Limitaciones que deben permanecer visibles

El repositorio original de adquisicion se perdio y el dataset heredado no
demostraba la procedencia de sus campos. Por esa razon, los resultados finales
se obtuvieron sobre un dataset reconstruido desde fuentes identificadas. El
dataset canonico no se redistribuye en Git y la reconstruccion de GDELT depende
de BigQuery, facturacion activa y precios vigentes. Consulte
`docs/REPRODUCIR.md` antes de ejecutar cualquier consulta con costo.
