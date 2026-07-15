# TesisProject

Repositorio de la tesis de maestria sobre pronostico multi-horizonte del NASDAQ-100
con modelos clasicos de series de tiempo y variantes LSTM con variables exogenas.

## Estado actual

- Rama de correcciones: `correcciones-auditoria-final`
- Documento final editable: `Tesis Maestro Final.docx`
- Set de resultados vigente: RC2.1
- Estudio principal: OOS 2024, horizontes `T in {1, 5, 10, 20}`
- Bloque adicional: robustez enero-abril 2025

## Estructura principal

- `src/`: codigo del pipeline reproducible.
- `tests/`: pruebas de consistencia, anti-fuga y sanidad de `y_true`.
- `scripts/`: scripts auxiliares de verificacion y entorno.
- `data/raw/data.csv`: fuente primaria versionada.
- `reports/data/`: tablas finales usadas por la tesis.
- `reports/figs/`: figuras canonicas usadas por la tesis.
- `docs/REPRODUCIR.md`: guia de reproduccion detallada.

## Reproduccion rapida

En PowerShell, desde la raiz del repositorio:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe run_all.py --quick
```

La corrida completa sin `--quick` reproduce tambien los bloques costosos de LSTM,
walk-forward y multi-semilla.

El modo `--quick` crea una copia temporal del proyecto, ejecuta allí el smoke test
y elimina la copia al terminar. No modifica `outputs/`, `reports/` ni los YAML del
árbol de trabajo principal.

## Notas de limpieza

Esta rama busca conservar solo los insumos necesarios para reproducir resultados y
entregar el documento final. Versiones antiguas, borradores, zips, caches locales y
artefactos marcados como buggy deben quedar fuera de la rama final limpia.
