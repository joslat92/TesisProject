# TesisProject

Repositorio de la tesis de maestria sobre pronostico multi-horizonte del
NASDAQ-100 con modelos clasicos de series de tiempo y variantes LSTM con
variables exogenas.

## Estado actual

- Rama de reconstruccion: `reconstruccion-datos-origen`.
- Dataset heredado: congelado solo para auditoria en `data/raw/data.csv`.
- Dataset canonico: se regenera localmente en `data/curated/model_input_ndx.csv`.
- Estudio principal: OOS 2024, horizontes `h in {1, 5, 10, 20}`.
- Bloque adicional: robustez enero-abril de 2025.
- Documento Word: pendiente de alineacion con los resultados reconstruidos.

## Capas principales

- `scripts/data/`: adquisicion, auditoria forense, integracion y comparacion.
- `data/manifests/`: fuentes, consultas, parametros y hashes SHA-256.
- `data/quality/`: controles de cobertura y comparacion con el legado.
- `src/`: pipeline de modelado y evaluacion.
- `tests/`: contrato, anti-fuga, sanidad de `y_true` y regresiones del orquestador.
- `outputs/preds/`: predicciones selladas por modelo y horizonte.
- `reports/data/` y `reports/figs/`: tablas y figuras derivadas.
- `docs/reconstruccion_datos.md`: decisiones y hallazgos de procedencia.
- `docs/REPRODUCIR.md`: instrucciones completas.

## Verificacion rapida

Despues de reconstruir el dataset canonico y crear el entorno:

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe run_all.py --quick
```

La corrida completa se inicia con:

```powershell
.venv\Scripts\python.exe run_all.py --fresh
```

`--fresh` mueve los artefactos existentes a `_run_archive/` y crea arboles
vacios antes de ejecutar. No elimina resultados previos. El modo `--quick`
trabaja en una copia temporal y no modifica los artefactos canonicos.
