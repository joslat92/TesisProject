# TesisProject - entrega final para tutor

Repositorio reproducible de la tesis de maestria sobre pronostico
multi-horizonte del NASDAQ-100 con modelos clasicos de series de tiempo y
variantes LSTM con VIX y tono de noticias GDELT.

## Entregable principal

- Documento final: `Tesis Final.docx`.
- Rama de entrega: `entrega-final-tutor`.
- Rama con antecedentes y archivos retirados: `reconstruccion-datos-origen`.
- Informe de actualizacion: `docs/reporte_actualizacion_tesis_datos_reconstruidos.md`.

## Evidencia reproducible

- `scripts/data/`: adquisicion, auditoria forense e integracion de fuentes.
- `data/manifests/`: URLs, consulta, parametros, versiones y hashes SHA-256.
- `data/quality/`: controles de cobertura, procedencia y comparacion con el legado.
- `src/`: preparacion, modelado, evaluacion estadistica y figuras.
- `tests/`: contrato, anti-fuga, reconstruccion, `y_true` y orquestacion.
- `outputs/preds/`: 512 predicciones selladas.
- `reports/data/` y `reports/figs/`: tablas y figuras usadas en la tesis.
- `logs/reproduction_final_summary_2026-07-16.log`: extracto verificable del cierre.

El dataset canonico no se publica en Git. Se reconstruye como
`data/curated/model_input_ndx.csv` y debe producir:

```text
2.558 filas
SHA-256 abf82d900314ce09cd00113785804e9f6742c38dd0867880ae9620c2a12f92ce
```

La guia completa, incluido el control de costo antes de consultar BigQuery,
esta en `docs/REPRODUCIR.md`.

## Verificacion rapida

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe run_all.py --quick
```

Resultado de cierre: `19 passed`. El modo `--quick` trabaja en una copia
temporal y no modifica los artefactos canonicos.

Despues de editar el Word en Windows, sus indices y listas se pueden actualizar
con `scripts/finalize_tutor_word.ps1`.

La reproduccion completa se ejecuta, despues de reconstruir el dataset, con:

```powershell
.venv\Scripts\python.exe run_all.py --fresh
```

`--fresh` archiva los resultados existentes en `_run_archive/` antes de crear
una corrida nueva. No se debe ejecutar la consulta GDELT sin revisar primero el
dry-run, la cuota y el presupuesto del proyecto de Google Cloud.
