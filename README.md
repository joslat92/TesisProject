# TesisProject - entrega final para tutor

Repositorio reproducible de la tesis de maestria sobre pronostico
multi-horizonte del NASDAQ-100 con modelos clasicos de series de tiempo y
variantes LSTM con VIX y tono de noticias GDELT.

## Entregable principal

- Documento final: `Tesis Final.docx`.
- Guia de construccion de datos: `docs/construccion_datos.md`.
- Guia de reproduccion: `docs/REPRODUCIR.md`.

## Evidencia reproducible

- `scripts/data/`: adquisicion, validacion e integracion de fuentes.
- `data/manifests/`: URLs, consulta, parametros, versiones y hashes SHA-256.
- `data/quality/`: controles de cobertura y calidad de las fuentes canonicas.
- `src/`: preparacion, modelado, evaluacion estadistica y figuras.
- `tests/`: 27 pruebas de contrato, anti-fuga, construccion de datos, `y_true`,
  inferencia y orquestacion.
- `outputs/preds/`: 512 predicciones selladas.
- `reports/data/` y `reports/figs/`: tablas y figuras usadas en la tesis.
- `logs/reproduction_latest_summary.log`: extracto verificable del ultimo cierre.

El dataset canonico no se publica en Git. Se construye como
`data/curated/model_input_ndx.csv` a partir de FRED/Nasdaq, Cboe y GDELT, y
debe producir:

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

Resultado de cierre: `27 passed`. El modo `--quick` trabaja en una copia
temporal y no modifica los artefactos canonicos.

Despues de editar el Word en Windows, sus indices y listas se pueden actualizar
con `scripts/finalize_tutor_word.ps1`.

La reproduccion completa se ejecuta, despues de construir el dataset, con:

```powershell
.venv\Scripts\python.exe run_all.py --fresh
```

`--fresh` archiva los resultados existentes en `_run_archive/`, ejecuta una
corrida nueva, valida las 27 pruebas y regenera el resumen y el manifiesto de
integridad v3. La corrida sellada del 22 de agosto de 2026 termino en 19,916
minutos, con 512 predicciones, 19 tablas y 28 figuras. No se debe ejecutar la
consulta GDELT sin revisar primero el dry-run, la cuota y el presupuesto del
proyecto de Google Cloud.

La version academica definitiva se identifica con el tag
`entrega-definitiva-2026-08-22`. El uso de asistencia de IA y los limites de
licencia y redistribucion se documentan en `docs/USO_IA.md` y `NOTICE.md`.
