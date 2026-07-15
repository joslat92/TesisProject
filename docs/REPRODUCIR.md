# Apéndice B — Reproducción de punta a punta

Instrucciones verificadas en limpio el 2026-06-12 (Windows 11, Python 3.11).

## Requisitos
- Python 3.11 (el pipeline está congelado contra las versiones de
  `requirements.txt`; torch 2.9.1 instala la rueda CPU en Windows).
- Git. ~3 GB de disco (entorno + datos + artefactos).

## Pasos exactos

```powershell
# 1. Clonar (rama de la reestructuración)
git clone --branch reestructura-dic2025 https://github.com/joslat92/TesisProject.git
cd TesisProject

# 2. Entorno congelado
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3a. Verificación rápida de mecánica (~15-25 min, recomendada primero)
.venv\Scripts\python.exe run_all.py --quick

# 3b. Reproducción COMPLETA (horas; incluye WF 144 entrenamientos y
#     multi-semilla 120)
.venv\Scripts\python.exe run_all.py
```

`run_all.py` ejecuta la cadena 00→30 y se detiene si cualquier stage falla.
El **gate** del pipeline (validación de contrato fail-fast + suite anti-fuga +
verificación del y_true contra la fuente primaria, `tests/`) corre dentro de
`20_evaluate_stats` — si la corrida termina con `RUN_ALL_OK`, pasó todos los
gates.

## Qué produce
- `outputs/preds/` — predicciones por modelo/horizonte (OOS, WF, MULTISEED, OOS_2025)
- `reports/data/` — métricas y tests (metrics_OOS, metrics_WF, tbl_DM_OOS,
  dm_summary, tbl_MZ_core, tbl_PT_OOS, multiseed_*, *_2025)
- `reports/figs/` — 25 figuras canónicas + boxplot multi-semilla

## Modo --quick: qué se verifica y qué no
| Aspecto | --quick | completo |
|---|---|---|
| Preparación de datos, RW, ARIMA, ARIMAX/SARIMAX (OOS 2024 y 2025) | ✔ números EXACTOS de la tesis (deterministas) | ✔ |
| Gates (contrato, anti-fuga, y_true vs fuente primaria) | ✔ completos | ✔ |
| Variantes LSTM | mecánica ✔, números NO (epochs=2) | ✔ (epochs=30, seed 42) |
| Walk-forward (14) y multi-semilla (15) | omitidos | ✔ |

En `--quick` los configs se parchean en memoria de la corrida (epochs=2) y se
restauran al terminar.

## Verificación realizada (2026-06-12, script: scripts/cleanroom_test.ps1)
- Clon limpio en carpeta temporal + venv desde cero + `pip install -r
  requirements.txt` (con descarga de torch CPU) + `run_all.py --quick`:
  **RUN_ALL_OK en 6.1 min**, todos los gates pasados dentro del clon
  (contrato fail-fast, anti-fuga, y_true contra fuente primaria).
- Comparación de las métricas de los modelos clásicos (RW/ARIMA/ARIMAX/
  SARIMAX) del clon contra las del repositorio (RC2.1): **IDÉNTICAS** en
  `metrics_OOS.csv` y `metrics_OOS_2025.csv` (igualdad exacta de
  RMSE/MAE/MDA por horizonte) — los clásicos no dependen del flag --quick.
- Verificado en modo rápido (no repetido en el clon por costo): los números
  LSTM (epochs=30), el walk-forward (14) y el multi-semilla (15). Esos
  corresponden a la cadena completa con la que se generó RC2.1 en el
  repositorio principal (bitácora 2026-06-11/12), ejecutable con
  `run_all.py` sin flags.

## Notas de entorno
- Los resultados LSTM son deterministas dada la semilla en CPU con estas
  versiones; cambios de versión de torch pueden mover decimales.
- `data/raw/data.csv` viaja en el repositorio como fuente primaria; el snapshot
  de reproducibilidad (hash SHA-256 del crudo) queda en
  `reports/data/metadata_snapshot.csv`.
