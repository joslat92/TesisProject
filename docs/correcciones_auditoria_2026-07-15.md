# Correcciones posteriores a la auditoría

Fecha: 2026-07-15
Rama: `correcciones-auditoria-final`
Base: `limpieza-repo-final`

## Alcance ejecutado

### Documento de tesis

- Se sincronizaron el resumen, abstract, capítulos 7 y 8, tablas y conclusiones con los resultados sellados del repositorio: DM/HLN, evaluación multisemilla y robustez 2025.
- Se corrigieron las ecuaciones ADF, KPSS y Mincer-Zarnowitz como ecuaciones nativas de Word.
- Se alineó la metodología con el pipeline: siete modelos activos, ARIMA(1,0,1), gate estacional, reestimación OOS/WF, HAC con `maxlags=h-1` y rutas reales de artefactos.
- Se corrigió la Figura 7.3 para representar niveles observados frente a pronosticados; también se regeneraron las cuatro figuras MZ canónicas.
- Se mantuvo el sistema de citación numérico y se incorporaron las referencias de Harvey-Leybourne-Newbold, Pesaran-Timmermann y Mincer-Zarnowitz. La bibliografía final contiene 29 entradas.
- Se actualizaron el índice, la lista de tablas y la lista de figuras.
- Se verificó que no quedaran las expresiones vetadas `SARIMAX_BOTH`, `7.29`, `β≈0.96` ni `β ≈ 0.96`.

### Código y reproducibilidad

- `run_all.py --quick` ahora se ejecuta en una copia temporal, usa dos épocas para LSTM, omite WF/multisemilla y elimina la copia al terminar. No modifica resultados canónicos.
- El contrato valida nombres, columnas, valores finitos, duplicados, orden temporal, consistencia de `y_true` y completitud esperada de OOS y WF.
- La preparación de datos falla de forma explícita ante columnas ausentes, nulos, fechas duplicadas o precios no positivos.
- La evaluación Mincer-Zarnowitz dejó de ocultar excepciones y exige entradas completas antes de producir resultados.
- Los tests de cordura omiten correctamente un bloque que todavía no contiene predicciones, aunque su directorio ya exista.
- Se eliminaron módulos y scripts heredados que no pertenecían al pipeline activo: `00_split_data.py`, `05_features.py`, `23_final_comparison.py`, `30_reports.py`, `dataio.py`, `features.py`, `trading_calendar.py` y `ask_gemini.py`.
- Se actualizaron `README.md`, `docs/REPRODUCIR.md` y los comentarios de configuración para reflejar el flujo real.

## Verificación realizada

1. Entorno limpio de Python 3.12 con las versiones exactas de `requirements.txt`.
2. `python -m pytest -q`: **10 pruebas aprobadas**.
3. `python src/core/contract.py`: **28 artefactos OOS y 336 artefactos WF validados**, consistencia de verdad observada aprobada y **10 pruebas aprobadas** dentro del gate.
4. `python run_all.py --quick`: **RUN_ALL_OK en 9,3 minutos**. Se ejecutaron preparación, RW, ARIMA, ARIMAX/SARIMAX, LSTM, robustez 2025, evaluación, figuras y regímenes; WF y multisemilla se omitieron por diseño.
5. Huella conjunta antes y después de `--quick`, sobre 556 archivos de `outputs/`, `reports/`, configuraciones y tesis: `78f868d57679da0ae4bbefbb65e1525fe1e5029175b6b9513775af985199b6f0` en ambos casos.
6. `git diff --check`: sin errores de whitespace.
7. Render del Word mediante Microsoft Word y revisión visual de las **58 páginas** con Poppler: sin páginas en blanco accidentales, recortes ni superposiciones detectadas.
8. Comparación interna del paquete DOCX: se conservaron los 11 recursos de imagen; únicamente cambió la Figura 7.3 prevista.

## Pendientes que requieren decisión humana

### Procedencia de `Target_Price`

`data/raw/data.csv` contiene 2.561 observaciones con `Target_Price` entre 89,9071 y 538,7167. Esa escala parece incompatible con niveles nominales del índice NASDAQ-100, mientras que la tesis presenta la variable como nivel del índice. El repositorio no contiene una fuente o transformación que permita decidir si se trata de QQQ, una serie ajustada/reescalada u otro instrumento. No se modificó el dato ni la redacción a falta de evidencia. Debe documentarse la fuente, símbolo, proveedor y transformaciones aplicadas.

### Accesibilidad del Word

La inspección detectó 8 imágenes sin texto alternativo y 12 tablas sin marcación formal de fila de encabezado. Son características heredadas del documento base y no afectan el cálculo ni la paginación, pero conviene corregirlas si la institución exige accesibilidad documental.

## Archivos principales

- Documento final: `Tesis Maestro Final.docx`
- Pipeline: `run_all.py`
- Contrato: `src/core/contract.py`
- Evaluación: `src/stages/20_evaluate_stats.py`
- Figuras: `src/stages/30_make_figures.py`
- Script reproducible de edición: `scripts/update_tesis_final.py`
- Finalización de campos Word: `scripts/finalize_tesis_word.ps1`
