# Limpieza de repo y tesis final - 2026-07-15

Rama: `limpieza-repo-final`

## Cambios en el documento Word

Archivo: `Tesis Maestro Final.docx`

- Se completo la lista de figuras: ahora incluye Figura 1 y Figuras 7.1 a 7.7.
- Se corrigio la tabla de apendices: ahora lista A, B y C con paginas actualizadas.
- Se cambio el glosario de `Apendice B` a `Apendice C`.
- Se convirtieron las leyendas de Figuras 7.1 a 7.7 al estilo `Caption`.
- Se eliminaron 53 parrafos de bibliografia APA duplicada y se conservo la tabla numerada `[1]`-`[26]`.
- Se limpiaron 28 encabezados vacios y 9 leyendas vacias que ensuciaban la estructura Word.
- Se corrigieron rutas obsoletas en metodologia: `src/stages/`, `reports/figs/` y `run_all.py`.
- Se corrigio la regla estacional de `p<0` a `p<0,10`.
- Se corrigio un artefacto tipografico en la explicacion de asteriscos de significancia.

Verificacion:

- Microsoft Word abre el archivo sin reparacion.
- Render visual por Word a PDF y PNG: 57 paginas revisadas mediante hojas de contacto y paginas criticas.
- El renderer del paquete Documents no pudo usarse porque no encontro LibreOffice/soffice en este entorno; se uso Word COM como fallback.

## Archivos eliminados de la rama

Versiones antiguas, borradores o auxiliares:

- `Documento Tesis 4.docx`
- `Capitulo7_Borrador_v1_3.docx`
- `contrato.docx`
- `Contrato_Control_actualizado.xlsx`
- `pasos replicacion 220925 2230.xlsx`
- `Auditoria_Proyecto_Tesis.txt`
- `script de consolidacion.txt`

Duplicados / redundantes:

- `data/data.csv` - duplicado exacto de `data/raw/data.csv`.
- `reports/data/mz_summary_OOS.csv` - copia de `reports/data/tbl_MZ_core.csv`.

Artefactos no finales:

- `reports/figs/_archive_corrida_buggy_20251229/`
- `reports/figs.zip` - no versionado, duplicaba figuras.

## Archivos conservados deliberadamente

- `outputs/preds/`: predicciones selladas RC2.1. Siguen versionadas para permitir auditoria sin reentrenamiento completo.
- `data/processed/`: features/parquets generados, utiles para reproducibilidad rapida.
- `logs/`: bitacoras historicas. Mantienen menciones a rutas o artefactos antiguos por trazabilidad.
- `reports/data/` y `reports/figs/`: tablas y figuras finales usadas por la tesis.

## Pendientes / notas

- Las listas de tablas, figuras y apendices del Word quedaron como entradas estaticas con paginas actualizadas. Si se edita contenido antes de ellas o antes de las figuras/apendices, conviene regenerarlas.
- La tabla de referencias numeradas empieza al final de la pagina 47. Es legible y no se corta, pero podria mejorarse con un salto de pagina manual si se desea pulido estetico.
- `logs/bitacora.md` conserva menciones historicas a `data/data.csv`, `mz_summary_OOS.csv` y la carpeta buggy; no se reescribio para no alterar bitacora historica.
