# Reconstruccion y procedencia de los datos

## Motivo

El repositorio original de adquisicion de datos se perdio antes de consolidar
esta version del proyecto. El archivo `data/raw/data.csv` es un producto ya
integrado, no una fuente primaria, y por si solo no permite demostrar de donde
proviene cada campo. Por eso se conserva intacto como dataset heredado y se
reconstruye una capa nueva desde proveedores identificables.

## Evidencia sobre el dataset heredado

El manifiesto `data/manifests/legacy_data_manifest.json` fija el archivo
heredado mediante SHA-256. Contiene 2.561 filas entre 2015-02-17 y 2025-04-22.
El sentimiento permanece constante durante las ultimas 75 filas, desde
2025-01-02, lo que evidencia un relleno o una interrupcion de la fuente que no
puede justificarse con los archivos activos.

### Objetivo

`scripts/data/20_forensic_legacy_target.py` compara `Target_Price` con el
NASDAQ-100 oficial y, solo con fines forenses, con QQQ. Los retornos del objetivo
heredado y el cierre ajustado de QQQ tienen correlacion 0,9999999997 y la razon
entre niveles presenta un coeficiente de variacion de 0,00000024. El historial
Git tambien conserva el mismo precio junto con volumen negociado. La inferencia,
con confianza alta, es que el objetivo heredado fue el cierre ajustado del ETF
QQQ y no el nivel del indice NASDAQ-100.

La reconstruccion canonica usara el cierre diario de la serie `NASDAQ100`
publicada por FRED con fuente original Nasdaq. QQQ quedara como analisis de
sensibilidad; no se usara para escoger modelos ni para preservar resultados
anteriores.

### VIX

`scripts/data/21_forensic_legacy_vix.py` reproduce las 2.561 filas heredadas a
partir de `5674181:data/vix.csv` y `159ff23:src/merge_vix.py`. El CSV historico no
tenia encabezado y ordenaba sus campos como cierre, maximo, minimo, apertura y
volumen. El script les asigno los nombres apertura, maximo, minimo, cierre y
volumen, y selecciono el cuarto campo numerico creyendo que era el cierre. Ese
campo era la apertura. El ultimo valor, correspondiente a 2025-04-22, fue ademas
rellenado desde 2025-04-21 mediante `ffill`.

La reconstruccion canonica usara el cierre diario oficial de Cboe. La apertura
se conserva en la capa de fuente para auditoria, pero no se renombra ni se
rellena implicitamente.

## Reconstruccion canonica

Las decisiones de instrumento, campo, periodo, calendario y faltantes se
versionan en `config_data.yaml`. Las descargas viven en `data/source/` y no se
publican en Git mientras se revisan sus condiciones de redistribucion. Los
manifiestos versionados registran URL, fecha de consulta, parametros, conteos y
SHA-256.

El calendario canonico es la interseccion de fechas observadas del NASDAQ-100 y
el VIX. Esta regla excluye de forma visible 2019-04-19, fecha presente en FRED
con un valor casi repetido pese a no existir observacion VIX. La exclusion se
registra en el reporte de calidad y no ocurre silenciosamente.

Para GDELT, cada dia calendario se asigna a la primera fecha de mercado igual o
posterior. Asi, las noticias del fin de semana se incorporan al lunes en vez de
perderse o asignarse retrospectivamente. El tono de varios dias se agrega como
media ponderada por cantidad de articulos.

## Estado y decisiones pendientes

| Componente | Estado | Evidencia o bloqueo |
| --- | --- | --- |
| Dataset heredado | Congelado | Manifiesto y SHA-256 versionados |
| Identidad del objetivo | Cerrada | QQQ ajustado, confianza alta |
| Identidad del VIX | Cerrada | Apertura mal etiquetada, 2.561/2.561 filas reproducidas |
| NASDAQ-100 oficial | Descargado | FRED/Nasdaq, cierre diario |
| VIX oficial | Descargado | Cboe, apertura y cierre diarios |
| Consulta GDELT | Versionada | Falta autenticacion y proyecto de facturacion BigQuery |
| Scope GDELT | Pendiente | Debe decidirse por cobertura y revision de relevancia |
| Dataset curado | Bloqueado por diseno | El constructor falla hasta sellar el scope GDELT |
| Nueva corrida del pipeline | Pendiente | Solo procede despues de sellar el dataset curado |
| Actualizacion de la tesis | Pendiente | Depende de los resultados de la nueva corrida |

No se actualizaran cifras, conclusiones ni afirmaciones del documento de tesis
antes de ejecutar y auditar el pipeline completo con el dataset reconstruido.
