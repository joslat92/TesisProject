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

La consulta evalua tres alcances sin usar resultados predictivos. La busqueda
exacta de NASDAQ-100 no produjo observaciones. `broad_nasdaq` y
`nasdaq_market` cubren 2.558 de 2.559 fechas, pero el segundo exige ademas el
tema GDELT `ECON_STOCKMARKET` y reduce el corpus de 6.936.770 a 5.277.627
articulos. La muestra documental confirma mayor presencia de noticias de
mercado, aunque conserva historias de empresas individuales. Por ello se sella
`nasdaq_market` y se interpreta como proxy de tono de noticias del mercado
Nasdaq/tecnologia, no como corpus exclusivo del indice NASDAQ-100.

La fecha 2017-08-29 no tiene tono en ninguno de los dos alcances con cobertura
alta. Se excluye y se registra en el reporte de calidad; no se aplica relleno ni
se imputa un valor neutral.

## Comparacion con el dataset heredado

El dataset curado contiene 2.558 filas entre 2015-02-19 y 2025-04-22. Frente al
heredado, quedan fuera las dos fechas anteriores al inicio de GDELT 2.0 y
2017-08-29 por falta de tono. La correlacion de retornos entre QQQ ajustado
heredado y NASDAQ-100 reconstruido es 0,998720. El VIX heredado (apertura) y el
cierre oficial reconstruido correlacionan 0,970155, con diferencia absoluta
media de 1,0376 puntos.

El cambio mayor esta en sentimiento: la correlacion entre la serie heredada de
origen desconocido y el proxy `nasdaq_market` es 0,107042. Durante las 75 fechas
en que el dato heredado repite exactamente -1,2462, la serie reconstruida tiene
75 valores distintos y desviacion estandar 0,7429. En consecuencia, los
resultados obtenidos con el dataset heredado no se transfieren al nuevo: deben
recalcularse mediante el pipeline completo antes de actualizar la tesis.

## Resultados de la reproduccion reconstruida

La corrida limpia uso el dataset con SHA-256
`abf82d900314ce09cd00113785804e9f6742c38dd0867880ae9620c2a12f92ce`.
Completo preparacion, modelos OOS, walk-forward, 120 entrenamientos
multi-semilla, robustez 2025, evaluacion, figuras y regimenes. El gate final
aprobo 17 pruebas y `run_all.py --fresh` termino con `RUN_ALL_OK` en 38,0
minutos.

En OOS 2024, LSTM_FULL obtuvo RMSE 0,03983 en h=20 frente a 0,04368 del RW,
pero el DM-HLN de la semilla canonica no rechazo igualdad predictiva
(p=0,1811). En diez semillas, la mediana del p-valor fue 0,2225, solo una
semilla quedo bajo 0,05 y el ensemble obtuvo p=0,2101. Por tanto, el resultado
no respalda una superioridad estadistica robusta de LSTM_FULL frente al RW.

En walk-forward, LSTM_FULL obtuvo RMSE medio 0,03593 en h=20 frente a 0,04136
del RW. Este resultado describe estabilidad temporal de error, pero no sustituye
la inferencia DM del OOS. En robustez enero-abril de 2025, RW tuvo el menor RMSE
en h=5, 10 y 20; ninguna comparacion DM contra RW rechazo al 5 %.

La primera corrida verdaderamente limpia tambien revelo que el analisis
multi-semilla dependia de `metrics_OOS.csv`, generado en una etapa posterior.
La dependencia quedaba oculta por artefactos antiguos. Se corrigio calculando la
referencia RW desde sus propias predicciones y se agrego una prueba de regresion.

Dos corridas independientes produjeron 512/512 predicciones y 14/14 tablas
comparables identicas byte por byte. Veinticinco de 26 figuras tambien fueron
identicas. La unica diferencia era el jitter no sembrado del boxplot
multi-semilla; tras fijar la semilla, dos regeneraciones consecutivas dieron el
mismo SHA-256. Los diagnosticos ADF/KPSS, los correlogramas y el resumen
multi-semilla por variante se agregaron despues de esa comparacion. El manifiesto
final `data/manifests/reproduction_results.json` sella 17 tablas, 28 figuras,
dataset, artefactos y resultados clave.

## Estado y decisiones

| Componente | Estado | Evidencia o bloqueo |
| --- | --- | --- |
| Dataset heredado | Congelado | Manifiesto y SHA-256 versionados |
| Identidad del objetivo | Cerrada | QQQ ajustado, confianza alta |
| Identidad del VIX | Cerrada | Apertura mal etiquetada, 2.561/2.561 filas reproducidas |
| NASDAQ-100 oficial | Descargado | FRED/Nasdaq, cierre diario |
| VIX oficial | Descargado | Cboe, apertura y cierre diarios |
| Consulta GDELT | Ejecutada | Job, consulta, bytes y SHA-256 versionados |
| Scope GDELT | Sellado | `nasdaq_market`, sin usar rendimiento predictivo |
| Dataset curado | Sellado | 2.558 filas y SHA-256 versionado |
| Nueva corrida del pipeline | Completada | Todas las etapas y gate final ejecutados |
| Actualizacion de la tesis | Completada | `Tesis Maestro Final - Datos Reconstruidos.docx`, 53 paginas verificadas |

No deben reutilizarse cifras, conclusiones ni afirmaciones de versiones de la
tesis que procedan del dataset heredado. La version reconstruida usa solo los
artefactos sellados de esta reproduccion; el maestro anterior se conserva como
evidencia historica y no como fuente de resultados vigentes.
