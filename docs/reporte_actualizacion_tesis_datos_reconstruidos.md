# Reporte de actualizacion de la tesis con datos reconstruidos

Fecha de cierre: 2026-07-16  
Rama: `entrega-final-tutor`

## Entregables

- Version vigente: `Tesis_Maestro_Final.docx`.
- El maestro historico y los generadores de integracion permanecen en
  `reconstruccion-datos-origen`, no en la rama de entrega.
- Manifiesto de resultados: `data/manifests/reproduction_results.json`.

## Procedencia y reconstruccion

La auditoria confirmo que el repositorio original de adquisicion se perdio y que
el CSV heredado no demostraba la procedencia de sus campos. Su objetivo era QQQ
ajustado, no el NASDAQ-100 oficial; `VIX_Close` reproducia la apertura del VIX y
su ultimo valor estaba rellenado; la procedencia del sentimiento no pudo
recuperarse. El archivo heredado se conserva solo como evidencia historica.

El conjunto vigente se reconstruyo con cierre NASDAQ-100 de FRED/Nasdaq, cierre
del VIX de Cboe y un proxy de tono de mercado Nasdaq/tecnologia derivado de
GDELT. Contiene 2.558 filas entre 2015-02-19 y 2025-04-22 y SHA-256
`abf82d900314ce09cd00113785804e9f6742c38dd0867880ae9620c2a12f92ce`.
La consulta GDELT proceso 1.209,42 GiB; el costo observado fue aproximadamente
USD 5,98, sujeto a las tarifas y cuotas del proveedor.

## Cambios en el Word

- Se reemplazaron resumen, abstract, resultados, conclusiones, tablas y figuras
  que dependian de resultados heredados.
- La metodologia ahora documenta fuentes, calendario, transformaciones,
  causalidad de exogenas, contrato anti-fuga, hash y limitaciones de procedencia.
- Se incorporaron ADF/KPSS sobre `log(P)` y `ret_1d = Δlog(P)`, junto con
  correlogramas de `log(P)`. El diagnostico se presenta como descriptivo y no
  como selector dinamico del modelo.
- Se actualizaron DM-HLN, Mincer-Zarnowitz, Pesaran-Timmermann, walk-forward,
  multi-semilla, robustez 2025 y la discusion por regimen.
- Se agregaron las referencias [30]-[32] para FRED/Nasdaq, Cboe y GDELT.
- El Apendice B contiene el procedimiento real de reconstruccion y reproduccion.
- Se sincronizaron tabla de contenido, listas de tablas, figuras y apendices.
- Se corrigio la leyenda heredada `Continuacion: Tabla 4` a `Tabla 3`, se
  repitieron encabezados y se impidio dividir filas entre paginas.

## Resultados vigentes

- OOS 2024: RW fue mejor en T+1; ARIMAX/SARIMAX fue significativamente peor que
  RW (`p_HLN=0,0373`).
- En T+20, LSTM+Sent+VIX logro RMSE 0,03983 frente a 0,04368 de RW, sin
  diferencia significativa (`p_HLN=0,1811`).
- La LSTM canonica fue significativa en T+20 con semilla 42
  (`p_HLN=0,0201`), pero la mediana de diez semillas fue 0,0562 y el ensemble
  0,0584; la inferencia depende de la inicializacion.
- Walk-forward: LSTM+Sent+VIX obtuvo RMSE medio 0,03593 frente a 0,04136 de RW
  en T+20.
- Enero-abril de 2025: RW fue mejor en T+5, T+10 y T+20; ningun contraste contra
  RW fue significativo bajo HLN. La unica celda significativa del bloque fue
  LSTM frente a SARIMAX en T+20 (`p_HLN=0,0496`), no frente a RW, y se conserva
  como hallazgo aislado.

## Verificaciones

- Pipeline completo: `RUN_ALL_OK` en 38,0 minutos.
- Gate consolidado: 17 controles aprobados.
- Pruebas: 19 aprobadas.
- Validacion independiente: 14 casos ejecutables en un entorno parcial fueron
  aprobados; los cinco restantes no se ejecutaron por ausencia de `torch` y
  `pyarrow`. Esto no contradice la corrida interna de 19 casos en el entorno
  completo, pero la auditoria externa la clasifica como no verificada por ella.
- Reproduccion: 512/512 predicciones y 14/14 tablas comparables identicas entre
  dos corridas; jitter grafico corregido y determinista.
- Manifiesto final: 17 tablas y 28 figuras.
- Word: 56 paginas inspeccionadas visualmente, 13 tablas, 10 figuras, 20 objetos
  OMML de ecuacion, 32 referencias, sin comentarios ni cambios controlados.
- Terminos vetados en el maestro de partida y en la version final:

| Termino | Encontrado en el maestro | Eliminado | Ocurrencias finales |
| --- | ---: | ---: | ---: |
| `SARIMAX_BOTH` | 0 | 0 | 0 |
| `7.29` | 0 | 0 | 0 |
| `β≈0.96` | 0 | 0 | 0 |
| `β ≈ 0.96` | 0 | 0 | 0 |

El maestro de partida ya habia eliminado esos cuatro rastros en una integracion
anterior; esta fase verifico que no reaparecieran. Tambien se verifico ausencia
de los valores precierre `p=0,048`, `p=0,038`, mediana `0,190`, ensemble `0,136`
y de la etiqueta `RC2.1`.

## Dudas y limitaciones no ocultadas

1. La procedencia exacta del sentimiento heredado no puede recuperarse porque el
   repositorio original de adquisicion fue eliminado.
2. `nasdaq_market` es un proxy documental de noticias Nasdaq/tecnologia, no un
   corpus exclusivo de las empresas del indice ni una medida semantica validada
   manualmente.
3. Los archivos fuente no se redistribuyen en Git mientras se revisan licencias;
   la reproduccion desde cero depende de proveedores externos.
4. El costo de BigQuery no es fijo. Siempre debe ejecutarse el dry-run y revisar
   tarifa, cuota y presupuesto antes de `--execute`.
5. La reconstruccion crea una nueva base de evidencia. No valida ni debe
   presentarse como continuidad empirica de las cifras historicas.
6. `reproduction_final.log` no se publica. El manifiesto registra su SHA-256,
   distingue la verificacion local basada en ese log de una reproduccion
   independiente desde un clon y enlaza el resumen versionado
   `logs/reproduction_final_summary_2026-07-16.log`.
