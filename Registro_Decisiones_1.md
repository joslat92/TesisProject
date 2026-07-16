# Registro de Decisiones — pendientes de ratificación del director

Proyecto: Tesis MMACC — Pronóstico multi-horizonte NASDAQ-100 (clásicos vs. LSTM con exógenas)
Contexto: decisiones tomadas durante la ausencia del director (junio 2026) para no detener
la retoma del proyecto. Cada decisión es reversible y está justificada para ratificación
o ajuste a su regreso. Detalle técnico completo en logs/bitacora.md del repositorio.

## D1 — Alcance del estudio: versión multi-horizonte
**Decisión:** la tesis final corresponde al estudio multi-horizonte (T ∈ {1,5,10,20} sobre
retornos acumulados), no al estudio T+1 en niveles del borrador "Documento Tesis 4".
**Justificación:** mayor aporte académico; los experimentos están completos; la formulación
en retornos evita que la calibración refleje mera persistencia del nivel (el R²≈0.94 del
estudio T+1 en MZ se explica principalmente por P(t+1)≈P(t)).
**Reversibilidad:** el estudio T+1 queda documentado y reproducible como antecedente.

## D2 — Hiperparámetros canónicos del LSTM
**Decisión:** ventana=40, 64 unidades, dropout=0.2, Adam lr=0.001, 30 épocas, batch=64,
seed canónica 42, early stopping (val = último 20% del train) con embargo de h días.
**Justificación:** es la especificación publicada en el borrador de tesis (las dos
alternativas encontradas en archivos de control carecen de documento asociado). El early
stopping con embargo corrige una omisión del código respecto de lo que el documento declara.

## D3 — Bloque de robustez con datos 2025
**Decisión:** se añade una subsección de robustez con OOS extendido (enero–abril 2025,
incluye el shock de volatilidad de abril 2025). El estudio principal permanece con
OOS = 2024; los datos de 2025 no participan en ningún entrenamiento del estudio principal.
**Justificación:** costo marginal mínimo (datos ya integrados, pipeline parametrizado);
prueba de estrés de la conclusión de regímenes con un evento distinto a agosto 2024.
**Reversibilidad:** total — es una subsección aditiva que puede excluirse sin afectar nada.

## D4 — Fuente canónica de resultados (decisión del autor, ya ejecutada)
**Decisión:** todas las tablas y figuras del documento final provienen del pipeline
reproducible auditado (RC1, rama reestructura-dic2025), no del borrador.
**Justificación:** los números del borrador provienen de código no recuperable; los de RC1
son reproducibles de punta a punta, con prueba anti-fuga automática como gate del pipeline.
Las diferencias (~7–11% en exógenos T=20) no alteran ninguna conclusión cualitativa.

## D5 — Reporte del hallazgo de exógenas en T=20
**Decisión:** se reporta como "ventaja direccional unánime (DM>0 en 10/10 semillas) con
significancia frágil al 5% (mediana p=0.062; 4/10 semillas p<0.05)". El ensemble
multi-semilla (p=0.049) se reporta como resultado secundario, no como conclusión principal.
**Justificación:** evita sobre-afirmación apoyada en una sola semilla o en un p-valor
marginal; convierte la fragilidad en hallazgo metodológico de la tesis.

## D6 — Estacionalidad
**Decisión:** la regla de activación (gate) con Kruskal–Wallis sobre efecto día-de-semana
no encontró evidencia (p=0.302 > 0.10); los términos m=5 no se activan y SARIMAX≡ARIMAX
en la muestra. Se reporta el veredicto del gate en lugar de filas duplicadas.
**Justificación:** aplicación literal de la regla pre-registrada en el contrato del proyecto.

## D7 — Mincer–Zarnowitz adopta L=h−1 en el kernel HAC (2026-06-16)
**Decisión:** la regresión MZ (y_t = α + β·ŷ_t + ε, en niveles) usa errores HAC con
maxlags = h−1, en lugar del maxlags=1 fijo previo.
**Justificación:** coherencia con el kernel del Diebold–Mariano (que ya usa L=h−1). Los
retornos acumulados solapados inducen autocorrelación MA(h−1) en el residuo de la MZ;
un lag fijo de 1 subestima la incertidumbre en horizontes largos. Para h=1 (sin solape)
maxlags=0, con lo que el HAC se reduce a robustez de heterocedasticidad.
**Impacto (verificado, no cambia ningún veredicto):** los p-valores de H0: β=1 en T+20
pasan de ≈0.0000 a ≈0.020–0.046; los de T+10 de ≈0.0000 a ≈0.008–0.017; T+5 de
≈0.001–0.004 a ≈0.008–0.018; T+1 sigue sin rechazarse (≈0.09–0.29). El veredicto
cualitativo se mantiene: β=1 se rechaza al 5% desde T+5 y no se rechaza en T+1; ningún
rechazo al 5% se invierte. En la rama limpia se conserva reports/data/tbl_MZ_core.csv
como salida MZ canónica; ninguna otra tabla se ve afectada.
**Reversibilidad:** total — es un cambio de un argumento del estimador de covarianza.

---

## Adenda 2026-06-11 — actualización de D5 tras corrección del pipeline

El bloque de robustez 2025 (D3) destapó un bug de alineación en las salidas LSTM
(y_true desplazado seq_len=40 filas; detalle en logs/bitacora.md 2026-06-11). Las
predicciones y el entrenamiento eran correctos; la verdad contra la que se evaluaban
las filas LSTM no. Todo el universo LSTM fue regenerado (set vigente: **RC2**).

**Impacto sobre D5 (números corregidos, LSTM_FULL T=20 vs RW):** la dirección se
mantiene (DM>0 en 10/10 semillas; RMSE bajo RW en 10/10), pero la significancia
desaparece: mediana p=0.190 (antes 0.062), 1/10 semillas con p<0.05 (antes 4/10),
ensemble p=0.136 (antes 0.049). [Números con kernel de Bartlett en la varianza
HAC, adoptado 2026-06-12 tras detectar varianza rectangular negativa en el bloque
2025; afinados 2026-06-16 con la convención HAC unificada del DM, ver
docs/cierre_auditoria_2026-06-16.md — el cambio es <0.0025 y no altera veredictos.]
En el OOS 2024
corregido la única celda DM-HLN significativa al 5% es LSTM_SENT T=5 vs RW
(p=0.048, marginal; con ~36 contrastes
es compatible con ruido de comparaciones múltiples y no se propone como hallazgo).

**Redacción propuesta para D5 (a ratificar):** "ventaja consistente en magnitud
(10/10 semillas con RMSE bajo RW) pero estadísticamente no significativa; sin
evidencia al 5% de aporte predictivo incremental de las exógenas en T=20. El
empate técnico es la conclusión transversal del OOS 2024."

D1, D2, D3 y D6 no se afectan. D4 se refuerza: la fuente canónica reproducible y
sus gates automáticos fueron precisamente lo que permitió detectar y corregir el
error (el guard de consistencia de y_true queda permanente en el validador).

---

**RC2 sellado: y_true verificado contra fuente primaria el 2026-06-12, commit
a06f473** (tests/test_ytrue_sanity.py: retorno acumulado recomputado a mano desde
data/raw/data.csv crudo en 10 fechas aleatorias del OOS 2024 + 5 del bloque 2025 por
horizonte, contrastado contra los 56 archivos de predicciones de los 7 modelos,
tolerancia 1e-10; integrado a la suite permanente del gate).

---

## Nota de nomenclatura (2026-06-16)
Se adopta **RC2.1** como etiqueta única del set sellado de resultados, alineada con
`config.yaml` (version 2.1.0) y los entregables externos. "RC2" en las bitácoras y
entradas fechadas de junio-2026 designa exactamente el mismo set; no se reescriben
los registros históricos. El tag git anotado `RC2.1` marca el commit sellado.

---

## D8 — Reconstruccion de procedencia y nueva corrida (2026-07-16)

El dataset RC2.1 queda conservado como evidencia historica, pero deja de ser la
base canonica para actualizar la tesis. La auditoria forense concluyo que su
`Target_Price` reproduce QQQ ajustado, no el nivel del NASDAQ-100; su
`VIX_Close` corresponde en realidad a la apertura del VIX por un error de
etiquetado; y el origen del sentimiento no puede demostrarse, con una cola de
75 valores constantes desde 2025-01-02.

Se adopta como nueva base el dataset reconstruido desde NASDAQ-100 oficial
(FRED/Nasdaq, cierre), VIX oficial (Cboe, cierre) y tono GDELT del alcance
`nasdaq_market`. Tiene 2.558 filas y SHA-256
`abf82d900314ce09cd00113785804e9f6742c38dd0867880ae9620c2a12f92ce`.
La fecha 2017-08-29 se excluye por falta de tono, sin imputacion.

La corrida completa desde arboles vacios termino con `RUN_ALL_OK` en 38,0
minutos y gate 17/17. Dos ejecuciones independientes dieron 512/512 predicciones
y 14/14 tablas comparables identicas byte por byte. Los resultados sellados
muestran para LSTM_FULL vs RW en h=20: RMSE 0,03983 vs 0,04368, pero DM-HLN
p=0,1811; mediana multi-semilla p=0,2225, 1/10 semillas bajo 0,05 y ensemble
p=0,2101. En robustez 2025, RW minimiza RMSE en h=5, 10 y 20.

**Decision:** ninguna cifra ni conclusion de RC2.1 se trasladara al Word. El
capitulo de resultados, resumen, abstract, conclusiones, tablas y figuras deben
actualizarse exclusivamente desde `data/manifests/reproduction_results.json` y
los artefactos sellados de esta corrida.
