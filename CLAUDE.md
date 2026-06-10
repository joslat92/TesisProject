# CLAUDE.md — Contexto del Proyecto de Tesis

## Qué es este proyecto

Tesis de maestría (MMACC, Universidad del Rosario) de José Luis Agudelo: comparación de
modelos clásicos de series de tiempo (RW, ARIMA, ARIMAX, SARIMA, SARIMAX) contra LSTM
(plain, +Sentiment, +Sentiment+VIX) para pronosticar retornos acumulados del NASDAQ-100
en horizontes T ∈ {1, 5, 10, 20}, evaluando el aporte de exógenas (VIX y sentimiento
GDELT). Protocolo: OOS 2024 + walk-forward de 12 bloques mensuales, métricas RMSE/MAE/MDA,
tests Diebold–Mariano (HAC, L=h−1, vs RW y vs SARIMAX) y Mincer–Zarnowitz en niveles.

El proyecto estuvo pausado ~1 año por motivos médicos del autor. Se retomó en junio 2026.

## Historia y estado de las versiones (IMPORTANTE)

1. **Versión T+1** (Documento Tesis 4.docx, ago-2025): estudio cerrado y coherente.
   T+1 en nivel, 6 modelos + RW1. Conclusión: empate técnico, SARIMAX_BOTH mejor RMSE OOS.
2. **Versión multi-horizonte** (doc "Numeros normales"): expansión con T∈{1,5,10,20} sobre
   retornos. Tablas completas de 8 modelos (incluye SARIMAX y LSTM con exógenas), pero
   resumen/abstract desactualizados y discusión a medio redactar. **Es la versión que se
   va a terminar** (pendiente de confirmar con el director).
3. **Historial de git (commits hasta e7ac2d3)**: la base de código ANTERIOR a la
   reestructuración. Probablemente contiene el código que generó las tablas de la versión
   multi-horizonte (SARIMAX, ablations, LSTM tuned). NO BORRAR NI REESCRIBIR ese historial.
4. **Árbol de trabajo actual (29-12-2025)**: reestructuración limpia y modular del
   pipeline, SIN commitear. Corrida preliminar solo con RW/ARIMA/LSTM. ⚠️ Sus resultados
   son SOSPECHOSOS (ver "Bug conocido") y sus figuras no son las canónicas de la tesis.

## Bug conocido (prioridad de auditoría)

En `reports/data/metrics_OOS.csv` del árbol actual, ARIMA supera a RW por márgenes
irreales en T≥5 (T=20: RMSE 0.0167 vs 0.044, MDA 0.89, DM p≈0.000). Para retornos
acumulados eso es atípico y sugiere FUGA DE INFORMACIÓN en el pronóstico multi-paso
(posible re-anclaje con valores reales dentro del horizonte). Auditar
`src/stages/11_train_arima.py` y `src/stages/20_evaluate_stats.py`: el pronóstico de
y_t(h) debe usar SOLO información disponible hasta t. Referencia de cordura: en la tesis,
todos los modelos tienen RMSE≈0.011 en T=1 y arimax≈0.039 en T=20, con diferencias
estrechas entre familias.

## Especificación canónica (fuente de la verdad)

- `contrato.docx` — contrato de diseño: esquema de predicciones (Date, h, model,
  y_true_ret, y_pred_ret, y_true_level, y_pred_level [+block en WF]), naming
  (preds_T{h}_{model}.csv / preds_T{h}_{model}_block{b}.csv), splits (IS→2023-12-29,
  OOS 2024, WF 12 bloques), escalado train-only sin fuga, DM HAC L=h−1, MZ, regímenes.
- `Contrato_Control_actualizado.xlsx` — libro de control (SSOT) con errores conocidos.
- `pasos replicacion 220925 2230.xlsx` — guía de replicación etapa por etapa.
- HPs LSTM canónicos (los del documento de tesis, salvo que el director indique otro):
  window=40, hidden=64, dropout=0.2, Adam lr=0.001, epochs=30, batch=64, seed=42,
  early-stopping con val 20%.

## Reglas de trabajo

- NUNCA reescribir el historial de git ni hacer force-push. La rama main debe seguir
  apuntando a la versión pre-reestructuración hasta que se valide la nueva.
- Trabajar la reestructuración en la rama `reestructura-dic2025`.
- Commit + push al final de CADA sesión de trabajo.
- Todo cambio al pipeline debe respetar el contrato (contrato.docx §8 y §11).
- Sin fuga de información: scalers y selección de hiperparámetros solo con IS o bloque
  WF previo; exógenas siempre rezagadas.
- Sesiones de ~2 horas diarias. Mantener bitácora en `logs/bitacora.md` (qué se hizo,
  qué sigue, decisiones tomadas).
- El plan completo de 12 semanas está en `Plan_Retoma_Tesis.md` (fases 0–5).

## Prioridades inmediatas (en orden)

1. Crear `.gitignore` (.venv/, venv/, __pycache__/, *.pyc) y commitear el árbol actual
   en la rama `reestructura-dic2025`; push al remoto.
2. Explorar el historial de git: localizar el código que generó las corridas SARIMAX y
   LSTM con exógenas de la tesis (buscar en commits previos a e7ac2d3). Extraerlo a una
   carpeta de referencia (`legacy/`) usando git worktree o checkout selectivo, sin tocar main.
3. Auditar y corregir el bug de fuga en ARIMA multi-paso.
4. Implementar el validador de contrato (contrato.docx §11) como gate del pipeline.
5. Completar stages SARIMAX y variantes LSTM con exógenas en la estructura nueva.
6. Re-correr OOS + WF completo y validar contra las tablas de la tesis.

## Datos

- `data/data.csv` (= data/raw/data.csv): 2,561 obs diarias, 2015-02-17 a 2025-04-22.
  Columnas: Date, Target_Price, Sentiment_GDELT, VIX_Close, logP, ret_log.
- El estudio principal usa OOS=2024. Los datos de ene–abr 2025 (incluyen el shock de
  volatilidad de abril) quedan reservados para un bloque opcional de robustez.
