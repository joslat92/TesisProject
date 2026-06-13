"""
Test de Diebold–Mariano con y sin corrección de muestra pequeña HLN
(Harvey, Leybourne & Newbold, 1997).

Convención del proyecto (contrato): pérdida cuadrática, d_t = e²_bench − e²_chall
(d>0 ⇒ el challenger pierde menos que el benchmark), varianza HAC tipo
Newey–West con lag de truncamiento L = h−1.

- DM clásico: estadístico ~ N(0,1), p-valor a dos colas con la normal.
- DM-HLN: estadístico multiplicado por √k con k = (n + 1 − 2h + h(h−1)/n) / n,
  y p-valor a dos colas con t de Student de n−1 grados de libertad. Corrige el
  sobre-rechazo del DM clásico en muestras pequeñas y horizontes largos.

Nota histórica del repo: la implementación embebida en 20_evaluate_stats
(RC1) ya aplicaba k y t(n−1), de modo que los p-valores publicados en RC1
eran de facto los HLN. Este módulo separa ambas versiones con nombre propio.
"""
import numpy as np
from scipy import stats


def diebold_mariano(y_true, y_pred_bench, y_pred_chall, h=1):
    """
    DM clásico y DM-HLN para pérdida cuadrática.

    Devuelve (dm_stat, p_value, dm_hln, p_hln):
    - dm_stat, p_value: estadístico clásico y p-valor N(0,1) dos colas.
    - dm_hln, p_hln: estadístico corregido HLN y p-valor t(n−1) dos colas.
    Si la varianza HAC degenera (≤0), devuelve (0, 1, 0, 1) — sin evidencia.
    """
    y_true = np.asarray(y_true, dtype=float)
    e_bench = (y_true - np.asarray(y_pred_bench, dtype=float)) ** 2
    e_chall = (y_true - np.asarray(y_pred_chall, dtype=float)) ** 2
    d = e_bench - e_chall

    n = len(d)
    mean_d = np.mean(d)

    # Varianza HAC Newey-West con kernel de BARTLETT (w_k = 1 − k/h),
    # truncamiento L = h−1. El kernel rectangular del DM(1995) original
    # puede arrojar varianza NEGATIVA (ocurrió con n=55, h=20 en el bloque
    # 2025 ⇒ DM espurio = 0.0); Bartlett garantiza semidefinida positiva.
    gamma_0 = np.var(d)
    gamma_sum = 0.0
    for lag in range(1, h):
        w = 1.0 - lag / h
        gamma_sum += w * np.cov(d[lag:], d[:-lag])[0][1]
    var_d = (gamma_0 + 2 * gamma_sum) / n

    if var_d <= 1e-16:
        # Caso degenerado genuino: predicciones idénticas (d constante ≈ 0,
        # p.ej. ARIMAX ≡ SARIMAX con gate estacional apagado) ⇒ sin evidencia.
        return 0.0, 1.0, 0.0, 1.0

    dm_stat = mean_d / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    # Corrección HLN (1997): factor sobre el estadístico + t de Student n−1
    k = (n + 1 - 2 * h + h * (h - 1) / n) / n
    dm_hln = dm_stat * np.sqrt(k)
    p_hln = 2 * (1 - stats.t.cdf(np.abs(dm_hln), df=n - 1))

    return dm_stat, p_value, dm_hln, p_hln
