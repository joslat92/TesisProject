import numpy as np
import pandas as pd
from scipy import stats

def pesaran_timmermann(y_true, y_pred):
    """
    Test de Pesaran–Timmermann (1992) de acierto direccional.

    H0: la dirección predicha es independiente de la realizada (sin habilidad
    direccional). Estadístico = (P̂ − P*) / √(var(P̂) − var(P*)) ~ N(0,1);
    p-valor a UNA cola (habilidad > azar). Dirección definida como retorno > 0.

    ⚠️ ADVERTENCIA (leer antes de citar): el test asume observaciones
    independientes. Con targets de ventanas SOLAPADAS (h>1, retornos
    acumulados diarios) los aciertos consecutivos están autocorrelacionados
    por construcción, lo que INFLA el estadístico y encoge el p-valor. Para
    h>1 el resultado es INDICATIVO, no inferencia formal; la inferencia
    formal del proyecto es el DM-HLN con HAC. No aplicar a RW (ŷ=0 ⇒
    dirección indefinida y varianza degenerada).

    Devuelve (pt_stat, p_value, p_hat, p_star):
    p_hat = tasa de acierto observada; p_star = esperada bajo independencia.
    Si la varianza degenera (predicciones de un solo signo y n chico),
    devuelve (nan, nan, p_hat, p_star).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)

    up_true = y_true > 0
    up_pred = y_pred > 0

    p_hat = np.mean(up_true == up_pred)
    py = np.mean(up_true)
    px = np.mean(up_pred)
    p_star = py * px + (1 - py) * (1 - px)

    var_phat = p_star * (1 - p_star) / n
    var_pstar = ((2 * px - 1) ** 2 * py * (1 - py) / n
                 + (2 * py - 1) ** 2 * px * (1 - px) / n
                 + 4 * py * px * (1 - py) * (1 - px) / n ** 2)

    denom = var_phat - var_pstar
    if denom <= 1e-16:
        return np.nan, np.nan, p_hat, p_star

    pt_stat = (p_hat - p_star) / np.sqrt(denom)
    p_value = 1 - stats.norm.cdf(pt_stat)
    return pt_stat, p_value, p_hat, p_star

def calculate_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    smape = 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    
    # MDA (Mean Directional Accuracy)
    actual_diff = np.diff(y_true) > 0
    pred_diff = (y_pred[1:] - y_true[:-1]) > 0
    mda = np.mean(actual_diff == pred_diff)
    
    return {'RMSE': rmse, 'MAE': mae, 'sMAPE': smape, 'MDA': mda}
