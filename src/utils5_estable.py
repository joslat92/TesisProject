# src/utils5_estable.py
"""
Utilidades comunes para la suite de predicción NASDAQ-100.

Todas las funciones expuestas aquí están testeadas y documentadas.
"""

from __future__ import annotations
import json
import typing as t
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------------------------------------
# ESCALADORES
# ----------------------------------------------------------------------


def fit_minmax(
    series: pd.Series | np.ndarray, feature_range: tuple[float, float] = (0, 1)
) -> MinMaxScaler:
    """
    Ajusta y devuelve un MinMaxScaler 1-D.

    Parameters
    ----------
    series : pd.Series | np.ndarray
        Vector con la señal a escalar.
    feature_range : tuple[float, float], default=(0, 1)
        Rango de salida deseado.

    Returns
    -------
    MinMaxScaler
        Objeto ya ajustado.
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(
        series.reshape(-1, 1)
        if isinstance(series, np.ndarray)
        else series.values.reshape(-1, 1)
    )
    return scaler


# ----------------------------------------------------------------------
# WINDOWS PARA SERIES DE TIEMPO
# ----------------------------------------------------------------------


def make_windows(array: np.ndarray, window: int) -> np.ndarray:
    """Genera ventanas deslizantes de tamaño fijo."""
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return np.stack([array[i : i + window] for i in range(len(array) - window)])


# ----------------------------------------------------------------------
# METADATOS DEL MODELO
# ----------------------------------------------------------------------


def save_metadata(
    path: str,
    features_used: list[str],
    lookback: int,
    target_column: str,
    scaler_path: str,
    model_path: str,
) -> None:
    """Graba un archivo JSON con el resumen del modelo."""
    meta = {
        "features_used": features_used,
        "lookback": lookback,
        "target_column": target_column,
        "scaler_path": scaler_path,
        "model_path": model_path,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)


# ----------------------------------------------------------------------
# AÑADE SOLO LO NECESARIO 🔽
# ----------------------------------------------------------------------
