# src/utils5_estable.py
"""
Utilidades comunes para la suite de predicción NASDAQ-100.

Todas las funciones expuestas aquí están testeadas y documentadas.
"""

from __future__ import annotations
import json
from pathlib import Path
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
# UTILIDADES DE VALIDACIÓN CRUZADA
# ----------------------------------------------------------------------

def expanding_cv_splits(y, n_folds=10):
    """
    Genera los índices para una validación cruzada de ventana expandible.
    """
    n_samples = len(y)
    if n_folds >= n_samples:
        raise ValueError("El número de folds debe ser menor que el número de muestras.")

    initial_train_size = n_samples - n_folds
    
    for i in range(n_folds):
        train_indices = range(initial_train_size + i)
        test_indices = range(initial_train_size + i, initial_train_size + i + 1)
        yield list(train_indices), list(test_indices)

def save_fold_preds(y_true: pd.Series, y_pred: pd.Series, fold_n: int, out_dir: Path) -> pd.DataFrame:
    """
    Combina los resultados de un fold en un DataFrame.
    """
    fold_df = pd.DataFrame({
        'Date': y_true.index,
        'y_true': y_true.values,
        'y_pred': y_pred.values,
        'fold': fold_n
    })
    return fold_df