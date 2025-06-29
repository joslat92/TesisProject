"""
evaluate_metrics.py
-------------------
Evalúa las predicciones de cada modelo guardadas como CSV dentro de la carpeta
`outputs/`.  Cada archivo .csv debe contener las columnas:

    y_true, y_pred

El script recorre todos los CSV, calcula MAE, RMSE, R² y precisión direccional,
muestra los resultados por modelo y finalmente un resumen ordenado por MAE.

Autor : Jose_agudelo
Fecha : 2025-06-28
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ----------------------------------------------------------------------
# Métricas auxiliares
# ----------------------------------------------------------------------
def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Precisión direccional (Direction Accuracy – DA).
    Compara el signo del cambio entre pasos consecutivos.

    Returns
    -------
    float
        Proporción de aciertos en la dirección del movimiento.
    """
    return np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))


# ----------------------------------------------------------------------
# Funciones principales
# ----------------------------------------------------------------------
def evaluate_model(file_path: str, model_name: str) -> dict:
    """
    Calcula métricas para un único archivo de predicciones.

    Parameters
    ----------
    file_path : str
        Ruta al CSV con columnas `y_true` y `y_pred`.
    model_name : str
        Nombre que se mostrará en pantalla y en el diccionario de resultados.

    Returns
    -------
    dict
        Diccionario con las métricas calculadas.
    """
    df = pd.read_csv(file_path)

    # Validación básica
    required_cols = {'y_true', 'y_pred'}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"{file_path} debe contener las columnas {required_cols}"
        )

    y_true = df['y_true'].values
    y_pred = df['y_pred'].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    da = directional_accuracy(y_true, y_pred)

    # Log en consola
    print(f"\n📊  Resultados para {model_name}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")
    print(f"  DA   : {da:.4f}")

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "DA": da,
    }


def evaluate_all(models_dir: str = 'outputs') -> None:
    """
    Busca todos los archivos *.csv en `models_dir`, evalúa y muestra
    un resumen ordenado por MAE (de menor a mayor).

    Parameters
    ----------
    models_dir : str, default 'outputs'
        Carpeta donde se encuentran los CSV de las predicciones.
    """
    results = []

    for file in os.listdir(models_dir):
        if file.lower().endswith('.csv'):
            file_path = os.path.join(models_dir, file)
            model_name = os.path.splitext(file)[0]   # quita la extensión
            results.append(evaluate_model(file_path, model_name))

    if not results:
        print("  No se encontraron archivos CSV en la carpeta especificada.")
        return

    df_results = (
        pd.DataFrame(results)
        .sort_values(by='MAE')
        .reset_index(drop=True)
    )

    print("\n========================== RESUMEN ==========================")
    print(df_results.to_string(index=False, float_format="{:.4f}".format))


# ----------------------------------------------------------------------
# Ejecución directa
# ----------------------------------------------------------------------
if __name__ == "__main__":
    evaluate_all()
