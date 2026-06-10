import pandas as pd
import numpy as np

def compute_log_returns(df, target_col='Target_Price'):
    """
    Calcula log-precios y retornos logarítmicos para asegurar estacionariedad.
    Basado en el marco teórico de la tesis (Cap. 4).
    """
    df = df.copy()
    # logP = ln(Price)
    df['logP'] = np.log(df[target_col])
    # ret_log = ln(P_t) - ln(P_{t-1})
    df['ret_log'] = df['logP'].diff()
    return df

def create_exogenous_lags(df, exog_cols, lags_list):
    """
    Genera rezagos para VIX y Sentimiento.
    REGLA DE ORO: Solo lags positivos para evitar 'Future Leakage'.
    """
    df = df.copy()
    for col in exog_cols:
        if col in df.columns:
            for lag in lags_list:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df

def prepare_horizon_target(df, target_col, horizon=1):
    """
    Crea la variable objetivo desplazada según el horizonte T+h.
    """
    df = df.copy()
    # Para pronóstico en nivel T+h
    df[f'target_T{horizon}'] = df[target_col].shift(-horizon)
    return df
