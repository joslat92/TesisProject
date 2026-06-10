import sys
import os
import pandas as pd
import yaml
import numpy as np

# Añadir raíz al path para importar desde src.core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.core.features import compute_log_returns, create_exogenous_lags

def generate_multi_horizon_targets(df, price_col='Target_Price', horizons=[1, 5, 10, 20]):
    """
    Calcula el retorno logarítmico acumulado desplazado para evitar leakage.
    target_R1 = log(P_{t+1}) - log(P_t)
    """
    df = df.copy()
    # Asegurar que tenemos log_price para el cálculo
    df['log_price'] = np.log(df[price_col])
    
    for h in horizons:
        # El target es el retorno futuro: log(Precio futuro) - log(Precio actual)
        # Shift(-h) trae el valor del futuro a la fila actual 't'
        df[f'target_R{h}'] = df['log_price'].shift(-h) - df['log_price']
        
    # Eliminamos las últimas filas que quedan con NaN por el desplazamiento futuro
    return df.dropna()

def run_features(config_path):
    print("--- Stage 05: Ingeniería de Variables ---")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    input_path = os.path.join(config['paths']['data_processed'], "dataset_canonico.parquet")
    # Cambiamos el nombre de salida a 'dataset_features' para que 10_baselines lo encuentre
    output_path = os.path.join(config['paths']['data_processed'], "dataset_features.parquet")
    
    if not os.path.exists(input_path):
        print(f"[ERROR] No se encuentra el archivo de entrada: {input_path}")
        return

    df = pd.read_parquet(input_path)
    
    # 1. Retornos logarítmicos actuales
    df = compute_log_returns(df)
    
    # 2. Lags de exógenas (VIX y Sentiment)
    exog_cols = config['schema']['exog_cols']
    lags = [1, 5, 10] 
    df = create_exogenous_lags(df, exog_cols, lags)
    
    # 3. GENERAR TARGETS (Lo que faltaba)
    # Usamos 'Target_Price' que es el nombre que confirmamos que tiene tu parquet
    df = generate_multi_horizon_targets(df, price_col='Target_Price')
    
    df.to_parquet(output_path, index=False)
    print(f"[OK] Matriz de features con targets guardada en: {output_path}")
    print(f"Columnas generadas: {[col for col in df.columns if 'target' in col]}")

if __name__ == "__main__":
    run_features("config.yaml")
